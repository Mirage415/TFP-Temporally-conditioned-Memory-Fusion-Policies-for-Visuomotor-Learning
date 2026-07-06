import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _get_wandb_settings() -> wandb.Settings:
    init_timeout_secs = int(os.environ.get("WANDB_INIT_TIMEOUT_SECS", "1800"))
    return wandb.Settings(init_timeout=init_timeout_secs)


def _init_wandb_with_offline_fallback(train_config: _config.TrainConfig, **kwargs) -> bool:
    """Returns True if online init succeeded, False if offline fallback was used."""
    try:
        wandb.init(**kwargs)
        return True
    except Exception as exc:
        logging.error("wandb online init failed, falling back to offline mode: %s", exc)
        wandb.init(
            mode="offline",
            name=train_config.exp_name,
            config=dataclasses.asdict(train_config),
            project=train_config.project_name,
            settings=_get_wandb_settings(),
        )
        return False


class TemporalHiddenBank:
    """Host-side detached hidden-state bank keyed by episode id."""

    def __init__(self, hidden_dim: int):
        self._hidden_dim = hidden_dim
        self._states: dict[int, np.ndarray] = {}

    @property
    def size(self) -> int:
        return len(self._states)

    def state_dict(self) -> dict[str, np.ndarray]:
        episode_ids = np.asarray(sorted(self._states.keys()), dtype=np.int32)
        if len(episode_ids) == 0:
            hidden_states = np.zeros((0, self._hidden_dim), dtype=np.float32)
        else:
            hidden_states = np.stack([self._states[int(episode_id)] for episode_id in episode_ids], axis=0).astype(
                np.float32
            )
        return {
            "episode_ids": episode_ids,
            "hidden_states": hidden_states,
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        episode_ids = np.asarray(state["episode_ids"], dtype=np.int32)
        hidden_states = np.asarray(state["hidden_states"], dtype=np.float32)
        if hidden_states.shape != (len(episode_ids), self._hidden_dim):
            raise ValueError(
                f"Hidden-bank checkpoint shape mismatch: got hidden_states {hidden_states.shape}, "
                f"expected ({len(episode_ids)}, {self._hidden_dim})."
            )
        self._states = {
            int(episode_id): np.asarray(hidden, dtype=np.float32).copy()
            for episode_id, hidden in zip(episode_ids, hidden_states, strict=True)
        }

    def load(self, episode_ids: np.ndarray, start_chunk_ids: np.ndarray) -> np.ndarray:
        hidden = np.zeros((len(episode_ids), self._hidden_dim), dtype=np.float32)
        for row, (episode_id, start_chunk_id) in enumerate(zip(episode_ids, start_chunk_ids, strict=True)):
            episode_id = int(episode_id)
            start_chunk_id = int(start_chunk_id)
            if episode_id < 0:
                continue
            if start_chunk_id == 0:
                continue
            if episode_id not in self._states:
                raise KeyError(
                    f"Missing hidden-bank entry for episode_id={episode_id}, start_chunk_id={start_chunk_id}. "
                    "Temporal samples must be consumed in order."
                )
            state = self._states[episode_id]
            if state.shape != (self._hidden_dim,):
                raise ValueError(
                    f"Hidden-bank shape mismatch for episode_id={episode_id}: got {state.shape}, "
                    f"expected ({self._hidden_dim},)."
                )
            hidden[row] = state
        return hidden

    def store(self, episode_ids: np.ndarray, final_hidden: np.ndarray) -> None:
        if final_hidden.shape != (len(episode_ids), self._hidden_dim):
            raise ValueError(
                f"Final hidden shape mismatch: got {final_hidden.shape}, expected ({len(episode_ids)}, {self._hidden_dim})."
            )
        for episode_id, hidden in zip(episode_ids, final_hidden, strict=True):
            episode_id = int(episode_id)
            if episode_id < 0:
                continue
            self._states[episode_id] = np.asarray(hidden, dtype=np.float32).copy()


def _temporal_batch_host_metadata(batch: dict, expected_num_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    episode_ids = np.asarray(jax.device_get(batch["episode_id"]), dtype=np.int32)
    start_chunk_ids = np.asarray(jax.device_get(batch["start_chunk_id"]), dtype=np.int32)
    chunk_ids = np.asarray(jax.device_get(batch["chunks"]["chunk_ids"]), dtype=np.int32)
    chunk_mask = np.asarray(jax.device_get(batch["chunks"]["mask"]), dtype=bool)
    row_has_chunks = np.any(chunk_mask, axis=1)
    active_episode_ids = episode_ids[row_has_chunks]

    if len(np.unique(active_episode_ids)) != len(active_episode_ids):
        raise ValueError("A temporal batch contains duplicate episode_ids; hidden-bank semantics would be ambiguous.")

    for row, (episode_id, start_chunk_id) in enumerate(zip(episode_ids, start_chunk_ids, strict=True)):
        if not row_has_chunks[row]:
            continue
        valid_chunk_ids = chunk_ids[row][chunk_mask[row]]
        if valid_chunk_ids.shape[0] == 0:
            raise ValueError(
                f"Temporal sample for episode_id={int(episode_id)} has no valid chunks."
            )
        if valid_chunk_ids.shape[0] > expected_num_chunks:
            raise ValueError(
                f"Temporal sample for episode_id={int(episode_id)} has {valid_chunk_ids.shape[0]} valid chunks; "
                f"expected at most {expected_num_chunks}."
            )
        expected_chunk_ids = np.arange(
            int(start_chunk_id),
            int(start_chunk_id) + valid_chunk_ids.shape[0],
            dtype=np.int32,
        )
        if not np.array_equal(valid_chunk_ids, expected_chunk_ids):
            raise ValueError(
                f"Temporal sample for episode_id={int(episode_id)} is not consecutive: "
                f"got {valid_chunk_ids.tolist()}, expected {expected_chunk_ids.tolist()}."
            )

    episode_ids = np.where(row_has_chunks, episode_ids, np.asarray(-1, dtype=np.int32))
    start_chunk_ids = np.where(row_has_chunks, start_chunk_ids, np.asarray(0, dtype=np.int32))

    return episode_ids, start_chunk_ids


def _temporal_batch_preview(
    episode_ids: np.ndarray,
    start_chunk_ids: np.ndarray,
    *,
    limit: int = 8,
) -> list[str]:
    pairs = [
        f"{int(episode_id)}:{int(start_chunk_id)}"
        for episode_id, start_chunk_id in zip(episode_ids, start_chunk_ids, strict=True)
        if int(episode_id) >= 0
    ]
    return pairs[:limit]


def _attach_initial_hidden(
    batch: dict,
    initial_hidden: np.ndarray,
    data_sharding: jax.sharding.Sharding,
) -> dict:
    batch_with_hidden = dict(batch)
    batch_with_hidden["initial_hidden"] = jax.make_array_from_process_local_data(data_sharding, initial_hidden)
    return batch_with_hidden


def _zero_like_param_state(params: nnx.State) -> nnx.State:
    return nnx_utils.state_map(params, nnx.Param, lambda p: p.replace(jnp.zeros_like(p.value)))


def _expand_grads_to_full_tree(full_params: nnx.State, partial_grads: nnx.State) -> nnx.State:
    full_grads = _zero_like_param_state(full_params)
    full_grads.replace_by_pure_dict(partial_grads.to_pure_dict())
    return full_grads


def _zero_like_leaf(leaf: Any) -> Any:
    if hasattr(leaf, "replace") and hasattr(leaf, "value"):
        return leaf.replace(jnp.zeros_like(leaf.value))
    return jnp.zeros_like(leaf)


def _mask_state_to_keys(state: nnx.State, keep_keys: set[Any]) -> nnx.State:
    return state.map(lambda k, v: v if k in keep_keys else _zero_like_leaf(v))


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        wandb_id_path = ckpt_dir / "wandb_id.txt"
        if wandb_id_path.exists():
            run_id = wandb_id_path.read_text().strip()
            _init_wandb_with_offline_fallback(
                config,
                id=run_id,
                resume="must",
                project=config.project_name,
                settings=_get_wandb_settings(),
            )
        else:
            logging.warning("Resuming checkpoint without wandb_id.txt; starting a new wandb run.")
            online_ok = _init_wandb_with_offline_fallback(
                config,
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
                settings=_get_wandb_settings(),
            )
            if online_ok:
                wandb_id_path.write_text(wandb.run.id)
    else:
        online_ok = _init_wandb_with_offline_fallback(
            config,
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            settings=_get_wandb_settings(),
        )
        if online_ok:
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step_temporal(
    config: _config.TrainConfig,
    active_trainable_filter: nnx.filterlib.Filter,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: dict,
) -> tuple[training_utils.TrainState, dict[str, at.Array], jax.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, temporal_batch: dict):
        if hasattr(model, "compute_temporal_loss_state_info"):
            return model.compute_temporal_loss_state_info(rng, temporal_batch, train=True)
        loss, final_hidden = model.compute_temporal_loss_and_state(rng, temporal_batch, train=True)
        return loss, (final_hidden, {})

    (loss, (final_hidden, temporal_info)), active_grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, active_trainable_filter),
        has_aux=True,
    )(model, train_rng, batch)

    params = state.params.filter(config.trainable_filter)
    active_param_keys = set(params.filter(active_trainable_filter).flat_state())
    full_grads = _expand_grads_to_full_tree(params, active_grads)
    updates, new_opt_state = state.tx.update(full_grads, state.opt_state, params)
    updates = _mask_state_to_keys(updates, active_param_keys)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(active_grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(temporal_info)
    return new_state, info, final_hidden


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    active_trainable_filter: nnx.filterlib.Filter,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: Any,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, active_trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    active_param_keys = set(params.filter(active_trainable_filter).flat_state())
    full_grads = _expand_grads_to_full_tree(params, grads)
    updates, new_opt_state = state.tx.update(full_grads, state.opt_state, params)
    updates = _mask_state_to_keys(updates, active_param_keys)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    temporal_training = getattr(config.model, "temporal_memory_enabled", False)
    if config.has_curriculum:
        logging.info(
            "Freeze curriculum enabled: freezing regexes=%s through step %d, then training the full backbone.",
            list(config.curriculum_freeze_patterns),
            config.curriculum_unfreeze_step,
        )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    default_jax_cache_dir = epath.Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser() / "jax"
    jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", str(default_jax_cache_dir))
    jax.config.update("jax_compilation_cache_dir", str(epath.Path(jax_cache_dir).expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=not temporal_training,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    if getattr(config.model, "temporal_memory_enabled", False):
        first_chunk_images = batch["chunks"]["images"]
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i, 0]) for img in first_chunk_images.values()], axis=1))
            for i in range(min(5, len(next(iter(first_chunk_images.values())))))
        ]
    else:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    hidden_bank = None
    warmup_ptrain_step = None
    if temporal_training:
        hidden_bank = TemporalHiddenBank(config.model.ltc_hidden_dim)
        logging.info(
            "Temporal TBPTT enabled: tbptt_num_chunks=%d, action_horizon=%d, effective horizon=%d chunk-steps.",
            config.model.tbptt_num_chunks,
            config.model.action_horizon,
            config.model.tbptt_num_chunks * config.model.action_horizon,
        )
        ptrain_step = jax.jit(
            functools.partial(train_step_temporal, config, config.trainable_filter),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding, data_sharding),
            donate_argnums=(1,),
        )
        if config.has_curriculum:
            warmup_ptrain_step = jax.jit(
                functools.partial(train_step_temporal, config, config.initial_trainable_filter),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=(train_state_sharding, replicated_sharding, data_sharding),
                donate_argnums=(1,),
            )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config, config.trainable_filter),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        if config.has_curriculum:
            warmup_ptrain_step = jax.jit(
                functools.partial(train_step, config, config.initial_trainable_filter),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )

    if resuming:
        if temporal_training:
            hidden_bank_template = hidden_bank.state_dict()
            try:
                train_state, restored_extras = _checkpoints.restore_state_with_extras(
                    checkpoint_manager,
                    train_state,
                    data_loader,
                    extra_items={"hidden_bank": hidden_bank_template},
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to restore temporal hidden bank from checkpoint. "
                    "Temporal resume requires a checkpoint written by the updated trainer."
                ) from exc
            hidden_bank.load_state_dict(restored_extras["hidden_bank"])
            logging.info("Restored temporal hidden bank with %d entries.", hidden_bank.size)
        else:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    ema_info = None
    try:
        log_ema_decay = float(os.environ.get("LOG_EMA_DECAY", "0.98"))
    except ValueError:
        log_ema_decay = 0.98
    for step in pbar:
        active_rows = None
        padded_rows = None
        if temporal_training:
            episode_ids, start_chunk_ids = _temporal_batch_host_metadata(batch, config.model.tbptt_num_chunks)
            active_rows = int(np.count_nonzero(episode_ids >= 0))
            padded_rows = int(len(episode_ids) - active_rows)
            initial_hidden = hidden_bank.load(episode_ids, start_chunk_ids)
            batch_for_step = _attach_initial_hidden(batch, initial_hidden, data_sharding)
            if step < start_step + 3 or step % config.log_interval == 0:
                logging.info(
                    "Temporal batch step=%d active_rows=%d padded_rows=%d head(ep:start)=%s episode_ids=%s start_chunk_ids=%s hidden_bank_entries=%d",
                    step,
                    active_rows,
                    padded_rows,
                    _temporal_batch_preview(episode_ids, start_chunk_ids),
                    episode_ids[: min(8, len(episode_ids))].tolist(),
                    start_chunk_ids[: min(8, len(start_chunk_ids))].tolist(),
                    hidden_bank.size,
                )
        else:
            batch_for_step = batch

        active_ptrain_step = ptrain_step
        if config.has_curriculum and step < config.curriculum_unfreeze_step:
            active_ptrain_step = warmup_ptrain_step
        elif config.has_curriculum and step == config.curriculum_unfreeze_step:
            logging.info("Reached curriculum boundary at step %d; unfreezing the VLA backbone.", step)

        with sharding.set_mesh(mesh):
            if temporal_training:
                train_state, info, final_hidden = active_ptrain_step(train_rng, train_state, batch_for_step)
            else:
                train_state, info = active_ptrain_step(train_rng, train_state, batch_for_step)
        infos.append(info)
        if temporal_training:
            final_hidden_host = np.asarray(jax.device_get(final_hidden), dtype=np.float32)
            hidden_bank.store(episode_ids, final_hidden_host)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info = {k: float(v) for k, v in reduced_info.items()}
            if ema_info is None:
                ema_info = dict(reduced_info)
            else:
                for key, value in reduced_info.items():
                    if key in ema_info:
                        ema_info[key] = log_ema_decay * ema_info[key] + (1 - log_ema_decay) * value
                    else:
                        ema_info[key] = value
            display_info = ema_info if ema_info is not None else reduced_info
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in display_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb_payload = dict(reduced_info)
            if ema_info is not None:
                wandb_payload.update({f"ema/{k}": v for k, v in ema_info.items()})
            wandb.log(wandb_payload, step=step)
            if temporal_training:
                wandb.log(
                    {
                        "hidden_bank_entries": hidden_bank.size,
                        "temporal_active_rows": active_rows if active_rows is not None else 0,
                        "temporal_padded_rows": padded_rows if padded_rows is not None else 0,
                    },
                    step=step,
                )
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            extra_items = None
            if temporal_training:
                extra_items = {"hidden_bank": hidden_bank.state_dict()}
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                extra_items=extra_items,
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
