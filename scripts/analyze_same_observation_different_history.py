from __future__ import annotations

import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    output_dir: str = "outputs/memory_ablation/same_obs_diff_history"
    num_histories: int = 4


def _load_real_batch(config: _config.TrainConfig):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id in (None, "fake"):
        raise ValueError("Same-observation analysis requires a real dataset; fake data is not allowed.")
    loader = _data_loader.create_data_loader(config, sharding=None, shuffle=False, num_batches=1)
    return next(iter(loader))


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    if not getattr(config.model, "temporal_memory_enabled", False):
        raise ValueError("Same-observation analysis requires a memory-enabled temporal config.")
    if config.model.memory_injection_strategy == "none":
        raise ValueError("No-memory baseline has no hidden-state intervention to analyze.")

    ckpt = pathlib.Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    batch = _load_real_batch(config)
    model = config.model.create(jax.random.key(config.seed))
    params_dir = ckpt / "params" if (ckpt / "params").exists() else ckpt
    model = model.load(_model.restore_params(params_dir, dtype=jnp.bfloat16), remove_extra_params=True)

    # Use real observations from the first temporal sample and intervene with hidden states from real rollout chunks.
    debug = model.debug_temporal_forward(jax.random.key(config.seed + 1), batch, train=False)
    hidden = np.asarray(jax.device_get(debug["final_hidden"]))
    if hidden.shape[0] < args.num_histories:
        raise ValueError(f"Need at least {args.num_histories} real hidden states, got {hidden.shape[0]}.")

    sample = jax.tree.map(lambda x: x[:1], batch)
    obs = model._prepare_temporal_observation(  # noqa: SLF001
        jax.tree.map(lambda x: x[:, 0], sample["chunks"]["images"]),
        jax.tree.map(lambda x: x[:, 0], sample["chunks"]["image_masks"]),
        sample["chunks"]["state"][:, 0],
        sample["prompt"],
        None,
        train=False,
    )
    actions = []
    for i in range(args.num_histories):
        h = jnp.asarray(hidden[i : i + 1])
        pred, _ = model.sample_actions_temporal(
            jax.random.key(config.seed + 100 + i),
            _model.preprocess_observation(None, obs, train=False),
            h,
            delta_t=jnp.asarray([config.model.default_delta_t], dtype=jnp.float32),
            num_steps=10,
        )
        actions.append(np.asarray(jax.device_get(pred[0]), dtype=np.float32))

    action_stack = np.stack(actions, axis=0)
    distances = np.linalg.norm(action_stack[:, None] - action_stack[None, :], axis=(-1, -2))
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predicted_actions.npy", action_stack)
    with (out_dir / "results.json").open("w") as f:
        json.dump({"config_name": args.config_name, "checkpoint_dir": args.checkpoint_dir}, f, indent=2)
    with (out_dir / "action_distance.csv").open("w") as f:
        f.write("history_i,history_j,distance\n")
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                f.write(f"{i},{j},{float(distances[i, j])}\n")
    (out_dir / "summary.md").write_text(
        f"# Same Observation Different History\n\nMean pairwise action distance: {float(distances.mean()):.6f}\n"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
