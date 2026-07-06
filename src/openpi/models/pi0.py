import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.temporal_memory as temporal_memory
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.asarray(mask_ar, dtype=jnp.bool_)
    # Keep cumsum on the original 1D AR template when possible. Broadcasting first
    # makes XLA constant-fold a [B, N] reduce_window and recompiles for each batch size.
    if mask_ar.ndim == 1:
        cumsum = jnp.broadcast_to(jnp.cumsum(mask_ar, axis=0)[None, :], input_mask.shape)
    else:
        cumsum = jnp.cumsum(jnp.broadcast_to(mask_ar, input_mask.shape), axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.temporal_memory_enabled = config.temporal_memory_enabled
        self.temporal_action_loss = config.temporal_action_loss
        self.tbptt_num_chunks = config.tbptt_num_chunks
        self.memory_token_count = config.memory_token_count
        self.memory_cross_attn_layers = config.memory_cross_attn_layers
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.action_expert_depth = action_expert_config.depth
        self.action_expert_width = action_expert_config.width
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                enable_memory_cross_attn=config.temporal_memory_enabled and config.memory_cross_attn_layers > 0,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
            memory_token_len=config.memory_token_count if config.temporal_memory_enabled else 0,
            memory_cross_attn_layers=config.memory_cross_attn_layers if config.temporal_memory_enabled else 0,
        )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # The diffusion action decoder is always part of pi0/pi05, including temporal-memory variants.
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        if self.temporal_memory_enabled:
            fake_image = next(iter(config.fake_obs().images.values()))
            fake_image_tokens, _ = self.PaliGemma.img(fake_image, train=False)
            self.ltc_sensory_size = fake_image_tokens.shape[1] * len(_model.IMAGE_KEYS) + 1
            self.state_proj = nnx.Linear(config.action_dim, paligemma_config.width, rngs=rngs)
            self.ltc_encoder = temporal_memory.LTCEncoder(
                paligemma_config.width,
                config.ltc_hidden_dim,
                self.ltc_sensory_size,
                ode_unfolds=config.ltc_ode_unfolds,
                rngs=rngs,
            )
            self.memory_token_proj = nnx.Linear(
                config.ltc_hidden_dim,
                config.memory_token_count * action_expert_config.width,
                rngs=rngs,
            )
            self.action_query_proj = nnx.Linear(config.action_horizon, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        else:
            if not config.pi05:
                self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _prepare_temporal_observation(
        self,
        image_batch: dict[str, jax.Array],
        image_masks: dict[str, jax.Array],
        state: jax.Array,
        prompt: dict[str, jax.Array],
        preprocess_rng,
        *,
        train: bool,
    ) -> _model.Observation:
        images = {}
        for name, image in image_batch.items():
            if image.dtype == jnp.uint8:
                images[name] = image.astype(jnp.float32) / 255.0 * 2.0 - 1.0
            else:
                images[name] = image.astype(jnp.float32)
        with at.disable_typechecking():
            observation = _model.Observation(
                images=images,
                image_masks=image_masks,
                state=state,
                tokenized_prompt=prompt["input_ids"],
                tokenized_prompt_mask=prompt["attention_mask"],
            )
        return _model.preprocess_observation(preprocess_rng, observation, train=train)

    def _encode_visual_tokens(
        self,
        obs: _model.Observation,
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        visual_tokens = []
        visual_masks = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            visual_tokens.append(image_tokens)
            visual_masks.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
        return visual_tokens, visual_masks

    def _build_ltc_input_sequence(
        self,
        obs: _model.Observation,
        visual_tokens: list[jax.Array],
        visual_masks: list[jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        state_token = self.state_proj(obs.state)[:, None, :]
        state_mask = jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_)
        token_sequence = jnp.concatenate([*visual_tokens, state_token], axis=1)
        token_mask = jnp.concatenate([*visual_masks, state_mask], axis=1)
        return token_sequence, token_mask

    def _embed_temporal_prefix(
        self,
        obs: _model.Observation,
        visual_tokens: list[jax.Array],
        visual_masks: list[jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        tokens = list(visual_tokens)
        input_mask = list(visual_masks)
        ar_mask = []
        for image_tokens in visual_tokens:
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [False]

        prefix_tokens = jnp.concatenate(tokens, axis=1)
        prefix_mask = jnp.concatenate(input_mask, axis=1)
        prefix_ar_mask = jnp.array(ar_mask, dtype=jnp.bool_)

        return prefix_tokens, prefix_mask, prefix_ar_mask

    def _embed_action_queries(self, batch_size: int, dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
        query_basis = jnp.broadcast_to(
            jnp.eye(self.action_horizon, dtype=dtype)[None, :, :],
            (batch_size, self.action_horizon, self.action_horizon),
        )
        query_tokens = self.action_query_proj(query_basis).astype(dtype)
        query_mask = jnp.ones((batch_size, self.action_horizon), dtype=jnp.bool_)
        query_ar_mask = jnp.array([True] + ([False] * (self.action_horizon - 1)), dtype=jnp.bool_)
        return query_tokens, query_mask, query_ar_mask

    def _temporal_cross_attn_enabled(self) -> jax.Array:
        return jnp.arange(self.action_expert_depth) >= max(
            self.action_expert_depth - self.memory_cross_attn_layers,
            0,
        )

    def _temporal_memory_from_hidden(
        self,
        hidden: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        memory_tokens = self.memory_token_proj(hidden).reshape(
            hidden.shape[0],
            self.memory_token_count,
            self.action_expert_width,
        )
        memory_mask = jnp.broadcast_to(mask[:, None], memory_tokens.shape[:2])
        return memory_tokens, memory_mask

    def _update_temporal_hidden(
        self,
        observation: _model.Observation,
        hidden_state: jax.Array,
        delta_t: jax.Array,
        *,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, list[jax.Array], list[jax.Array]]:
        visual_tokens, visual_masks = self._encode_visual_tokens(observation)
        ltc_tokens, ltc_token_mask = self._build_ltc_input_sequence(observation, visual_tokens, visual_masks)
        next_hidden = self.ltc_encoder.step(hidden_state, ltc_tokens, ltc_token_mask, delta_t, mask)
        return next_hidden, ltc_tokens, ltc_token_mask, visual_tokens, visual_masks

    def _compute_diffusion_loss_with_memory(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        memory_tokens: jax.Array | None = None,
        memory_mask: jax.Array | None = None,
        cross_attn_enabled: jax.Array | None = None,
    ) -> jax.Array:
        noise_rng, time_rng = jax.random.split(rng)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            cross_attn_enabled=cross_attn_enabled,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _sample_actions_with_memory(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        memory_tokens: jax.Array | None = None,
        memory_mask: jax.Array | None = None,
        cross_attn_enabled: jax.Array | None = None,
    ) -> _model.Actions:
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
                cross_attn_enabled=cross_attn_enabled,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def compute_temporal_outputs(
        self,
        rng: at.KeyArrayLike,
        batch: dict,
        *,
        train: bool = False,
        return_debug: bool = False,
    ) -> tuple[jax.Array, jax.Array, dict[str, tuple[int, ...]]]:
        num_chunks = batch["chunks"]["state"].shape[1]
        if num_chunks != self.tbptt_num_chunks:
            raise ValueError(
                f"Temporal batch has {num_chunks} chunks, expected tbptt_num_chunks={self.tbptt_num_chunks}."
            )
        preprocess_rngs = jax.random.split(rng, num_chunks)
        cross_attn_enabled = self._temporal_cross_attn_enabled()
        initial_hidden = batch.get("initial_hidden")
        if initial_hidden is None:
            initial_hidden = self.ltc_encoder.initial_state(
                batch["chunks"]["state"].shape[0],
                dtype=batch["chunks"]["state"].dtype,
            )
        initial_hidden = jax.lax.stop_gradient(initial_hidden)

        scan_inputs = {
            "chunk_ids": jnp.swapaxes(batch["chunks"]["chunk_ids"], 0, 1),
            "mask": jnp.swapaxes(batch["chunks"]["mask"], 0, 1),
            "delta_t": jnp.swapaxes(batch["chunks"]["delta_t"], 0, 1),
            "state": jnp.swapaxes(batch["chunks"]["state"], 0, 1),
            "target_actions": jnp.swapaxes(batch["chunks"]["target_actions"], 0, 1),
            "images": jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch["chunks"]["images"]),
            "image_masks": jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch["chunks"]["image_masks"]),
            "rng": preprocess_rngs,
        }

        def scan_step(hidden, xs):
            preprocess_rng = xs["rng"] if train else None
            observation = self._prepare_temporal_observation(
                xs["images"],
                xs["image_masks"],
                xs["state"],
                batch["prompt"],
                preprocess_rng,
                train=train,
            )
            hidden, ltc_tokens, _, _, _ = self._update_temporal_hidden(
                observation,
                hidden,
                xs["delta_t"],
                mask=xs["mask"],
            )
            memory_tokens, memory_mask = self._temporal_memory_from_hidden(hidden, xs["mask"])
            step_loss = self._compute_diffusion_loss_with_memory(
                xs["rng"],
                observation,
                xs["target_actions"],
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
                cross_attn_enabled=cross_attn_enabled,
            )

            return hidden, {
                "step_loss": step_loss,
                "ltc_tokens": ltc_tokens,
                "h_t": hidden,
                "memory_tokens": memory_tokens,
            }

        final_hidden, scan_outputs = jax.lax.scan(scan_step, initial_hidden, scan_inputs)
        step_loss = jnp.swapaxes(scan_outputs["step_loss"], 0, 1)

        debug_shapes = {}
        if return_debug:
            debug_shapes = {
                "ltc_tokens": tuple(jnp.swapaxes(scan_outputs["ltc_tokens"], 0, 1).shape),
                "h_t": tuple(jnp.swapaxes(scan_outputs["h_t"], 0, 1).shape),
                "memory_tokens": tuple(jnp.swapaxes(scan_outputs["memory_tokens"], 0, 1).shape),
                "step_loss": tuple(step_loss.shape),
                "final_hidden": tuple(final_hidden.shape),
            }
        return step_loss, jax.lax.stop_gradient(final_hidden), debug_shapes

    def sample_actions_temporal(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        hidden_state: jax.Array,
        *,
        delta_t: jax.Array | float | None = None,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, jax.Array]:
        if not self.temporal_memory_enabled:
            raise NotImplementedError("sample_actions_temporal is only available for temporal-memory pi05 models.")

        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if delta_t is None:
            delta_t = jnp.ones((batch_size,), dtype=jnp.float32)
        else:
            delta_t = jnp.asarray(delta_t, dtype=jnp.float32)
            if delta_t.ndim == 0:
                delta_t = jnp.broadcast_to(delta_t[None], (batch_size,))
            elif delta_t.shape != (batch_size,):
                delta_t = jnp.broadcast_to(delta_t.reshape(-1)[:1], (batch_size,))

        next_hidden, _, _, _, _ = self._update_temporal_hidden(
            observation,
            hidden_state,
            delta_t,
            mask=jnp.ones((batch_size,), dtype=jnp.bool_),
        )
        memory_tokens, memory_mask = self._temporal_memory_from_hidden(
            next_hidden,
            jnp.ones((batch_size,), dtype=jnp.bool_),
        )
        pred_actions = self._sample_actions_with_memory(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            cross_attn_enabled=self._temporal_cross_attn_enabled(),
        )
        return pred_actions, jax.lax.stop_gradient(next_hidden)

    def compute_temporal_loss(self, rng: at.KeyArrayLike, batch: dict, *, train: bool = False) -> jax.Array:
        step_loss, _, _ = self.compute_temporal_outputs(rng, batch, train=train, return_debug=False)
        step_weights = batch["chunks"]["target_mask"].astype(step_loss.dtype)
        chunk_denominator = jnp.clip(jnp.sum(step_weights, axis=-1), 1.0)
        chunk_loss = jnp.sum(step_loss * step_weights, axis=-1) / chunk_denominator
        chunk_loss = chunk_loss * batch["chunks"]["mask"].astype(chunk_loss.dtype)
        return jnp.sum(chunk_loss) / jnp.clip(jnp.sum(batch["chunks"]["mask"]), 1)

    def compute_temporal_loss_and_state(
        self, rng: at.KeyArrayLike, batch: dict, *, train: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        step_loss, final_hidden, _ = self.compute_temporal_outputs(rng, batch, train=train, return_debug=False)
        step_weights = batch["chunks"]["target_mask"].astype(step_loss.dtype)
        chunk_denominator = jnp.clip(jnp.sum(step_weights, axis=-1), 1.0)
        chunk_loss = jnp.sum(step_loss * step_weights, axis=-1) / chunk_denominator
        chunk_loss = chunk_loss * batch["chunks"]["mask"].astype(chunk_loss.dtype)
        loss = jnp.sum(chunk_loss) / jnp.clip(jnp.sum(batch["chunks"]["mask"]), 1)
        return loss, final_hidden

    def debug_temporal_forward(self, rng: at.KeyArrayLike, batch: dict, *, train: bool = False) -> dict:
        step_loss, final_hidden, debug_shapes = self.compute_temporal_outputs(rng, batch, train=train, return_debug=True)
        step_weights = batch["chunks"]["target_mask"].astype(step_loss.dtype)
        chunk_denominator = jnp.clip(jnp.sum(step_weights, axis=-1), 1.0)
        chunk_loss = jnp.sum(step_loss * step_weights, axis=-1) / chunk_denominator
        chunk_loss = chunk_loss * batch["chunks"]["mask"].astype(chunk_loss.dtype)
        loss = jnp.sum(chunk_loss) / jnp.clip(jnp.sum(batch["chunks"]["mask"]), 1)
        return {
            "loss": loss,
            "step_loss": step_loss,
            "chunk_loss": chunk_loss,
            "final_hidden": final_hidden,
            "shapes": debug_shapes,
        }

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, model_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        return self._compute_diffusion_loss_with_memory(model_rng, observation, actions)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        return self._sample_actions_with_memory(rng, observation, num_steps=num_steps, noise=noise)
