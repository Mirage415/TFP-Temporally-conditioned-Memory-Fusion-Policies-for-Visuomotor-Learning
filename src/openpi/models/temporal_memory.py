from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class LTCEncoder(nnx.Module):
    """Semi-implicit Liquid Time-Constant cell following the official LTC dynamics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sensory_size: int,
        *,
        ode_unfolds: int,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.sensory_size = sensory_size
        self.ode_unfolds = ode_unfolds
        self.epsilon = 1e-8

        # Affine sensory input mapping before the LTC dynamics.
        self.sensory_in_proj = nnx.Linear(input_dim, 1, rngs=rngs)

        mu_init = nnx.initializers.uniform(scale=0.5)
        sigma_init = nnx.initializers.uniform(scale=5.0)
        erev_init = nnx.initializers.uniform(scale=1.0)
        weight_init = nnx.initializers.uniform(scale=1.0)

        self.sensory_mu = nnx.Param(mu_init(rngs.params(), (sensory_size, hidden_dim), jnp.float32))
        self.sensory_sigma = nnx.Param(sigma_init(rngs.params(), (sensory_size, hidden_dim), jnp.float32))
        self.sensory_w = nnx.Param(weight_init(rngs.params(), (sensory_size, hidden_dim), jnp.float32))
        self.sensory_erev = nnx.Param(erev_init(rngs.params(), (sensory_size, hidden_dim), jnp.float32))

        self.mu = nnx.Param(mu_init(rngs.params(), (hidden_dim, hidden_dim), jnp.float32))
        self.sigma = nnx.Param(sigma_init(rngs.params(), (hidden_dim, hidden_dim), jnp.float32))
        self.w = nnx.Param(weight_init(rngs.params(), (hidden_dim, hidden_dim), jnp.float32))
        self.erev = nnx.Param(erev_init(rngs.params(), (hidden_dim, hidden_dim), jnp.float32))

        self.vleak = nnx.Param(jnp.zeros((hidden_dim,), dtype=jnp.float32))
        self.gleak = nnx.Param(jnp.ones((hidden_dim,), dtype=jnp.float32))
        self.cm = nnx.Param(jnp.ones((hidden_dim,), dtype=jnp.float32) * 0.5)

    def initial_state(self, batch_size: int, *, dtype=jnp.float32) -> jax.Array:
        return jnp.zeros((batch_size, self.hidden_dim), dtype=dtype)

    def _positive(self, value: jax.Array) -> jax.Array:
        return jax.nn.softplus(value) + self.epsilon

    def _sigmoid(self, source: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
        source = source[..., :, None]
        sigma = self._positive(sigma)
        return jax.nn.sigmoid((source - mu[None, ...]) * sigma[None, ...])

    def _map_sensory_inputs(self, input_tokens: jax.Array) -> jax.Array:
        if input_tokens.shape[1] != self.sensory_size:
            raise ValueError(
                f"LTC expected sensory_size={self.sensory_size}, got {input_tokens.shape[1]} input tokens."
            )
        return self.sensory_in_proj(input_tokens)[..., 0].astype(jnp.float32)

    def _ode_step(
        self,
        inputs: jax.Array,
        input_mask: jax.Array,
        state: jax.Array,
        elapsed_time: jax.Array,
    ) -> jax.Array:
        sensory_gate = self._sigmoid(inputs, self.sensory_mu.value, self.sensory_sigma.value)
        sensory_w = self._positive(self.sensory_w.value)[None, ...] * sensory_gate
        sensory_w = sensory_w * input_mask[..., None].astype(sensory_w.dtype)
        sensory_rev = sensory_w * self.sensory_erev.value[None, ...]
        sensory_num = jnp.sum(sensory_rev, axis=1)
        sensory_den = jnp.sum(sensory_w, axis=1)

        gleak = self._positive(self.gleak.value)[None, :]
        cm = self._positive(self.cm.value)[None, :]
        dt = jnp.maximum(elapsed_time[:, None], self.epsilon)
        cm_t = cm / (dt / float(self.ode_unfolds))

        recurrent_w = self._positive(self.w.value)
        v_pre = state.astype(jnp.float32)
        for _ in range(self.ode_unfolds):
            recurrent_gate = self._sigmoid(v_pre, self.mu.value, self.sigma.value)
            w_activation = recurrent_w[None, ...] * recurrent_gate
            rev_activation = w_activation * self.erev.value[None, ...]
            w_num = jnp.sum(rev_activation, axis=1) + sensory_num
            w_den = jnp.sum(w_activation, axis=1) + sensory_den

            numerator = cm_t * v_pre + gleak * self.vleak.value[None, :] + w_num
            denominator = cm_t + gleak + w_den
            v_pre = numerator / (denominator + self.epsilon)

        return v_pre

    def step(
        self,
        hidden: jax.Array,
        input_tokens: jax.Array,
        token_mask: jax.Array,
        delta_t: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        sensory_inputs = self._map_sensory_inputs(input_tokens)
        updated = self._ode_step(
            sensory_inputs,
            token_mask,
            hidden,
            jnp.asarray(delta_t, dtype=jnp.float32),
        )
        return jnp.where(mask[:, None], updated.astype(hidden.dtype), hidden)


class LTCDiagnostics(dict):
    """Dictionary marker for LTC diagnostic arrays."""


class LiquidTimeConstantMemory(nnx.Module):
    """Compact formula-based LTC memory cell shared by all memory-injection ablations."""

    def __init__(
        self,
        vision_dim: int,
        state_dim: int,
        memory_dim: int,
        *,
        eps: float,
        use_delta_t: bool,
        default_delta_t: float,
        rngs: nnx.Rngs,
    ):
        self.memory_dim = memory_dim
        self.eps = eps
        self.use_delta_t = use_delta_t
        self.default_delta_t = default_delta_t
        self.vision_proj = nnx.Linear(vision_dim, memory_dim, rngs=rngs)
        self.state_proj = nnx.Linear(state_dim, memory_dim, rngs=rngs)
        self.hidden_proj = nnx.Linear(memory_dim * 3, memory_dim, rngs=rngs)
        self.tau_proj = nnx.Linear(memory_dim * 3, memory_dim, rngs=rngs)

    def init_hidden(self, batch_size: int, device=None, dtype=jnp.float32) -> jax.Array:
        hidden = jnp.zeros((batch_size, self.memory_dim), dtype=dtype)
        if device is not None:
            hidden = jax.device_put(hidden, device)
        return hidden

    def initial_state(self, batch_size: int, *, dtype=jnp.float32) -> jax.Array:
        return self.init_hidden(batch_size, dtype=dtype)

    def reset_hidden(self, hidden: jax.Array, reset_mask: jax.Array | None) -> jax.Array:
        if reset_mask is None:
            return hidden
        return jnp.where(jnp.asarray(reset_mask, dtype=jnp.bool_)[:, None], jnp.zeros_like(hidden), hidden)

    def detach_hidden(self, hidden: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient(hidden)

    def observation_summary(
        self,
        visual_tokens: list[jax.Array],
        visual_masks: list[jax.Array],
        state: jax.Array,
    ) -> jax.Array:
        if not visual_tokens:
            raise ValueError("LiquidTimeConstantMemory requires real visual tokens from the VLA image encoder.")
        vision = jnp.concatenate(visual_tokens, axis=1)
        mask = jnp.concatenate(visual_masks, axis=1).astype(vision.dtype)
        pooled = jnp.sum(vision * mask[..., None], axis=1) / jnp.clip(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        return jnp.concatenate([self.vision_proj(pooled), self.state_proj(state)], axis=-1)

    def step(
        self,
        hidden: jax.Array,
        x_t: jax.Array,
        delta_t: jax.Array | None,
        *,
        mask: jax.Array | None = None,
        reset_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, LTCDiagnostics]:
        hidden = self.reset_hidden(hidden, reset_mask)
        if delta_t is None or not self.use_delta_t:
            delta = jnp.full((hidden.shape[0], 1), self.default_delta_t, dtype=jnp.float32)
        else:
            delta = jnp.asarray(delta_t, dtype=jnp.float32)
            if delta.ndim == 1:
                delta = delta[:, None]
            elif delta.ndim == 0:
                delta = jnp.broadcast_to(delta[None, None], (hidden.shape[0], 1))
        cell_input = jnp.concatenate([x_t.astype(hidden.dtype), hidden], axis=-1)
        h_hat = jnp.tanh(self.hidden_proj(cell_input))
        tau = jax.nn.softplus(self.tau_proj(cell_input)) + self.eps
        k = jnp.exp(-delta / tau)
        updated = k * hidden + (1.0 - k) * h_hat
        if mask is not None:
            updated = jnp.where(jnp.asarray(mask, dtype=jnp.bool_)[:, None], updated, hidden)
        diagnostics = LTCDiagnostics(
            hidden_norm=jnp.linalg.norm(updated, axis=-1),
            hidden_delta_norm=jnp.linalg.norm(updated - hidden, axis=-1),
            tau_mean=jnp.mean(tau, axis=-1),
            tau_min=jnp.min(tau, axis=-1),
            tau_max=jnp.max(tau, axis=-1),
            k_mean=jnp.mean(k, axis=-1),
            k_min=jnp.min(k, axis=-1),
            k_max=jnp.max(k, axis=-1),
        )
        return updated.astype(hidden.dtype), tau.astype(hidden.dtype), k.astype(hidden.dtype), diagnostics


def empty_retrieval_bank(batch_size: int, bank_size: int, x_dim: int, hidden_dim: int, *, dtype=jnp.float32) -> dict:
    return {
        "x": jnp.zeros((batch_size, bank_size, x_dim), dtype=dtype),
        "h": jnp.zeros((batch_size, bank_size, hidden_dim), dtype=dtype),
        "mask": jnp.zeros((batch_size, bank_size), dtype=jnp.bool_),
        "time": jnp.zeros((batch_size, bank_size), dtype=dtype),
        "step": jnp.zeros((batch_size, bank_size), dtype=jnp.int32),
        "ptr": jnp.zeros((batch_size,), dtype=jnp.int32),
        "size": jnp.zeros((batch_size,), dtype=jnp.int32),
        "elapsed": jnp.zeros((batch_size,), dtype=dtype),
    }


def reset_retrieval_bank(bank: dict, reset_mask: jax.Array | None) -> dict:
    if reset_mask is None:
        return bank
    reset = jnp.asarray(reset_mask, dtype=jnp.bool_)
    return jax.tree.map(
        lambda value: jnp.where(reset.reshape((reset.shape[0],) + (1,) * (value.ndim - 1)), jnp.zeros_like(value), value),
        bank,
    )


def append_retrieval_bank(bank: dict, x_t: jax.Array, h_t: jax.Array, delta_t: jax.Array, mask: jax.Array) -> dict:
    batch_size, bank_size = bank["mask"].shape
    rows = jnp.arange(batch_size)
    ptr = bank["ptr"]
    active = jnp.asarray(mask, dtype=jnp.bool_)
    elapsed = bank["elapsed"] + jnp.where(active, jnp.asarray(delta_t, dtype=bank["elapsed"].dtype), 0.0)
    new_bank = dict(bank)
    new_bank["x"] = bank["x"].at[rows, ptr].set(jnp.where(active[:, None], jax.lax.stop_gradient(x_t), bank["x"][rows, ptr]))
    new_bank["h"] = bank["h"].at[rows, ptr].set(jnp.where(active[:, None], jax.lax.stop_gradient(h_t), bank["h"][rows, ptr]))
    new_bank["mask"] = bank["mask"].at[rows, ptr].set(active)
    new_bank["time"] = bank["time"].at[rows, ptr].set(jnp.where(active, elapsed, bank["time"][rows, ptr]))
    new_bank["step"] = bank["step"].at[rows, ptr].set(jnp.where(active, bank["size"], bank["step"][rows, ptr]))
    new_bank["ptr"] = jnp.where(active, (ptr + 1) % bank_size, ptr)
    new_bank["size"] = jnp.where(active, jnp.minimum(bank["size"] + 1, bank_size), bank["size"])
    new_bank["elapsed"] = elapsed
    return new_bank


class EventEncoder(nnx.Module):
    """A simple masked-pooling encoder for longer control history."""

    def __init__(self, input_dim: int, embedding_dim: int, rngs: nnx.Rngs):
        self.embedding_dim = embedding_dim
        self.in_proj = nnx.Linear(input_dim + 1, embedding_dim, rngs=rngs)
        self.hidden_proj = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array, delta_t: jax.Array, mask: jax.Array) -> jax.Array:
        x = jnp.concatenate([inputs, delta_t[..., None]], axis=-1)
        x = nnx.swish(self.in_proj(x))
        x = nnx.swish(self.hidden_proj(x))

        weights = mask[..., None].astype(x.dtype)
        pooled = jnp.sum(x * weights, axis=1) / jnp.clip(jnp.sum(weights, axis=1), 1.0)
        return self.out_proj(pooled)


def masked_action_chunk_loss(
    pred_actions: jax.Array,
    target_actions: jax.Array,
    target_mask: jax.Array,
    *,
    loss_type: str = "mse",
) -> tuple[jax.Array, jax.Array]:
    if loss_type == "mse":
        per_step = jnp.mean(jnp.square(pred_actions - target_actions), axis=-1)
    elif loss_type == "l1":
        per_step = jnp.mean(jnp.abs(pred_actions - target_actions), axis=-1)
    else:
        raise ValueError(f"Unsupported temporal action loss: {loss_type}")

    masked = per_step * target_mask.astype(per_step.dtype)
    loss = jnp.sum(masked) / jnp.clip(jnp.sum(target_mask), 1)
    return loss, masked


def masked_chunkwise_action_loss(
    pred_actions: jax.Array,
    target_actions: jax.Array,
    target_mask: jax.Array,
    chunk_mask: jax.Array,
    *,
    loss_type: str = "mse",
) -> tuple[jax.Array, jax.Array]:
    if loss_type == "mse":
        per_step = jnp.mean(jnp.square(pred_actions - target_actions), axis=-1)
    elif loss_type == "l1":
        per_step = jnp.mean(jnp.abs(pred_actions - target_actions), axis=-1)
    else:
        raise ValueError(f"Unsupported temporal action loss: {loss_type}")

    step_weights = target_mask.astype(per_step.dtype)
    chunk_denominator = jnp.clip(jnp.sum(step_weights, axis=-1), 1.0)
    chunk_loss = jnp.sum(per_step * step_weights, axis=-1) / chunk_denominator
    chunk_loss = chunk_loss * chunk_mask.astype(chunk_loss.dtype)
    loss = jnp.sum(chunk_loss) / jnp.clip(jnp.sum(chunk_mask), 1)
    return loss, chunk_loss
