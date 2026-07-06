from __future__ import annotations

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


MEMORY_STRATEGIES = (
    "memory_as_input_token",
    "memory_to_vlm_backbone",
    "memory_by_retrieved_context_token",
    "memory_to_action_head_concat",
    "memory_to_action_head_adaln",
)


@dataclasses.dataclass
class InjectionOutputs:
    prefix_tokens: jax.Array | None = None
    prefix_mask: jax.Array | None = None
    prefix_ar_mask: jax.Array | None = None
    suffix_tokens: jax.Array | None = None
    suffix_cond: jax.Array | None = None
    memory_tokens: jax.Array | None = None
    memory_mask: jax.Array | None = None
    cross_attn_enabled: jax.Array | None = None
    vlm_adapter_gamma: jax.Array | None = None
    vlm_adapter_beta: jax.Array | None = None
    vlm_adapter_scale: jax.Array | None = None
    diagnostics: dict[str, jax.Array] = dataclasses.field(default_factory=dict)


class VLMBackboneMemoryAdapters(nnx.Module):
    def __init__(self, memory_dim: int, num_layers: int, hidden_dim: int, *, init_scale: float, rngs: nnx.Rngs):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.modulation_proj = nnx.Linear(memory_dim, num_layers * hidden_dim * 2, rngs=rngs)
        self.adapter_scale = nnx.Param(jnp.full((num_layers,), init_scale, dtype=jnp.float32))

    def __call__(self, hidden: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        gamma_beta = self.modulation_proj(hidden)
        gamma_beta = gamma_beta.reshape(hidden.shape[0], self.num_layers, 2, self.hidden_dim)
        gamma = jnp.swapaxes(gamma_beta[:, :, 0, :], 0, 1)
        beta = jnp.swapaxes(gamma_beta[:, :, 1, :], 0, 1)
        return gamma, beta, self.adapter_scale.value


class RetrievalContextMemory(nnx.Module):
    def __init__(
        self,
        x_dim: int,
        memory_dim: int,
        context_dim: int,
        *,
        bank_size: int,
        top_k: int,
        rngs: nnx.Rngs,
    ):
        self.bank_size = bank_size
        self.top_k = top_k
        query_dim = x_dim + memory_dim
        self.q_proj = nnx.Linear(query_dim, context_dim, rngs=rngs)
        self.k_proj = nnx.Linear(query_dim, context_dim, rngs=rngs)
        self.v_proj = nnx.Linear(query_dim, context_dim, rngs=rngs)
        self.time_proj = nnx.Linear(1, context_dim, rngs=rngs)

    def __call__(self, x_t: jax.Array, h_t: jax.Array, bank: dict) -> tuple[jax.Array, dict[str, jax.Array]]:
        query = self.q_proj(jnp.concatenate([x_t, h_t], axis=-1))
        entries = jnp.concatenate([bank["x"], bank["h"]], axis=-1)
        keys = self.k_proj(entries)
        values = self.v_proj(entries)
        query_n = query / jnp.clip(jnp.linalg.norm(query, axis=-1, keepdims=True), 1e-6)
        keys_n = keys / jnp.clip(jnp.linalg.norm(keys, axis=-1, keepdims=True), 1e-6)
        sim = jnp.einsum("bd,bkd->bk", query_n, keys_n)
        tie_break = jnp.arange(self.bank_size, dtype=sim.dtype)[None, :] * 1e-7
        sim = jnp.where(bank["mask"], sim - tie_break, -jnp.inf)
        top_sim, top_idx = jax.lax.top_k(sim, self.top_k)
        batch_idx = jnp.arange(x_t.shape[0])[:, None]
        selected_values = values[batch_idx, top_idx]
        rel_time = (bank["elapsed"][:, None] - bank["time"][batch_idx, top_idx])[..., None]
        tokens = selected_values + self.time_proj(rel_time)
        valid = bank["mask"][batch_idx, top_idx]
        tokens = tokens * valid[..., None].astype(tokens.dtype)
        diagnostics = {
            "retrieval_similarity_mean": jnp.mean(jnp.where(valid, top_sim, 0.0), axis=-1),
            "retrieval_indices": top_idx,
            "retrieval_bank_size": bank["size"],
            "retrieved_token_norm": jnp.linalg.norm(tokens, axis=-1).mean(axis=-1),
        }
        return tokens, diagnostics


class MemoryInjectionRouter(nnx.Module):
    def __init__(
        self,
        strategy: str,
        memory_dim: int,
        vlm_hidden_dim: int,
        action_hidden_dim: int,
        action_horizon: int,
        num_vlm_layers: int,
        x_dim: int,
        *,
        action_head_memory_dim: int | None,
        retrieval_bank_size: int,
        retrieval_top_k: int,
        vlm_adapter_init_scale: float,
        rngs: nnx.Rngs,
    ):
        if strategy != "none" and strategy not in MEMORY_STRATEGIES:
            raise ValueError(f"Unknown memory injection strategy: {strategy}")
        self.strategy = strategy
        self.action_horizon = action_horizon
        self.memory_type_embedding = nnx.Param(jnp.zeros((vlm_hidden_dim,), dtype=jnp.float32))
        self.input_token_proj = nnx.Linear(memory_dim, vlm_hidden_dim, rngs=rngs)
        self.vlm_adapters = VLMBackboneMemoryAdapters(
            memory_dim, num_vlm_layers, vlm_hidden_dim, init_scale=vlm_adapter_init_scale, rngs=rngs
        )
        concat_dim = action_hidden_dim if action_head_memory_dim is None else action_head_memory_dim
        self.action_concat_proj = nnx.Linear(memory_dim, concat_dim, rngs=rngs)
        self.action_concat_out = nnx.Linear(action_hidden_dim + concat_dim, action_hidden_dim, rngs=rngs)
        self.memory_adaln_proj = nnx.Linear(memory_dim, action_hidden_dim, rngs=rngs)
        self.retrieval = RetrievalContextMemory(
            x_dim,
            memory_dim,
            action_hidden_dim,
            bank_size=retrieval_bank_size,
            top_k=retrieval_top_k,
            rngs=rngs,
        )

    def _assert_strategy(self, expected: str) -> None:
        if self.strategy != expected:
            raise AssertionError(f"Router strategy is {self.strategy}, not {expected}.")

    def inject_input_token(
        self, prefix_tokens: jax.Array, prefix_mask: jax.Array, prefix_ar_mask: jax.Array, h_t: jax.Array
    ) -> InjectionOutputs:
        self._assert_strategy("memory_as_input_token")
        mem_token = self.input_token_proj(h_t) + self.memory_type_embedding.value[None, :]
        mem_token = mem_token[:, None, :]
        # Current pi0.5 prefix order in Pi0AdaLN is visual tokens, language tokens, state token.
        # Insert before the final state token so memory remains prefix context and does not alter action suffix order.
        insert_at = prefix_tokens.shape[1] - 1
        tokens = jnp.concatenate([prefix_tokens[:, :insert_at], mem_token, prefix_tokens[:, insert_at:]], axis=1)
        mask = jnp.concatenate(
            [prefix_mask[:, :insert_at], jnp.ones((prefix_mask.shape[0], 1), dtype=jnp.bool_), prefix_mask[:, insert_at:]],
            axis=1,
        )
        ar = jnp.concatenate([prefix_ar_mask[:insert_at], jnp.asarray([False]), prefix_ar_mask[insert_at:]], axis=0)
        return InjectionOutputs(
            prefix_tokens=tokens,
            prefix_mask=mask,
            prefix_ar_mask=ar,
            diagnostics={"injection_norm": jnp.linalg.norm(mem_token[:, 0, :], axis=-1)},
        )

    def vlm_adapter_inputs(self, h_t: jax.Array) -> InjectionOutputs:
        self._assert_strategy("memory_to_vlm_backbone")
        gamma, beta, scale = self.vlm_adapters(h_t)
        adapter_norm = jnp.mean(jnp.linalg.norm(gamma, axis=-1) + jnp.linalg.norm(beta, axis=-1), axis=0)
        return InjectionOutputs(
            vlm_adapter_gamma=gamma,
            vlm_adapter_beta=beta,
            vlm_adapter_scale=scale,
            diagnostics={"injection_norm": adapter_norm},
        )

    def inject_retrieved_context(self, x_t: jax.Array, h_t: jax.Array, bank: dict) -> InjectionOutputs:
        self._assert_strategy("memory_by_retrieved_context_token")
        tokens, diagnostics = self.retrieval(x_t, h_t, bank)
        return InjectionOutputs(
            memory_tokens=tokens,
            memory_mask=jnp.ones(tokens.shape[:2], dtype=jnp.bool_),
            cross_attn_enabled=jnp.ones((self.vlm_adapters.num_layers,), dtype=jnp.bool_),
            diagnostics={**diagnostics, "injection_norm": diagnostics["retrieved_token_norm"]},
        )

    def inject_action_concat(self, suffix_tokens: jax.Array, h_t: jax.Array) -> InjectionOutputs:
        self._assert_strategy("memory_to_action_head_concat")
        action_tokens = suffix_tokens[:, -self.action_horizon :]
        memory = self.action_concat_proj(h_t)
        memory_tokens = einops.repeat(memory, "b d -> b t d", t=action_tokens.shape[1])
        projected = self.action_concat_out(jnp.concatenate([action_tokens, memory_tokens], axis=-1))
        suffix_tokens = suffix_tokens.at[:, -self.action_horizon :].set(projected.astype(suffix_tokens.dtype))
        return InjectionOutputs(
            suffix_tokens=suffix_tokens,
            diagnostics={"injection_norm": jnp.linalg.norm(projected - action_tokens, axis=-1).mean(axis=-1)},
        )

    def inject_action_adaln(self, timestep_cond: jax.Array | None, h_t: jax.Array) -> InjectionOutputs:
        self._assert_strategy("memory_to_action_head_adaln")
        memory_cond = self.memory_adaln_proj(h_t)
        if timestep_cond is None:
            combined = memory_cond
            timestep_norm = jnp.zeros((h_t.shape[0],), dtype=h_t.dtype)
        else:
            combined = timestep_cond + memory_cond.astype(timestep_cond.dtype)
            timestep_norm = jnp.linalg.norm(timestep_cond, axis=-1)
        return InjectionOutputs(
            suffix_cond=combined,
            diagnostics={
                "memory_condition_norm": jnp.linalg.norm(memory_cond, axis=-1),
                "timestep_condition_norm": timestep_norm,
                "combined_condition_norm": jnp.linalg.norm(combined, axis=-1),
                "injection_norm": jnp.linalg.norm(memory_cond, axis=-1),
            },
        )
