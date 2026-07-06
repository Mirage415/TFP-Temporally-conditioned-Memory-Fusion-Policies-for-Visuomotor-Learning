import dataclasses
from typing import TYPE_CHECKING
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Enables the direct action-chunk temporal-memory policy on top of the pi05 backbone.
    temporal_memory_enabled: bool = False
    # Preferred public name for the same temporal memory switch used by the ablation framework.
    memory_enabled: bool | None = None
    # Unified memory injection strategy. "memory_to_action_head_adaln" is the full TFP method.
    memory_injection_strategy: Literal[
        "none",
        "memory_as_input_token",
        "memory_to_vlm_backbone",
        "memory_by_retrieved_context_token",
        "memory_to_action_head_concat",
        "memory_to_action_head_adaln",
    ] = "memory_to_action_head_adaln"
    # Public alias for ltc_hidden_dim used by the ablation configs.
    memory_dim: int | None = None
    ltc_eps: float = 1e-6
    ltc_input_dim: int | None = None
    use_delta_t: bool = True
    default_delta_t: float = 1.0
    reset_hidden_on_episode_start: bool = True
    action_head_memory_dim: int | None = None
    vlm_adapter_init_scale: float = 0.0
    vlm_adapter_layers: tuple[int, ...] | None = None
    retrieval_bank_size: int = 32
    retrieval_top_k: int = 4
    # History length for the observation-based LTC pathway.
    ltc_history_len: int = 8
    # History length for the control-aware event pathway.
    event_history_len: int = 32
    # Latent hidden size for the LTC encoder.
    ltc_hidden_dim: int = 256
    # Number of internal ODE unfolds per recurrent step for the LTC solver.
    ltc_ode_unfolds: int = 6
    # Latent embedding size for the event encoder.
    event_embedding_dim: int = 256
    # Number of chunks per truncated-BPTT segment.
    tbptt_chunk_len: int = 8
    # Preferred name for the number of chunks per truncated-BPTT sample.
    tbptt_num_chunks: int | None = None
    # Number of memory tokens projected from the LTC hidden state.
    memory_token_count: int = 8
    # Number of suffix transformer layers that read memory tokens via cross-attention.
    memory_cross_attn_layers: int = 4
    # Loss used for future action chunk supervision.
    temporal_action_loss: Literal["mse", "l1"] = "mse"

    def __post_init__(self):
        if self.memory_enabled is None:
            object.__setattr__(self, "memory_enabled", self.temporal_memory_enabled)
        else:
            object.__setattr__(self, "temporal_memory_enabled", self.memory_enabled)
        if self.memory_dim is None:
            object.__setattr__(self, "memory_dim", self.ltc_hidden_dim)
        else:
            object.__setattr__(self, "ltc_hidden_dim", self.memory_dim)
        if not self.temporal_memory_enabled and self.memory_injection_strategy != "none":
            object.__setattr__(self, "memory_injection_strategy", "none")
        if self.temporal_memory_enabled and self.memory_injection_strategy == "none":
            raise ValueError("memory_injection_strategy='none' requires memory_enabled=False.")
        valid_strategies = {
            "none",
            "memory_as_input_token",
            "memory_to_vlm_backbone",
            "memory_by_retrieved_context_token",
            "memory_to_action_head_concat",
            "memory_to_action_head_adaln",
        }
        if self.memory_injection_strategy not in valid_strategies:
            raise ValueError(f"Unknown memory_injection_strategy: {self.memory_injection_strategy}")
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive.")
        if self.retrieval_bank_size <= 0:
            raise ValueError("retrieval_bank_size must be positive.")
        if self.retrieval_top_k > self.retrieval_bank_size:
            raise ValueError("retrieval_top_k cannot exceed retrieval_bank_size.")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.tbptt_num_chunks is None:
            object.__setattr__(self, "tbptt_num_chunks", self.tbptt_chunk_len)
        if self.tbptt_chunk_len <= 0:
            raise ValueError(f"tbptt_chunk_len must be positive, got {self.tbptt_chunk_len}.")
        if self.tbptt_num_chunks is None or self.tbptt_num_chunks <= 0:
            raise ValueError(f"tbptt_num_chunks must be positive, got {self.tbptt_num_chunks}.")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    @staticmethod
    def backbone_path_patterns() -> tuple[str, ...]:
        """Returns top-level parameter path regexes that belong to the pretrained pi0/pi05 backbone."""
        return (
            "PaliGemma/.*",
            "action_in_proj/.*",
            "time_mlp_in/.*",
            "time_mlp_out/.*",
            "action_time_mlp_in/.*",
            "action_time_mlp_out/.*",
            "action_out_proj/.*",
        )

    @staticmethod
    def memory_adaln_warmup_freeze_patterns() -> tuple[str, ...]:
        """Freeze non-memory-adaln branches during the initial temporal-head curriculum."""
        return (
            *Pi0Config.backbone_path_patterns(),
            "memory_router/memory_type_embedding/.*",
            "memory_router/input_token_proj/.*",
            "memory_router/vlm_adapters/.*",
            "memory_router/action_concat_proj/.*",
            "memory_router/action_concat_out/.*",
            "memory_router/retrieval/.*",
        )
