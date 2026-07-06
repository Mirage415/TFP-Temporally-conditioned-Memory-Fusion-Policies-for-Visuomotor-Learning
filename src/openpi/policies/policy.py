import atexit
from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class _LatentEpisodeRecorder:
    """Writes per-episode temporal hidden states for offline analysis."""

    def __init__(self, root_dir: str | os.PathLike[str]):
        self._root_dir = pathlib.Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._episode_counter = 0
        self._rows: list[dict[str, Any]] = []
        atexit.register(self.close)

    def record(self, hidden: np.ndarray, obs: dict[str, Any]) -> None:
        analysis = obs.get("_analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}
        self._rows.append(
            {
                "hidden": np.asarray(hidden, dtype=np.float32).copy(),
                "task_id": int(analysis.get("task_id", -1)),
                "episode_idx": int(analysis.get("episode_idx", -1)),
                "chunk_idx": int(analysis.get("chunk_idx", len(self._rows))),
                "env_step": int(analysis.get("env_step", -1)),
                "task_description": str(analysis.get("task_description", "")),
                "task_suite_name": str(analysis.get("task_suite_name", "")),
            }
        )

    def flush_episode(self) -> None:
        if not self._rows:
            return

        hidden = np.stack([row["hidden"] for row in self._rows], axis=0).astype(np.float32)
        task_ids = np.asarray([row["task_id"] for row in self._rows], dtype=np.int32)
        episode_indices = np.asarray([row["episode_idx"] for row in self._rows], dtype=np.int32)
        chunk_indices = np.asarray([row["chunk_idx"] for row in self._rows], dtype=np.int32)
        env_steps = np.asarray([row["env_step"] for row in self._rows], dtype=np.int32)
        progress = (
            np.linspace(0.0, 1.0, num=len(self._rows), dtype=np.float32)
            if len(self._rows) > 1
            else np.asarray([0.0], dtype=np.float32)
        )
        task_description = np.asarray(self._rows[0]["task_description"])
        task_suite_name = np.asarray(self._rows[0]["task_suite_name"])

        output_path = self._root_dir / (
            f"episode_{self._episode_counter:06d}_task{int(task_ids[0]):02d}_ep{int(episode_indices[0]):03d}.npz"
        )
        np.savez_compressed(
            output_path,
            hidden=hidden,
            task_id=task_ids,
            episode_idx=episode_indices,
            chunk_idx=chunk_indices,
            env_step=env_steps,
            progress=progress,
            task_description=task_description,
            task_suite_name=task_suite_name,
        )
        self._rows.clear()
        self._episode_counter += 1

    def close(self) -> None:
        self.flush_episode()


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._is_temporal_jax_model = (not is_pytorch) and bool(getattr(model, "temporal_memory_enabled", False))
        self._temporal_hidden_state = None
        self._temporal_retrieval_bank = None
        self._temporal_last_timestamp = None
        self._temporal_returns_diagnostics = hasattr(model, "memory_router")
        latent_dir = os.environ.get("OPENPI_RECORD_LATENTS_DIR")
        self._latent_recorder = (
            _LatentEpisodeRecorder(latent_dir) if latent_dir and self._is_temporal_jax_model else None
        )

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if self._is_temporal_jax_model:
                self._sample_actions_temporal = nnx_utils.module_jit(model.sample_actions_temporal)
            self._rng = rng or jax.random.key(0)
        self._metadata = {
            **self._metadata,
            "action_horizon": getattr(model, "action_horizon", None),
            "temporal_memory_enabled": self._is_temporal_jax_model,
            "stateful_inference": self._is_temporal_jax_model,
            "tbptt_chunk_len": getattr(model, "tbptt_chunk_len", None),
            "tbptt_num_chunks": getattr(model, "tbptt_num_chunks", None),
            "latent_recording_enabled": self._latent_recorder is not None,
        }

    def _lookup_numeric_value(self, obs: dict, path: tuple[str, ...]) -> float | None:
        current = obs
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        value = np.asarray(current)
        if value.size == 0 or value.dtype.kind not in {"i", "u", "f"}:
            return None
        return float(value.reshape(-1)[0].item())

    def _extract_temporal_delta_t(self, *obs_sources: dict, batch_size: int) -> jax.Array:
        timestamp = None
        for obs in obs_sources:
            if not isinstance(obs, dict):
                continue
            if timestamp is None:
                for path in (("timestamp",), ("observation", "timestamp"), ("observation", "timestamps")):
                    timestamp = self._lookup_numeric_value(obs, path)
                    if timestamp is not None:
                        break
            if timestamp is not None:
                break

        if timestamp is None:
            timestamp = time.monotonic()

        if self._temporal_last_timestamp is None:
            delta_t = 0.0
        else:
            delta_t = max(timestamp - self._temporal_last_timestamp, 0.0)
        self._temporal_last_timestamp = timestamp

        return jnp.full((batch_size,), float(delta_t), dtype=jnp.float32)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        raw_obs = jax.tree.map(lambda x: x, obs)
        transformed_inputs = self._input_transform(raw_obs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed_inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], transformed_inputs
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if self._is_temporal_jax_model:
            if self._temporal_hidden_state is None:
                self._temporal_hidden_state = self._model.ltc_encoder.initial_state(
                    observation.state.shape[0],
                    dtype=observation.state.dtype,
                )
            temporal_delta_t = self._extract_temporal_delta_t(
                transformed_inputs,
                raw_obs,
                batch_size=observation.state.shape[0],
            )
            if self._temporal_returns_diagnostics:
                sample_result = self._sample_actions_temporal(
                    sample_rng_or_pytorch_device,
                    observation,
                    self._temporal_hidden_state,
                    delta_t=temporal_delta_t,
                    retrieval_bank=self._temporal_retrieval_bank,
                    return_diagnostics=True,
                    **sample_kwargs,
                )
                actions, self._temporal_hidden_state, temporal_diagnostics, self._temporal_retrieval_bank = sample_result
            else:
                actions, self._temporal_hidden_state = self._sample_actions_temporal(
                    sample_rng_or_pytorch_device,
                    observation,
                    self._temporal_hidden_state,
                    delta_t=temporal_delta_t,
                    **sample_kwargs,
                )
                temporal_diagnostics = {}
            if self._latent_recorder is not None:
                self._latent_recorder.record(np.asarray(self._temporal_hidden_state[0]), raw_obs)
            outputs = {
                "state": inputs["state"],
                "actions": actions,
                "memory_diagnostics": temporal_diagnostics,
            }
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def reset(self) -> None:
        if self._latent_recorder is not None:
            self._latent_recorder.flush_episode()
        self._temporal_hidden_state = None
        self._temporal_retrieval_bank = None
        self._temporal_last_timestamp = None


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
