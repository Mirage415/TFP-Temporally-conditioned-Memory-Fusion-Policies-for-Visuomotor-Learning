from __future__ import annotations

import copy
from collections.abc import Sequence
import dataclasses
import functools
import gc
import io
import json
import logging
import pathlib
from concurrent import futures

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.lerobot_compat as _lerobot_compat
import openpi.transforms as _transforms

logger = logging.getLogger("openpi")
_CACHE_ROW_GROUP_SIZE = 16


@dataclasses.dataclass(frozen=True)
class EpisodeInfo:
    start: int
    end: int
    episode_index: int
    task_index: int = -1


@dataclasses.dataclass(frozen=True)
class WindowInfo:
    episode_offset: int
    episode_index: int
    task_index: int
    raw_start: int
    raw_end: int
    start_chunk_id: int


def create_temporal_memory_dataset(
    data_config: _config.DataConfig,
    model_config: pi0_config.Pi0Config,
    *,
    skip_norm_stats: bool = False,
    cache_dir: str | None = None,
) -> "TemporalMemoryDataset | FakeTemporalMemoryDataset":
    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create temporal-memory dataset.")

    if cache_dir is not None:
        return CachedTemporalMemoryDataset(
            cache_dir,
            model_config.tbptt_num_chunks,
            model_config.tbptt_chunk_len,
        )

    if data_config.repo_id == "fake":
        return FakeTemporalMemoryDataset(model_config, num_samples=256)

    _lerobot_compat.ensure_local_episodes_stats(data_config.repo_id)
    raw_dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, tolerance_s=data_config.lerobot_tolerance_s)
    _lerobot_compat.patch_local_image_transform(raw_dataset)
    if data_config.prompt_from_task:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
        raw_dataset = _TransformedTemporalDataset(
            raw_dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )

    return TemporalMemoryDataset(raw_dataset, data_config, model_config, skip_norm_stats=skip_norm_stats)


class _TransformedTemporalDataset:
    def __init__(self, dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: int):
        return self._transform(copy.deepcopy(self._dataset[index]))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        dataset = object.__getattribute__(self, "_dataset")
        return getattr(dataset, name)


class FakeTemporalMemoryDataset:
    def __init__(self, model_config: pi0_config.Pi0Config, num_samples: int):
        self._config = model_config
        self._num_samples = num_samples
        self._num_chunks = model_config.tbptt_num_chunks

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(index)

        prompt_len = min(16, self._config.max_token_len)
        prompt_ids = np.zeros((self._config.max_token_len,), dtype=np.int32)
        prompt_ids[:prompt_len] = rng.integers(1, 1024, size=(prompt_len,), dtype=np.int32)
        prompt_mask = np.zeros((self._config.max_token_len,), dtype=np.bool_)
        prompt_mask[:prompt_len] = True

        t_len = self._num_chunks
        k_len = self._config.action_horizon
        state_dim = self._config.action_dim

        return {
            "episode_id": np.asarray(index, dtype=np.int32),
            "start_chunk_id": np.asarray(0, dtype=np.int32),
            "num_chunks": np.asarray(t_len, dtype=np.int32),
            "meta": {
                "episode_index": np.asarray(index, dtype=np.int32),
                "task_index": np.asarray(index % 7, dtype=np.int32),
                "num_chunks": np.asarray(t_len, dtype=np.int32),
                "start_chunk_id": np.asarray(0, dtype=np.int32),
            },
            "prompt": {
                "input_ids": prompt_ids,
                "attention_mask": prompt_mask,
            },
            "chunks": {
                "chunk_ids": np.arange(t_len, dtype=np.int32),
                "mask": np.ones((t_len,), dtype=np.bool_),
                "delta_t": np.linspace(0.05, 0.2, num=t_len, dtype=np.float32),
                "images": {
                    name: rng.integers(0, 255, size=(t_len, *_model.IMAGE_RESOLUTION, 3), dtype=np.uint8)
                    for name in _model.IMAGE_KEYS
                },
                "image_masks": {name: np.ones((t_len,), dtype=np.bool_) for name in _model.IMAGE_KEYS},
                "state": rng.standard_normal((t_len, state_dim), dtype=np.float32),
                "target_actions": rng.standard_normal((t_len, k_len, state_dim), dtype=np.float32),
                "target_mask": np.ones((t_len, k_len), dtype=np.bool_),
            },
        }

    def __len__(self) -> int:
        return self._num_samples


class TemporalMemoryDataset:
    def __init__(
        self,
        dataset,
        data_config: _config.DataConfig,
        model_config: pi0_config.Pi0Config,
        *,
        skip_norm_stats: bool = False,
    ):
        self._dataset = dataset
        self._data_config = data_config
        self._model_config = model_config
        self._tbptt_num_chunks = model_config.tbptt_num_chunks
        self._tbptt_chunk_len = model_config.tbptt_chunk_len
        self._episodes = self._build_episode_index()
        self._sample_to_episode = np.zeros((len(self._dataset),), dtype=np.int32)
        for episode_offset, episode in enumerate(self._episodes):
            self._sample_to_episode[episode.start : episode.end] = episode_offset
        self._windows, self._window_groups, padded_tail_windows = self._build_window_index()
        if not self._windows:
            raise ValueError(
                f"No temporal windows were created. Check tbptt_num_chunks={self._tbptt_num_chunks}, "
                f"tbptt_chunk_len={self._tbptt_chunk_len} "
                f"against dataset episode lengths."
            )
        if padded_tail_windows:
            logger.warning(
                "Temporal-memory loader padded %d tail windows to tbptt_num_chunks=%d, tbptt_chunk_len=%d.",
                padded_tail_windows,
                self._tbptt_num_chunks,
                self._tbptt_chunk_len,
            )
        logger.info(
            "Temporal-memory loader created %d windows across %d episodes (tbptt_num_chunks=%d, tbptt_chunk_len=%d).",
            len(self._windows),
            len(self._episodes),
            self._tbptt_num_chunks,
            self._tbptt_chunk_len,
        )

        shared_model_transforms: list[_transforms.DataTransformFn] = []
        current_only_transforms: list[_transforms.DataTransformFn] = []
        for transform in data_config.model_transforms.inputs:
            if isinstance(transform, (_transforms.ResizeImages, _transforms.PadStatesAndActions)):
                shared_model_transforms.append(transform)
            elif isinstance(transform, (_transforms.InjectDefaultPrompt, _transforms.TokenizePrompt)):
                current_only_transforms.append(transform)
            else:
                raise NotImplementedError(
                    f"Temporal-memory loader does not support model transform {type(transform).__name__}."
                )

        normalize = _transforms.Normalize(
            None if skip_norm_stats else data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        shared = [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            normalize,
            *shared_model_transforms,
        ]
        current = [*shared, *current_only_transforms]
        self._shared_transform = _transforms.compose(shared)
        self._current_transform = _transforms.compose(current)

        example = self._current_step(0)
        self._image_keys = tuple(example["image"].keys())
        self._image_shapes = {key: example["image"][key].shape for key in self._image_keys}
        self._image_dtypes = {key: example["image"][key].dtype for key in self._image_keys}
        self._state_dim = example["state"].shape[-1]
        self._action_dim = example["actions"].shape[-1]

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def tbptt_num_chunks(self) -> int:
        return self._tbptt_num_chunks

    @property
    def tbptt_chunk_len(self) -> int:
        return self._tbptt_chunk_len

    def make_batch_sampler(self, batch_size: int, *, shuffle: bool, seed: int, multiple_of: int = 1):
        return _TemporalWindowBatchSampler(
            self._window_groups,
            self._windows,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            multiple_of=multiple_of,
        )

    def __getitem__(self, index: int) -> dict:
        window = self._windows[index]
        episode = self._episodes[window.episode_offset]
        prompt = self._current_step(window.raw_start)

        chunk_ids = np.full((self._tbptt_num_chunks,), -1, dtype=np.int32)
        chunk_mask = np.zeros((self._tbptt_num_chunks,), dtype=np.bool_)
        delta_t = np.zeros((self._tbptt_num_chunks,), dtype=np.float32)
        state = np.zeros((self._tbptt_num_chunks, self._state_dim), dtype=np.float32)
        images = {
            key: np.zeros((self._tbptt_num_chunks, *self._image_shapes[key]), dtype=self._image_dtypes[key])
            for key in self._image_keys
        }
        image_masks = {key: np.zeros((self._tbptt_num_chunks,), dtype=np.bool_) for key in self._image_keys}
        target_actions = np.zeros(
            (self._tbptt_num_chunks, self._model_config.action_horizon, self._action_dim),
            dtype=np.float32,
        )
        target_mask = np.zeros((self._tbptt_num_chunks, self._model_config.action_horizon), dtype=np.bool_)

        if window.raw_start >= episode.end:
            actual_num_chunks = 0
        else:
            actual_num_chunks = min(
                self._tbptt_num_chunks,
                (episode.end - window.raw_start + self._tbptt_chunk_len - 1) // self._tbptt_chunk_len,
            )

        for local_chunk_id in range(actual_num_chunks):
            step_index = window.raw_start + local_chunk_id * self._tbptt_chunk_len
            if step_index >= episode.end:
                break
            step = self._shared_step(step_index)
            chunk_id = window.start_chunk_id + local_chunk_id
            chunk_ids[local_chunk_id] = chunk_id
            chunk_mask[local_chunk_id] = True
            delta_t[local_chunk_id] = (
                0.0 if chunk_id == 0 else self._delta_t(step_index - self._tbptt_chunk_len, step_index)
            )
            state[local_chunk_id] = np.asarray(step["state"], dtype=np.float32)

            for key in self._image_keys:
                images[key][local_chunk_id] = np.asarray(step["image"][key])
                image_masks[key][local_chunk_id] = np.asarray(step["image_mask"][key], dtype=np.bool_)

            for target_slot, target_index in enumerate(range(step_index, step_index + self._model_config.action_horizon)):
                if target_index >= episode.end:
                    continue
                target_step = self._shared_step(target_index)
                target_actions[local_chunk_id, target_slot] = np.asarray(target_step["actions"], dtype=np.float32)
                target_mask[local_chunk_id, target_slot] = True

        task_index = window.task_index
        if task_index < 0:
            task_index = self._task_index(window.raw_start)

        return {
            "episode_id": np.asarray(window.episode_index, dtype=np.int32),
            "start_chunk_id": np.asarray(window.start_chunk_id, dtype=np.int32),
            "num_chunks": np.asarray(actual_num_chunks, dtype=np.int32),
            "meta": {
                "episode_index": np.asarray(window.episode_index, dtype=np.int32),
                "task_index": np.asarray(task_index, dtype=np.int32),
                "num_chunks": np.asarray(actual_num_chunks, dtype=np.int32),
                "start_chunk_id": np.asarray(window.start_chunk_id, dtype=np.int32),
            },
            "prompt": {
                "input_ids": np.asarray(prompt["tokenized_prompt"], dtype=np.int32),
                "attention_mask": np.asarray(prompt["tokenized_prompt_mask"], dtype=np.bool_),
            },
            "chunks": {
                "chunk_ids": chunk_ids,
                "mask": chunk_mask,
                "delta_t": delta_t,
                "images": images,
                "image_masks": image_masks,
                "state": state,
                "target_actions": target_actions,
                "target_mask": target_mask,
            },
        }

    @functools.lru_cache(maxsize=16_384)
    def _shared_step(self, index: int) -> dict:
        return self._shared_transform(self._raw_step(index))

    @functools.lru_cache(maxsize=16_384)
    def _current_step(self, index: int) -> dict:
        return self._current_transform(self._raw_step(index))

    @functools.lru_cache(maxsize=16_384)
    def _timestamp(self, index: int) -> float:
        flat = _transforms.flatten_dict(self._raw_step(index))
        timestamp = _find_first_scalar(flat, ("timestamp", "observation/timestamp", "observation/timestamps"))
        if timestamp is not None:
            return float(timestamp)
        raise KeyError(
            "Temporal-memory training requires a real per-step timestamp in the dataset. "
            f"Missing timestamp for sample index {index}; expected one of "
            "['timestamp', 'observation/timestamp', 'observation/timestamps']."
        )

    def clear_runtime_caches(self) -> None:
        self._raw_step.cache_clear()
        self._shared_step.cache_clear()
        self._current_step.cache_clear()
        self._timestamp.cache_clear()

    def _delta_t(self, current_index: int, next_index: int) -> np.float32:
        episode = self._episode_for_index(current_index)
        if next_index >= episode.end:
            raise IndexError(
                f"Temporal delta_t requested past the end of episode {episode.episode_index}: "
                f"{current_index} -> {next_index}."
            )
        delta_t = self._timestamp(next_index) - self._timestamp(current_index)
        if delta_t <= 0:
            raise ValueError(
                "Temporal-memory training requires strictly increasing real timestamps. "
                f"Got non-positive delta_t={delta_t} between dataset indices {current_index} and {next_index}."
            )
        return np.asarray(delta_t, dtype=np.float32)

    def _build_episode_index(self) -> list[EpisodeInfo]:
        episode_data_index = getattr(self._dataset, "episode_data_index", None)
        if episode_data_index is not None:
            starts = _find_array_field(episode_data_index, ("from", "start", "episode_start"))
            ends = _find_array_field(episode_data_index, ("to", "end", "episode_end"))
            if starts is not None and ends is not None:
                episodes = []
                for episode_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                    task_index = self._task_index(int(start))
                    episodes.append(
                        EpisodeInfo(
                            start=int(start),
                            end=int(end),
                            episode_index=episode_index,
                            task_index=task_index,
                        )
                    )
                return episodes

        episodes: list[EpisodeInfo] = []
        current_episode = None
        start = 0
        for index in range(len(self._dataset)):
            episode_index = self._episode_index(index)
            if current_episode is None:
                current_episode = episode_index
                start = index
                continue
            if episode_index != current_episode:
                episodes.append(
                    EpisodeInfo(
                        start=start,
                        end=index,
                        episode_index=int(current_episode),
                        task_index=self._task_index(start),
                    )
                )
                current_episode = episode_index
                start = index

        if current_episode is None:
            return [EpisodeInfo(start=0, end=len(self._dataset), episode_index=0)]
        episodes.append(
            EpisodeInfo(
                start=start,
                end=len(self._dataset),
                episode_index=int(current_episode),
                task_index=self._task_index(start),
            )
        )
        return episodes

    def _build_window_index(self) -> tuple[list[WindowInfo], list[list[int]], int]:
        windows: list[WindowInfo] = []
        window_groups: list[list[int]] = []
        padded_tail_windows = 0

        for episode_offset, episode in enumerate(self._episodes):
            episode_num_steps = episode.end - episode.start
            episode_num_chunks = max((episode_num_steps + self._tbptt_chunk_len - 1) // self._tbptt_chunk_len, 1)
            num_windows = max((episode_num_chunks + self._tbptt_num_chunks - 1) // self._tbptt_num_chunks, 1)
            for window_index in range(num_windows):
                if len(window_groups) <= window_index:
                    window_groups.append([])
                start_chunk_id = window_index * self._tbptt_num_chunks
                raw_start = episode.start + start_chunk_id * self._tbptt_chunk_len
                raw_end = min(raw_start + self._tbptt_num_chunks * self._tbptt_chunk_len, episode.end)
                actual_num_chunks = max((max(raw_end - raw_start, 0) + self._tbptt_chunk_len - 1) // self._tbptt_chunk_len, 0)
                if actual_num_chunks < self._tbptt_num_chunks:
                    padded_tail_windows += 1
                sample_index = len(windows)
                windows.append(
                    WindowInfo(
                        episode_offset=episode_offset,
                        episode_index=episode.episode_index,
                        task_index=episode.task_index,
                        raw_start=raw_start,
                        raw_end=raw_end,
                        start_chunk_id=start_chunk_id,
                    )
                )
                window_groups[window_index].append(sample_index)

        return windows, window_groups, padded_tail_windows

    def _episode_for_index(self, index: int) -> EpisodeInfo:
        if index < 0 or index >= len(self._sample_to_episode):
            raise IndexError(f"Index {index} is out of range for temporal-memory dataset.")
        return self._episodes[int(self._sample_to_episode[index])]

    @functools.lru_cache(maxsize=16_384)
    def _raw_step(self, index: int) -> dict:
        return copy.deepcopy(self._dataset[index])

    def _episode_index(self, index: int) -> int:
        flat = _transforms.flatten_dict(self._raw_step(index))
        episode_index = _find_first_scalar(flat, ("episode_index", "episode_id"))
        if episode_index is None:
            return 0
        return int(episode_index)

    def _task_index(self, index: int) -> int:
        flat = _transforms.flatten_dict(self._raw_step(index))
        task_index = _find_first_scalar(flat, ("task_index",))
        if task_index is None:
            return -1
        return int(task_index)

def _find_array_field(tree, names: Sequence[str]) -> np.ndarray | None:
    if isinstance(tree, dict):
        for name in names:
            if name in tree:
                return np.asarray(tree[name])
    for name in names:
        value = getattr(tree, name, None)
        if value is not None:
            return np.asarray(value)
    return None


def _find_first_scalar(flat: dict[str, object], names: Sequence[str]) -> int | float | None:
    for name in names:
        if name not in flat:
            continue
        value = flat[name]
        value = np.asarray(value)
        if value.size == 0:
            continue
        if value.dtype.kind in {"i", "u", "f"}:
            return value.reshape(-1)[0].item()
    return None


class _TemporalWindowBatchSampler:
    """Yield fixed-size batches without violating per-episode TBPTT ordering."""

    def __init__(
        self,
        window_groups: Sequence[Sequence[int]],
        windows: Sequence[WindowInfo],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
        multiple_of: int = 1,
    ):
        self._window_groups = [tuple(group) for group in window_groups]
        self._windows = tuple(windows)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        self._multiple_of = max(1, multiple_of)
        self._episode_windows = self._build_episode_windows()

    def _build_episode_windows(self) -> list[tuple[int, tuple[int, ...]]]:
        episode_to_indices: dict[int, list[int]] = {}
        for sample_index, window in enumerate(self._windows):
            episode_to_indices.setdefault(window.episode_index, []).append(sample_index)
        return [
            (
                episode_id,
                tuple(
                    sorted(
                        indices,
                        key=lambda idx: (
                            self._windows[idx].start_chunk_id,
                            self._windows[idx].raw_start,
                        ),
                    )
                ),
            )
            for episode_id, indices in episode_to_indices.items()
        ]

    def _trim_to_multiple(self, indices: list[int], *, cohort_index: int, round_index: int, emit_logs: bool) -> list[int]:
        if self._multiple_of <= 1:
            return indices
        usable = len(indices) - (len(indices) % self._multiple_of)
        if usable == len(indices):
            return indices
        if usable == 0:
            if emit_logs:
                logger.warning(
                    "Skipping cohort %d round %d because it has only %d windows, which is fewer than the required batch multiple %d.",
                    cohort_index,
                    round_index,
                    len(indices),
                    self._multiple_of,
                )
            return []
        dropped = len(indices) - usable
        if emit_logs:
            logger.warning(
                "Dropping %d windows from cohort %d round %d to keep batch size divisible by %d.",
                dropped,
                cohort_index,
                round_index,
                self._multiple_of,
            )
        return indices[:usable]

    def _iter_round_batches(self, *, emit_logs: bool):
        rng = np.random.default_rng(self._seed)
        episode_windows = list(self._episode_windows)
        if self._shuffle:
            rng.shuffle(episode_windows)

        for cohort_index, cohort_start in enumerate(range(0, len(episode_windows), self._batch_size)):
            cohort = episode_windows[cohort_start : cohort_start + self._batch_size]
            if self._drop_last and len(cohort) < self._batch_size:
                if emit_logs:
                    logger.warning(
                        "Skipping final cohort %d because it has only %d episodes for batch_size=%d.",
                        cohort_index,
                        len(cohort),
                        self._batch_size,
                    )
                break

            active_episodes = list(cohort)
            round_index = 0
            while active_episodes:
                indices = []
                next_active_episodes = []
                for episode_id, windows_for_episode in active_episodes:
                    if round_index >= len(windows_for_episode):
                        continue
                    indices.append(windows_for_episode[round_index])
                    if round_index + 1 < len(windows_for_episode):
                        next_active_episodes.append((episode_id, windows_for_episode))

                if not indices:
                    break

                if self._drop_last:
                    if len(indices) < self._batch_size:
                        if emit_logs:
                            logger.warning(
                                "Stopping cohort %d at round %d because only %d episodes remain for batch_size=%d.",
                                cohort_index,
                                round_index,
                                len(indices),
                                self._batch_size,
                            )
                        break
                    batch_indices = indices
                else:
                    batch_indices = self._trim_to_multiple(
                        indices,
                        cohort_index=cohort_index,
                        round_index=round_index,
                        emit_logs=emit_logs,
                    )
                    if not batch_indices:
                        break
                    kept_episode_ids = {self._windows[idx].episode_index for idx in batch_indices}
                    next_active_episodes = [
                        (episode_id, windows_for_episode)
                        for episode_id, windows_for_episode in next_active_episodes
                        if episode_id in kept_episode_ids
                    ]

                yield batch_indices
                active_episodes = next_active_episodes
                round_index += 1

    def __iter__(self):
        yield from self._iter_round_batches(emit_logs=True)

    def __len__(self) -> int:
        return sum(1 for _ in self._iter_round_batches(emit_logs=False))


class CachedTemporalMemoryDataset:
    """Reads precomputed temporal TBPTT samples from a local cache directory."""

    def __init__(self, cache_dir: str | pathlib.Path, tbptt_num_chunks: int, tbptt_chunk_len: int):
        self._cache_dir = pathlib.Path(cache_dir)
        metadata_path = self._cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing temporal cache metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        timing_source = metadata.get("timing_source")
        if timing_source != "timestamp":
            raise ValueError(
                "Temporal cache is stale or incompatible: expected metadata timing_source='timestamp'. "
                f"Got {timing_source!r}. Rebuild the temporal cache from dataset timestamps."
            )
        self._tbptt_num_chunks = tbptt_num_chunks
        cached_tbptt = int(metadata["tbptt_num_chunks"])
        if cached_tbptt != tbptt_num_chunks:
            raise ValueError(
                f"Temporal cache tbptt_num_chunks={cached_tbptt} does not match config tbptt_num_chunks={tbptt_num_chunks}."
            )
        cached_chunk_len = int(metadata.get("tbptt_chunk_len", 1))
        if cached_chunk_len != tbptt_chunk_len:
            raise ValueError(
                "Temporal cache tbptt_chunk_len="
                f"{cached_chunk_len} does not match config tbptt_chunk_len={tbptt_chunk_len}. "
                "Rebuild the temporal cache."
            )
        self._window_groups = [[int(v) for v in group] for group in metadata["window_groups"]]
        self._legacy_sample_files = None
        self._shards = None
        self._shard_starts = None
        self._windows = None
        self._column_specs = metadata.get("column_specs")
        if "shard_files" in metadata:
            self._shards = [
                {
                    "path": self._cache_dir / shard["file"],
                    "start_index": int(shard["start_index"]),
                    "num_rows": int(shard["num_rows"]),
                }
                for shard in metadata["shard_files"]
            ]
            self._shard_starts = [int(shard["start_index"]) for shard in self._shards]
            self._windows = [
                WindowInfo(
                    episode_offset=sample_index,
                    episode_index=int(window["episode_id"]),
                    task_index=-1,
                    raw_start=0,
                    raw_end=0,
                    start_chunk_id=int(window["start_chunk_id"]),
                )
                for sample_index, window in enumerate(metadata["windows"])
            ]
            self._num_samples = int(metadata["num_samples"])
        else:
            self._legacy_sample_files = [self._cache_dir / name for name in metadata["sample_files"]]
            self._num_samples = len(self._legacy_sample_files)

    def __len__(self) -> int:
        return self._num_samples

    def make_batch_sampler(self, batch_size: int, *, shuffle: bool, seed: int, multiple_of: int = 1):
        if self._windows is None:
            windows = []
            assert self._legacy_sample_files is not None
            for sample_index, path in enumerate(self._legacy_sample_files):
                encoded = path.stem.split("_")
                start_chunk_id = int(encoded[-1])
                episode_id = int(encoded[-2])
                windows.append(
                    WindowInfo(
                        episode_offset=sample_index,
                        episode_index=episode_id,
                        task_index=-1,
                        raw_start=0,
                        raw_end=0,
                        start_chunk_id=start_chunk_id,
                    )
                )
        else:
            windows = self._windows
        return _TemporalWindowBatchSampler(
            self._window_groups,
            windows,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            multiple_of=multiple_of,
        )

    def __getitem__(self, index: int) -> dict:
        if self._legacy_sample_files is not None:
            with np.load(self._legacy_sample_files[index], allow_pickle=False) as data:
                flat = {key: data[key] for key in data.files}
            return _transforms.unflatten_dict(flat)

        shard_index, row_index = self._locate_shard(index)
        if self._column_specs is not None:
            table = self._load_shard_table(shard_index)
            return _deserialize_sample_columns(table, row_index, self._column_specs)
        payloads = self._load_shard_payloads(shard_index)
        return _deserialize_sample_bytes(payloads[row_index].as_py())

    def _locate_shard(self, index: int) -> tuple[int, int]:
        assert self._shards is not None and self._shard_starts is not None
        shard_index = int(np.searchsorted(self._shard_starts, index, side="right") - 1)
        if shard_index < 0 or shard_index >= len(self._shards):
            raise IndexError(f"Temporal cache sample index {index} is out of range.")
        shard = self._shards[shard_index]
        row_index = index - shard["start_index"]
        if row_index < 0 or row_index >= shard["num_rows"]:
            raise IndexError(f"Temporal cache sample index {index} resolved to invalid row {row_index}.")
        return shard_index, row_index

    @functools.lru_cache(maxsize=2)
    def _load_shard_payloads(self, shard_index: int):
        assert self._shards is not None
        table = pq.read_table(self._shards[shard_index]["path"], columns=["payload"])
        return table.column("payload")

    @functools.lru_cache(maxsize=2)
    def _load_shard_table(self, shard_index: int):
        assert self._shards is not None
        return pq.read_table(self._shards[shard_index]["path"])


def save_temporal_cache(
    output_dir: str | pathlib.Path,
    data_config: _config.DataConfig,
    model_config: pi0_config.Pi0Config,
    *,
    skip_norm_stats: bool = False,
    overwrite: bool = False,
    shard_size: int = 512,
    num_workers: int = 1,
) -> pathlib.Path:
    dataset = create_temporal_memory_dataset(
        data_config,
        model_config,
        skip_norm_stats=skip_norm_stats,
    )

    output_path = pathlib.Path(output_dir)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Temporal cache directory already exists: {output_path}")
        for child in output_path.iterdir():
            if child.is_dir():
                raise IsADirectoryError(f"Temporal cache directory contains subdirectory {child}, refusing to overwrite.")
            child.unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}.")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}.")

    sample_count = len(dataset)
    shard_ranges: list[tuple[int, int, int]] = []
    for shard_index, start_index in enumerate(range(0, sample_count, shard_size)):
        end_index = min(start_index + shard_size, sample_count)
        shard_ranges.append((shard_index, start_index, end_index))

    shard_files: list[dict[str, int | str]] = []
    windows = [
        {
            "episode_id": int(window.episode_index),
            "start_chunk_id": int(window.start_chunk_id),
        }
        for window in getattr(dataset, "_windows", [])
    ]

    writer = functools.partial(_write_cache_shard, dataset=dataset, output_path=output_path)
    if num_workers == 1:
        shard_results = [writer(shard_index, start_index, end_index) for shard_index, start_index, end_index in shard_ranges]
    else:
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            shard_results = list(
                executor.map(
                    lambda args: writer(*args),
                    shard_ranges,
                )
            )

    shard_results.sort(key=lambda item: item["start_index"])
    shard_files.extend(shard_results)

    metadata = {
        "format": "parquet_columns_v2",
        "timing_source": "timestamp",
        "tbptt_chunk_len": int(model_config.tbptt_chunk_len),
        "tbptt_num_chunks": int(model_config.tbptt_num_chunks),
        "num_samples": sample_count,
        "shard_files": shard_files,
        "windows": windows,
        "window_groups": getattr(dataset, "_window_groups", []),
        "column_specs": _infer_cache_column_specs(dataset[0]),
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata))
    return output_path


def _serialize_sample_bytes(sample: dict) -> bytes:
    flat = {key: np.asarray(value) for key, value in _transforms.flatten_dict(sample).items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **flat)
    return buffer.getvalue()


def _deserialize_sample_bytes(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        flat = {key: data[key] for key in data.files}
    return _transforms.unflatten_dict(flat)


def _write_cache_shard(
    shard_index: int,
    start_index: int,
    end_index: int,
    *,
    dataset: TemporalMemoryDataset | FakeTemporalMemoryDataset,
    output_path: pathlib.Path,
) -> dict[str, int | str]:
    filename = f"shard_{shard_index:05d}.parquet"
    writer = None
    column_specs = None
    try:
        for group_start in range(start_index, end_index, _CACHE_ROW_GROUP_SIZE):
            group_end = min(group_start + _CACHE_ROW_GROUP_SIZE, end_index)
            flat_samples = [
                {key: np.asarray(value) for key, value in _transforms.flatten_dict(dataset[sample_index]).items()}
                for sample_index in range(group_start, group_end)
            ]
            if column_specs is None:
                column_specs = _infer_cache_column_specs(_transforms.unflatten_dict(flat_samples[0]))
            table = _build_explicit_column_table(flat_samples, column_specs)
            if writer is None:
                writer = pq.ParquetWriter(output_path / filename, table.schema, compression="zstd")
            writer.write_table(table)
            if hasattr(dataset, "clear_runtime_caches"):
                dataset.clear_runtime_caches()
            del flat_samples, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    logger.info(
        "Wrote temporal cache shard %s with %d samples (sample indices [%d, %d)).",
        filename,
        end_index - start_index,
        start_index,
        end_index,
    )
    return {
        "file": filename,
        "start_index": start_index,
        "num_rows": end_index - start_index,
    }


def _infer_cache_column_specs(sample: dict) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    flat = {key: np.asarray(value) for key, value in _transforms.flatten_dict(sample).items()}
    for key, value in flat.items():
        specs[key] = {
            "dtype": np.dtype(value.dtype).name,
            "shape": list(value.shape),
        }
    return specs


def _arrow_type_for_dtype(dtype: np.dtype) -> pa.DataType:
    if np.issubdtype(dtype, np.bool_):
        return pa.bool_()
    if np.issubdtype(dtype, np.uint8):
        return pa.uint8()
    if np.issubdtype(dtype, np.int32):
        return pa.int32()
    if np.issubdtype(dtype, np.int64):
        return pa.int64()
    if np.issubdtype(dtype, np.float32):
        return pa.float32()
    if np.issubdtype(dtype, np.float64):
        return pa.float64()
    raise TypeError(f"Unsupported temporal cache dtype: {dtype}")


def _build_explicit_column_table(
    flat_samples: list[dict[str, np.ndarray]],
    column_specs: dict[str, dict[str, object]],
) -> pa.Table:
    arrays: dict[str, pa.Array] = {}
    for key, spec in column_specs.items():
        dtype = np.dtype(spec["dtype"])
        shape = tuple(int(v) for v in spec["shape"])
        value_type = _arrow_type_for_dtype(dtype)
        values = [sample[key] for sample in flat_samples]
        if shape:
            stacked = np.stack([value.reshape(shape) for value in values], axis=0).astype(dtype, copy=False)
            flat = stacked.reshape(-1)
            arrays[key] = pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=value_type),
                list_size=int(np.prod(shape)),
            )
        else:
            stacked = np.asarray([value.item() for value in values], dtype=dtype)
            arrays[key] = pa.array(stacked, type=value_type)
    return pa.table(arrays)


def _deserialize_sample_columns(
    table: pa.Table,
    row_index: int,
    column_specs: dict[str, dict[str, object]],
) -> dict:
    flat: dict[str, np.ndarray] = {}
    for key, spec in column_specs.items():
        dtype = np.dtype(spec["dtype"])
        shape = tuple(int(v) for v in spec["shape"])
        value = table.column(key)[row_index].as_py()
        if shape:
            flat[key] = np.asarray(value, dtype=dtype).reshape(shape)
        else:
            flat[key] = np.asarray(value, dtype=dtype)
    return _transforms.unflatten_dict(flat)
