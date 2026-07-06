from __future__ import annotations

import copy
import os
import sys
from collections.abc import Sequence
from concurrent import futures
import dataclasses
import functools
import io
import json
import logging
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_OPENPI_CLIENT_SRC = pathlib.Path(
    os.environ.get("OPENPI_CLIENT_SRC", str(_REPO_ROOT / "packages" / "openpi-client" / "src"))
)
if _OPENPI_CLIENT_SRC.exists() and str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import lerobot_compat as _lerobot_compat
from openpi.training import temporal_memory_loader as temporal_memory_loader
import openpi.transforms as _transforms

logger = logging.getLogger("openpi")
_CACHE_ROW_GROUP_SIZE = 16


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
        value = np.asarray(flat[name])
        if value.size == 0:
            continue
        if value.dtype.kind in {"i", "u", "f"}:
            return value.reshape(-1)[0].item()
    return None


@functools.lru_cache(maxsize=16_384)
def _raw_step(dataset, index: int) -> dict:
    return copy.deepcopy(dataset[index])


def _episode_index(raw_dataset, index: int) -> int:
    flat = _transforms.flatten_dict(_raw_step(raw_dataset, index))
    episode_index = _find_first_scalar(flat, ("episode_index", "episode_id"))
    if episode_index is None:
        return 0
    return int(episode_index)


def _task_index(raw_dataset, index: int) -> int:
    flat = _transforms.flatten_dict(_raw_step(raw_dataset, index))
    task_index = _find_first_scalar(flat, ("task_index",))
    if task_index is None:
        return -1
    return int(task_index)


@functools.lru_cache(maxsize=16_384)
def _timestamp(raw_dataset, index: int) -> float:
    flat = _transforms.flatten_dict(_raw_step(raw_dataset, index))
    timestamp = _find_first_scalar(flat, ("timestamp", "observation/timestamp", "observation/timestamps"))
    if timestamp is not None:
        return float(timestamp)
    raise KeyError(
        "tfp_final-style cache build requires a real per-step timestamp in the dataset. "
        f"Missing timestamp for sample index {index}; expected one of "
        "['timestamp', 'observation/timestamp', 'observation/timestamps']."
    )


def _build_episode_index(raw_dataset) -> list[temporal_memory_loader.EpisodeInfo]:
    episode_data_index = getattr(raw_dataset, "episode_data_index", None)
    if episode_data_index is not None:
        starts = _find_array_field(episode_data_index, ("from", "start", "episode_start"))
        ends = _find_array_field(episode_data_index, ("to", "end", "episode_end"))
        if starts is not None and ends is not None:
            episodes = []
            for episode_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                episodes.append(
                    temporal_memory_loader.EpisodeInfo(
                        start=int(start),
                        end=int(end),
                        episode_index=episode_index,
                        task_index=_task_index(raw_dataset, int(start)),
                    )
                )
            return episodes

    episodes: list[temporal_memory_loader.EpisodeInfo] = []
    current_episode = None
    start = 0
    for index in range(len(raw_dataset)):
        ep = _episode_index(raw_dataset, index)
        if current_episode is None:
            current_episode = ep
            start = index
            continue
        if ep != current_episode:
            episodes.append(
                temporal_memory_loader.EpisodeInfo(
                    start=start,
                    end=index,
                    episode_index=int(current_episode),
                    task_index=_task_index(raw_dataset, start),
                )
            )
            current_episode = ep
            start = index
    if current_episode is None:
        return [temporal_memory_loader.EpisodeInfo(start=0, end=len(raw_dataset), episode_index=0)]
    episodes.append(
        temporal_memory_loader.EpisodeInfo(
            start=start,
            end=len(raw_dataset),
            episode_index=int(current_episode),
            task_index=_task_index(raw_dataset, start),
        )
    )
    return episodes


def _build_window_index(
    episodes: Sequence[temporal_memory_loader.EpisodeInfo], tbptt_num_chunks: int, tbptt_chunk_len: int
) -> tuple[list[temporal_memory_loader.WindowInfo], list[list[int]], int]:
    windows: list[temporal_memory_loader.WindowInfo] = []
    window_groups: list[list[int]] = []
    padded_tail_windows = 0
    for episode_offset, episode in enumerate(episodes):
        episode_num_steps = episode.end - episode.start
        episode_num_chunks = max((episode_num_steps + tbptt_chunk_len - 1) // tbptt_chunk_len, 1)
        num_windows = max((episode_num_chunks + tbptt_num_chunks - 1) // tbptt_num_chunks, 1)
        for window_index in range(num_windows):
            if len(window_groups) <= window_index:
                window_groups.append([])
            start_chunk_id = window_index * tbptt_num_chunks
            raw_start = episode.start + start_chunk_id * tbptt_chunk_len
            raw_end = min(raw_start + tbptt_num_chunks * tbptt_chunk_len, episode.end)
            actual_num_chunks = max((max(raw_end - raw_start, 0) + tbptt_chunk_len - 1) // tbptt_chunk_len, 0)
            if actual_num_chunks < tbptt_num_chunks:
                padded_tail_windows += 1
            sample_index = len(windows)
            windows.append(
                temporal_memory_loader.WindowInfo(
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


def _serialize_sample_bytes(sample: dict) -> bytes:
    flat = {key: np.asarray(value) for key, value in _transforms.flatten_dict(sample).items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **flat)
    return buffer.getvalue()


def _make_sample_builder(
    transformed_dataset,
    raw_dataset,
    windows: Sequence[temporal_memory_loader.WindowInfo],
    episodes: Sequence[temporal_memory_loader.EpisodeInfo],
    model_config,
):
    example = transformed_dataset[0]
    image_keys = tuple(example["image"].keys())
    image_shapes = {key: np.asarray(example["image"][key]).shape for key in image_keys}
    image_dtypes = {key: np.asarray(example["image"][key]).dtype for key in image_keys}
    state_dim = int(np.asarray(example["state"]).shape[-1])
    action_dim = int(np.asarray(example["actions"]).shape[-1])
    action_horizon = int(np.asarray(example["actions"]).shape[0])
    chunk_len = int(model_config.tbptt_chunk_len)

    def build_sample(sample_index: int) -> dict:
        window = windows[sample_index]
        episode = episodes[window.episode_offset]
        prompt = transformed_dataset[window.raw_start]

        chunk_ids = np.full((model_config.tbptt_num_chunks,), -1, dtype=np.int32)
        chunk_mask = np.zeros((model_config.tbptt_num_chunks,), dtype=np.bool_)
        delta_t = np.zeros((model_config.tbptt_num_chunks,), dtype=np.float32)
        state = np.zeros((model_config.tbptt_num_chunks, state_dim), dtype=np.float32)
        images = {
            key: np.zeros((model_config.tbptt_num_chunks, *image_shapes[key]), dtype=image_dtypes[key]) for key in image_keys
        }
        image_masks = {key: np.zeros((model_config.tbptt_num_chunks,), dtype=np.bool_) for key in image_keys}
        target_actions = np.zeros((model_config.tbptt_num_chunks, action_horizon, action_dim), dtype=np.float32)
        target_mask = np.zeros((model_config.tbptt_num_chunks, action_horizon), dtype=np.bool_)

        if window.raw_start >= episode.end:
            actual_num_chunks = 0
        else:
            actual_num_chunks = min(
                model_config.tbptt_num_chunks,
                (episode.end - window.raw_start + chunk_len - 1) // chunk_len,
            )
        for local_chunk_id in range(actual_num_chunks):
            step_index = window.raw_start + local_chunk_id * chunk_len
            if step_index >= episode.end:
                break
            step = transformed_dataset[step_index]
            chunk_id = window.start_chunk_id + local_chunk_id
            chunk_ids[local_chunk_id] = chunk_id
            chunk_mask[local_chunk_id] = True
            delta_t[local_chunk_id] = 0.0 if chunk_id == 0 else np.asarray(
                _timestamp(raw_dataset, step_index) - _timestamp(raw_dataset, step_index - chunk_len), dtype=np.float32
            )
            state[local_chunk_id] = np.asarray(step["state"], dtype=np.float32)
            for key in image_keys:
                images[key][local_chunk_id] = np.asarray(step["image"][key])
                image_masks[key][local_chunk_id] = np.asarray(step["image_mask"][key], dtype=np.bool_)
            target_actions[local_chunk_id] = np.asarray(step["actions"], dtype=np.float32)
            valid_horizon = min(action_horizon, episode.end - step_index)
            target_mask[local_chunk_id, :valid_horizon] = True

        return {
            "episode_id": np.asarray(window.episode_index, dtype=np.int32),
            "start_chunk_id": np.asarray(window.start_chunk_id, dtype=np.int32),
            "num_chunks": np.asarray(actual_num_chunks, dtype=np.int32),
            "meta": {
                "episode_index": np.asarray(window.episode_index, dtype=np.int32),
                "task_index": np.asarray(window.task_index, dtype=np.int32),
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

    return build_sample


def _write_payload_shard(
    shard_index: int,
    start_index: int,
    end_index: int,
    *,
    build_sample,
    output_path: pathlib.Path,
) -> dict[str, int | str]:
    filename = f"shard_{shard_index:05d}.parquet"
    writer = None
    try:
        for group_start in range(start_index, end_index, _CACHE_ROW_GROUP_SIZE):
            group_end = min(group_start + _CACHE_ROW_GROUP_SIZE, end_index)
            payloads = [_serialize_sample_bytes(build_sample(sample_index)) for sample_index in range(group_start, group_end)]
            table = pa.table({"payload": pa.array(payloads, type=pa.large_binary())})
            if writer is None:
                writer = pq.ParquetWriter(output_path / filename, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    logger.info(
        "Wrote tfp_final-style temporal cache shard %s with %d samples (sample indices [%d, %d)).",
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


def main(
    config_name: str = "tfp_short",
    output_dir: str | None = None,
    overwrite: bool = False,
    skip_norm_stats: bool = False,
    shard_size: int = 512,
    num_workers: int = 4,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if output_dir is None:
        output_dir = str((pathlib.Path(config.assets_dirs) / "temporal_cache_tfp_final_style").resolve())

    print(f"[cache] config={config.name}")
    print(f"[cache] repo_id={data_config.repo_id}")
    print(f"[cache] asset_id={data_config.asset_id}")
    print(f"[cache] tbptt_chunk_len={config.model.tbptt_chunk_len}")
    print(f"[cache] tbptt_num_chunks={config.model.tbptt_num_chunks}")
    print(f"[cache] output_dir={output_dir}")
    print(f"[cache] shard_size={shard_size}")
    print(f"[cache] num_workers={num_workers}")
    print("[cache] organizing samples with tfp_final-style create_torch_dataset + transform_dataset")

    transformed_dataset = _data_loader.transform_dataset(
        _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model),
        data_config,
        skip_norm_stats=skip_norm_stats,
    )
    raw_dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, tolerance_s=data_config.lerobot_tolerance_s)
    _lerobot_compat.patch_local_image_transform(raw_dataset)
    episodes = _build_episode_index(raw_dataset)
    windows, window_groups, padded_tail_windows = _build_window_index(
        episodes,
        config.model.tbptt_num_chunks,
        config.model.tbptt_chunk_len,
    )
    print(f"[cache] episodes={len(episodes)} windows={len(windows)} padded_tail_windows={padded_tail_windows}")

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

    build_sample = _make_sample_builder(transformed_dataset, raw_dataset, windows, episodes, config.model)
    sample_count = len(windows)
    shard_ranges = []
    for shard_index, start_index in enumerate(range(0, sample_count, shard_size)):
        end_index = min(start_index + shard_size, sample_count)
        shard_ranges.append((shard_index, start_index, end_index))

    writer = functools.partial(_write_payload_shard, build_sample=build_sample, output_path=output_path)
    if num_workers == 1:
        shard_results = [writer(shard_index, start_index, end_index) for shard_index, start_index, end_index in shard_ranges]
    else:
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            shard_results = list(executor.map(lambda args: writer(*args), shard_ranges))
    shard_results.sort(key=lambda item: item["start_index"])

    metadata = {
        "format": "parquet_payload_v1",
        "timing_source": "timestamp",
        "tbptt_chunk_len": int(config.model.tbptt_chunk_len),
        "tbptt_num_chunks": int(config.model.tbptt_num_chunks),
        "num_samples": sample_count,
        "shard_files": shard_results,
        "windows": [
            {"episode_id": int(window.episode_index), "start_chunk_id": int(window.start_chunk_id)} for window in windows
        ],
        "window_groups": window_groups,
        "source_layout": "tfp_final_style",
        "source_config": config_name,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata))
    print(f"[cache] wrote tfp_final-style temporal cache to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
