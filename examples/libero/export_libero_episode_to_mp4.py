"""
Export a single LIBERO episode to an mp4 video.

Usage:
python examples/libero/export_libero_episode_to_mp4.py \
    --data-dir /path/to/lerobot/libero \
    --episode-idx 0

The script supports two input layouts:
- A LeRobot dataset root (preferred if you already converted LIBERO), e.g.
  /path/to/lerobot/libero
- A raw TFDS/RLDS root containing builders such as libero_10_no_noops

The raw LIBERO episodes contain image observations for each step:
- observation["image"]
- observation["wrist_image"]

This script stitches one episode's image sequence into a replay video.
"""

from collections.abc import Iterable
import io
import json
import pathlib
from typing import Literal

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import tyro


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}.")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _build_frame(step: dict, camera: Literal["agent", "wrist", "both"]) -> np.ndarray:
    image = _to_uint8_hwc(step["observation"]["image"])
    wrist_image = _to_uint8_hwc(step["observation"]["wrist_image"])

    if camera == "agent":
        return image
    if camera == "wrist":
        return wrist_image
    return np.concatenate([image, wrist_image], axis=1)


def _iter_episode_frames(episode: dict, camera: Literal["agent", "wrist", "both"]) -> Iterable[np.ndarray]:
    for step in episode["steps"].as_numpy_iterator():
        yield _build_frame(step, camera)


def _is_lerobot_dataset_root(path: pathlib.Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _load_lerobot_tasks(meta_dir: pathlib.Path) -> dict[int, str]:
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = {}
    with tasks_path.open() as f:
        for line in f:
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
    return tasks


def _decode_lerobot_image(image_value: dict, dataset_root: pathlib.Path) -> np.ndarray:
    if image_value is None:
        raise ValueError("Missing image payload in LeRobot row.")

    image_bytes = image_value.get("bytes")
    image_path = image_value.get("path")

    if image_bytes is not None:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"))

    if image_path is not None:
        image_path = pathlib.Path(image_path)
        if not image_path.is_absolute():
            image_path = dataset_root / image_path
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))

    raise ValueError("LeRobot image payload does not contain bytes or path.")


def _build_lerobot_frame(row: dict, dataset_root: pathlib.Path, camera: Literal["agent", "wrist", "both"]) -> np.ndarray:
    image = _decode_lerobot_image(row["image"], dataset_root)
    wrist_image = _decode_lerobot_image(row["wrist_image"], dataset_root)

    if camera == "agent":
        return image
    if camera == "wrist":
        return wrist_image
    return np.concatenate([image, wrist_image], axis=1)


def _load_lerobot_episode(
    dataset_root: pathlib.Path,
    episode_idx: int,
    camera: Literal["agent", "wrist", "both"],
) -> tuple[list[np.ndarray], str | None]:
    meta_dir = dataset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info["chunks_size"])

    if episode_idx < 0 or episode_idx >= total_episodes:
        raise IndexError(f"episode_idx={episode_idx} is out of range for LeRobot dataset with {total_episodes} episodes.")

    episode_chunk = episode_idx // chunks_size
    episode_path = dataset_root / info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_idx,
    )

    parquet_file = pq.ParquetFile(episode_path)
    frames = []
    task_index = None
    for row_group_idx in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(row_group_idx)
        for row in row_group.to_pylist():
            frames.append(_build_lerobot_frame(row, dataset_root, camera))
            if task_index is None:
                task_index = int(row["task_index"])

    tasks = _load_lerobot_tasks(meta_dir)
    task = tasks.get(task_index) if task_index is not None else None
    return frames, task


def main(
    data_dir: str,
    *,
    dataset_name: str = "libero_10_no_noops",
    episode_idx: int = 0,
    split: str = "train",
    output_path: str | None = None,
    camera: Literal["agent", "wrist", "both"] = "agent",
    fps: int = 10,
) -> None:
    dataset_root = pathlib.Path(data_dir)
    output_stem = dataset_name
    if _is_lerobot_dataset_root(dataset_root):
        frames, task = _load_lerobot_episode(dataset_root, episode_idx, camera)
        output_stem = dataset_root.name
    else:
        import tensorflow_datasets as tfds

        dataset = tfds.load(dataset_name, data_dir=data_dir, split=split)

        if episode_idx < 0 or episode_idx >= len(dataset):
            raise IndexError(
                f"episode_idx={episode_idx} is out of range for split {split!r} with {len(dataset)} episodes."
            )

        episode = next(iter(dataset.skip(episode_idx).take(1)))
        frames = list(_iter_episode_frames(episode, camera))
        if not frames:
            raise ValueError(f"Episode {episode_idx} from dataset {dataset_name!r} has no frames.")

        task = None
        for step in episode["steps"].take(1).as_numpy_iterator():
            task = step["language_instruction"].decode()

    if output_path is None:
        output_dir = pathlib.Path("data/libero/episode_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_stem}_{split}_ep{episode_idx:05d}_{camera}.mp4"
    else:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(output_path, frames, fps=fps)

    print(f"Saved {len(frames)} frames to {output_path}")
    if task is not None:
        print(f"Task: {task}")


if __name__ == "__main__":
    tyro.cli(main)
