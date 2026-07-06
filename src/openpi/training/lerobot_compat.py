from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import jsonlines
from PIL import Image

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.utils import hf_transform_to_torch


@dataclasses.dataclass(frozen=True)
class LocalImageTorchTransform:
    root: Path
    image_keys: frozenset[str]

    def __call__(self, items_dict: dict) -> dict:
        converted = dict(items_dict)
        for key in self.image_keys.intersection(converted):
            images = []
            for item in converted[key]:
                if isinstance(item, dict):
                    if item.get("bytes") is not None:
                        images.append(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
                    else:
                        images.append(Image.open(self.root / item["path"]).convert("RGB"))
                else:
                    images.append(item)
            converted[key] = images
        return hf_transform_to_torch(converted)


def ensure_local_episodes_stats(repo_id: str) -> None:
    """Backfill v2.1 metadata for local LeRobot datasets that only have stats.json.

    Some local exports do not include meta/episodes_stats.jsonl yet. The upstream
    LeRobot loader expects that file for v2.1 datasets and otherwise falls back to
    Hugging Face. For local offline runs, we synthesize a compatible file by
    repeating the dataset-level stats for each episode.
    """

    root = Path(HF_LEROBOT_HOME) / repo_id
    meta_dir = root / "meta"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    stats_path = meta_dir / "stats.json"
    episodes_path = meta_dir / "episodes.jsonl"
    info_path = meta_dir / "info.json"
    if not stats_path.exists() or not episodes_path.exists():
        return

    if info_path.exists():
        info = json.loads(info_path.read_text())
        if "chunks_size" not in info:
            info["chunks_size"] = 1000
            info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False))

    stats = json.loads(stats_path.read_text())
    episodes: list[dict] = []
    with jsonlines.open(episodes_path, "r") as reader:
        for item in reader:
            episodes.append(item)

    if not episodes:
        return

    meta_dir.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(episodes_stats_path, "w") as writer:
        for item in episodes:
            episode_index = int(item["episode_index"])
            episode_length = int(item["length"])
            episode_stats = {
                feature_name: {**feature_stats, "count": [episode_length]}
                for feature_name, feature_stats in stats.items()
            }
            writer.write({"episode_index": episode_index, "stats": episode_stats})


def patch_local_image_transform(dataset) -> None:
    """Load relative image paths from a local LeRobot dataset before torch formatting."""

    root = Path(dataset.root)
    image_keys = frozenset(getattr(dataset.meta, "image_keys", []))
    if not image_keys:
        return
    dataset.hf_dataset.set_transform(LocalImageTorchTransform(root=root, image_keys=image_keys))
