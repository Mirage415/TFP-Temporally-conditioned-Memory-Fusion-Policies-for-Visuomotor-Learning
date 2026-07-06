import dataclasses
import pathlib

import tyro

from openpi.training import config as _config
from openpi.training import temporal_memory_loader


def main(
    config_name: str = "tfp",
    output_dir: str | None = None,
    overwrite: bool = False,
    skip_norm_stats: bool = False,
    shard_size: int = 512,
    num_workers: int = 4,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if output_dir is None:
        output_dir = str((pathlib.Path(config.assets_dirs) / "temporal_cache").resolve())

    print(f"[cache] config={config.name}")
    print(f"[cache] repo_id={data_config.repo_id}")
    print(f"[cache] asset_id={data_config.asset_id}")
    print(f"[cache] tbptt_chunk_len={config.model.tbptt_chunk_len}")
    print(f"[cache] tbptt_num_chunks={config.model.tbptt_num_chunks}")
    print(f"[cache] output_dir={output_dir}")
    print(f"[cache] shard_size={shard_size}")
    print(f"[cache] num_workers={num_workers}")

    path = temporal_memory_loader.save_temporal_cache(
        output_dir,
        data_config,
        dataclasses.replace(config.model),
        skip_norm_stats=skip_norm_stats,
        overwrite=overwrite,
        shard_size=shard_size,
        num_workers=num_workers,
    )
    print(f"[cache] wrote temporal cache to {path}")


if __name__ == "__main__":
    tyro.cli(main)
