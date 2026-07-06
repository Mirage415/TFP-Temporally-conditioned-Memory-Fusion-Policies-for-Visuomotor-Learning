import dataclasses

import tyro

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import utils as training_utils


def main(
    config_name: str = "tfp",
    num_workers: int = 2,
    batch_size: int = 1,
):
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, num_workers=num_workers, batch_size=batch_size)
    print(f"[warmup] config={config.name} batch_size={config.batch_size} num_workers={config.num_workers}")

    loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=1,
    )
    batch = next(iter(loader))
    print("[warmup] first batch ready")
    print(training_utils.array_tree_to_info(batch))


if __name__ == "__main__":
    tyro.cli(main)
