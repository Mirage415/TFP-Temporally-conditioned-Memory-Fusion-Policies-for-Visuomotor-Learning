import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import pi0_adaln_config
from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


def test_tfp_adaln_curriculum_config():
    config = _config.get_config("tfp_short_adaln_curriculum")

    assert config.has_curriculum
    assert config.curriculum_unfreeze_step == 5_000
    assert config.curriculum_freeze_patterns == pi0_adaln_config.Pi0AdaLNConfig.backbone_path_patterns()
