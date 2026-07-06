import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _minimal_temporal_policy() -> _policy.Policy:
    policy = object.__new__(_policy.Policy)
    policy._temporal_last_timestamp = None
    return policy


def test_extract_temporal_delta_t_prefers_timestamp():
    policy = _minimal_temporal_policy()

    policy._temporal_last_timestamp = 10.0
    delta_t = policy._extract_temporal_delta_t({"timestamp": np.asarray(10.05, dtype=np.float32)}, batch_size=1)

    np.testing.assert_allclose(np.asarray(delta_t), np.array([0.05], dtype=np.float32))


def test_extract_temporal_delta_t_uses_wall_clock(monkeypatch: pytest.MonkeyPatch):
    policy = _minimal_temporal_policy()
    monotonic_values = iter([100.0, 100.3])
    monkeypatch.setattr(_policy.time, "monotonic", lambda: next(monotonic_values))

    first = policy._extract_temporal_delta_t({}, batch_size=1)
    second = policy._extract_temporal_delta_t({}, batch_size=1)

    np.testing.assert_allclose(np.asarray(first), np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(second), np.array([0.3], dtype=np.float32), atol=1e-6)


def test_extract_temporal_delta_t_uses_timestamp_delta():
    policy = _minimal_temporal_policy()

    first = policy._extract_temporal_delta_t({"timestamp": np.asarray(10.0, dtype=np.float32)}, batch_size=1)
    second = policy._extract_temporal_delta_t({"timestamp": np.asarray(10.3, dtype=np.float32)}, batch_size=1)

    np.testing.assert_allclose(np.asarray(first), np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(second), np.array([0.3], dtype=np.float32), atol=1e-6)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
