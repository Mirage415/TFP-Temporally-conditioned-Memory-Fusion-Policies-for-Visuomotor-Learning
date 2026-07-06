from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_temporal_memory_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        temporal_memory_enabled=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        tbptt_chunk_len=4,
        ltc_hidden_dim=32,
    )
    model = config.create(key)

    batch_size = 2
    num_chunks = config.tbptt_num_chunks
    chunk_images = jax.numpy.zeros((batch_size, num_chunks, *_model.IMAGE_RESOLUTION, 3), dtype=jax.numpy.float32)
    batch = {
        "meta": {
            "episode_index": jax.numpy.zeros((batch_size,), dtype=jax.numpy.int32),
            "task_index": jax.numpy.zeros((batch_size,), dtype=jax.numpy.int32),
            "num_chunks": jax.numpy.full((batch_size,), num_chunks, dtype=jax.numpy.int32),
        },
        "prompt": {
            "input_ids": jax.numpy.ones((batch_size, config.max_token_len), dtype=jax.numpy.int32),
            "attention_mask": jax.numpy.ones((batch_size, config.max_token_len), dtype=jax.numpy.bool_),
        },
        "chunks": {
            "chunk_ids": jax.numpy.broadcast_to(jax.numpy.arange(num_chunks, dtype=jax.numpy.int32), (batch_size, num_chunks)),
            "mask": jax.numpy.ones((batch_size, num_chunks), dtype=jax.numpy.bool_),
            "delta_t": jax.numpy.ones((batch_size, num_chunks), dtype=jax.numpy.float32),
            "images": {name: chunk_images for name in _model.IMAGE_KEYS},
            "image_masks": {
                name: jax.numpy.ones((batch_size, num_chunks), dtype=jax.numpy.bool_) for name in _model.IMAGE_KEYS
            },
            "state": jax.numpy.ones((batch_size, num_chunks, config.action_dim), dtype=jax.numpy.float32),
            "target_actions": jax.numpy.ones(
                (batch_size, num_chunks, config.action_horizon, config.action_dim), dtype=jax.numpy.float32
            ),
            "target_mask": jax.numpy.ones((batch_size, num_chunks, config.action_horizon), dtype=jax.numpy.bool_),
        },
    }

    loss = nnx_utils.module_jit(model.compute_temporal_loss)(key, batch)
    assert loss.shape == ()

    obs = config.fake_obs(batch_size)
    hidden = model.ltc_encoder.initial_state(batch_size)
    actions, next_hidden = nnx_utils.module_jit(model.sample_actions_temporal)(
        key,
        obs,
        hidden,
        delta_t=jax.numpy.ones((batch_size,), dtype=jax.numpy.float32),
        num_steps=4,
    )
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)
    assert next_hidden.shape == (batch_size, config.ltc_hidden_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
