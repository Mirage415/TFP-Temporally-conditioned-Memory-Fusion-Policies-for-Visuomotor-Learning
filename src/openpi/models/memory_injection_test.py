import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import memory_injection
from openpi.models import temporal_memory


def test_ltc_memory_update_reset_delta_and_gradients():
    key = jax.random.key(0)
    module = temporal_memory.LiquidTimeConstantMemory(
        vision_dim=6,
        state_dim=4,
        memory_dim=8,
        eps=1e-6,
        use_delta_t=True,
        default_delta_t=1.0,
        rngs=nnx.Rngs(key),
    )
    hidden = module.init_hidden(2)
    visual = [jnp.ones((2, 3, 6))]
    masks = [jnp.ones((2, 3), dtype=jnp.bool_)]
    x_t = module.observation_summary(visual, masks, jnp.ones((2, 4)))
    next_hidden, tau, k, diag = module.step(
        hidden + 1.0,
        x_t,
        jnp.asarray([0.1, 2.0]),
        mask=jnp.asarray([True, True]),
        reset_mask=jnp.asarray([True, False]),
    )
    assert next_hidden.shape == (2, 8)
    assert tau.shape == (2, 8)
    assert k.shape == (2, 8)
    assert diag["hidden_norm"].shape == (2,)
    assert not jnp.allclose(k[0], k[1])

    no_delta = temporal_memory.LiquidTimeConstantMemory(
        vision_dim=6,
        state_dim=4,
        memory_dim=8,
        eps=1e-6,
        use_delta_t=False,
        default_delta_t=1.0,
        rngs=nnx.Rngs(jax.random.key(1)),
    )
    x2 = no_delta.observation_summary(visual, masks, jnp.ones((2, 4)))
    _, _, k2, _ = no_delta.step(hidden, x2, jnp.asarray([0.1, 2.0]), mask=jnp.asarray([True, True]))
    assert jnp.allclose(k2[0], k2[1])


def test_strategy_input_token_exclusive_shapes():
    router = memory_injection.MemoryInjectionRouter(
        "memory_as_input_token",
        8,
        16,
        12,
        4,
        3,
        16,
        action_head_memory_dim=None,
        retrieval_bank_size=4,
        retrieval_top_k=2,
        vlm_adapter_init_scale=0.0,
        rngs=nnx.Rngs(jax.random.key(2)),
    )
    prefix = jnp.ones((2, 5, 16))
    mask = jnp.ones((2, 5), dtype=jnp.bool_)
    ar = jnp.zeros((5,), dtype=jnp.bool_)
    out = router.inject_input_token(prefix, mask, ar, jnp.ones((2, 8)))
    assert out.prefix_tokens.shape == (2, 6, 16)
    assert out.prefix_mask.shape == (2, 6)
    assert out.prefix_ar_mask.shape == (6,)
    assert out.suffix_cond is None


def test_retrieval_bank_topk_and_reset():
    bank = temporal_memory.empty_retrieval_bank(2, 4, 16, 8)
    bank = temporal_memory.append_retrieval_bank(
        bank,
        jnp.ones((2, 16)),
        jnp.ones((2, 8)),
        jnp.asarray([0.1, 0.1]),
        jnp.asarray([True, True]),
    )
    assert jnp.all(bank["size"] == 1)
    bank = temporal_memory.reset_retrieval_bank(bank, jnp.asarray([True, False]))
    assert bank["size"][0] == 0
    assert bank["size"][1] == 1

    router = memory_injection.MemoryInjectionRouter(
        "memory_by_retrieved_context_token",
        8,
        16,
        12,
        4,
        3,
        16,
        action_head_memory_dim=None,
        retrieval_bank_size=4,
        retrieval_top_k=2,
        vlm_adapter_init_scale=0.0,
        rngs=nnx.Rngs(jax.random.key(3)),
    )
    out = router.inject_retrieved_context(jnp.ones((2, 16)), jnp.ones((2, 8)), bank)
    assert out.memory_tokens.shape == (2, 2, 12)
    assert out.memory_mask.shape == (2, 2)


def test_action_concat_and_adaln_shapes_are_exclusive():
    concat_router = memory_injection.MemoryInjectionRouter(
        "memory_to_action_head_concat",
        8,
        16,
        12,
        4,
        3,
        16,
        action_head_memory_dim=5,
        retrieval_bank_size=4,
        retrieval_top_k=2,
        vlm_adapter_init_scale=0.0,
        rngs=nnx.Rngs(jax.random.key(4)),
    )
    suffix = jnp.ones((2, 4, 12))
    out = concat_router.inject_action_concat(suffix, jnp.ones((2, 8)))
    assert out.suffix_tokens.shape == suffix.shape
    assert out.suffix_cond is None

    adaln_router = memory_injection.MemoryInjectionRouter(
        "memory_to_action_head_adaln",
        8,
        16,
        12,
        4,
        3,
        16,
        action_head_memory_dim=None,
        retrieval_bank_size=4,
        retrieval_top_k=2,
        vlm_adapter_init_scale=0.0,
        rngs=nnx.Rngs(jax.random.key(5)),
    )
    cond = adaln_router.inject_action_adaln(jnp.ones((2, 12)), jnp.ones((2, 8)))
    assert cond.suffix_cond.shape == (2, 12)
    assert cond.suffix_tokens is None
