# Memory Injection Ablation

This implementation adds one shared LTC memory state `h_t` and routes it to exactly one injection site selected by
`memory_injection_strategy`.

Strategies:

- `memory_as_input_token`: project `h_t` to one prefix token and insert it into the real pi0.5 prefix sequence before the temporal state token.
- `memory_to_vlm_backbone`: compute per-layer `gamma_l, beta_l` from `h_t` and modulate the VLM stream as `y + s_l (gamma_l LN(y) + beta_l)`.
- `memory_by_retrieved_context_token`: store detached past `(x_i, h_i)` in an episode FIFO bank, retrieve top-k by cosine similarity, and pass retrieved tokens through the existing action-expert memory cross-attention path.
- `memory_to_action_head_concat`: concatenate a projection of `h_t` to action suffix tokens, then project back to the original action hidden size.
- `memory_to_action_head_adaln`: full TFP method. Project `h_t` and add it to the diffusion timestep AdaLN condition used by every pi0.5 action-expert layer.

Common LTC:

`x_t = concat(phi_vision(V_t), phi_state(s_t))`

`h_hat_t = tanh(W_h [x_t ; h_{t-1}] + b_h)`

`tau_t = softplus(W_tau [x_t ; h_{t-1}] + b_tau) + eps`

`k_t = exp(-delta_t / tau_t)`

`h_t = k_t h_{t-1} + (1-k_t) h_hat_t`

Modified files:

- `src/openpi/models/pi0_config.py`
- `src/openpi/models/temporal_memory.py`
- `src/openpi/models/memory_injection.py`
- `src/openpi/models/gemma.py`
- `src/openpi/models/pi0_adaln.py`
- `src/openpi/training/config.py`
- `scripts/train.py`
- `src/openpi/policies/policy.py`
- `scripts/analyze_same_observation_different_history.py`
- `scripts/run_memory_injection_ablation_smoke.sh`
- `scripts/run_memory_injection_ablation_full.sh`
- `scripts/eval_memory_injection_ablation.sh`

Configs:

- `pi05_tfp_no_memory`
- `pi05_tfp_input_token`
- `pi05_tfp_vlm_backbone`
- `pi05_tfp_retrieved_context`
- `pi05_tfp_action_concat`
- `pi05_tfp_action_adaln`

Run smoke training:

```bash
OPENPI_MEMORY_ABLATION_ASSETS_DIR=/path/to/real/assets \
  scripts/run_memory_injection_ablation_smoke.sh
```

Run full training:

```bash
OPENPI_MEMORY_ABLATION_ASSETS_DIR=/path/to/real/assets \
  scripts/run_memory_injection_ablation_full.sh
```

Run evaluation:

```bash
OPENPI_MEMORY_ABLATION_CHECKPOINT_ROOT=outputs/memory_ablation/checkpoints \
OPENPI_MEMORY_ABLATION_EVAL_CMD='your real LIBERO eval command wrapper' \
  scripts/eval_memory_injection_ablation.sh
```

Run same-observation different-history analysis:

```bash
uv run scripts/analyze_same_observation_different_history.py \
  --config-name=pi05_tfp_action_adaln \
  --checkpoint-dir=/path/to/checkpoint/step/params
```

Outputs are written under `outputs/memory_ablation`. The CSV reports strategy, checkpoint, task metadata, success, episode length, and memory statistics when available.

Known limitations:

- The repository has no single built-in batch LIBERO evaluator that writes the requested aggregate CSV, so `eval_memory_injection_ablation.sh` requires a real evaluation command wrapper through `OPENPI_MEMORY_ABLATION_EVAL_CMD`.
- The smoke script intentionally refuses fake data; it requires real LIBERO assets and dataset availability.
