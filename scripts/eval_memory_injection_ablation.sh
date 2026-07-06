#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENPI_MEMORY_ABLATION_CHECKPOINT_ROOT:-}" ]]; then
  echo "OPENPI_MEMORY_ABLATION_CHECKPOINT_ROOT must point to trained ablation checkpoints." >&2
  exit 2
fi

mkdir -p outputs/memory_ablation/results
out="outputs/memory_ablation/results/memory_injection_ablation.csv"
echo "strategy,checkpoint_path,task_suite,task_name,seed,success,stage_success,episode_length,failure_reason,hidden_norm_mean,tau_mean,k_mean,injection_norm_mean" > "${out}"

for checkpoint in "${OPENPI_MEMORY_ABLATION_CHECKPOINT_ROOT}"/*; do
  [[ -d "${checkpoint}" ]] || continue
  strategy="$(basename "${checkpoint}")"
  if [[ -z "${OPENPI_MEMORY_ABLATION_EVAL_CMD:-}" ]]; then
    echo "OPENPI_MEMORY_ABLATION_EVAL_CMD must be set to the repository evaluation command for real rollouts." >&2
    exit 2
  fi
  OPENPI_RECORD_MEMORY_DIAGNOSTICS_DIR="outputs/memory_ablation/results/${strategy}" \
    ${OPENPI_MEMORY_ABLATION_EVAL_CMD} "${strategy}" "${checkpoint}" "${out}"
done
