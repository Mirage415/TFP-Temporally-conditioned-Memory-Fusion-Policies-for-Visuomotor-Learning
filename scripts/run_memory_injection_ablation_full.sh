#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  pi05_tfp_no_memory
  pi05_tfp_input_token
  pi05_tfp_vlm_backbone
  pi05_tfp_retrieved_context
  pi05_tfp_action_concat
  pi05_tfp_action_adaln
)

OPENPI_MEMORY_ABLATION_ASSETS_DIR=${OPENPI_MEMORY_ABLATION_ASSETS_DIR:-./assets}
OPENPI_MEMORY_ABLATION_TEMPORAL_CACHE_DIR=${OPENPI_MEMORY_ABLATION_TEMPORAL_CACHE_DIR:-./assets/tfp_adaln_curriculum/temporal_cache_chunklen6}
OPENPI_MEMORY_ABLATION_NORM_STATS_PATH=${OPENPI_MEMORY_ABLATION_NORM_STATS_PATH:-${OPENPI_MEMORY_ABLATION_ASSETS_DIR}/tfp/physical-intelligence/libero/norm_stats.json}

if [[ ! -f "${OPENPI_MEMORY_ABLATION_NORM_STATS_PATH}" ]]; then
  echo "Missing real LIBERO norm stats at ${OPENPI_MEMORY_ABLATION_NORM_STATS_PATH}" >&2
  exit 2
fi
if [[ ! -f "${OPENPI_MEMORY_ABLATION_TEMPORAL_CACHE_DIR}/metadata.json" ]]; then
  echo "Missing real temporal cache metadata at ${OPENPI_MEMORY_ABLATION_TEMPORAL_CACHE_DIR}/metadata.json" >&2
  exit 2
fi

for config in "${CONFIGS[@]}"; do
  uv run scripts/train.py "$config" \
    --exp-name="${OPENPI_MEMORY_ABLATION_EXP_PREFIX:-corl}_${config}" \
    --checkpoint-base-dir=outputs/memory_ablation/checkpoints \
    --assets-base-dir="${OPENPI_MEMORY_ABLATION_ASSETS_DIR}" \
    --temporal-cache-dir="${OPENPI_MEMORY_ABLATION_TEMPORAL_CACHE_DIR}" \
    "$@"
done
