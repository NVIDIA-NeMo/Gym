#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

usage() {
  cat <<'EOF'
Usage: 12_run_shard_eval.sh [SHARD_INDEX]

Runs one SWE-Atlas QnA shard against already-running vLLM and Gym servers.
Set NUM_SHARDS in config.env, or export it before running this script.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SHARD_INDEX="${1:-${SHARD_INDEX}}"
if ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "SHARD_INDEX must be an integer, got: ${SHARD_INDEX}" >&2
  exit 2
fi
if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "SHARD_INDEX must be in [0, ${NUM_SHARDS}), got: ${SHARD_INDEX}" >&2
  exit 2
fi

cd "${GYM_DIR}"
mkdir -p "${SHARD_OUTPUT_DIR}"
export RAY_TMPDIR

HOST_INPUT="${SHARD_INPUT_DIR}/input_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.jsonl"
CONTAINER_INPUT="${SHARD_CONTAINER_INPUT_DIR}/input_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.jsonl"
HOST_OUTPUT="${SHARD_OUTPUT_DIR}/shard_${SHARD_INDEX}_of_${NUM_SHARDS}.jsonl"
CONTAINER_OUTPUT="${SHARD_CONTAINER_OUTPUT_DIR}/shard_${SHARD_INDEX}_of_${NUM_SHARDS}.jsonl"

if [[ ! -s "${HOST_INPUT}" ]]; then
  echo "Shard input missing: ${HOST_INPUT}. Run 11_prepare_full_shards.sh first." >&2
  exit 2
fi

echo "Running shard ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Input: ${HOST_INPUT}"
echo "Output: ${HOST_OUTPUT}"
echo "Concurrency: ${CONCURRENCY}"

if [[ "${RUN_GYM_IN_CONTAINER}" == "true" ]]; then
  srun --overlap \
    --container-image "${GYM_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-workdir "${GYM_CONTAINER_DIR}" \
    bash -lc "cd \"${GYM_CONTAINER_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_CONTAINER_DIR}:\${PYTHONPATH:-}\" && gym eval run --no-serve --agent swe_atlas_qna_benchmark_mini_swe_agent --input \"${CONTAINER_INPUT}\" --output \"${CONTAINER_OUTPUT}\" --prompt-config \"${PROMPT_CONFIG_PATH}\" --num-repeats 3 --concurrency \"${CONCURRENCY}\" --resume ++disable_aggregation=true"
else
  srun --overlap \
    bash -lc "cd \"${GYM_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_DIR}:\${PYTHONPATH:-}\" && gym eval run --no-serve --agent swe_atlas_qna_benchmark_mini_swe_agent --input \"${HOST_INPUT}\" --output \"${HOST_OUTPUT}\" --prompt-config \"${HOST_PROMPT_CONFIG_PATH}\" --num-repeats 3 --concurrency \"${CONCURRENCY}\" --resume ++disable_aggregation=true"
fi

echo "Shard rollouts: ${HOST_OUTPUT}"
echo "Shard materialized inputs: ${HOST_OUTPUT%.jsonl}_materialized_inputs.jsonl"
