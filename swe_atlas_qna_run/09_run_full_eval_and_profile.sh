#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
mkdir -p results "${PROFILE_OUTPUT_DIR}"
export RAY_TMPDIR

INPUT="benchmarks/swe_atlas_qna/data/swe_atlas_qna_benchmark.jsonl"
CONTAINER_INPUT="${GYM_CONTAINER_DIR}/${INPUT}"
HOST_INPUT="${GYM_DIR}/${INPUT}"
if [[ ! -s "${INPUT}" ]]; then
  echo "Full benchmark input missing: ${INPUT}. Run 01_prepare_smoke_slice.sh first." >&2
  exit 2
fi

if [[ "${RUN_GYM_IN_CONTAINER}" == "true" ]]; then
  srun --overlap \
    --container-image "${GYM_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-workdir "${GYM_CONTAINER_DIR}" \
    bash -lc "cd \"${GYM_CONTAINER_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_CONTAINER_DIR}:\${PYTHONPATH:-}\" && gym eval run --no-serve --agent swe_atlas_qna_benchmark_mini_swe_agent --input \"${CONTAINER_INPUT}\" --output \"${FULL_CONTAINER_OUTPUT}\" --prompt-config \"${PROMPT_CONFIG_PATH}\" --num-repeats 3 --concurrency \"${CONCURRENCY}\" --resume && gym eval profile --inputs \"${CONTAINER_INPUT}\" --rollouts \"${FULL_CONTAINER_OUTPUT}\" --output-dir \"${PROFILE_CONTAINER_OUTPUT_DIR}\""
else
  srun --overlap \
    bash -lc "cd \"${GYM_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_DIR}:\${PYTHONPATH:-}\" && gym eval run --no-serve --agent swe_atlas_qna_benchmark_mini_swe_agent --input \"${HOST_INPUT}\" --output \"${FULL_OUTPUT}\" --prompt-config \"${HOST_PROMPT_CONFIG_PATH}\" --num-repeats 3 --concurrency \"${CONCURRENCY}\" --resume && gym eval profile --inputs \"${HOST_INPUT}\" --rollouts \"${FULL_OUTPUT}\" --output-dir \"${PROFILE_OUTPUT_DIR}\""
fi

echo "Full rollouts: ${FULL_OUTPUT}"
echo "Profile output: ${PROFILE_OUTPUT_DIR}"
