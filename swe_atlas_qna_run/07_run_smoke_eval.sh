#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
mkdir -p results
export RAY_TMPDIR

if [[ ! -s "${SMOKE_PATH}" ]]; then
  echo "Smoke input missing: ${SMOKE_PATH}. Run 01_prepare_smoke_slice.sh first." >&2
  exit 2
fi

if [[ "${RUN_GYM_IN_CONTAINER}" == "true" ]]; then
  srun --overlap \
    --container-image "${GYM_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-workdir "${GYM_CONTAINER_DIR}" \
    bash -lc "cd \"${GYM_CONTAINER_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_CONTAINER_DIR}:\${PYTHONPATH:-}\" && gym eval run --no-serve --agent swe_atlas_qna_benchmark_mini_swe_agent --input \"${SMOKE_CONTAINER_PATH}\" --output \"${SMOKE_CONTAINER_OUTPUT}\" --prompt-config \"${PROMPT_CONFIG_PATH}\" --num-repeats 1"
else
  srun --overlap \
    bash -lc "cd \"${GYM_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_DIR}:\${PYTHONPATH:-}\" && gym eval run --no-serve --agent swe_atlas_qna_benchmark_mini_swe_agent --input \"${SMOKE_PATH}\" --output \"${SMOKE_OUTPUT}\" --prompt-config \"${HOST_PROMPT_CONFIG_PATH}\" --num-repeats 1"
fi

echo "Wrote ${SMOKE_OUTPUT}"
