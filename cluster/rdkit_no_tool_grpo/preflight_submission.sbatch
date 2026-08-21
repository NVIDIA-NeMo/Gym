#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
GYM_MAIN_DIR=${GYM_MAIN_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
TRAIN_PYTHON=${TRAIN_PYTHON:-/opt/nemo_rl_venv/bin/python}
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0-gym-peft.sqsh}
SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/inferno/data/rdkit-chemistry-no-tool/prepared-train1024-test1000}
ES_ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
ES140_MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-${BUNDLE_DIR}/results/es140_lora/baseline_step0.json}
MERGE_PARITY_SUMMARY=${MERGE_PARITY_SUMMARY:-${BUNDLE_DIR}/results/es140_lora/merged_model_validation.json}
PINNED_GYM_COMMIT=1a4912e231bb2795b062f7de97496caaf382c7f6
SOURCE_GYM_COMMIT=$(git -C "${GYM_DIR}" rev-parse HEAD)
git -C "${GYM_DIR}" merge-base --is-ancestor "${PINNED_GYM_COMMIT}" "${SOURCE_GYM_COMMIT}"

if mkdir -p "/raid/scratch/${USER}" 2>/dev/null && [[ -w "/raid/scratch/${USER}" ]]; then
  NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-submit-preflight-${SLURM_JOB_ID}"
elif [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR}" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR}/rdkit-submit-preflight-${SLURM_JOB_ID}"
else
  NODE_LOCAL_ROOT="/tmp/rdkit-submit-preflight-${USER}-${SLURM_JOB_ID}"
fi
LOCAL_GYM_DIR="${NODE_LOCAL_ROOT}/Gym"
LOCAL_BUNDLE_DIR="${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo"
ENROOT_CONTAINER_NAME="rdkit-submit-preflight-${SLURM_JOB_ID}"
mkdir -p \
  "${LOCAL_GYM_DIR}" \
  "${NODE_LOCAL_ROOT}/enroot/cache" \
  "${NODE_LOCAL_ROOT}/enroot/data" \
  "${NODE_LOCAL_ROOT}/enroot/runtime" \
  "${NODE_LOCAL_ROOT}/enroot/tmp" \
  "${NODE_LOCAL_ROOT}/hf-home" \
  "${NODE_LOCAL_ROOT}/uv-cache"

cleanup() {
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-submit-preflight-*|/tmp/rdkit-submit-preflight-*|*/rdkit-submit-preflight-*)
      rm -rf "${NODE_LOCAL_ROOT}"
      ;;
  esac
}
trap cleanup EXIT

rsync -a \
  --exclude=.git/ \
  --exclude=artifacts/ \
  --exclude=cluster/rdkit_no_tool_grpo/enroot/ \
  --exclude=cluster/rdkit_no_tool_grpo/logs/ \
  --exclude=cluster/rdkit_no_tool_grpo/results/ \
  --exclude=cluster/rdkit_no_tool_grpo/sqsh/ \
  --exclude=cluster/rdkit_no_tool_grpo/venvs/ \
  "${GYM_DIR}/" "${LOCAL_GYM_DIR}/"

export ENROOT_CACHE_PATH="${NODE_LOCAL_ROOT}/enroot/cache"
export ENROOT_DATA_PATH="${NODE_LOCAL_ROOT}/enroot/data"
export ENROOT_RUNTIME_PATH="${NODE_LOCAL_ROOT}/enroot/runtime"
export ENROOT_TEMP_PATH="${NODE_LOCAL_ROOT}/enroot/tmp"

enroot create --name "${ENROOT_CONTAINER_NAME}" "${CONTAINER_IMAGE_PATH}"

DATA_TOKEN_ARGS=()
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  DATA_TOKEN_ARGS=(--max-output-tokens "${MAX_NEW_TOKENS}")
fi
BASELINE_ARGS=()
if [[ "${REQUIRE_BASELINE_GATE:-0}" == "1" ]]; then
  BASELINE_ARGS=(--require-baseline-gate --baseline-summary "${BASELINE_SUMMARY}")
fi
COMMON_SRUN_ARGS=(
  --nodes=1
  --ntasks=1
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}"
  /usr/bin/env
  ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH}"
  ENROOT_DATA_PATH="${ENROOT_DATA_PATH}"
  ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH}"
  ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH}"
  enroot start
  --mount "${NODE_LOCAL_ROOT}:${NODE_LOCAL_ROOT}"
  --mount "/lustre:/lustre"
  "${ENROOT_CONTAINER_NAME}"
)
COMMON_ENV=(
  /usr/bin/env
  PYTHONPATH="${LOCAL_GYM_DIR}"
  PYTHONDONTWRITEBYTECODE=1
  HF_HOME="${NODE_LOCAL_ROOT}/hf-home"
  UV_CACHE_DIR="${NODE_LOCAL_ROOT}/uv-cache"
)

srun "${COMMON_SRUN_ARGS[@]}" "${COMMON_ENV[@]}" \
  "${TRAIN_PYTHON}" "${LOCAL_BUNDLE_DIR}/prepare_direct_data.py" \
    --root "${LOCAL_GYM_DIR}" \
    --source-dir "${SOURCE_DATA_DIR}" \
    "${DATA_TOKEN_ARGS[@]}"

srun "${COMMON_SRUN_ARGS[@]}" "${COMMON_ENV[@]}" \
  "${TRAIN_PYTHON}" "${LOCAL_BUNDLE_DIR}/validate_bundle.py" \
    --root "${LOCAL_GYM_DIR}" \
    --source-gym-commit "${SOURCE_GYM_COMMIT}" \
    --container-path "${CONTAINER_IMAGE_PATH}" \
    --require-container-preflight \
    --require-adapter --adapter-dir "${ES_ADAPTER_DIR}" \
    --require-merged-model --merged-model-dir "${ES140_MERGED_MODEL_DIR}" \
    --require-merge-parity --merge-parity-summary "${MERGE_PARITY_SUMMARY}" \
    "${BASELINE_ARGS[@]}" \
    "${DATA_TOKEN_ARGS[@]}"

mkdir -p "${BUNDLE_DIR}/data"
rsync -a "${LOCAL_BUNDLE_DIR}/data/" "${BUNDLE_DIR}/data/"
echo "Published validated task data: ${BUNDLE_DIR}/data"
