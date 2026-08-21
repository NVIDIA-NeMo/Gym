#!/usr/bin/env bash
#SBATCH --job-name=rdkit-es140-merge
#SBATCH --partition=cpu
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
GYM_MAIN_DIR=${GYM_MAIN_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym}
INFERNO_DIR=${INFERNO_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/inferno}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
BASE_MODEL_CACHE_ROOT=${BASE_MODEL_CACHE_ROOT:-${INFERNO_DIR}/.local/cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
OUTPUT_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
MERGE_PYTHON=${MERGE_PYTHON:-/opt/nemo_rl_venv/bin/python}
VALIDATION_OUTPUT=${MERGE_VALIDATION_OUTPUT:-${BUNDLE_DIR}/results/es140_lora/merged_model_validation.json}
PINNED_GYM_COMMIT=1a4912e231bb2795b062f7de97496caaf382c7f6
SOURCE_GYM_COMMIT=$(git -C "${GYM_DIR}" rev-parse HEAD)
git -C "${GYM_DIR}" merge-base --is-ancestor "${PINNED_GYM_COMMIT}" "${SOURCE_GYM_COMMIT}"

if [[ ! -f "${BASE_MODEL_CACHE_ROOT}/refs/main" ]]; then
  echo "Missing cached base-model ref: ${BASE_MODEL_CACHE_ROOT}/refs/main" >&2
  exit 1
fi
BASE_SNAPSHOT_SHA=$(<"${BASE_MODEL_CACHE_ROOT}/refs/main")
BASE_SNAPSHOT_DIR="${BASE_MODEL_CACHE_ROOT}/snapshots/${BASE_SNAPSHOT_SHA}"
if [[ ! -d "${BASE_SNAPSHOT_DIR}" ]]; then
  echo "Missing cached base-model snapshot: ${BASE_SNAPSHOT_DIR}" >&2
  exit 1
fi
if mkdir -p "/raid/scratch/${USER}" 2>/dev/null && [[ -w "/raid/scratch/${USER}" ]]; then
  NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-es140-merge-${SLURM_JOB_ID}"
elif [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR}" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR}/rdkit-es140-merge-${SLURM_JOB_ID}"
else
  NODE_LOCAL_ROOT="/tmp/rdkit-es140-merge-${USER}-${SLURM_JOB_ID}"
fi
LOCAL_GYM_DIR="${NODE_LOCAL_ROOT}/Gym"
LOCAL_CONTAINER="${NODE_LOCAL_ROOT}/container/$(basename "${CONTAINER_IMAGE_PATH}")"
LOCAL_BASE_MODEL="${NODE_LOCAL_ROOT}/base-model"
LOCAL_ADAPTER="${NODE_LOCAL_ROOT}/adapter"
LOCAL_OUTPUT="${NODE_LOCAL_ROOT}/merged-model"
LOCAL_VALIDATION="${NODE_LOCAL_ROOT}/merged_model_validation.json"
PERSIST_TMP="${OUTPUT_DIR}.tmp-${SLURM_JOB_ID}"
mkdir -p \
  "${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo" \
  "${NODE_LOCAL_ROOT}/container" \
  "${LOCAL_BASE_MODEL}" \
  "${LOCAL_ADAPTER}" \
  "${NODE_LOCAL_ROOT}/hf-home" \
  "${NODE_LOCAL_ROOT}/uv-cache" \
  "$(dirname "${VALIDATION_OUTPUT}")"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing merged-model output: ${OUTPUT_DIR}" >&2
  exit 1
fi

cleanup() {
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-es140-merge-*|/tmp/rdkit-es140-merge-*|*/rdkit-es140-merge-*)
      rm -rf "${NODE_LOCAL_ROOT}"
      ;;
  esac
  case "${PERSIST_TMP}" in
    *.tmp-"${SLURM_JOB_ID}")
      rm -rf "${PERSIST_TMP}"
      ;;
  esac
}
trap cleanup EXIT

install -m 0644 "${BUNDLE_DIR}/merge_es_adapter.py" \
  "${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo/merge_es_adapter.py"
rsync -a "${CONTAINER_IMAGE_PATH}" "${LOCAL_CONTAINER}"
rsync -aL "${BASE_SNAPSHOT_DIR}/" "${LOCAL_BASE_MODEL}/"
rsync -a "${ADAPTER_DIR}/" "${LOCAL_ADAPTER}/"

mkdir -p "${BUNDLE_DIR}/logs" "${GYM_MAIN_DIR}/artifacts"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --no-container-mount-home \
  --container-image="${LOCAL_CONTAINER}" \
  --container-mounts="${NODE_LOCAL_ROOT}:${NODE_LOCAL_ROOT},/lustre:/lustre" \
  --container-workdir="${LOCAL_GYM_DIR}" \
  --chdir="${LOCAL_GYM_DIR}" \
  /bin/bash -lc "
    set -euo pipefail
    export UV_CACHE_DIR='${NODE_LOCAL_ROOT}/uv-cache'
    export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
    '${MERGE_PYTHON}' \
      '${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo/merge_es_adapter.py' \
      --base-model '${LOCAL_BASE_MODEL}' \
      --adapter-dir '${LOCAL_ADAPTER}' \
      --output-dir '${LOCAL_OUTPUT}' \
      --validation-output '${LOCAL_VALIDATION}' \
      --gym-commit '${SOURCE_GYM_COMMIT}'
  "

mkdir -p "${PERSIST_TMP}"
rsync -a "${LOCAL_OUTPUT}/" "${PERSIST_TMP}/"
mv "${PERSIST_TMP}" "${OUTPUT_DIR}"
install -m 0644 "${LOCAL_VALIDATION}" "${VALIDATION_OUTPUT}"
echo "Published merged model: ${OUTPUT_DIR}"
echo "Published direct-merge validation: ${VALIDATION_OUTPUT}"
