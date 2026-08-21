#!/usr/bin/env bash
#SBATCH --job-name=rdkit-es140-merge-parity
#SBATCH --partition=pool0
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
GYM_MAIN_DIR=${GYM_MAIN_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym}
INFERNO_DIR=${INFERNO_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/inferno}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0-gym-peft.sqsh}
BASE_MODEL_CACHE_ROOT=${BASE_MODEL_CACHE_ROOT:-${INFERNO_DIR}/.local/cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
OUTPUT=${MERGE_PARITY_OUTPUT:-${BUNDLE_DIR}/results/es140_lora/merged_model_validation.json}
MERGE_PYTHON=${MERGE_PYTHON:-/opt/nemo_rl_venv/bin/python}

BASE_SNAPSHOT_SHA=$(<"${BASE_MODEL_CACHE_ROOT}/refs/main")
BASE_SNAPSHOT_DIR="${BASE_MODEL_CACHE_ROOT}/snapshots/${BASE_SNAPSHOT_SHA}"
if [[ ! -d "${BASE_SNAPSHOT_DIR}" ]]; then
  echo "Missing cached base-model snapshot: ${BASE_SNAPSHOT_DIR}" >&2
  exit 1
fi
if mkdir -p "/raid/scratch/${USER}" 2>/dev/null && [[ -w "/raid/scratch/${USER}" ]]; then
  NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-es140-parity-${SLURM_JOB_ID}"
elif [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR}" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR}/rdkit-es140-parity-${SLURM_JOB_ID}"
else
  NODE_LOCAL_ROOT="/tmp/rdkit-es140-parity-${USER}-${SLURM_JOB_ID}"
fi
LOCAL_GYM_DIR="${NODE_LOCAL_ROOT}/Gym"
LOCAL_CONTAINER="${NODE_LOCAL_ROOT}/container/$(basename "${CONTAINER_IMAGE_PATH}")"
LOCAL_BASE_MODEL="${NODE_LOCAL_ROOT}/base-model"
LOCAL_ADAPTER="${NODE_LOCAL_ROOT}/adapter"
LOCAL_OUTPUT="${NODE_LOCAL_ROOT}/merged_model_validation.json"
mkdir -p \
  "${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo" \
  "${NODE_LOCAL_ROOT}/container" \
  "${LOCAL_BASE_MODEL}" \
  "${LOCAL_ADAPTER}" \
  "${NODE_LOCAL_ROOT}/hf-home" \
  "$(dirname "${OUTPUT}")"

cleanup() {
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-es140-parity-*|/tmp/rdkit-es140-parity-*|*/rdkit-es140-parity-*)
      rm -rf "${NODE_LOCAL_ROOT}"
      ;;
  esac
}
trap cleanup EXIT

install -m 0644 "${BUNDLE_DIR}/validate_merged_model.py" \
  "${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo/validate_merged_model.py"
rsync -a "${CONTAINER_IMAGE_PATH}" "${LOCAL_CONTAINER}"
rsync -aL "${BASE_SNAPSHOT_DIR}/" "${LOCAL_BASE_MODEL}/"
rsync -a "${ADAPTER_DIR}/" "${LOCAL_ADAPTER}/"

srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --no-container-mount-home \
  --container-image="${LOCAL_CONTAINER}" \
  --container-mounts="${NODE_LOCAL_ROOT}:${NODE_LOCAL_ROOT},/lustre:/lustre" \
  --container-workdir="${LOCAL_GYM_DIR}" \
  --chdir="${LOCAL_GYM_DIR}" \
  /usr/bin/env \
    HF_HOME="${NODE_LOCAL_ROOT}/hf-home" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${MERGE_PYTHON}" \
    "${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo/validate_merged_model.py" \
      --base-model "${LOCAL_BASE_MODEL}" \
      --adapter-dir "${LOCAL_ADAPTER}" \
      --merged-model-dir "${MERGED_MODEL_DIR}" \
      --output "${LOCAL_OUTPUT}"

install -m 0644 "${LOCAL_OUTPUT}" "${OUTPUT}"
echo "Published merge parity summary: ${OUTPUT}"
