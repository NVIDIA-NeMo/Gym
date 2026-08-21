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
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
CONTAINER=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
HF_CACHE_DIR=${HF_CACHE_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/inferno/.local/cache/huggingface}
ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
OUTPUT_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
MERGE_PYTHON=${MERGE_PYTHON:-${BUNDLE_DIR}/venvs/nemo-rl-gym-peft/bin/python}

if [[ ! -x "${MERGE_PYTHON}" ]]; then
  echo "Missing PEFT merge environment: ${MERGE_PYTHON}" >&2
  exit 1
fi

mkdir -p "${BUNDLE_DIR}/logs" "${GYM_MAIN_DIR}/artifacts"
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre" \
  --container-workdir="${GYM_DIR}" \
  --chdir="${GYM_DIR}" \
  /usr/bin/env \
    HF_HOME="${HF_HOME}" \
    HF_HUB_CACHE="${HF_HUB_CACHE}" \
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    "${MERGE_PYTHON}" "${BUNDLE_DIR}/merge_es_adapter.py" \
      --adapter-dir "${ADAPTER_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --gym-dir "${GYM_DIR}"
