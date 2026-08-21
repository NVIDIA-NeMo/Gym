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
CONTAINER=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
HF_CACHE_DIR=${HF_CACHE_DIR:-${INFERNO_DIR}/.local/cache/huggingface}
ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
OUTPUT=${MERGE_PARITY_OUTPUT:-${BUNDLE_DIR}/results/es140_lora/merged_model_validation.json}
MERGE_PYTHON=${MERGE_PYTHON:-${BUNDLE_DIR}/venvs/nemo-rl-gym-peft/bin/python}

if [[ ! -x "${MERGE_PYTHON}" ]]; then
  echo "Missing PEFT merge environment: ${MERGE_PYTHON}" >&2
  exit 1
fi

mkdir -p "${BUNDLE_DIR}/logs" "$(dirname "${OUTPUT}")"
srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre" \
  --container-workdir="${GYM_DIR}" \
  --chdir="${GYM_DIR}" \
  /usr/bin/env \
    HF_HOME="${HF_CACHE_DIR}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${MERGE_PYTHON}" "${BUNDLE_DIR}/validate_merged_model.py" \
      --base-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --adapter-dir "${ADAPTER_DIR}" \
      --merged-model-dir "${MERGED_MODEL_DIR}" \
      --output "${OUTPUT}"
