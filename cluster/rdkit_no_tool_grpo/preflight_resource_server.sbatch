#!/usr/bin/env bash
#SBATCH --account=healthcareeng_research
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=rdkit-server-preflight
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo/cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo/cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
CONTAINER=${CONTAINER:-${GYM_DIR}/cluster/rdkit_no_tool_grpo/sqsh/nemo-rl-v0.6.0.sqsh}
PREFLIGHT=${GYM_DIR}/cluster/rdkit_no_tool_grpo/preflight_resource_server.py
TRAIN_PYTHON=${TRAIN_PYTHON:-${GYM_DIR}/cluster/rdkit_no_tool_grpo/venvs/nemo-rl-gym-peft/bin/python}

mkdir -p "${GYM_DIR}/cluster/rdkit_no_tool_grpo/logs"
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre" \
  --container-workdir="${GYM_DIR}" \
  --chdir="${GYM_DIR}" \
  /usr/bin/env GYM_DIR="${GYM_DIR}" PYTHONDONTWRITEBYTECODE=1 \
  "${TRAIN_PYTHON}" "${PREFLIGHT}"
