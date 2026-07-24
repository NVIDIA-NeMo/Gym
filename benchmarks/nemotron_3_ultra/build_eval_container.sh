#!/bin/bash
# 
#SBATCH --output=slurm-logs/%j-%x.log
#SBATCH --job-name=gym-build_eval_container
# 
# Example run:
# SBATCH_ACCOUNT=my-slurm-account \
# SBATCH_PARTITION=batch \
# INPUT_CONTAINER=/path/to/vllm/container \
# OUTPUT_CONTAINER=/path/to/vllm/container___with_gym.sqsh \
# MOUNTS=/path/to/env.yaml:/opt/Gym/env.yaml,/path/to/config.yaml:/opt/Gym/config.yaml \
# GYM_CONFIG=benchmarks/nemotron_3_ultra/eval_container_config.yaml \
# sbatch --gres=gpu:4 \
#   benchmarks/nemotron_3_ultra/build_eval_container.sh
# 

set -euo pipefail

# Input arguments and validation
INPUT_CONTAINER=$INPUT_CONTAINER
OUTPUT_CONTAINER=$OUTPUT_CONTAINER
MOUNTS=$MOUNTS
GYM_CONFIG=$GYM_CONFIG
NEMO_GYM_GIT_URL=${NEMO_GYM_GIT_URL:-https://github.com/NVIDIA-NeMo/Gym}
NEMO_GYM_GIT_REF=${NEMO_GYM_GIT_REF:-main}

srun --nodes=1 --ntasks=1 \
    --container-image=$INPUT_CONTAINER \
    --container-mounts=$MOUNTS \
    --container-save=$OUTPUT_CONTAINER \
    bash -s <<INNER_BUILD
set -euo pipefail

ray_dependency="ray[default]==2.55.1"
uv pip install --system "\$ray_dependency"

apt-get update
apt-get install -y --no-install-recommends \
    git
rm -rf /var/lib/apt/lists/*

cd /opt
# Reuse the vLLM container's python3 so we strongly align the Python versions across vLLM and Gym.
uv venv --python \$(which python3) Gym_venv
source Gym_venv/bin/activate

git clone $NEMO_GYM_GIT_URL Gym
cd Gym
git checkout $NEMO_GYM_GIT_REF

uv sync --active
uv pip install "\$ray_dependency"

gym eval prepare $GYM_CONFIG

gym eval run \
    $GYM_CONFIG \
    ++uv_venv_dir=/opt/uv_venvs \
    ++dry_run=true

echo ">>> Inner build complete. Container will now be packed into sqsh."
INNER_BUILD
