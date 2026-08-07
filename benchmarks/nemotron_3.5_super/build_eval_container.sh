#!/bin/bash
# 
#SBATCH --output=slurm-logs/%j-%x.log
#SBATCH --job-name=gym-build_eval_container

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
    --no-container-mount-home \
    --container-save=$OUTPUT_CONTAINER \
    bash -s <<INNER_BUILD
set -euo pipefail

# Hardlink, not clone to save space
export UV_LINK_MODE=hardlink

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

# We use this flow to support use cases where env.yaml, etc config files are mounted
# In these cases, git clone throws a non-empty directory error.
mkdir -p Gym
cd Gym
git init
git remote add origin $NEMO_GYM_GIT_URL
git fetch origin $NEMO_GYM_GIT_REF
git checkout $NEMO_GYM_GIT_REF

# Gym main has upgraded to Python 3.13.14, but vLLM still uses 3.12.13
# Rather than entirely rebuild the vLLM container from scratch, we just manually downgrade the Gym python here.
sed -i 's/requires-python = ">=3\.13\.14"/requires-python = ">=3.12.13"/' pyproject.toml

uv sync --active
uv pip install "\$ray_dependency"

gym eval prepare --config $GYM_CONFIG

gym env start \
    --config $GYM_CONFIG \
    ++dry_run=true \
    ++uv_venv_dir=/opt/uv_venvs

echo ">>> Inner build complete. Container will now be packed into sqsh."
INNER_BUILD
