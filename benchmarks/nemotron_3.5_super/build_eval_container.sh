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
set -xeuo pipefail

# Hardlink, not clone to save space
export UV_LINK_MODE=hardlink

ray_dependency="ray[default]==2.55.1"
uv pip install --system "\$ray_dependency" fastokens vllm-router

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

# See the script for more information.
python benchmarks/nemotron_3.5_super/downgrade_python.py


uv sync --active
# gym eval prepare imports nemo_gym.profiling at startup. In this Python 3.12 build,
# uv sync --active can leave gprof2dot absent, so install the uv.lock version and verify the import.
uv pip install "\$ray_dependency" "gprof2dot==2025.4.14"
python -c 'import gprof2dot'

########################################
# START Benchmark specific preparation
########################################

# See benchmarks/scicode/README.md
uv pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR" \
    -O benchmarks/scicode/data

########################################
# END Benchmark specific preparation
########################################

gym eval prepare --config $GYM_CONFIG

gym env start \
    --config $GYM_CONFIG \
    ++dry_run=true \
    ++uv_venv_dir=/opt/uv_venvs

echo ">>> Inner build complete. Container will now be packed into sqsh."
INNER_BUILD
