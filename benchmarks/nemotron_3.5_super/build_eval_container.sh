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
VLLM_ROUTER_WHEEL=$VLLM_ROUTER_WHEEL
NEMO_GYM_GIT_URL=${NEMO_GYM_GIT_URL:-https://github.com/NVIDIA-NeMo/Gym}
NEMO_GYM_GIT_REF=${NEMO_GYM_GIT_REF:-main}
TAU_2_MOUNT_BASE_GYM_DIR=${TAU_2_MOUNT_BASE_GYM_DIR:-""}


if [[ -n "$TAU_2_MOUNT_BASE_GYM_DIR" ]]; then
    MOUNTS="$MOUNTS,$TAU_2_MOUNT_BASE_GYM_DIR:$TAU_2_MOUNT_BASE_GYM_DIR"
fi

VLLM_ROUTER_WHEEL=$(readlink -f "$VLLM_ROUTER_WHEEL")
MOUNTS="$MOUNTS,$(dirname "$VLLM_ROUTER_WHEEL"):$(dirname "$VLLM_ROUTER_WHEEL")"

# pyxis --container-save exports the image when the step tears down, whatever the
# inner script exited with, and it overwrites whatever already sits at the target.
# So stage the build and publish only on success; otherwise a failed build silently
# replaces a good container with a broken one.
staged_container="$OUTPUT_CONTAINER.partial"
rm -f "$staged_container"
save_status=0

srun --nodes=1 --ntasks=1 \
    --container-image=$INPUT_CONTAINER \
    --container-mounts=$MOUNTS \
    --no-container-mount-home \
    --container-save="$staged_container" \
    bash -s <<INNER_BUILD || save_status=$?
set -xeuo pipefail

# Hardlink, not clone to save space
export UV_LINK_MODE=hardlink

uv pip install --system --reinstall-package vllm-router "$VLLM_ROUTER_WHEEL"
uv pip show --system vllm-router

apt-get update
apt-get install -y --no-install-recommends \
    git
rm -rf /var/lib/apt/lists/*

cd /opt
# Python 3.13.14 is Gym main's Python version.
uv venv --python 3.13.14 Gym_venv
source Gym_venv/bin/activate

# We use this flow to support use cases where env.yaml, etc config files are mounted
# In these cases, git clone throws a non-empty directory error.
mkdir -p Gym
cd Gym
git init
git remote add origin $NEMO_GYM_GIT_URL
git fetch origin $NEMO_GYM_GIT_REF
git checkout $NEMO_GYM_GIT_REF

uv sync --active

########################################
# START Benchmark specific preparation
########################################

# See benchmarks/scicode/README.md
uv pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR" \
    -O benchmarks/scicode/data

if [[ -n "$TAU_2_MOUNT_BASE_GYM_DIR" ]]; then
    echo "Copying Tau2 and Tau3 data from mounted Gym dir: $TAU_2_MOUNT_BASE_GYM_DIR"
    cp -r "$TAU_2_MOUNT_BASE_GYM_DIR/benchmarks/tau2/nemo_gym_data" benchmarks/tau2/nemo_gym_data
    cp -r "$TAU_2_MOUNT_BASE_GYM_DIR/responses_api_agents/tau2/tau2_data" responses_api_agents/tau2/tau2_data
fi

########################################
# END Benchmark specific preparation
########################################

gym eval prepare +num_prepare_benchmark_processes=4 --config $GYM_CONFIG

gym env start \
    --config $GYM_CONFIG \
    ++dry_run=true \
    ++uv_venv_dir=/opt/uv_venvs

echo ">>> Inner build complete. Container will now be packed into sqsh."
INNER_BUILD

if (( save_status != 0 )); then
    rm -f "$staged_container"
    echo "Build failed (exit $save_status). $OUTPUT_CONTAINER left untouched." >&2
    exit "$save_status"
fi
mv -f "$staged_container" "$OUTPUT_CONTAINER"
echo ">>> Published $OUTPUT_CONTAINER"
