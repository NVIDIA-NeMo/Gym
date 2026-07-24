#!/bin/bash
# 
#SBATCH --output=slurm-logs/%j-%x.log
#SBATCH --job-name=gym-build_eval_container-%u
# 
# Example run:
# SBATCH_ACCOUNT=my-slurm-account \
# SBATCH_PARTITION=batch \
# INPUT_CONTAINER=/path/to/vllm/container \
# OUTPUT_CONTAINER=/path/to/vllm/container___with_gym.sqsh \
# sbatch --gres=gpu:4 \
#   benchmarks/nemotron_3_ultra/sbatch_eval_with_external_vllm.sh \
#   --config benchmarks/my-benchmark/config.yaml
# 

set -euo pipefail

# Input arguments and validation
INPUT_CONTAINER=$INPUT_CONTAINER
OUTPUT_CONTAINER=$OUTPUT_CONTAINER
NEMO_GYM_GIT_URL=${NEMO_GYM_GIT_URL:-https://github.com/NVIDIA-NeMo/Gym}
NEMO_GYM_GIT_REF=${NEMO_GYM_GIT_REF:-main}

CONFIGS=$(cat <<EOF
    --config benchmarks/gpqa/config.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    ++policy_base_url="dummy" \
    ++policy_api_key="dummy" \
    ++policy_model_name="dummy"
EOF
)

srun --nodes=1 --ntasks=1 \
    --container-image=$INPUT_CONTAINER \
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
uv venv --python $(which python3) Gym_venv
source Gym_venv/bin/activate

git clone $NEMO_GYM_GIT_URL Gym
cd Gym
git checkout $NEMO_GYM_GIT_REF

uv sync --active
uv pip install "\$ray_dependency"

gym eval prepare $CONFIGS

gym eval run \
    $CONFIGS \
    ++uv_venv_dir=/opt/uv_venvs \
    ++dry_run=true

echo ">>> Inner build complete. Container will now be packed into sqsh."
INNER_BUILD
