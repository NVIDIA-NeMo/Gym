#!/bin/bash
# 
# Example run:
# MODEL=/path/to/model \
# EXPERIMENT_NAME=my-experiment-name \
# NUM_NODES=4 \
# SBATCH_ACCOUNT=my-slurm-account \
# SBATCH_PARTITION=batch \
# CONTAINER=/path/to/vllm/container \
# MOUNTS=/shared/fs:/shared/fs \
# bash benchmarks/nemotron_3_ultra/sbatch_eval_with_external_vllm.sh \
# --config benchmarks/my-benchmark/config.yaml
# 
# This script assumes:
# - The container is one built via benchmarks/nemotron_3_ultra/build_eval_container.sh
# - GB200s which are 4 GPUs per node. If you want to use 8 GPUs per node, update the --tensor-parallel-size and --gres=gpu arguments to 8.
# - Nemotron 3 Ultra configs e.g. with the parser configs.
# 
# If you want to use your own custom local Gym, please mount:
# MOUNTS=/shared/fs:/shared/fs,/opt/Gym:/path/to/custom/local/Gym
# The existing Gym venv and individual server venvs will still use the ones baked into the container.
# 

set -euo pipefail

# Input arguments and validation
EXPERIMENT_NAME=$EXPERIMENT_NAME
NUM_NODES=$NUM_NODES
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS

command=$(cat <<EOF
set -euo pipefail

host=\$(hostname -I | awk '{print \$1}')
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
vllm serve $MODEL \
    --gpu-memory-utilization 0.9 \
    --distributed-executor-backend ray \
    --data-parallel-backend ray \
    --data-parallel-size $NUM_NODES \
    --data-parallel-size-local 1 \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    --api-server-count 1 \
    --kv-cache-dtype fp8 \
    -cc.pass_config.fuse_allreduce_rms=False \
    --mamba-ssm-cache-dtype float32 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' \
    --enable-expert-parallel \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
    --max-num-batched-tokens 8480 \
    --host \$host \
    --port 8000 &

# Activate environment and cd into Gym
source /opt/Gym_venv/bin/activate
cd /opt/Gym

gym eval prepare $@ +use_cached_prepared_benchmarks=true

ip=http://\$host:8000/v1
until curl -s \$ip >/dev/null; do
    sleep 5
done

experiment_name=$EXPERIMENT_NAME-\$(date +%Y%m%d_%H%M%S)
# +uv_venv_dir=/opt/uv_venvs is from the container.
# +skip_venv_if_present=true will reuse the venvs baked into the container if possible.
gym eval run \
    $@ \
    +wandb_project=$USER-gym-eval \
    +wandb_name=\$experiment_name \
    +uv_venv_dir=/opt/uv_venvs \
    +skip_venv_if_present=true \
    ++output_jsonl_fpath=results/\$experiment_name.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=\$ip \
    ++policy_api_key=dummy_api_key \
    ++policy_model_name=$MODEL \
    ++global_aiohttp_connector_limit_per_host=16384

EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
sbatch \
    --nodes=$NUM_NODES \
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=gym-vllm-eval-$EXPERIMENT_NAME-$USER \
    --exclusive \
    --segment=$NUM_NODES \
    scripts/sbatch_base.sh bash -lc "$command"
