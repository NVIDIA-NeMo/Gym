#!/bin/bash
# 
# Example run:
# MODEL=/path/to/model \
# NUM_NODES=4 \
# SBATCH_ACCOUNT=my-slurm-account \
# SBATCH_PARTITION=batch \
# CONTAINER=/path/to/vllm/container \
# MOUNTS=/shared/fs:/shared/fs \
# bash scripts/sbatch_external_vllm.sh
# 
# This script assumes:
# - GB200s which are 4 GPUs per node. If you want to use 8 GPUs per node, update the --tensor-parallel-size and --gres=gpu arguments to 8.
# - Nemotron 3 Ultra configs e.g. with the parser configs.

set -euo pipefail

# Input arguments and validation
NUM_NODES=$NUM_NODES
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS

command=$(cat <<EOF
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
vllm serve $MODEL \
    --served-model-name nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
    --gpu-memory-utilization 0.9 \
    --distributed-executor-backend ray \
    --data-parallel-backend ray \
    --data-parallel-size 4 \
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
    --max-num-batched-tokens 16384 \
    --host \$(hostname -I | awk '{print \$1}') \
    --port 8000
EOF
)

CONTAINER=$CONTAINER \
MOUNTS=$MOUNTS \
sbatch \
    --nodes=$NUM_NODES \
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=vllm-$USER \
    --exclusive \
    --segment=$NUM_NODES \
    scripts/sbatch_base.sh bash -lc "$command"
