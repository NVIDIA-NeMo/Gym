#!/bin/bash

command=$(cat <<EOF
    pip install ray==2.55.1

    VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
    vllm serve $MODEL \
        --served-model-name nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
        --gpu-memory-utilization 0.85 \
        --distributed-executor-backend ray \
        --data-parallel-backend ray \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser nemotron_v3 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --api-server-count 1 \
        --kv-cache-dtype fp8 \
        -cc.pass_config.fuse_allreduce_rms=False \
        --mamba-ssm-cache-dtype float32 \
        --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 96}' \
        --moe-backend triton \
        --max-num-batched-tokens 32768
EOF
)

CONTAINER=/lustre/fs1/portfolios/nemotron/projects/nemotron_evals_dev/users/bxyu/vllm/vllm-openai:v0.25.1___with_ray.sqsh \
MOUNTS=/lustre:/lustre \
sbatch \
    --nodes=4 \
    --account=nemotron_evals_research \
    --partition=batch \
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=vllm-$USER \
    --exclusive \
    scripts/sbatch_base.sh bash "$command"
