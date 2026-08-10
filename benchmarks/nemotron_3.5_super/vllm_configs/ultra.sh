#!/bin/bash

VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
vllm serve "$MODEL" \
    --gpu-memory-utilization 0.9 \
    --distributed-executor-backend ray \
    --data-parallel-backend ray \
    --data-parallel-size "$NUM_NODES" \
    --data-parallel-size-local 1 \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    --api-server-count 1 \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --mamba-ssm-cache-dtype float32 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' \
    --enable-expert-parallel \
    --max-num-batched-tokens 8192 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
    --host "$host" \
    --port 8000 &
