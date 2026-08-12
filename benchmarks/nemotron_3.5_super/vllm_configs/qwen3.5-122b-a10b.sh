#!/bin/bash

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --max-model-len 262144
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --mm-encoder-tp-mode data
    --enable-chunked-prefill
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --no-async-scheduling
    --block-size 128
    --mamba-cache-mode align
    --mamba-ssm-cache-dtype float32
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --max-num-batched-tokens 33920
    --max-num-seqs 512
)
