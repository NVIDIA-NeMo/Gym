#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Lightning 3.5 BF16, TP1, native MTP from the served checkpoint.
MODEL_NAME=lightning35-bf16
# Pin to the nightly the Sep 20 TB 2.1 runs used (0.29.1rc1.dev422+gd05da62e9).
# Later nightlies renamed --enable-mamba-fine-grained-prefix-cache.
export VLLM_CONTAINER="registry-1.docker.io#vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af"
export ROUTER_BALANCE_ABS_THRESHOLD=40
export ROUTER_BALANCE_REL_THRESHOLD=2
# The checkpoint declares 256K; explicitly allow the 1M serving limit.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --tensor-parallel-size 1
    --data-parallel-size 1
    --distributed-executor-backend mp
    --api-server-count 16
    --renderer-num-workers 16
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-model-len 1048576
    --max-num-batched-tokens 32768
    --max-num-seqs 128
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --no-async-scheduling
    --block-size 128
    --mamba-cache-mode align
    --enable-mamba-fine-grained-prefix-cache
    --prefix-match-unit 16
    --mamba-ssm-cache-dtype float32
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'
)

VLLM_COMMON_ARGS+=(
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
)
