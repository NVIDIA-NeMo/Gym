#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen3.8-Flash-Next with its vision encoder enabled, for the visual_agent benchmark where both
# the policy and the agentic judge read screenshots. Same engine settings as qwen_3.8_flash_next.sh
# minus --language-model-only (which skips the vision tower) and plus a per-prompt image cap.
# Meant for VLLM_MODE=aggregated: one TP4 replica per node behind vllm-router, so image
# preprocessing and KV stay on one engine (no P/D transfer of multimodal prompts).
# The image cap bounds how many screenshots one OpenCode session can carry in its context.

GYM_MODEL_PARAMS=(
)

export VLLM_SSM_CONV_STATE_LAYOUT=DS

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-chunked-prefill
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
    --no-disable-hybrid-kv-cache-manager
    --block-size 128
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
    --limit-mm-per-prompt '{"image": 96, "video": 0}'
)
VLLM_AGGREGATED_ARGS=(
    --max-num-batched-tokens 33920
    --max-num-seqs 512
)
