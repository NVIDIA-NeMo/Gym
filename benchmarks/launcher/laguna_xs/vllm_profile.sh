#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Laguna XS 2.1 BF16, TP1, DFlash with 7 speculative tokens.
# vLLM downloads the official draft from Hugging Face at startup.

MODEL_NAME=laguna-xs-2.1-bf16
export ROUTER_BALANCE_ABS_THRESHOLD=40
export ROUTER_BALANCE_REL_THRESHOLD=2

# Evaluation output-token cap, forwarded by eval.sh to Gym.
BENCHMARK_EXTRA_ARGS+=(
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.max_tokens=49152"
)

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --tensor-parallel-size 1
    --data-parallel-size 1
    --distributed-executor-backend mp
    --enable-auto-tool-choice
    --tool-call-parser poolside_v1
    --reasoning-parser poolside_v1
    --default-chat-template-kwargs '{"enable_thinking": true}'
    --generation-config auto
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-model-len 262144
    --kv-cache-dtype fp8
    --speculative-config "{\"model\":\"poolside/Laguna-XS-2.1-DFlash\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}"
)
