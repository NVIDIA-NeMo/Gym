#!/usr/bin/env bash
# Laguna S 2.1 FP8, TP1, DFlash with 7 speculative tokens.
# vLLM downloads the official draft from Hugging Face at startup.

MODEL_NAME=laguna-s-2.1-fp8
export ROUTER_BALANCE_ABS_THRESHOLD=40
export ROUTER_BALANCE_REL_THRESHOLD=2
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0

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
    --moe-backend triton
    --speculative-config "{\"model\":\"poolside/Laguna-S-2.1-DFlash-FP8\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}"
)
