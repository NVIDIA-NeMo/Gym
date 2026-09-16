#!/bin/bash

GYM_MODEL_PARAMS=(
)

# @bxyu-nvidia: V2 model runner is the new default in vLLM 0.29.0, but it has quite a large speed regression
export VLLM_USE_V2_MODEL_RUNNER=0

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
    --max-model-len 262144
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --block-size 128
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    --language-model-only
)
VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
