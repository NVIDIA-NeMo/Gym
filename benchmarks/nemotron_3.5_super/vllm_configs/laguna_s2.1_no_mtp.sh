#!/bin/bash

GYM_MODEL_PARAMS=()

# @bxyu-nvidia: V2 model runner is the new default in vLLM 0.29.0, but it has quite a large speed regression
export VLLM_USE_V2_MODEL_RUNNER=0

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser poolside_v1
    --reasoning-parser poolside_v1
    --enable-chunked-prefill
    --enable-prefix-caching
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --moe-backend triton
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
    --default-chat-template-kwargs '{"enable_thinking": true}'
)
PREFILL_KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
DECODE_KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
if [[ "${ENABLE_MOONCAKE:-0}" == 1 ]]; then
    PREFILL_KV_TRANSFER_CONFIG='{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":true}}]}}'
    DECODE_KV_TRANSFER_CONFIG='{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":true}}]}}'
fi

VLLM_PREFILL_ARGS=(
    --kv-transfer-config "$PREFILL_KV_TRANSFER_CONFIG"
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config "$DECODE_KV_TRANSFER_CONFIG"
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
