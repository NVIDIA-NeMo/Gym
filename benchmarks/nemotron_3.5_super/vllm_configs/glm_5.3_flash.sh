#!/bin/bash

# See https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes
GYM_MODEL_PARAMS=(
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=1.0"
)

# Nemotron's three-read Mamba SSM state must use the dimension-sequence layout when KV transfer is enabled.
# Not used when the model has no Mamba layers.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.85
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser glm47
    --reasoning-parser glm45
    --enable-chunked-prefill
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --block-size 128
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
    --compilation-config '{"cudagraph_mm_encoder": true}'
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
