#!/bin/bash

GYM_MODEL_PARAMS=(
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=0.95"
)

# Nemotron's three-read Mamba SSM state must use the dimension-sequence layout when KV transfer is enabled.
# Not used when the model has no Mamba layers.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

# @bxyu-nvidia: V2 model runner is the new default in vLLM 0.29.0, but it has quite a large speed regression
export VLLM_USE_V2_MODEL_RUNNER=0

# @bxyu-nvidia: `--skip-mm-profiling` Is needed to get Super VL checkpoint working, even with text benchmarks
VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.85
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-model-len 262144
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --block-size 128
    --mamba-backend flashinfer
    --mamba-ssm-cache-dtype float16
    --enable-mamba-cache-stochastic-rounding
    --mamba-cache-philox-rounds 5
    --mamba-cache-mode align
    --prefix-match-unit 16
    --enable-mamba-fine-grained-prefix-cache
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
)
VLLM_PREFILL_ARGS=(
    --speculative-config '{"method":"mtp","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,1024,0]]}'
    --max-cudagraph-capture-size 1200
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
VLLM_DECODE_ARGS=(
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
