#!/bin/bash

# Nemotron 3.5 Nano on vLLM 0.30: four independent TP1 engines per node behind the cache-aware router.
# Launch with VLLM_MODE=aggregated VLLM_ENGINES_PER_NODE=4 and
# ROUTER_ARGS="--balance-abs-threshold 16 --balance-rel-threshold 1.2" (see the README).
# One DP4 x TP1 + EP server per node let vLLM's internal load balancer split an agent's turns across ranks, so
# prefix-cache hits fell to ~60% and lockstep ranks cut decode speed per request roughly in half.

GYM_MODEL_PARAMS=(
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=0.95"
)

# Nemotron's three-read Mamba SSM state must use the dimension-sequence layout when KV transfer is enabled.
# Not used when the model has no Mamba layers.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

# @bxyu-nvidia: V2 model runner is the new default in vLLM 0.29.0, but it has quite a large speed regression
export VLLM_USE_V2_MODEL_RUNNER=0

# A TP1 engine only uses one GPU, so the vLLM step must run four of them per node. The eval step sources this
# config too (NEMO_GYM_RUN_ID is set there) and needs no engines.
if [[ -z "${NEMO_GYM_RUN_ID:-}" && "${VLLM_ENGINES_PER_NODE:-1}" != 4 ]]; then
    echo "ERROR: nemotron_3.5_lightning.sh serves TP1 engines; set VLLM_ENGINES_PER_NODE=4." >&2
    exit 1
fi

# fp8 KV is for aggregated serving; with P/D on GB300 it produced degenerate generations.
# The fine-grained Mamba prefix cache lets the next agent turn reuse the previous turn's full context.
VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
    --enable-chunked-prefill
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --async-scheduling
    --block-size 128
    --prefix-match-unit 128
    --mamba-cache-mode align
    --enable-mamba-fine-grained-prefix-cache
    --mamba-ssm-cache-dtype float32
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --tensor-parallel-size 1
    --api-server-count 1
)
VLLM_AGGREGATED_ARGS=(
    --max-cudagraph-capture-size 1536
    --max-num-batched-tokens 33920
    --max-num-seqs 512
)
VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --max-cudagraph-capture-size 1200
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-cudagraph-capture-size 1536
    --max-num-batched-tokens 33920
    --max-num-seqs 512
)
