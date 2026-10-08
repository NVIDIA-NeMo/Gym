#!/bin/bash

# DS v4 Flash takes a super long time to start.
GYM_MODEL_PARAMS=(
    "++model_endpoint_readiness_timeout_seconds=1200"
    "++policy_model.responses_api_models.vllm_model.chat_template_kwargs.reasoning_effort=max"
)

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
    --tokenizer-mode deepseek_v4
    --enable-chunked-prefill
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --block-size 256
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
    --attention-config '{"indexer_kv_dtype":"mxfp4"}'
    --moe-backend deep_gemm_mega_moe
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
)

VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
    --max-num-batched-tokens 131072
    --max-num-seqs 1024
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-cudagraph-capture-size 2048
    --max-num-batched-tokens 32768
    --max-num-seqs 1024
)
