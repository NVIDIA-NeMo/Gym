#!/bin/bash

# Inkling's official vLLM recipe requires the V2 model runner and enables the
# FlashAttention CuTe DSL kernel cache to avoid recompiling kernels at startup.
# https://recipes.vllm.ai/thinkingmachines/Inkling-Small?hardware=gb200&strategy=single_node_tep&variant=bf16

export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

# The complete-response parsers skip unused partial tool-argument processing. On any vLLM
# other than the validated 0.29.0 they fall back to the stock Inkling parsers with a warning.
inkling_plugin_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../vllm_plugins" && pwd)

GYM_MODEL_PARAMS=(
    "++policy_model.responses_api_models.vllm_model.chat_template_kwargs.reasoning_effort=max"
)

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --tokenizer-mode inkling
    --kernel-config.enable_flashinfer_autotune=False
    --enable-auto-tool-choice
    --tool-call-parser inkling_complete_fast
    --reasoning-parser inkling_count_fast
    --tool-parser-plugin "$inkling_plugin_dir/inkling_complete_tool_parser.py"
    --reasoning-parser-plugin "$inkling_plugin_dir/inkling_complete_reasoning_parser.py"
    --enable-chunked-prefill
    --enable-prefix-caching
    --enable-expert-parallel
    --max-cudagraph-capture-size 256
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)

VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
    --max-num-batched-tokens 16384
    --max-num-seqs 256
)

VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 8192
    --max-num-seqs 256
)
