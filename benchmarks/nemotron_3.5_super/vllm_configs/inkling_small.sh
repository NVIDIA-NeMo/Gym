#!/bin/bash

# Inkling's official vLLM recipe requires the V2 model runner and enables the
# FlashAttention CuTe DSL kernel cache to avoid recompiling kernels at startup.
# https://recipes.vllm.ai/thinkingmachines/Inkling-Small?hardware=gb200&strategy=single_node_tep&variant=bf16

export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

# The shared launcher enables Fastokens globally. Keep this first baseline on
# Inkling's upstream tokenizer path so its effect can be measured separately.
unset VLLM_USE_FASTOKENS

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --dtype bfloat16
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --max-model-len 1048576
    --tokenizer-mode inkling
    --kernel-config.enable_flashinfer_autotune=False
    --enable-auto-tool-choice
    --tool-call-parser inkling
    --reasoning-parser inkling
    --enable-chunked-prefill
    --enable-prefix-caching
    --enable-expert-parallel
    --no-async-scheduling
    --max-cudagraph-capture-size 256
)

VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
    --max-num-batched-tokens 16384
    --max-num-seqs 256
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)

VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 8192
    --max-num-seqs 256
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)

# This checkpoint contains eight trained MTP prediction layers. Keep speculative
# decoding opt-in so non-MTP and MTP runs remain directly comparable. Default to
# all eight layers while allowing controlled draft-width experiments. Prefill and
# decode must use the same setting because NIXL transfers their KV-cache state.
if [[ "${INKLING_ENABLE_MTP:-0}" == "1" ]]; then
    INKLING_MTP_NUM_SPECULATIVE_TOKENS="${INKLING_MTP_NUM_SPECULATIVE_TOKENS:-8}"
    VLLM_PREFILL_ARGS+=(
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${INKLING_MTP_NUM_SPECULATIVE_TOKENS}}"
    )
    VLLM_DECODE_ARGS+=(
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${INKLING_MTP_NUM_SPECULATIVE_TOKENS}}"
    )
fi
