#!/bin/bash
#
# Nemotron Super 3.5 — low-concurrency serving recipe.
#
#
# Differs from nemotron_3.5_super_mtp.sh in exactly two ways, each measured at full scale:
#   * one engine instead of prefill/decode disaggregation — so VLLM_SERVE_ARGS replaces 
#     VLLM_PREFILL/DECODE_ARGS and the NIXL kv-transfer flags are gone.
#   * --enable-mamba-fine-grained-prefix-cache is NOT set: it costs +20% SWE call p50 and
#     +13% TB avg model call here. It buys prefix reuse that only pays when the KV cache is
#     scarce (production runs concurrency 1024); at C=32-48 we sit at 4-30% occupancy and
#     pay only its per-step overhead.
#
# MAX_NUM_SEQS is set by the launcher to max(8, 4 x concurrency); the default below matches
# C=48. Measured peak fan-out is 2 model calls per trajectory, so 4x leaves margin while a
# runaway still shows up as Waiting > 0 rather than hiding inside a 1024-wide ceiling.

GYM_MODEL_PARAMS=(
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=0.95"
)

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
    --mamba-cache-mode align
    --mamba-ssm-cache-dtype float32
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
)
VLLM_SERVE_ARGS=(
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
    --max-num-batched-tokens 33920
    --max-num-seqs "${MAX_NUM_SEQS:-192}"
)
