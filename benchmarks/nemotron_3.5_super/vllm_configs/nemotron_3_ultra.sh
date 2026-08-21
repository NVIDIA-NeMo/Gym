#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nemotron 3 Ultra BF16 baseline for disaggregated prefill/decode on 4-GPU
# GB200 nodes. This launcher's TP4-per-node layout needs four coupled DP ranks
# per tier so expert parallelism can shard the 512 experts over 16 GPUs.

# InstantTensor's io_uring loader returned EIO while 24 ranks concurrently read
# the Lustre-hosted checkpoint in Run 001. Keep the loader selectable for later
# experiments, but use vLLM's standard safetensors path for the stable baseline.
ULTRA_LOAD_FORMAT="${ULTRA_LOAD_FORMAT:-safetensors}"
# Run 021's all-eager P4/D4 baseline needs more prefill activation headroom,
# while decode can retain a larger KV cache. ULTRA_GPU_MEMORY_UTILIZATION is
# preserved as an optional global override; tier-specific overrides take priority.
ULTRA_GPU_MEMORY_UTILIZATION="${ULTRA_GPU_MEMORY_UTILIZATION:-}"
ULTRA_PREFILL_GPU_MEMORY_UTILIZATION="${ULTRA_PREFILL_GPU_MEMORY_UTILIZATION:-${ULTRA_GPU_MEMORY_UTILIZATION:-0.90}}"
ULTRA_DECODE_GPU_MEMORY_UTILIZATION="${ULTRA_DECODE_GPU_MEMORY_UTILIZATION:-${ULTRA_GPU_MEMORY_UTILIZATION:-0.95}}"
ULTRA_PREFILL_MAX_NUM_BATCHED_TOKENS="${ULTRA_PREFILL_MAX_NUM_BATCHED_TOKENS:-16384}"
ULTRA_DECODE_MAX_NUM_BATCHED_TOKENS="${ULTRA_DECODE_MAX_NUM_BATCHED_TOKENS:-8192}"
ULTRA_MAX_NUM_SEQS="${ULTRA_MAX_NUM_SEQS:-64}"
ULTRA_ENABLE_PREFIX_CACHING="${ULTRA_ENABLE_PREFIX_CACHING:-0}"
ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS="${ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS:-0}"
ULTRA_PREFILL_ENFORCE_EAGER="${ULTRA_PREFILL_ENFORCE_EAGER:-1}"
ULTRA_DECODE_CUDAGRAPH_MODE="${ULTRA_DECODE_CUDAGRAPH_MODE:-NONE}"
ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS="${ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS:-0}"
ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES="${ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES:-}"
ULTRA_DISABLE_DECODE_CUSTOM_ALL_REDUCE="${ULTRA_DISABLE_DECODE_CUSTOM_ALL_REDUCE:-0}"
ULTRA_DECODE_ALL2ALL_BACKEND="${ULTRA_DECODE_ALL2ALL_BACKEND:-}"
ULTRA_DECODE_ENFORCE_EAGER="${ULTRA_DECODE_ENFORCE_EAGER:-1}"
export SAFETENSORS_FAST_GPU=1

case "$ULTRA_DECODE_CUDAGRAPH_MODE" in
    FULL_DECODE_ONLY | PIECEWISE | NONE) ;;
    *)
        echo "ERROR: ULTRA_DECODE_CUDAGRAPH_MODE must be FULL_DECODE_ONLY, PIECEWISE, or NONE; got '$ULTRA_DECODE_CUDAGRAPH_MODE'." >&2
        exit 2
        ;;
esac

case "$ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS" in
    0)
        ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS_JSON=false
        ;;
    1)
        ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS_JSON=true
        ;;
    *)
        echo "ERROR: ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS must be 0 or 1; got '$ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS'." >&2
        exit 2
        ;;
esac

case "$ULTRA_PREFILL_ENFORCE_EAGER" in
    0 | 1) ;;
    *)
        echo "ERROR: ULTRA_PREFILL_ENFORCE_EAGER must be 0 or 1; got '$ULTRA_PREFILL_ENFORCE_EAGER'." >&2
        exit 2
        ;;
esac

case "$ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS" in
    0)
        ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS_JSON=false
        ;;
    1)
        ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS_JSON=true
        ;;
    *)
        echo "ERROR: ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS must be 0 or 1; got '$ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS'." >&2
        exit 2
        ;;
esac

ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES_JSON=""
if [[ -n "$ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES" ]]; then
    if [[ ! "$ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]]; then
        echo "ERROR: ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES must be a compact JSON integer array; got '$ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES'." >&2
        exit 2
    fi
    ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES_JSON=",\"cudagraph_capture_sizes\":$ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES"
fi

case "$ULTRA_DISABLE_DECODE_CUSTOM_ALL_REDUCE" in
    0 | 1) ;;
    *)
        echo "ERROR: ULTRA_DISABLE_DECODE_CUSTOM_ALL_REDUCE must be 0 or 1; got '$ULTRA_DISABLE_DECODE_CUSTOM_ALL_REDUCE'." >&2
        exit 2
        ;;
esac

case "$ULTRA_DECODE_ALL2ALL_BACKEND" in
    "" | allgather_reducescatter | deepep_high_throughput | deepep_low_latency | deepep_v2 | flashinfer_all2allv | flashinfer_nvlink_one_sided | flashinfer_nvlink_two_sided | mori_high_throughput | mori_low_latency | nixl_ep) ;;
    *)
        echo "ERROR: unsupported ULTRA_DECODE_ALL2ALL_BACKEND '$ULTRA_DECODE_ALL2ALL_BACKEND'." >&2
        exit 2
        ;;
esac

case "$ULTRA_DECODE_ENFORCE_EAGER" in
    0 | 1) ;;
    *)
        echo "ERROR: ULTRA_DECODE_ENFORCE_EAGER must be 0 or 1; got '$ULTRA_DECODE_ENFORCE_EAGER'." >&2
        exit 2
        ;;
esac

VLLM_COMMON_ARGS=(
    --disable-uvicorn-access-log
    --trust-remote-code
    --dtype bfloat16
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --max-model-len 262144
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
    --enable-chunked-prefill
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --no-async-scheduling
    --block-size 128
    --mamba-cache-mode align
    --mamba-ssm-cache-dtype float16
    --mamba-backend flashinfer
    --enable-mamba-cache-stochastic-rounding
    --mamba-cache-philox-rounds 5
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --load-format "$ULTRA_LOAD_FORMAT"
    --enable-expert-parallel
    --distributed-timeout-seconds 3600
)

if [[ "$ULTRA_ENABLE_PREFIX_CACHING" == "1" ]]; then
    VLLM_COMMON_ARGS+=(--enable-prefix-caching)
else
    # vLLM 0.25.1's NIXL pull connector cannot reconcile multiple locally
    # prefix-cached Mamba/SSM blocks with transferred prefill state.
    VLLM_COMMON_ARGS+=(--no-enable-prefix-caching)
fi

VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
    --compilation-config "{\"cudagraph_copy_inputs\":$ULTRA_PREFILL_CUDAGRAPH_COPY_INPUTS_JSON,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
    --gpu-memory-utilization "$ULTRA_PREFILL_GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$ULTRA_PREFILL_MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$ULTRA_MAX_NUM_SEQS"
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)

if [[ "$ULTRA_PREFILL_ENFORCE_EAGER" == "1" ]]; then
    # Bypass torch.compile as well as CUDA graphs on the prefill tier.
    VLLM_PREFILL_ARGS+=(--enforce-eager)
fi

VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    --compilation-config "{\"cudagraph_mode\":\"$ULTRA_DECODE_CUDAGRAPH_MODE\",\"cudagraph_copy_inputs\":$ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS_JSON$ULTRA_DECODE_CUDAGRAPH_CAPTURE_SIZES_JSON,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
    --gpu-memory-utilization "$ULTRA_DECODE_GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$ULTRA_DECODE_MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$ULTRA_MAX_NUM_SEQS"
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)

if [[ "$ULTRA_DISABLE_DECODE_CUSTOM_ALL_REDUCE" == "1" ]]; then
    # Isolate custom all-reduce from the graph-enabled distributed decode path.
    # vLLM falls back to its standard NCCL-backed tensor-parallel reductions.
    VLLM_DECODE_ARGS+=(--disable-custom-all-reduce)
fi

if [[ -n "$ULTRA_DECODE_ALL2ALL_BACKEND" ]]; then
    # Select the collective used to dispatch tokens to the expert-parallel
    # decode ranks and combine the expert outputs. Prefill stays on the default.
    VLLM_DECODE_ARGS+=(--all2all-backend "$ULTRA_DECODE_ALL2ALL_BACKEND")
fi

if [[ "$ULTRA_DECODE_ENFORCE_EAGER" == "1" ]]; then
    # Bypass torch.compile as well as CUDA graphs on the decode tier.
    VLLM_DECODE_ARGS+=(--enforce-eager)
fi

if [[ "${ULTRA_ENABLE_MTP:-1}" == "1" ]]; then
    # Producer and consumer must expose matching cache layouts for NIXL KV transfer.
    VLLM_PREFILL_ARGS+=(--speculative-config '{"method":"nemotron_h_mtp","num_speculative_tokens":5}')
    VLLM_DECODE_ARGS+=(--speculative-config '{"method":"nemotron_h_mtp","num_speculative_tokens":5}')
fi
