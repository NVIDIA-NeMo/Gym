#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nemotron 3 Ultra BF16 configuration for disaggregated prefill/decode on
# 4-GPU GB200 nodes. Each tier uses four coupled data-parallel ranks so expert
# parallelism can shard the 512 experts over 16 GPUs. Launch this config with
# VLLM_PD_DEPLOYMENT_MODE=coupled and VLLM_SLURM_SEGMENT=4.

# The defaults reproduce the full-suite MTP3 configuration validated in Run
# 060: eager prefill and piecewise decode CUDA graphs with graph-owned inputs.
ULTRA_PREFILL_GPU_MEMORY_UTILIZATION="${ULTRA_PREFILL_GPU_MEMORY_UTILIZATION:-0.90}"
ULTRA_DECODE_GPU_MEMORY_UTILIZATION="${ULTRA_DECODE_GPU_MEMORY_UTILIZATION:-0.95}"
ULTRA_PREFILL_MAX_NUM_BATCHED_TOKENS="${ULTRA_PREFILL_MAX_NUM_BATCHED_TOKENS:-16384}"
ULTRA_DECODE_MAX_NUM_BATCHED_TOKENS="${ULTRA_DECODE_MAX_NUM_BATCHED_TOKENS:-8192}"
ULTRA_MAX_NUM_SEQS="${ULTRA_MAX_NUM_SEQS:-64}"
ULTRA_DECODE_CUDAGRAPH_MODE="${ULTRA_DECODE_CUDAGRAPH_MODE:-PIECEWISE}"
ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS="${ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS:-1}"
ULTRA_DECODE_ENFORCE_EAGER="${ULTRA_DECODE_ENFORCE_EAGER:-0}"
ULTRA_ENABLE_MTP="${ULTRA_ENABLE_MTP:-1}"
ULTRA_NUM_SPECULATIVE_TOKENS="${ULTRA_NUM_SPECULATIVE_TOKENS:-3}"

# Standard safetensors loading avoided the InstantTensor io_uring failures seen
# against the Lustre-hosted checkpoint.
export SAFETENSORS_FAST_GPU=1

case "$ULTRA_DECODE_CUDAGRAPH_MODE" in
    FULL_DECODE_ONLY | PIECEWISE | NONE) ;;
    *)
        echo "ERROR: ULTRA_DECODE_CUDAGRAPH_MODE must be FULL_DECODE_ONLY, PIECEWISE, or NONE; got '$ULTRA_DECODE_CUDAGRAPH_MODE'." >&2
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

case "$ULTRA_DECODE_ENFORCE_EAGER" in
    0 | 1) ;;
    *)
        echo "ERROR: ULTRA_DECODE_ENFORCE_EAGER must be 0 or 1; got '$ULTRA_DECODE_ENFORCE_EAGER'." >&2
        exit 2
        ;;
esac

case "$ULTRA_ENABLE_MTP" in
    0 | 1) ;;
    *)
        echo "ERROR: ULTRA_ENABLE_MTP must be 0 or 1; got '$ULTRA_ENABLE_MTP'." >&2
        exit 2
        ;;
esac

if [[ ! "$ULTRA_NUM_SPECULATIVE_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ULTRA_NUM_SPECULATIVE_TOKENS must be a positive integer; got '$ULTRA_NUM_SPECULATIVE_TOKENS'." >&2
    exit 2
fi

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
    --block-size 128
    --mamba-cache-mode align
    --mamba-ssm-cache-dtype float16
    --mamba-backend flashinfer
    --enable-mamba-cache-stochastic-rounding
    --mamba-cache-philox-rounds 5
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --load-format safetensors
    --enable-expert-parallel
    --distributed-timeout-seconds 3600
    # NIXL-transferred Mamba state must not coexist with locally retained
    # prefix-cache blocks; doing so triggers the multiple-local-block assertion.
    --no-enable-prefix-caching
)

VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
    --gpu-memory-utilization "$ULTRA_PREFILL_GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$ULTRA_PREFILL_MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$ULTRA_MAX_NUM_SEQS"
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --no-async-scheduling
    # Eager prefill avoids the compiled/CUDA-graph stalls observed during tuning.
    --enforce-eager
)

VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    --compilation-config "{\"cudagraph_mode\":\"$ULTRA_DECODE_CUDAGRAPH_MODE\",\"cudagraph_copy_inputs\":$ULTRA_DECODE_CUDAGRAPH_COPY_INPUTS_JSON,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
    --gpu-memory-utilization "$ULTRA_DECODE_GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$ULTRA_DECODE_MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$ULTRA_MAX_NUM_SEQS"
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --no-async-scheduling
)

if [[ "$ULTRA_DECODE_ENFORCE_EAGER" == "1" ]]; then
    # This fallback disables compilation and CUDA graphs for decode.
    VLLM_DECODE_ARGS+=(--enforce-eager)
fi

if [[ "$ULTRA_ENABLE_MTP" == "1" ]]; then
    # Prefill and decode must use the same speculative width so the transferred
    # cache layouts agree.
    ULTRA_SPECULATIVE_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$ULTRA_NUM_SPECULATIVE_TOKENS}"
    VLLM_PREFILL_ARGS+=(--speculative-config "$ULTRA_SPECULATIVE_CONFIG")
    VLLM_DECODE_ARGS+=(--speculative-config "$ULTRA_SPECULATIVE_CONFIG")
fi
