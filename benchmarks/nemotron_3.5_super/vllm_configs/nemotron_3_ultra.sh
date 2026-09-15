#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nemotron 3 Ultra BF16 configuration for disaggregated prefill/decode on
# 4-GPU GB200 nodes. Each tier uses four coupled data-parallel ranks so expert
# parallelism can shard the 512 experts over 16 GPUs. Launch this config with
# VLLM_PD_DEPLOYMENT_MODE=coupled and VLLM_SLURM_SEGMENT=4.

# Both tiers use piecewise CUDA graphs with graph-owned inputs and MTP3.
# Settings are fixed in this recipe; asynchronous scheduling is prefill-only.

# Standard safetensors loading avoided the InstantTensor io_uring failures seen
# against the Lustre-hosted checkpoint.
export SAFETENSORS_FAST_GPU=1

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
    --enable-prefix-caching
    # Both tiers need the same speculative width for compatible cache layouts.
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)

VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
    --gpu-memory-utilization 0.90
    --max-num-batched-tokens 16384
    --max-num-seqs 64
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --async-scheduling
    --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_copy_inputs":true,"pass_config":{"fuse_allreduce_rms":false}}'
)

VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_copy_inputs":true,"cudagraph_capture_sizes":
      [1,2,3,4,5,8,10,12,15,16,20,24,25,28,30,32,35,40,45,50,55,60,65,70,75,80,128,256,512],"pass_config":{"fuse_allreduce_rms":false}}'
    --gpu-memory-utilization 0.95
    --max-num-batched-tokens 8192
    --max-num-seqs 64
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --no-async-scheduling
)
