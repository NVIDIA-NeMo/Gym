#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reconstructed from the vLLM 0.30.0 startup arguments in swe-bench-verified-juhim.log.
# Sampling overrides follow the other Nemotron Super configs.
GYM_MODEL_PARAMS=(
    "++model_endpoint_readiness_timeout_seconds=1200"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=0.95"
)

# Use the dimension-sequence Mamba state layout for KV transfer, as in the other Super configs.
export VLLM_SSM_CONV_STATE_LAYOUT=DS
# The log uses gpu_model_runner.py.
export VLLM_USE_V2_MODEL_RUNNER=0

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.8
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-model-len 262144
    --kv-cache-dtype nvfp4
    --no-disable-hybrid-kv-cache-manager
    --block-size 128
    --mamba-cache-dtype float32
    --mamba-ssm-cache-dtype float16
    --enable-mamba-cache-stochastic-rounding
    --mamba-cache-philox-rounds 5
    --mamba-cache-mode align
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --async-scheduling
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
)
VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
    --max-num-batched-tokens 135680
    --max-num-seqs 1024
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
