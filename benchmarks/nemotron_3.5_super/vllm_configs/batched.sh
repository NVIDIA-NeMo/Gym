#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Serving flags from the batch pilots: use a compatible Super 3.5 vLLM image.
# Sourced in both containers; only the evaluation container installs Gym dependencies.
if [[ -n "${ROUTER_NODE:-}" ]]; then
    : "${GYM_BATCH_ARGS:?Submit with benchmarks/nemotron_3.5_super/submit_batch.sh.}"
    export UV_CACHE_DIR="/tmp/gym-batch-uv-${SLURM_JOB_ID:?Missing Slurm job ID.}"
    uv sync --active --frozen --inexact --no-dev
    source /opt/Gym_venv/bin/activate
    python -c 'import nemo_gym.cli.eval, nemo_gym.cli.env, ray'

    # submit_batch.sh encodes each original argument with printf %q, including overrides.
    # Decode only that generated string so prefetch and evaluation see identical settings.
    eval "batch_args=($GYM_BATCH_ARGS)"
    gym env prefetch \
        "${batch_args[@]}" \
        --config benchmarks/nemotron_3.5_super/sandbox_utils.yaml \
        --config benchmarks/nemotron_3.5_super/policy_model_override.yaml \
        ++uv_venv_dir=/opt/uv_venvs \
        ++skip_venv_if_present=false \
        ++dry_run=false \
        "++nemo_gym_log_dir=results/$EXPERIMENT_NAME/setup/logs" \
        "++policy_base_url=http://$ROUTER_NODE:8000/v1" \
        ++policy_api_key=dummy_api_key \
        "++policy_model_name=$MODEL_NAME"
fi

# Sampling is in batch_configs/*.yaml so CLI overrides remain effective.
GYM_MODEL_PARAMS=()
VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --chat-template "$MODEL/chat_template.jinja"
    --reasoning-parser-plugin "$MODEL/ultra_v3_reasoning_parser.py"
    --reasoning-parser ultra_v3
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-model-len 262144
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --async-scheduling
    --block-size 128
    --mamba-cache-mode align
    --mamba-cache-dtype auto
    --mamba-ssm-cache-dtype float32
    --no-enable-mamba-cache-stochastic-rounding
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --data-parallel-size 1
    --api-server-count 1
)
VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
    --max-num-batched-tokens 135680
    --max-num-seqs 1024
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)
