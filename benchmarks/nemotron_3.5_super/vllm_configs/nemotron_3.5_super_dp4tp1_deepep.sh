#!/bin/bash

NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"

# Optional detailed Nsight Systems capture on PD prefill nodes only.
# PROFILE_PREFILL=1 auto-installs nsys if needed in root Ubuntu containers.
# After /health is ready, wait PREFILL_PROFILE_DELAY_SECONDS, then capture
# PREFILL_PROFILE_STEPS worker steps. Set the delay to manual to POST /start_profile yourself.
# Reports go to ${SLURM_SUBMIT_DIR}/results/nsys; override with PREFILL_PROFILE_DIR.
PROFILE_PREFILL="${PROFILE_PREFILL:-0}"
PREFILL_PROFILE_STEPS="${PREFILL_PROFILE_STEPS:-20}"
PREFILL_PROFILE_DELAY_SECONDS="${PREFILL_PROFILE_DELAY_SECONDS:-300}"

GYM_MODEL_PARAMS=(
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=0.95"
)

# Nemotron's three-read Mamba SSM state must use the dimension-sequence layout when KV transfer is enabled.
# Not used when the model has no Mamba layers.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

export VLLM_USE_V2_MODEL_RUNNER=1

export VLLM_USE_RUST_FRONTEND=1

export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:8

# @bxyu-nvidia: `--skip-mm-profiling` Is needed to get Super VL checkpoint working, even with text benchmarks
# @bxyu-nvidia: We set --prefix-match-unit 128 because the Mooncake store prefill lookup gets more expensive the smaller this number is
# and caused prefill bottlenecks as a result when using Mooncake.
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
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --block-size 128
    --mamba-cache-mode align
    --mamba-backend flashinfer
    --mamba-ssm-cache-dtype float32
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --data-parallel-size 4
    --data-parallel-size-local 4
    --tensor-parallel-size 1
    --api-server-count 1
    --enable-mamba-shared-prefix-checkpoint
    --prefix-match-unit 128
    --use-replayssm
    --replayssm-buffer-len 16
)
# Diagnostic run only: record allocator stacks on both prefill and decode workers.
# Stops after GYM_ALLOCATION_TRACE_STEPS serving steps (default 120).
if [[ "${TRACE_ALLOCATIONS:-0}" == 1 ]]; then
    allocation_trace_module_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    export PYTHONPATH="${allocation_trace_module_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export GYM_ALLOCATION_TRACE_DIR="${GYM_ALLOCATION_TRACE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/results/allocation-traces/${SLURM_JOB_ID:-local}}"
    VLLM_COMMON_ARGS+=(--worker-cls allocation_trace_worker.AllocationTraceWorker)
fi
# @bxyu-nvidia: Lease duration is set to 180s since to give nodes a chance to pick up the request. Helps with resilience.
# @bxyu-nvidia: We set num_speculative_tokens_per_batch_size to 0 here since prefill does not need to speculate any tokens, it just needs to know that we are speculating.
VLLM_PREFILL_ARGS=(
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"num_speculative_tokens_per_batch_size\":[[1,1024,0]]}"
    # MRV2 captures a one-token graph for the zero-draft schedule, but Mamba's
    # full-graph capture requires 1 + num_speculative_tokens. Keep prefill
    # piecewise until that dynamic-speculation capture path supports Mamba.
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_producer",
                    "kv_load_failure_policy": "fail",
                    "kv_connector_extra_config": {
                        "kv_lease_duration": 180
                    }
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "load_async": true,
                        "lookup_async": true
                    }
                }
            ]
        }
    }'
    --max-cudagraph-capture-size 1200
    --max-num-batched-tokens 16960
    --max-num-seqs 256
    --all2all-backend deepep_high_throughput
    --moe-backend flashinfer_trtllm
)
# @bxyu-nvidia: We set "enable_lookup": false for the Mooncake store connector because the prefill worker handles the prefix cache lookup and decode just needs to receive the transferred kv cache.
VLLM_DECODE_ARGS=(
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}"
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_consumer",
                    "kv_load_failure_policy": "fail",
                    "kv_connector_extra_config": {
                        "kv_lease_duration": 180
                    }
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {
                        "load_async": true,
                        "lookup_async": true,
                        "enable_lookup": false
                    }
                }
            ]
        }
    }'
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [
            6, 12, 18, 24, 30, 36, 42, 48,
            72, 96, 144, 192, 240, 288, 336, 384,
            480, 576, 672, 768,
            864, 960, 1056, 1152, 1248, 1344, 1440, 1536
        ]
    }'
    --max-cudagraph-capture-size 1536
    --max-num-batched-tokens 1536
    --max-num-seqs 256
)
