#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_pro
export NUM_NODES=${NUM_NODES:-8}
BENCHMARK_CONFIG=benchmarks/swebench/pro/opencode.yaml
BENCHMARK_EXTRA_ARGS+=(
    "++swebench_pro_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.opencode_chunk_timeout_ms=1800000"
)
