#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_verified
BENCHMARK_CONFIG=benchmarks/swebench/verified/opencode.yaml
BENCHMARK_EXTRA_ARGS+=(
    "++swebench_verified_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.opencode_chunk_timeout_ms=1800000"
)
