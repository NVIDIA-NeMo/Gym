#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=tb21
BENCHMARK_CONFIG=benchmarks/terminal_bench_2_1/opencode.yaml
BENCHMARK_EXTRA_ARGS+=(
    "++terminal_bench_2_1_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.opencode_chunk_timeout_ms=1800000"
)
