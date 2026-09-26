#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_multilingual
BENCHMARK_CONFIG=benchmarks/swebench/multilingual/opencode.yaml
BENCHMARK_EXTRA_ARGS+=(
    "++swebench_multilingual_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.opencode_chunk_timeout_ms=1800000"
)
