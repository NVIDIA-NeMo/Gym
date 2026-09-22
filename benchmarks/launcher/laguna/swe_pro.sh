#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_pro
BENCHMARK_CONFIG=benchmarks/swebench/pro/opencode.yaml
BENCHMARK_CONCURRENCY=1024
