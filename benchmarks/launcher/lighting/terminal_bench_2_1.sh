#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=tb21
BENCHMARK_CONFIG=benchmarks/terminal_bench_2_1/opencode.yaml
BENCHMARK_CONCURRENCY=512
