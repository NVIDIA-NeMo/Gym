#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_verified
BENCHMARK_CONFIG=benchmarks/swebench/verified/opencode.yaml
BENCHMARK_CONCURRENCY=512
