#!/usr/bin/env bash
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_multilingual
BENCHMARK_CONFIG=benchmarks/swebench/multilingual/opencode.yaml
BENCHMARK_CONCURRENCY=512
