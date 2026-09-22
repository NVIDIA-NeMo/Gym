#!/usr/bin/env bash
# IFBench (AllenAI instruction following). No sandboxes; the resources server clones github.com/allenai/IFBench at startup.
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=ifbench
BENCHMARK_CONFIG=benchmarks/ifbench/config.yaml
BENCHMARK_CONCURRENCY=512
# the checked-in ifbench_benchmark_metrics.json lacks the newer grading_mode key; the benchmark README runs with this override
BENCHMARK_EXTRA_ARGS=(++overwrite_metrics_conflicts=true)
