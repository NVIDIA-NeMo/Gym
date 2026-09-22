#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
source "$(dirname -- "${BASH_SOURCE[0]}")/fast_common.sh"
BENCHMARK=ifbench_fast
BENCHMARK_CONFIG=benchmarks/ifbench/config_fast.yaml
BENCHMARK_CONCURRENCY=512
BENCHMARK_EXTRA_ARGS+=(++overwrite_metrics_conflicts=true)
