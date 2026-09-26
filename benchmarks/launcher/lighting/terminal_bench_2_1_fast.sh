#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
source "$(dirname -- "${BASH_SOURCE[0]}")/fast_common.sh"
BENCHMARK=tb21fast
BENCHMARK_CONFIG=benchmarks/terminal_bench_2_1/terminus_2_fast.yaml
BENCHMARK_CONCURRENCY=512
