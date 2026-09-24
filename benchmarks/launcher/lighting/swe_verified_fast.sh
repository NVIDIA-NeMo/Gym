#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
source "$(dirname -- "${BASH_SOURCE[0]}")/fast_common.sh"
BENCHMARK=swe_verified_fast
BENCHMARK_CONFIG=benchmarks/swebench/verified/opencode_fast.yaml
BENCHMARK_CONCURRENCY=256
