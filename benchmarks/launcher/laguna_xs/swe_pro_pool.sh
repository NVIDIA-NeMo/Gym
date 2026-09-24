#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_pro_pool
export NUM_NODES=${NUM_NODES:-2}
BENCHMARK_CONFIG=benchmarks/swebench/pro/pool.yaml
BENCHMARK_CONCURRENCY=$((128 * NUM_NODES))
