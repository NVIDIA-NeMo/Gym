#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile.sh"
BENCHMARK=swe_multilingual
BENCHMARK_CONFIG=benchmarks/swebench/multilingual/opencode.yaml
BENCHMARK_CONCURRENCY=128
