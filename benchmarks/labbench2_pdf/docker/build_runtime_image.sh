#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="${1:-nemo-gym-labbench2-pdf:2.0}"

docker build \
  --tag "${IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/labbench2-pdf-runtime.Dockerfile" \
  "${BENCHMARK_DIR}"
