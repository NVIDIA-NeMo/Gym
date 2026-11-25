#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: bash $0 --worker-urls http://replica1:8000 http://replica2:8000 [router options]" >&2
    exit 2
fi

# Build the patched wheel with environments/synclaw/build_vllm_router_wheel.sh.
# Native CLI options may override these defaults. A small imbalance threshold
# matters with only 64 live rollouts.
exec vllm-router \
    --host 127.0.0.1 --port 8090 \
    --prometheus-host 127.0.0.1 --prometheus-port 8091 \
    --policy cache_aware --cache-threshold 0.3 \
    --balance-abs-threshold 4 --balance-rel-threshold 1.5 \
    --intra-node-data-parallel-size 1 \
    --max-payload-size 536870912 --request-timeout-secs 3600 \
    --health-check-timeout-secs 60 --health-check-interval-secs 120 \
    --health-failure-threshold 10 \
    --disable-retries --disable-circuit-breaker --log-level info \
    "$@"
