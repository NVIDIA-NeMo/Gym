#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Submit one cold-start trial. Reuse MODEL, CONTAINER, VLLM_CONFIG and Gym arguments.
set -euo pipefail
case "${1:-}" in
    baseline)
        export ROUTER_CACHE_THRESHOLD=0.3 ROUTER_BALANCE_ABS_THRESHOLD=64 ROUTER_BALANCE_REL_THRESHOLD=1.5
        ;;
    affinity)
        export ROUTER_CACHE_THRESHOLD=0.1 ROUTER_BALANCE_ABS_THRESHOLD=256 ROUTER_BALANCE_REL_THRESHOLD=2.0
        ;;
    selective)
        export ROUTER_CACHE_THRESHOLD=0.8 ROUTER_BALANCE_ABS_THRESHOLD=256 ROUTER_BALANCE_REL_THRESHOLD=2.0
        ;;
    *)
        echo 'Usage: tune_router.sh {baseline|affinity|selective} [Gym --config/override arguments]' >&2
        echo 'Trials are candidates, not established improvements. Compare equal workloads and cache counter deltas.' >&2
        exit 2
        ;;
esac
trial=$1
shift
export ROUTER_PREFILL_POLICY=cache_aware ROUTER_DECODE_POLICY=cache_aware
export ROUTER_EVICTION_INTERVAL=120 ROUTER_MAX_TREE_SIZE=67108864
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opencode_swe_verified/router-tuning}/$trial"
exec bash "$(dirname "${BASH_SOURCE[0]}")/sbatch_cluster_vllm.sh" "$@"
