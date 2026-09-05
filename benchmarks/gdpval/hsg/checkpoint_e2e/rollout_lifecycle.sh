#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared lifecycle helpers for the GDPVal rollout batch wrapper.

# The caller owns `gym_pid` and `serve_pids`. Cleanup is deliberately
# idempotent because the walltime handler cleans up before exiting, which then
# also runs the EXIT trap.
GDPVAL_ROLLOUT_CLEANUP_DONE=${GDPVAL_ROLLOUT_CLEANUP_DONE:-false}

gdpval_rollout_materialize_dataset() {
    local source=$1 destination=$2 temporary
    [[ -f $source && ! -L $source ]] || {
        echo "FATAL: rollout dataset is not a regular non-symlink file: $source" >&2
        return 64
    }
    mkdir -p "$(dirname "$destination")"
    if [[ -e $destination || -L $destination ]]; then
        [[ -f $destination && ! -L $destination ]] || {
            echo "FATAL: materialized rollout dataset is not a regular file: $destination" >&2
            return 64
        }
        cmp -s "$source" "$destination" || {
            echo "FATAL: materialized rollout dataset drift: $destination" >&2
            return 64
        }
        return 0
    fi

    temporary="${destination}.tmp.$$"
    [[ ! -e $temporary && ! -L $temporary ]] || {
        echo "FATAL: stale rollout dataset staging path: $temporary" >&2
        return 64
    }
    if ! cp -- "$source" "$temporary"; then
        rm -f -- "$temporary"
        return 1
    fi
    chmod 0400 "$temporary"
    if ! cmp -s "$source" "$temporary"; then
        rm -f -- "$temporary"
        echo "FATAL: rollout dataset changed while materializing: $source" >&2
        return 64
    fi
    mv -- "$temporary" "$destination"
}

gdpval_rollout_cleanup() {
    local pid
    [[ ${GDPVAL_ROLLOUT_CLEANUP_DONE:-false} != true ]] || return 0
    GDPVAL_ROLLOUT_CLEANUP_DONE=true

    if [[ -n ${gym_pid:-} ]]; then
        kill -TERM -- "-${gym_pid}" 2>/dev/null || true
        kill -TERM "$gym_pid" 2>/dev/null || true
    fi
    for pid in "${serve_pids[@]:-}"; do
        [[ -n $pid ]] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done
    return 0
}

# An EXIT trap must preserve the command's original status. In particular, a
# harmless ESRCH from killing an already-exited Gym process must never turn a
# successful rollout shard into a failed Slurm job.
gdpval_rollout_on_exit() {
    local rc=$?
    trap - EXIT
    gdpval_rollout_cleanup
    exit "$rc"
}
