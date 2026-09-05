#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Foreground-owner lifecycle helpers for the long-running Gym judge process.

GYM_PID="${GYM_PID:-}"
GYM_RC="${GYM_RC:-}"
GDPVAL_CONTAINER_PYTHON=()

configure_gdpval_container_python() {
    local apptainer=$1 sif=$2 e2e_dir=$3 run_dir=$4 reference_overlay=$5 path
    local users_alias=/lustre/fsw/portfolios/llmservice/users user physical found
    local -a reference_roots=() reference_users=()
    for path in "$e2e_dir" "$run_dir" "$reference_overlay"; do
        [[ $path == /* && $path != *:* && $path != *$'\n'* ]] || return 64
    done
    [[ -x $apptainer && -f $sif && -d $e2e_dir && -d $run_dir && -f $reference_overlay ]] \
        || return 64
    while IFS= read -r path; do
        reference_roots+=("$path")
    done < <(
        sed -n 's|^[[:space:]]*deliverables_dir:[[:space:]]*\(/.*\)$|\1|p' \
            "$reference_overlay"
    )
    (( ${#reference_roots[@]} == 9 )) || return 64
    for path in "${reference_roots[@]}"; do
        [[ $path == /* && $path != *:* && $path != *$'\n'* && -d $path ]] || return 64
        if [[ $path =~ ^/lustre/fsw/portfolios/llmservice/users/([^/]+)/ ]]; then
            user=${BASH_REMATCH[1]}
            found=0
            for physical in "${reference_users[@]}"; do
                [[ $physical != "$user" ]] || found=1
            done
            (( found == 1 )) || reference_users+=("$user")
        fi
    done
    GDPVAL_CONTAINER_PYTHON=("$apptainer" exec)
    if (( ${#reference_users[@]} > 0 )); then
        [[ -d $users_alias ]] || return 64
        GDPVAL_CONTAINER_PYTHON+=(--bind "$users_alias:$users_alias:ro")
        for user in "${reference_users[@]}"; do
            physical=$(readlink -f "$users_alias/$user") || return 64
            [[ $physical == /* && $physical != *:* && -d $physical ]] || return 64
            GDPVAL_CONTAINER_PYTHON+=(--bind "$physical:$physical:ro")
        done
    fi
    GDPVAL_CONTAINER_PYTHON+=(
        --bind "$e2e_dir:$e2e_dir:ro"
        --bind "$run_dir:$run_dir"
        --bind "$reference_overlay:$reference_overlay:ro"
    )
    for path in "${reference_roots[@]}"; do
        GDPVAL_CONTAINER_PYTHON+=(--bind "$path:$path:ro")
    done
    GDPVAL_CONTAINER_PYTHON+=("$sif" python3)
}

start_judge_process_group() {
    local log=$1
    shift
    [[ -z $GYM_PID && $# -gt 0 ]] || return 64
    setsid "$@" >> "$log" 2>&1 &
    GYM_PID=$!
    [[ $GYM_PID =~ ^[1-9][0-9]*$ ]] || return 64
}

stop_judge_process_group() {
    local grace=${JUDGE_PROCESS_TERM_GRACE_SECONDS:-30} second
    [[ $grace =~ ^[1-9][0-9]*$ && $grace -le 300 ]] || return 64
    if [[ $GYM_PID =~ ^[1-9][0-9]*$ ]] && kill -0 -- "-$GYM_PID" 2>/dev/null; then
        kill -TERM -- "-$GYM_PID" 2>/dev/null || true
        for (( second=0; second<grace; second++ )); do
            kill -0 -- "-$GYM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 -- "-$GYM_PID" 2>/dev/null; then
            kill -KILL -- "-$GYM_PID" 2>/dev/null || true
        fi
    fi
    if [[ $GYM_PID =~ ^[1-9][0-9]*$ ]]; then
        wait "$GYM_PID" 2>/dev/null || true
    fi
    GYM_PID=""
}

wait_judge_process_group() {
    [[ $GYM_PID =~ ^[1-9][0-9]*$ ]] || return 64
    if wait "$GYM_PID"; then
        GYM_RC=0
    else
        GYM_RC=$?
    fi
    GYM_PID=""
}
