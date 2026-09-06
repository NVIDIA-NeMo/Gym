#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Low-cost progress evidence for long, buffered GDPVal judge stages.

gdpval_judge_cache_count() {
    [[ $# == 1 ]] || return 64
    local cache_root=$1 task_dir cache_path cache_name count=0
    if [[ ! -d $cache_root || -L $cache_root ]]; then
        printf '0\n'
        return
    fi
    # Exact-assignment verify caches are direct children of task_<id>. Avoid a
    # recursive Lustre walk: each newly published cache is one completed
    # task/reference assignment even while the aggregate JSONL is buffered.
    for task_dir in "$cache_root"/task_*; do
        [[ -d $task_dir && ! -L $task_dir ]] || continue
        for cache_path in "$task_dir"/repeat_*_verify_response_*.json; do
            [[ -f $cache_path && ! -L $cache_path ]] || continue
            cache_name=${cache_path##*/}
            [[ $cache_name =~ ^repeat_[0-9]+_verify_response_[0-9a-f]{12,16}\.json$ ]] \
                || continue
            count=$((count + 1))
        done
    done
    printf '%s\n' "$count"
}

gdpval_judge_progress_signature() {
    [[ $# == 4 ]] || return 64
    local output=$1 journal=$2 failures=$3 cache_root=$4
    local output_bytes=0 journal_bytes=0 failure_bytes=0 cache_count
    [[ ! -f $output ]] || read -r output_bytes < <(wc -c < "$output")
    [[ ! -f $journal ]] || read -r journal_bytes < <(wc -c < "$journal")
    [[ ! -f $failures ]] || read -r failure_bytes < <(wc -c < "$failures")
    cache_count=$(gdpval_judge_cache_count "$cache_root") || return
    printf '%s:%s:%s:%s\n' "$output_bytes" "$journal_bytes" "$failure_bytes" "$cache_count"
}
