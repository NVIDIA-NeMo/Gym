#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Process supervisor uploaded beside a sandboxed harness. Needs only bash, a readable /proc, a sleep
# that accepts fractions of a second, and mv.
#
# Runs COMMAND in its own process group until it exits, the deadline passes or the stop file
# appears. A stopped harness gets TERM, then the whole group gets KILL, and so do the descendants
# that were still running in other process groups when it was stopped (tools a CLI started in their
# own session). The receipt confirms that none of them can still run. Processes that left the
# harness before the stop (detached services) are left for the sandbox owner.
#
# usage: process_supervisor.sh --timeout SECONDS --cleanup-timeout SECONDS --receipt PATH --stop-file PATH -- COMMAND...

# An inherited BASH_ENV or SHELLOPTS must not turn an expected failure into a missing receipt.
set +o errexit +o nounset +o pipefail

usage() {
    echo "usage: process_supervisor.sh --timeout S --cleanup-timeout S --receipt PATH --stop-file PATH -- COMMAND..." >&2
    exit 2
}

# Seconds as integer centiseconds, rounded down; fails unless positive.
centiseconds() {
    [[ $1 =~ ^([0-9]*)\.?([0-9]*)$ ]] || return 1
    local fraction="${BASH_REMATCH[2]}00"
    local value=$((10#0${BASH_REMATCH[1]} * 100 + 10#${fraction:0:2}))
    ((value > 0)) && echo "$value"
}

# Linux uptime in centiseconds, without forking.
now() {
    local uptime
    read -r uptime _ </proc/uptime || return 1
    [[ $uptime =~ ^[0-9]+\.[0-9]+$ ]] || return 1
    NOW=$((10#${uptime/./}))
}

# Fields of /proc/PID/stat after the command name: [0] state, [1] ppid, [2] pgrp, [19] start time.
# The kernel prints the name unescaped, so it can hold spaces, parentheses and newlines: read the
# whole file, split after the last ") ", and never evaluate a field as arithmetic.
stat_fields() {
    local stat=
    IFS= read -r -d '' stat 2>/dev/null <"/proc/$1/stat"
    [[ $stat == *') '* ]] || return 1
    fields=(${stat##*) })
    [[ ${fields[1]} =~ ^[0-9]+$ && ${fields[2]} =~ ^[0-9]+$ && ${fields[19]} =~ ^[0-9]+$ ]]
}

# Succeeds while a member of one of the given process groups can still run. Zombies cannot, and
# only their parent can reap them. A process whose state cannot be read counts as running.
group_alive() {
    local entry group
    for entry in /proc/[0-9]*; do
        if ! stat_fields "${entry#/proc/}"; then
            [[ -e $entry ]] && return 0
            continue
        fi
        [[ ${fields[0]} == [ZX] ]] && continue
        for group; do
            [[ ${fields[2]} == "$group" ]] && return 0
        done
    done
    return 1
}

# Record the harness's descendants outside its process group as "PID START", and the groups they
# lead. They stop being descendants once the harness dies, so this runs before the harness is stopped.
track_descendants() {
    local entry pid changed=1
    local -a parent=() group=() start=() inside=()
    for entry in /proc/[0-9]*; do
        pid=${entry#/proc/}
        [[ $pid =~ ^[0-9]+$ ]] && stat_fields "$pid" || continue
        parent[pid]=${fields[1]} group[pid]=${fields[2]} start[pid]=${fields[19]}
    done
    inside[harness]=1
    while ((changed)); do
        changed=0
        for pid in "${!parent[@]}"; do
            [[ -z ${inside[pid]} && -n ${inside[parent[pid]]} ]] || continue
            inside[pid]=1 changed=1
        done
    done
    for pid in "${!inside[@]}"; do
        [[ ${group[pid]} == "$harness" ]] && continue
        tracked+=("$pid ${start[pid]}")
        [[ ${group[pid]} == "$pid" ]] && tracked_groups+=("$pid")
    done
}

# Succeeds while a tracked descendant (same PID and start time) can still run.
tracked_alive() {
    local entry
    for entry in "${tracked[@]}"; do
        stat_fields "${entry% *}" && [[ ${fields[19]} == "${entry#* }" && ${fields[0]} != [ZX] ]] && return 0
    done
    return 1
}

kill_harness() {
    local entry group
    kill -KILL -- "-$harness" 2>/dev/null
    for entry in "${tracked[@]}"; do
        stat_fields "${entry% *}" && [[ ${fields[19]} == "${entry#* }" ]] && kill -KILL "${entry% *}" 2>/dev/null
    done
    for group in "${tracked_groups[@]}"; do
        kill -KILL -- "-$group" 2>/dev/null
    done
    return 0
}

stopped() { ((stopping)) || [[ -e $stop_file ]]; }

write_receipt() {
    printf '{"cleanup_confirmed": %s, "return_code": %s, "error": %s, "timed_out": %s}' \
        "$1" "$return_code" "$2" "$timed_out" >"$receipt.tmp" && mv -f "$receipt.tmp" "$receipt"
}

while (($#)) && [[ $1 != -- ]]; do
    (($# >= 2)) || usage
    case $1 in
    --timeout) timeout=$(centiseconds "$2") || { echo "--timeout must be finite and positive" >&2; exit 2; } ;;
    --cleanup-timeout) cleanup=$(centiseconds "$2") || { echo "--cleanup-timeout must be finite and positive" >&2; exit 2; } ;;
    --receipt) receipt=$2 ;;
    --stop-file) stop_file=$2 ;;
    *) echo "unexpected argument: $1" >&2; exit 2 ;;
    esac
    shift 2
done
[[ $1 == -- ]] || usage
shift
[[ -n ${timeout-} && -n ${cleanup-} && -n ${receipt-} && -n ${stop_file-} && $# -gt 0 ]] || usage

return_code=null timed_out=false stopping=0 tracked=() tracked_groups=()
trap 'stopping=1' TERM
if stopped; then
    write_receipt true null # Nothing was launched.
    exit 0
fi
if ! now; then
    write_receipt true '"Sandbox process supervision requires a readable /proc"' # Nothing was launched.
    exit 1
fi

# Job control gives the harness its own process group.
set -m
"$@" &
harness=$!
set +m

now
deadline=$((NOW + timeout))
while kill -0 "$harness" 2>/dev/null && ! stopped && ((NOW < deadline)); do
    sleep 0.05
    now
done
if kill -0 "$harness" 2>/dev/null; then
    stopped || timed_out=true
    track_descendants
    kill -TERM "$harness" 2>/dev/null
    grace=$((NOW + cleanup))
    while kill -0 "$harness" 2>/dev/null && ((NOW < grace)); do
        sleep 0.05
        now
    done
fi
trap '' TERM

kill_harness
# A process in uninterruptible sleep outlives KILL; report it rather than wait without bound.
now
deadline=$((NOW + cleanup))
while kill -0 "$harness" 2>/dev/null; do
    now
    if ((NOW >= deadline)); then
        write_receipt false '"Harness termination was not confirmed: the harness process did not exit"'
        exit 1
    fi
    sleep 0.01
done
wait "$harness"
return_code=$?
now
deadline=$((NOW + cleanup))
while group_alive "$harness" "${tracked_groups[@]}" || tracked_alive; do
    kill_harness
    now
    if ((NOW >= deadline)); then
        write_receipt false '"Harness termination was not confirmed: its process group still runs"'
        exit 1
    fi
    sleep 0.01
done
write_receipt true null
