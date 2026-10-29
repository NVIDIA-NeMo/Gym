#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Process supervisor uploaded beside a sandboxed harness. Needs only bash, a readable /proc, a sleep
# that accepts fractions of a second, and mv.
#
# Runs COMMAND in its own process group until it exits, the deadline passes or the stop file
# appears. When the harness exits by itself, its process group gets KILL. A stopped harness gets
# TERM, then KILL together with its process group and every process the stop reaches: each process
# that descends from the harness, from a member of its group or from another reached process, and
# each process that inherited the harness's NEMO_GYM_SUPERVISOR environment marker, such as a tool
# a CLI started in its own session or a service a tool daemonized during the stop. /proc is scanned
# when the stop begins and on every polling round until nothing reached still runs.
#
# Not reached: processes that had already left the harness when the stop began (detached
# services), which are left for the sandbox owner, and a process that leaves the harness between two
# scans after dropping the marker from its environment. The receipt confirms that no reached
# process can still run; it is not a sandbox-empty check.
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
# whole file, split after the last ") ", and never evaluate a field as arithmetic. Scans parse every
# process, so this strips the shortest prefix until no ") " is left and tests digits with globs:
# ${stat##*) } takes quadratic time and each =~ compiles a regex.
stat_fields() {
    local stat=
    IFS= read -r -d '' stat 2>/dev/null <"/proc/$1/stat"
    [[ $stat == *') '* ]] || return 1
    while [[ $stat == *') '* ]]; do
        stat=${stat#*) }
    done
    fields=($stat)
    [[ ${fields[1]} && ${fields[2]} && ${fields[19]} && ${fields[1]}${fields[2]}${fields[19]} != *[!0-9]* ]]
}

# Succeeds when process $1, started at $2, inherited this supervisor's marker. A process that started
# before the supervisor cannot have, so its environment is not read.
marked() {
    local variable
    ((10#$2 >= 10#$started)) || return 1
    while IFS= read -r -d '' variable; do
        [[ $variable == "$marker" ]] && return 0
    done 2>/dev/null <"/proc/$1/environ"
    return 1
}

# Succeeds while process group $1, whose leader started at $2, can still be the recorded one. A
# group keeps its ID while it has members, so a different process holding that PID means the group
# emptied and the ID was reused.
owned_group() {
    if stat_fields "$1"; then
        [[ ${fields[19]} == "$2" ]]
    else
        [[ ! -e /proc/$1 ]]
    fi
}

# Record in tracked[PID]=START and groups[PGID]=LEADER_START what a forced stop reaches: the
# harness, members of the groups already recorded, processes already tracked, processes that carry
# the marker without descending from a preserved one, and the descendants of all of these. The first
# scan of a stop preserves marked processes that it does not reach. Succeeds, like reached_alive,
# while a reached process can still run.
scan() {
    local entry pid leader changed=1 unreadable=0
    local -a parent=() group=() start=() inside=() kept=()
    for entry in /proc/[0-9]*; do
        pid=${entry#/proc/}
        if ! [[ $pid =~ ^[0-9]+$ ]] || ! stat_fields "$pid"; then
            [[ -e $entry ]] && unreadable=1
            continue
        fi
        start[pid]=${fields[19]}
        [[ ${fields[0]} == [ZX] ]] || parent[pid]=${fields[1]} group[pid]=${fields[2]}
    done
    for pid in "${!preserved[@]}"; do
        [[ ${start[pid]} == "${preserved[pid]}" ]] && kept[pid]=1
    done
    while ((changed)); do
        changed=0
        for pid in "${!parent[@]}"; do
            [[ -z ${kept[pid]} && -n ${kept[parent[pid]]} ]] || continue
            kept[pid]=1 changed=1
        done
    done
    for pid in "${!parent[@]}"; do
        leader=${group[pid]}
        if [[ -n ${kept[pid]} || $pid == "$$" ]]; then
            continue
        elif [[ -n ${tracked[pid]} && ${tracked[pid]} == "${start[pid]}" ]]; then
            inside[pid]=1
        elif [[ -n ${groups[leader]+set} && ${start[leader]-${groups[leader]}} == "${groups[leader]}" ]]; then
            inside[pid]=1
        elif ((preserving)) && marked "$pid" "${start[pid]}"; then
            inside[pid]=1
        fi
    done
    changed=1
    while ((changed)); do
        changed=0
        for pid in "${!parent[@]}"; do
            [[ -z ${inside[pid]} && -z ${kept[pid]} && -n ${inside[parent[pid]]} ]] || continue
            inside[pid]=1 changed=1
        done
    done
    if ((!preserving)); then
        for pid in "${!parent[@]}"; do
            [[ -z ${inside[pid]} && $pid != "$$" ]] && marked "$pid" "${start[pid]}" && preserved[pid]=${start[pid]}
        done
        preserving=1
    fi
    for pid in "${!inside[@]}"; do
        tracked[pid]=${start[pid]}
        [[ ${group[pid]} == "$pid" ]] && groups[pid]=${start[pid]}
    done
    ((${#inside[@]} || unreadable))
}

kill_reached() {
    local pid leader
    for leader in "${!groups[@]}"; do
        owned_group "$leader" "${groups[leader]}" && kill -KILL -- "-$leader" 2>/dev/null
    done
    for pid in "${!tracked[@]}"; do
        stat_fields "$pid" && [[ ${fields[19]} == "${tracked[pid]}" ]] && kill -KILL "$pid" 2>/dev/null
    done
    return 0
}

# Succeeds while a reached process can still run: a member of a recorded group that was not reused,
# or a tracked process with its start time. Zombies cannot run, and only their parent can reap them.
# A process whose state cannot be read counts as running.
reached_alive() {
    local entry pid leader
    local -a owned=()
    for leader in "${!groups[@]}"; do
        owned_group "$leader" "${groups[leader]}" && owned[leader]=1
    done
    for entry in /proc/[0-9]*; do
        pid=${entry#/proc/}
        if ! [[ $pid =~ ^[0-9]+$ ]] || ! stat_fields "$pid"; then
            [[ -e $entry ]] && return 0
            continue
        fi
        [[ ${fields[0]} == [ZX] ]] && continue
        [[ -n ${owned[fields[2]]} ]] && return 0
        [[ -n ${tracked[pid]} && ${tracked[pid]} == "${fields[19]}" ]] && return 0
    done
    return 1
}

stopped() { ((stopping)) || [[ -e $stop_file ]]; }

write_receipt() {
    printf '{"cleanup_confirmed": %s, "return_code": %s, "error": %s, "timed_out": %s}' \
        "$1" "$return_code" "$2" "$timed_out" >"$receipt.tmp" && mv -f "$receipt.tmp" "$receipt"
}

# Tests source the functions above without running the supervisor.
[[ ${BASH_SOURCE[0]} == "$0" ]] || return 0

arguments=("$@")
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

# Leave the provider's process group, so that a KILL sent to it cannot end the supervisor before
# the receipt. This shell stays behind to pass on TERM and the exit status.
if [[ ${NEMO_GYM_SUPERVISOR-} != detached ]] && stat_fields $$ && [[ ${fields[2]} != "$$" ]]; then
    set -m
    NEMO_GYM_SUPERVISOR=detached "$BASH" "$0" "${arguments[@]}" &
    supervisor=$!
    set +m
    trap 'kill -TERM "$supervisor" 2>/dev/null' TERM
    while kill -0 "$supervisor" 2>/dev/null; do
        wait "$supervisor"
    done
    wait "$supervisor"
    exit
fi

return_code=null timed_out=false stopping=0 preserving=0 forced=0
tracked=() groups=() preserved=()
trap 'stopping=1' TERM
if stopped; then
    write_receipt true null # Nothing was launched.
    exit 0
fi
if ! now || ! stat_fields $$; then
    write_receipt true '"Sandbox process supervision requires a readable /proc"' # Nothing was launched.
    exit 1
fi
started=${fields[19]}
identity=$$.$started
marker=NEMO_GYM_SUPERVISOR=$identity

# Job control gives the harness its own process group.
set -m
NEMO_GYM_SUPERVISOR=$identity "$@" &
harness=$!
set +m
stat_fields "$harness" && groups[harness]=${fields[19]} tracked[harness]=${fields[19]} || groups[harness]=

now
deadline=$((NOW + timeout))
while kill -0 "$harness" 2>/dev/null && ! stopped && ((NOW < deadline)); do
    sleep 0.05
    now
done
if kill -0 "$harness" 2>/dev/null; then
    stopped || timed_out=true
    forced=1
    scan
    kill -TERM "$harness" 2>/dev/null
    grace=$((NOW + cleanup))
    # Once the harness has exited, the drain below scans for what it left.
    while sleep 0.05; kill -0 "$harness" 2>/dev/null; do
        scan
        now
        ((NOW < grace)) || break
    done
fi
trap '' TERM

kill_reached
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
while :; do
    if ((forced)); then
        scan || break
    else
        reached_alive || break
    fi
    kill_reached
    now
    if ((NOW >= deadline)); then
        write_receipt false '"Harness termination was not confirmed: a process it started still runs"'
        exit 1
    fi
    sleep 0.01
done
write_receipt true null
