#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Process supervisor uploaded beside a sandboxed harness. Needs only bash, /proc, sleep and mv.
#
# Runs COMMAND in its own process group until it exits, the deadline passes or the stop file
# appears. A stopped harness gets TERM, then the whole group gets KILL. The receipt confirms that
# no group member can still run. Processes that left the group (detached services) are left for
# the sandbox owner.
#
# usage: process_supervisor.sh --timeout SECONDS --cleanup-timeout SECONDS --receipt PATH --stop-file PATH -- COMMAND...

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
    read -r uptime _ </proc/uptime
    NOW=$((10#${uptime/./}))
}

# Succeeds while a group member can still run. Zombies cannot, and only their parent can reap them.
group_alive() {
    local stat fields
    for stat in /proc/[0-9]*/stat; do
        read -r stat 2>/dev/null <"$stat" || continue
        fields=(${stat##*) })
        ((fields[2] == harness)) && [[ ${fields[0]} != [ZX] ]] && return 0
    done
    return 1
}

stopped() { ((stopping)) || [[ -e $stop_file ]]; }

write_receipt() {
    printf '{"cleanup_confirmed": %s, "return_code": %s, "error": %s, "timed_out": %s}' \
        "$1" "$return_code" "$2" "$timed_out" >"$receipt.tmp" && mv -f "$receipt.tmp" "$receipt"
}

while [[ $1 == --* && $1 != -- ]]; do
    case $1 in
    --timeout) timeout=$(centiseconds "$2") || { echo "--timeout must be finite and positive" >&2; exit 2; } ;;
    --cleanup-timeout) cleanup=$(centiseconds "$2") || { echo "--cleanup-timeout must be finite and positive" >&2; exit 2; } ;;
    --receipt) receipt=$2 ;;
    --stop-file) stop_file=$2 ;;
    *) echo "unexpected argument: $1" >&2; exit 2 ;;
    esac
    shift 2
done
shift
[[ -n ${timeout-} && -n ${cleanup-} && -n ${receipt-} && -n ${stop_file-} && $# -gt 0 ]] || {
    echo "usage: process_supervisor.sh --timeout S --cleanup-timeout S --receipt PATH --stop-file PATH -- COMMAND..." >&2
    exit 2
}

return_code=null timed_out=false stopping=0
trap 'stopping=1' TERM
if stopped; then
    write_receipt true null # Nothing was launched.
    exit 0
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
    kill -TERM "$harness" 2>/dev/null
    grace=$((NOW + cleanup))
    while kill -0 "$harness" 2>/dev/null && ((NOW < grace)); do
        sleep 0.05
        now
    done
fi
trap '' TERM

kill -KILL -- "-$harness" 2>/dev/null
wait "$harness"
return_code=$?
now
deadline=$((NOW + cleanup))
while group_alive; do
    kill -KILL -- "-$harness" 2>/dev/null
    now
    if ((NOW >= deadline)); then
        write_receipt false '"Harness termination was not confirmed: its process group still runs"'
        exit 1
    fi
    sleep 0.01
done
write_receipt true null
