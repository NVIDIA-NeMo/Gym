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
# process can still run; it is not a sandbox-empty check. Its "stopped" field says the stop file,
# not the harness's own exit or the deadline, ended the harness or fenced its launch.
#
# Pause and continue, for a checkpoint that snapshots the sandbox while the harness keeps its state:
# writing a number N greater than any earlier one to the pause file sends STOP to every process the
# harness reaches, as a stop would find them, and to every process carrying the marker, including
# services that left the harness: unlike a kill, a pause is undone. The supervisor waits until each
# is stopped (longer while some are in uninterruptible sleep) and acknowledges in the paused file:
# {"seq": N, "confirmed": BOOL, "pids": [...], "error": STRING|null}. Writing N to the continue file
# sends CONT to the same processes and acknowledges in the continued file the same way. The time
# spent paused does not count against the deadline. A pause requested before the launch holds the
# launch until it is continued. A stop requested while paused continues the processes first, so the
# harness receives TERM and can save its state. The receipt, written when the harness ends, answers
# any pause the supervisor never acknowledged.
#
# usage: process_supervisor.sh --timeout SECONDS --cleanup-timeout SECONDS --receipt PATH --stop-file PATH
#        [--pause-file PATH --paused PATH --continue-file PATH --continued PATH] -- COMMAND...

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
    printf '{"cleanup_confirmed": %s, "return_code": %s, "error": %s, "timed_out": %s, "stopped": %s}' \
        "$1" "$return_code" "$2" "$timed_out" "$requested_stop" >"$receipt.tmp" && mv -f "$receipt.tmp" "$receipt"
}

# Set SEQ to the sequence number in request file $1, or 0 when it is missing or malformed. Polled
# every round, so it does not fork.
request_seq() {
    local value=
    SEQ=0
    [[ -n $1 && -e $1 ]] && IFS= read -r value 2>/dev/null <"$1"
    [[ $value =~ ^[0-9]{1,15}$ ]] && SEQ=$((10#$value))
    return 0
}

# Write acknowledgement $1 for request $2 atomically.
acknowledge() {
    printf '{"seq": %s, "confirmed": %s, "pids": [%s], "error": %s}' "$2" "$3" "$4" "$5" >"$1.tmp" && mv -f "$1.tmp" "$1"
}

# Record in paused[PID]=START every process a pause reaches: what a stop would reach now, and
# every marked process, including services that left the harness. Runs the scan in a subshell so
# that the stop's own bookkeeping is unchanged.
pause_scan() {
    local pid start
    while read -r pid start; do
        [[ -n $pid ]] && paused[pid]=$start
    done < <(
        preserving=1 preserved=()
        scan >/dev/null
        for pid in "${!tracked[@]}"; do echo "$pid ${tracked[pid]}"; done
    )
}

# Send signal $1 to every process recorded in paused whose start time still matches.
signal_paused() {
    local pid
    for pid in "${!paused[@]}"; do
        stat_fields "$pid" && [[ ${fields[19]} == "${paused[pid]}" ]] && kill "-$1" "$pid" 2>/dev/null
    done
    return 0
}

# Succeed once no recorded process is in a state other than $1 (a glob of /proc states); set
# blocked to 1 when only processes in uninterruptible sleep are left.
paused_settled() {
    local pid others=0 waiting=0
    for pid in "${!paused[@]}"; do
        stat_fields "$pid" && [[ ${fields[19]} == "${paused[pid]}" ]] || continue
        case ${fields[0]} in
        [ZX]) ;;
        $1) ;;
        D) waiting=1 ;;
        *) others=1 ;;
        esac
    done
    blocked=$((waiting && !others))
    ((!others && !waiting))
}

# Pause request $1: stop every reached process and acknowledge. A harness not launched yet has
# nothing to stop; the launch waits instead.
pause() {
    local limit error=null pid list=
    paused=()
    if [[ -n ${harness-} ]]; then
        pause_scan
        [[ -n ${groups[harness]} ]] && owned_group "$harness" "${groups[harness]}" && kill -STOP -- "-$harness" 2>/dev/null
        signal_paused STOP
        # A process forked between the scan and its group's stop is found again.
        pause_scan
        signal_paused STOP
    fi
    now
    limit=$((NOW + pause_wait))
    while ! paused_settled '[Tt]'; do
        now
        if ((NOW >= limit + (blocked ? pause_blocked_wait : 0))); then
            error='"A process the pause reached did not stop"'
            break
        fi
        signal_paused STOP
        sleep 0.02
    done
    for pid in "${!paused[@]}"; do list+="${list:+, }$pid"; done
    acknowledge "$paused_ack" "$1" "$([[ $error == null ]] && echo true || echo false)" "$list" "$error"
}

# Continue request $1: continue every process the pause stopped and acknowledge.
resume_paused() {
    local limit error=null
    now
    limit=$((NOW + pause_wait))
    signal_paused CONT
    while ! paused_settled '[!Tt]'; do
        now
        if ((NOW >= limit)); then
            error='"A paused process did not continue"'
            break
        fi
        signal_paused CONT
        sleep 0.02
    done
    paused=()
    acknowledge "$continued_ack" "$1" "$([[ $error == null ]] && echo true || echo false)" "" "$error"
}

# Serve a new pause request: pause, wait for its continue or a stop, then continue. Succeeds when a
# pause was served; the paused time is added to paused_total.
serve_pause() {
    # Not named started: pause_scan reads the supervisor's own start time under that name.
    local paused_since
    [[ -n $pause_file && -e $pause_file ]] || return 1
    request_seq "$pause_file"
    ((SEQ > pause_seq)) || return 1
    pause_seq=$SEQ
    now
    paused_since=$NOW
    pause "$pause_seq"
    while request_seq "$continue_file" && ((SEQ < pause_seq)) && ! stopped; do
        sleep 0.05
    done
    # A stop while paused continues the processes first, so TERM reaches a running harness.
    request_seq "$continue_file"
    resume_paused "$SEQ"
    now
    paused_total=$((paused_total + NOW - paused_since))
    return 0
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
    --pause-file) pause_file=$2 ;;
    --paused) paused_ack=$2 ;;
    --continue-file) continue_file=$2 ;;
    --continued) continued_ack=$2 ;;
    *) echo "unexpected argument: $1" >&2; exit 2 ;;
    esac
    shift 2
done
[[ $1 == -- ]] || usage
shift
[[ -n ${timeout-} && -n ${cleanup-} && -n ${receipt-} && -n ${stop_file-} && $# -gt 0 ]] || usage
pause_file=${pause_file-} paused_ack=${paused_ack-} continue_file=${continue_file-} continued_ack=${continued_ack-}
if [[ -n $pause_file$paused_ack$continue_file$continued_ack ]] &&
    [[ -z $pause_file || -z $paused_ack || -z $continue_file || -z $continued_ack ]]; then
    echo "--pause-file, --paused, --continue-file and --continued go together" >&2
    exit 2
fi

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

return_code=null timed_out=false stopping=0 preserving=0 forced=0 requested_stop=false
tracked=() groups=() preserved=() paused=()
# A pause waits this long (centiseconds) for reached processes to stop, and this much longer while
# only processes in uninterruptible sleep are left.
pause_wait=500 pause_blocked_wait=2500 pause_seq=0 paused_total=0 blocked=0
# A pause already continued before this supervisor started is not served: the launch starts running,
# and the continue is acknowledged for a client that waits for it.
if [[ -n $pause_file ]]; then
    request_seq "$continue_file"
    pause_seq=$SEQ
    ((pause_seq)) && acknowledge "$continued_ack" "$pause_seq" true "" null
fi
trap 'stopping=1' TERM
if stopped; then
    [[ -e $stop_file ]] && requested_stop=true
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
# A pause requested before the launch holds it until continued.
while ! stopped && serve_pause; do :; done
if stopped; then
    [[ -e $stop_file ]] && requested_stop=true
    write_receipt true null # Nothing was launched.
    exit 0
fi

# Job control gives the harness its own process group.
set -m
NEMO_GYM_SUPERVISOR=$identity "$@" &
harness=$!
set +m
stat_fields "$harness" && groups[harness]=${fields[19]} tracked[harness]=${fields[19]} || groups[harness]=

now
deadline=$((NOW + timeout))
while kill -0 "$harness" 2>/dev/null && ! stopped && ((NOW < deadline)); do
    if serve_pause; then
        # Paused time does not count against the deadline.
        deadline=$((deadline + paused_total))
        paused_total=0
    fi
    sleep 0.05
    now
done
if kill -0 "$harness" 2>/dev/null; then
    stopped || timed_out=true
    [[ $timed_out == false && -e $stop_file ]] && requested_stop=true
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
