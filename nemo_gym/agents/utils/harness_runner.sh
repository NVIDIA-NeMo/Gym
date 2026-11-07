#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Harness runner uploaded into task sandboxes; runs under process_supervisor.sh with only bash.
#
# usage: harness_runner.sh PARAMS
#
# PARAMS is bash source written by shell_assignments() on the agent server. It sets:
#   cwd, command (array), env (array of KEY=VALUE), stdin, stdout, stderr
# and optionally:
#   setup      bash code run first, before env is applied
#   unset_env  array of extglob patterns naming inherited variables to drop
#   events     file receiving each JSON object line of stdout as [seen_at, object]
#   MODEL_RELAY_PORT, MODEL_RELAY_ENDPOINT, MODEL_RELAY_STATUS, MODEL_RELAY_SCRIPT
#              run model_relay.py (MODEL_RELAY_SCRIPT) on 127.0.0.1:MODEL_RELAY_PORT for the command's model calls,
#              on the first Python 3.6+ found, restarted on the same port if it dies, and export
#              NEMO_GYM_MODEL_RELAY=1. MODEL_RELAY_SESSION_DIR (default: the directory of PARAMS) holds the
#              supervisor's pause and continue requests; MODEL_RELAY_PYTHONS (array) overrides the interpreters
#              tried. MODEL_RELAY_COMMAND (array: a program and its leading arguments, such as a harness binary and
#              model_relay.mjs) runs the relay instead of a Python, with MODEL_RELAY_COMMAND_ENV (array of
#              KEY=VALUE) set for the relay alone. Without a usable Python or program, or when the relay does not
#              come up, the runner writes "unavailable" beside MODEL_RELAY_STATUS, exports MODEL_RELAY_FALLBACK_ENV
#              (array of KEY=VALUE, such as a direct model URL) and runs the command without a relay.
# Writes runtime.json beside PARAMS. Exits with the command's status, 125 when the runner fails before
# starting it (including an exit from setup), or 126 or 127 when the command could not be started.

# An inherited BASH_ENV or SHELLOPTS must not stop the runner at an expected failure.
set +o errexit +o nounset +o pipefail
IFS=$' \t\n'
shopt -s extglob
# Inherited variables with these names must not configure the runner or leak into the CLI.
unset -v cwd command env stdin stdout stderr setup unset_env events MODEL_RELAY_PORT MODEL_RELAY_ENDPOINT \
    MODEL_RELAY_STATUS MODEL_RELAY_SCRIPT MODEL_RELAY_SESSION_DIR MODEL_RELAY_PYTHONS MODEL_RELAY_FALLBACK_ENV \
    MODEL_RELAY_COMMAND MODEL_RELAY_COMMAND_ENV
trap 'exit 125' EXIT
source "$1" || exit
printf '{"hostname": "%s", "pid": %d}\n' "$HOSTNAME" "$$" >"$(dirname "$1")/runtime.json"
eval "${setup-}" || exit
for pattern in "${unset_env[@]}"; do
    for name in $(compgen -e); do
        [[ $name == $pattern ]] && unset "$name"
    done
done
((${#env[@]})) && export "${env[@]}"
cd "$cwd" || exit

# Whether process $1 runs: a zombie, which kill -0 still finds, does not.
running() {
    local stat=
    IFS= read -r stat 2>/dev/null <"/proc/$1/stat" || return 1
    stat=${stat##*) }
    [[ ${stat:0:1} != [ZX] ]]
}

# The first interpreter that is Python 3.6 or later, as the binary it runs, in relay_python.
find_relay_python() {
    local candidates=(python3 python /usr/local/bin/python3 /usr/bin/python3 /usr/bin/python) candidate
    ((${#MODEL_RELAY_PYTHONS[@]})) && candidates=("${MODEL_RELAY_PYTHONS[@]}")
    for candidate in "${candidates[@]}"; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        relay_python=$("$candidate" -I -c 'import os, sys; assert sys.version_info >= (3, 6); print(os.readlink("/proc/self/exe"))' 2>/dev/null) && [[ -n $relay_python ]] && return 0
    done
    return 1
}

# The port the relay's status file names, in port, when it names one.
relay_status_port() {
    local status=
    IFS= read -r -d '' status 2>/dev/null <"$MODEL_RELAY_STATUS"
    [[ $status =~ \"port\":\ ([1-9][0-9]*) ]] && port=${BASH_REMATCH[1]}
}

# Run the relay until the runner (or the command it became) ends, restarting it on the port it took first. Its
# program runs through a link with a neutral name, so kills in the task that match python or the harness skip it.
relay_watchdog() {
    local port=$MODEL_RELAY_PORT relay
    trap '[[ -n ${relay-} ]] && kill -TERM "$relay" 2>/dev/null; exit 0' TERM
    while running "$runner"; do
        rm -f "$MODEL_RELAY_STATUS"
        (
            ((${#MODEL_RELAY_COMMAND_ENV[@]})) && export "${MODEL_RELAY_COMMAND_ENV[@]}"
            exec "$relay_link" "${relay_args[@]}" --port "$port" --endpoint "$MODEL_RELAY_ENDPOINT" \
                --session-dir "${MODEL_RELAY_SESSION_DIR:-$params_dir}" --status "$MODEL_RELAY_STATUS"
        ) >>"$relay_log" 2>&1 &
        relay=$!
        while running "$relay" && running "$runner"; do
            ((port)) || relay_status_port
            sleep 0.2
        done
        ((port)) || relay_status_port
        running "$runner" || break
        wait "$relay" 2>/dev/null
        echo "model relay exited with status $?; restarting it on port $port" >>"$relay_log"
        sleep 0.2
    done
    kill -TERM "$relay" 2>/dev/null
}

# Centiseconds since boot in clock, from /proc/uptime, else whole seconds from SECONDS; without forking.
read_clock() {
    local uptime=
    read -r uptime _ 2>/dev/null </proc/uptime
    if [[ $uptime =~ ^([0-9]+)\.([0-9]{2}) ]]; then
        clock=$((10#${BASH_REMATCH[1]} * 100 + 10#${BASH_REMATCH[2]}))
    else
        clock=$((SECONDS * 100))
    fi
}

# Start the relay and wait until it listens; fails, leaving no relay, when it does not within 10 seconds of the
# runner's own running time. A checkpoint can freeze the runner (SIGSTOP) or stop the whole sandbox between two
# polls: a gap counts as at most a second, so the time it spent stopped does not use up the wait, and a relay that
# came up meanwhile is seen before the wait is checked.
start_relay() {
    local status= waited=0 last step clock
    relay_log=$(dirname "$MODEL_RELAY_STATUS")/relay.log
    relay_link=$(dirname "$MODEL_RELAY_STATUS")/gym-model-relay
    read_clock
    last=$clock
    if ((${#MODEL_RELAY_COMMAND[@]})); then
        [[ -x ${MODEL_RELAY_COMMAND[0]} ]] && ln -sfn "${MODEL_RELAY_COMMAND[0]}" "$relay_link" || return 1
        relay_args=("${MODEL_RELAY_COMMAND[@]:1}")
    else
        find_relay_python && ln -sfn "$relay_python" "$relay_link" || return 1
        relay_args=(-I "$MODEL_RELAY_SCRIPT")
    fi
    rm -f "$MODEL_RELAY_STATUS"
    relay_watchdog 2>>"$relay_log" &
    watchdog=$!
    while running "$watchdog"; do
        IFS= read -r -d '' status 2>/dev/null <"$MODEL_RELAY_STATUS"
        [[ $status =~ \"port\":\ [1-9] ]] && return 0
        read_clock
        step=$((clock - last))
        last=$clock
        ((step < 0)) && step=5
        ((step > 100)) && step=100
        waited=$((waited + step))
        ((waited <= 1000)) || break
        sleep 0.05
    done
    kill -TERM "$watchdog" 2>/dev/null
    return 1
}

if [[ -n ${MODEL_RELAY_PORT-} ]]; then
    runner=$$
    params_dir=$(dirname "$1")
    if start_relay; then
        export NEMO_GYM_MODEL_RELAY=1
    else
        : >"$(dirname "$MODEL_RELAY_STATUS")/unavailable"
        ((${#MODEL_RELAY_FALLBACK_ENV[@]})) && export "${MODEL_RELAY_FALLBACK_ENV[@]}"
    fi
fi
# A failed redirection does not end bash at exec; the CLI did not start.
if [[ -z ${events-} ]]; then
    trap - EXIT
    exec "${command[@]}" <"$stdin" >"$stdout" 2>"$stderr"
    exit 126
fi

# Stamp stdout lines as they appear. A file, unlike a pipe, cannot be held open by a detached tool.
: >"$stdout" && exec 3<"$stdout" 4>"$events" || exit
trap - EXIT
# The runner's own descriptors stay out of the CLI.
{ exec "${command[@]}" <"$stdin" >>"$stdout" 2>"$stderr" 3<&- 4>&-; exit 126; } &
cli=$!
trap 'kill -TERM "$cli" 2>/dev/null' TERM
pending=""
emit() {
    # ${1%$'\r'} would try the suffix at every position of a long line.
    local line=$1
    [[ ${line: -1} == $'\r' ]] && line=${line%?}
    [[ $line == {* ]] && printf '[%s, %s]\n' "$seen" "$line" >&4
}
# Stamp new complete lines, keeping a partial one for the next pass. Bash < 5 has whole seconds;
# bash < 4.2 has no strftime in printf and forks date.
stamp() {
    local line
    seen=${EPOCHREALTIME/,/.}
    [[ -n $seen ]] || printf -v seen '%(%s)T' -1 2>/dev/null
    [[ $seen =~ ^[0-9.]+$ ]] || seen=$(date +%s)
    while IFS= read -r line <&3; do
        emit "$pending$line"
        pending=""
    done
    pending+=$line
}
while kill -0 "$cli" 2>/dev/null; do
    stamp
    sleep 0.05
done
wait "$cli"
status=$?
stamp
emit "$pending"
exit "$status"
