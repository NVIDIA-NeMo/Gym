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
# Writes runtime.json beside PARAMS. Exits with the command's status, 125 when the runner fails before
# starting it (including an exit from setup), or 126 or 127 when the command could not be started.

# An inherited BASH_ENV or SHELLOPTS must not stop the runner at an expected failure.
set +o errexit +o nounset +o pipefail
IFS=$' \t\n'
shopt -s extglob
# Inherited variables with these names must not configure the runner or leak into the CLI.
unset -v cwd command env stdin stdout stderr setup unset_env events
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
