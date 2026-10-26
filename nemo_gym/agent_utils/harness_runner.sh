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
# Writes runtime.json beside PARAMS.

shopt -s extglob
source "$1" || exit 1
printf '{"hostname": "%s", "pid": %d}\n' "$HOSTNAME" "$$" >"$(dirname "$1")/runtime.json"
eval "${setup-}" || exit 1
for pattern in "${unset_env[@]}"; do
    for name in $(compgen -e); do
        [[ $name == $pattern ]] && unset "$name"
    done
done
((${#env[@]})) && export "${env[@]}"
cd "$cwd" || exit 1
[[ -n ${events-} ]] || exec "${command[@]}" <"$stdin" >"$stdout" 2>"$stderr"

# Stamp stdout lines as they appear. A file, unlike a pipe, cannot be held open by a detached tool.
: >"$stdout"
exec 3<"$stdout" 4>"$events"
"${command[@]}" <"$stdin" >>"$stdout" 2>"$stderr" &
cli=$!
trap 'kill -TERM "$cli" 2>/dev/null' TERM
pending=""
emit() { [[ $1 == {* ]] && printf '[%s, %s]\n' "$seen" "$1" >&4; }
# Stamp new complete lines, keeping a partial one for the next pass. Bash < 5 has whole seconds.
stamp() {
    local line
    seen=${EPOCHREALTIME/,/.}
    [[ -n $seen ]] || printf -v seen '%(%s)T' -1
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
