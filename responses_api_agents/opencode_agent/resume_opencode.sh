#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Resume a saved OpenCode session's existing user turn without adding a prompt. Runs in the task sandbox,
# under the process supervisor, with only bash and the OpenCode runtime: the task image may have no Python.
#
# OpenCode's CLI resumes a session only with a new prompt, which adds a user turn the task never had.
# Its server's prompt API upserts the user message named by `messageID` and, with empty `parts`, keeps that
# message's parts and runs the session loop again from it. The agent server builds that request body from the
# saved conversation and uploads it as PAYLOAD.
#
# usage: resume_opencode.sh OPENCODE SESSION_ID PAYLOAD
#
# Exits 0 once the resumed turn finished, 3 when the server never became ready, 4 when the prompt request
# failed. OpenCode writes the conversation to its database as the CLI would.

set +o errexit +o nounset +o pipefail
opencode=$1 session=$2 payload=$3
[[ -x $opencode && -n $session && -f $payload ]] || { echo "usage: resume_opencode.sh OPENCODE SESSION_ID PAYLOAD" >&2; exit 2; }

password=
for _ in 1 2 3 4; do password+=$(printf '%08x' "$((RANDOM * 32768 + RANDOM))"); done
export OPENCODE_SERVER_USERNAME=opencode OPENCODE_SERVER_PASSWORD=$password
log=$(mktemp "${TMPDIR:-/tmp}/opencode-resume.XXXXXX") || exit 3

"$opencode" serve --hostname 127.0.0.1 --port 0 >"$log" 2>&1 &
server=$!
stop_server() {
    kill -INT "$server" 2>/dev/null
    for _ in $(seq 1 150); do kill -0 "$server" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$server" 2>/dev/null
    wait "$server" 2>/dev/null
    rm -f "$log"
}
trap 'stop_server; exit 143' TERM
trap 'stop_server; exit 130' INT

port=
for _ in $(seq 1 600); do
    if [[ $(<"$log") =~ opencode\ server\ listening\ on\ http://127\.0\.0\.1:([0-9]+) ]]; then
        port=${BASH_REMATCH[1]}
        break
    fi
    kill -0 "$server" 2>/dev/null || break
    sleep 0.1
done
if [[ -z $port ]]; then
    echo "OpenCode resume server did not become ready:" >&2
    cat "$log" >&2
    stop_server
    exit 3
fi

# OpenCode's CLI scopes sessions by the working directory, as PWD names it.
directory=$PWD
encode() {
    local value=$1 encoded= character index
    for ((index = 0; index < ${#value}; index++)); do
        character=${value:index:1}
        case $character in
        [A-Za-z0-9._~-]) encoded+=$character ;;
        *) printf -v character '%%%02X' "'$character"; encoded+=$character ;;
        esac
    done
    printf '%s' "$encoded"
}
path="/session/$(encode "$session")/message?directory=$(encode "$directory")"
authorization="Basic $(printf '%s' "opencode:$password" | base64 | tr -d '\n')"
status=
if command -v curl >/dev/null 2>&1; then
    status=$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' --max-time 0 \
        -H "Authorization: $authorization" -H 'Content-Type: application/json' \
        --data-binary "@$payload" "http://127.0.0.1:$port$path")
else
    # No curl in the image: one HTTP/1.0 request over bash's /dev/tcp, which closes when the turn is done.
    exec 5<>"/dev/tcp/127.0.0.1/$port" || { stop_server; exit 4; }
    body=$(<"$payload")
    printf 'POST %s HTTP/1.0\r\nHost: 127.0.0.1\r\nAuthorization: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s' \
        "$path" "$authorization" "$(printf '%s' "$body" | wc -c)" "$body" >&5
    IFS= read -r line <&5
    [[ $line =~ ^HTTP/[0-9.]+\ ([0-9]+) ]] && status=${BASH_REMATCH[1]}
    cat <&5 >/dev/null
    exec 5<&-
fi
stop_server
if [[ $status != 200 ]]; then
    echo "OpenCode resume request failed with HTTP status ${status:-none}" >&2
    exit 4
fi
echo "OpenCode resumed the existing user turn without adding a prompt"
