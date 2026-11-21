#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reprompt a saved OpenCode session: run its existing user turn again without adding a prompt. Runs in the
# task sandbox, under the process supervisor, with only bash and the OpenCode runtime: the task image may have
# no Python.
#
# OpenCode's CLI continues a session only with a new prompt, which adds a user turn the task never had.
# Its server's prompt API upserts the user message named by `messageID` and, with empty `parts`, keeps that
# message's parts and runs the session loop again from it. The agent server builds that request body from the
# saved conversation and uploads it as PAYLOAD.
#
# `opencode run` rejects every permission request itself. Under `opencode serve` nothing answers one, and the
# prompt request would wait for it until the harness deadline, so this script rejects them the same way
# (tested against the API of OpenCode 1.17.11: GET /permission, POST /permission/ID/reply).
#
# usage: reprompt_opencode.sh OPENCODE SESSION_ID PAYLOAD
#
# Exits 0 once the turn finished, 3 when the server never became ready, 4 when the prompt request failed.
# OpenCode writes the conversation to its database as the CLI would. The server's log and the request's
# status are kept beside PAYLOAD, in the launch directory, which is removed with the session.

set +o errexit +o nounset +o pipefail
opencode=$1 session=$2 payload=$3
[[ -x $opencode && -n $session && -f $payload ]] || { echo "usage: reprompt_opencode.sh OPENCODE SESSION_ID PAYLOAD" >&2; exit 2; }

password=
for _ in 1 2 3 4; do password+=$(printf '%08x' "$((RANDOM * 32768 + RANDOM))"); done
export OPENCODE_SERVER_USERNAME=opencode OPENCODE_SERVER_PASSWORD=$password
# The launch directory, without a fork: the image may have few tools.
case $payload in */*) files=${payload%/*} ;; *) files=. ;; esac
log=$files/reprompt-server.log response=$files/reprompt-status
: >"$log" || exit 3

"$opencode" serve --hostname 127.0.0.1 --port 0 >"$log" 2>&1 &
server=$!
request= rejecter=
stop_server() {
    [[ -n $rejecter ]] && kill "$rejecter" 2>/dev/null
    [[ -n $request ]] && kill "$request" 2>/dev/null
    # TERM, not INT: bash starts a background command with INT ignored.
    kill -TERM "$server" 2>/dev/null
    for _ in $(seq 1 150); do kill -0 "$server" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$server" 2>/dev/null
    wait "$server" 2>/dev/null
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
    echo "OpenCode server did not become ready for the reprompt:" >&2
    cat "$log" >&2
    stop_server
    exit 3
fi

# OpenCode's CLI scopes sessions by the working directory, as PWD names it.
directory=$PWD
encode() {
    # Bytes, not characters: a character outside ASCII is encoded as its UTF-8 bytes.
    local LC_ALL=C
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
: >"$response" || { stop_server; exit 4; }

# One short request to the server; prints what it answers (with /dev/tcp, the headers too).
call() {
    local method=$1 target=$2 body=${3-}
    if command -v curl >/dev/null 2>&1; then
        curl --silent --max-time 10 --noproxy '*' -X "$method" -H "Authorization: $authorization" \
            -H 'Content-Type: application/json' ${body:+--data-binary "$body"} "http://127.0.0.1:$port$target"
    else
        (
            exec 6<>"/dev/tcp/127.0.0.1/$port" || exit 1
            printf '%s %s HTTP/1.0\r\nHost: 127.0.0.1\r\nAuthorization: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s' \
                "$method" "$target" "$authorization" "${#body}" "$body" >&6
            cat <&6
        )
    fi
}

# Reject every pending permission request once a second until killed. All of them, also a subagent session's:
# this server runs only this turn, and an unanswered request blocks it.
reject_permissions() {
    local query pending id
    query="?directory=$(encode "$directory")"
    while :; do
        pending=$(call GET "/permission$query" 2>/dev/null)
        while [[ $pending =~ \"id\":[[:space:]]*\"(per[A-Za-z0-9_]+)\" ]]; do
            id=${BASH_REMATCH[1]}
            pending=${pending#*"${BASH_REMATCH[0]}"}
            call POST "/permission/$id/reply$query" '{"reply":"reject"}' >/dev/null 2>&1
            echo "OpenCode asked for a permission ($id); rejected, as opencode run rejects it" >&2
        done
        sleep 1
    done
}
# The request runs in the background and is waited for: bash runs a trap only between foreground commands, and
# this request lasts the whole turn, so in the foreground it would hold the supervisor's TERM until the turn ended.
if command -v curl >/dev/null 2>&1; then
    # No proxy: the task image may export one, and the server listens on loopback.
    curl --silent --show-error --output /dev/null --write-out '%{http_code}' --max-time 0 --noproxy '*' \
        -H "Authorization: $authorization" -H 'Content-Type: application/json' \
        --data-binary "@$payload" "http://127.0.0.1:$port$path" >"$response" &
else
    # No curl in the image: one HTTP/1.0 request over bash's /dev/tcp, which closes when the turn is done.
    (
        exec 5<>"/dev/tcp/127.0.0.1/$port" || exit 1
        body=$(<"$payload")
        printf 'POST %s HTTP/1.0\r\nHost: 127.0.0.1\r\nAuthorization: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s' \
            "$path" "$authorization" "$(printf '%s' "$body" | wc -c)" "$body" >&5
        IFS= read -r line <&5
        [[ $line =~ ^HTTP/[0-9.]+\ ([0-9]+) ]] && printf '%s' "${BASH_REMATCH[1]}"
        cat <&5 >/dev/null
    ) >"$response" &
fi
request=$!
reject_permissions &
rejecter=$!
wait "$request"
request=
status=$(<"$response")
stop_server
if [[ $status != 200 ]]; then
    echo "OpenCode reprompt request failed with HTTP status ${status:-none}; the server's log ends:" >&2
    tail -n 50 "$log" >&2
    exit 4
fi
echo "OpenCode ran the existing user turn again without adding a prompt"
