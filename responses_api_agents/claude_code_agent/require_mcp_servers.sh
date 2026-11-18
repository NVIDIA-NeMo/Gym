#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run Claude Code and stop it once its init event reports that a required MCP server did not connect.
#
# usage: require_mcp_servers.sh EVENTS NAME... -- COMMAND...
#
# EVENTS is the file COMMAND's stdout goes to. In print mode Claude Code first tries every configured MCP server,
# waiting up to MCP_TIMEOUT, then writes its stream-json init event with each server's status, and only then sends
# its first model request. When the init event gives a NAMEd server any status other than "connected", COMMAND is
# stopped, so the turn ends at that first request instead of running without the server. The agent server reads the
# init event again and fails the turn; this script only cuts the turn short. It reads the init event as text, so a
# format it does not recognize stops nothing. Exits with COMMAND's status. Needs only bash and sleep.

set +o errexit +o nounset +o pipefail
events=$1
shift
required=()
while (($#)) && [[ $1 != -- ]]; do
    required+=("$1")
    shift
done
shift

"$@" &
cli=$!
trap 'kill -TERM "$cli" 2>/dev/null' TERM INT

pending=""
# Reads the lines EVENTS has gained; returns 0 once it has checked the init event.
check_init() {
    local line name pattern
    while IFS= read -r line <&3; do
        line=$pending$line
        pending=""
        [[ $line == '{"type":"system","subtype":"init",'* ]] || continue
        for name in "${required[@]}"; do
            pattern='\{"name":"'"${name//./[.]}"'","status":"([^"]*)"'
            if [[ $line =~ $pattern && ${BASH_REMATCH[1]} != connected ]]; then
                echo "Required MCP server $name is ${BASH_REMATCH[1]}; stopping Claude Code" >&2
                kill -TERM "$cli" 2>/dev/null
            fi
        done
        return 0
    done
    # A line still being written stays for the next pass.
    pending+=$line
    return 1
}

if exec 3<"$events"; then
    until check_init; do
        kill -0 "$cli" 2>/dev/null || break
        sleep 0.05
    done
    exec 3<&-
fi
# The TERM trap interrupts wait; wait again until the CLI has exited.
wait "$cli"
status=$?
while kill -0 "$cli" 2>/dev/null; do
    wait "$cli"
    status=$?
done
exit "$status"
