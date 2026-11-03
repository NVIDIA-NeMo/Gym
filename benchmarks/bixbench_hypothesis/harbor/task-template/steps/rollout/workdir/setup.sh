#!/usr/bin/env bash

set -euo pipefail

BBH_MCP_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# shellcheck disable=SC2155
export BBH_MCP_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

nohup \
python -m bbh_mcp.server \
    -f task.jsonl \
    -d data/ \
    --id 0 \
    -b local \
    -p "${BBH_MCP_PORT}" \
    -w "$(pwd)" \
    --log-dir .harbor/artifacts \
    >/logs/agent/bbh-mcp.log 2>&1 </dev/null & mcp_pid=$!

mcp_ready=0
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
    if health_token=$(
        curl -fsS "http://127.0.0.1:${BBH_MCP_PORT}/health" \
            | python3 -c 'import json, sys; print(json.load(sys.stdin).get("id_token") or "")'
    ) && [[ "${health_token}" == "${BBH_MCP_TOKEN}" ]]; then
        mcp_ready=1
        break
    fi
    sleep 3
done

if [[ ${mcp_ready} -ne 1 ]]; then
    printf "MCP server ready timeout (or token mismatch).\n" >&2
    exit 1
fi

sed -i "s/{env:BBH_MCP_PORT}/${BBH_MCP_PORT}/g" opencode.json

rm task.jsonl
kill -0 "${mcp_pid}"
