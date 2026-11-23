# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference solution: send the row's ground-truth calls, in order, to the task's MCP tool server.

Standard library only, so it runs in the agent's image without an MCP client. Each call is one JSON-RPC
``tools/call`` over streamable HTTP; the server is stateless, so no session is opened first.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path


def rpc(url: str, request_id: int, method: str, params: dict) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def wait_ready(url: str, deadline_s: float = 300.0) -> None:
    start = time.monotonic()
    while True:
        try:
            rpc(url, 0, "tools/list", {})
            return
        except Exception as error:
            if time.monotonic() - start > deadline_s:
                raise SystemExit(f"tool server {url} not ready: {error!r}")
            time.sleep(2)


def main(url: str, calls_path: str) -> None:
    calls = json.loads(Path(calls_path).read_text())
    wait_ready(url)
    for index, call in enumerate(calls, start=1):
        reply = rpc(url, index, "tools/call", {"name": call["name"], "arguments": json.loads(call["arguments"])})
        print(call["name"], json.dumps(reply.get("result", reply))[:300])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
