# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific greeting tool served over MCP.

Each call is logged, so tests/test.sh can tell a greeting the tool returned from one the agent wrote by itself.
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP


# Harnesses start this server inside the task sandbox, where tests/test.sh runs afterwards. /logs/agent is the
# sandbox folder for the agent's logs: writable by every user, and kept for the verifier.
CALLS_LOG = Path("/logs/agent/hello-tools-calls.jsonl")

mcp = FastMCP("hello-tools")


@mcp.tool()
def get_greeting(name: str) -> str:
    """Return a greeting for the supplied name."""

    CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CALLS_LOG.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"name": name}) + "\n")
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()
