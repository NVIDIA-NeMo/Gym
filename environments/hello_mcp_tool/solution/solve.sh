#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The reference solution: `gym dataset validate` plays it with the oracle and expects reward 1.
# It does what the agent is asked to: call get_greeting over MCP, starting the server the way task.toml declares it,
# and save the greeting it returns. The image's python has the `mcp` package (environment/Dockerfile).
set -eu
python - <<'PY'
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server = StdioServerParameters(command="python", args=["/opt/hello-tools/server.py"])
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("get_greeting", {"name": "NeMo Gym"})
    if result.isError:
        raise SystemExit(f"get_greeting failed: {result.content}")
    Path("/workspace/tool-greeting.txt").write_text(result.content[0].text + "\n")


asyncio.run(main())
PY
