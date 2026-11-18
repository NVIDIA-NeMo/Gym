# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific greeting tool served over MCP."""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("hello-tools")


@mcp.tool()
def get_greeting(name: str) -> str:
    """Return a greeting for the supplied name."""

    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()
