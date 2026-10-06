# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal stdio MCP server wrapping Parallel's raw Search API with an explicit,
equal-to-Exa result/character budget, bypassing the hosted search.parallel.ai/mcp
endpoint's hardcoded ~25k char cap.
"""
import os

import aiohttp
from mcp.server.fastmcp import FastMCP

MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "20"))
MAX_CHARS_PER_RESULT = int(os.environ.get("SEARCH_MAX_CHARS_PER_RESULT", "2000"))
MAX_CHARS_TOTAL = int(os.environ.get("SEARCH_MAX_CHARS_TOTAL", "8000"))

mcp = FastMCP("parallel")


@mcp.tool()
async def web_search(query: str, objective: str = "") -> str:
    """Search the web and return ranked results with excerpts."""
    api_key = os.environ["PARALLEL_API_KEY"]
    payload = {
        "search_queries": [query],
        "objective": objective or query,
        "max_chars_total": MAX_CHARS_TOTAL,
        "advanced_settings": {
            "max_results": MAX_RESULTS,
            "excerpt_settings": {"max_chars_per_result": MAX_CHARS_PER_RESULT},
        },
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.parallel.ai/v1/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            response.raise_for_status()
            data = await response.json()
    parts = []
    for result in data.get("results", []):
        excerpts = "\n".join(result.get("excerpts") or [])
        parts.append(f"Title: {result.get('title', 'N/A')}\nURL: {result.get('url', 'N/A')}\n{excerpts}")
    return "\n\n---\n\n".join(parts) if parts else "No results found."


if __name__ == "__main__":
    mcp.run(transport="stdio")
