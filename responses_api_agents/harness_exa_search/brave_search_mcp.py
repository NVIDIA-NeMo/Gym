# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal stdio MCP server wrapping Brave's LLM Context endpoint with an
explicit, equal-to-Exa/Parallel result/character budget.
"""
import os

import aiohttp
from mcp.server.fastmcp import FastMCP

MAX_URLS = int(os.environ.get("SEARCH_MAX_RESULTS", "20"))
MAX_TOKENS = int(os.environ.get("BRAVE_MAX_TOKENS", "32768"))  # API ceiling

mcp = FastMCP("brave")


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web and return ranked results with excerpts."""
    api_key = os.environ["BRAVE_API_KEY"]
    params = {
        "q": query,
        "maximum_number_of_urls": MAX_URLS,
        "maximum_number_of_tokens": MAX_TOKENS,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.search.brave.com/res/v1/llm/context",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params=params,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            response.raise_for_status()
            data = await response.json()
    parts = []
    for item in (data.get("grounding") or {}).get("generic") or []:
        snippets = "\n".join(item.get("snippets") or [])
        parts.append(f"Title: {item.get('title', 'N/A')}\nURL: {item.get('url', 'N/A')}\n{snippets}")
    return "\n\n---\n\n".join(parts) if parts else "No results found."


if __name__ == "__main__":
    mcp.run(transport="stdio")
