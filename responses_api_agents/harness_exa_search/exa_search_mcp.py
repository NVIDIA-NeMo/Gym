# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal stdio MCP server wrapping Exa's raw Search API with an explicit,
equal-to-Parallel/Brave result/character budget, instead of the stock
exa-mcp-server (which lets the model silently request more per Khushi's finding).
"""
import os

import aiohttp
from mcp.server.fastmcp import FastMCP

MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "20"))
MAX_CHARS_PER_RESULT = int(os.environ.get("SEARCH_MAX_CHARS_PER_RESULT", "2000"))
MAX_CHARS_TOTAL = int(os.environ.get("SEARCH_MAX_CHARS_TOTAL", "8000"))

mcp = FastMCP("exa")


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web and return ranked results with excerpts."""
    api_key = os.environ["EXA_API_KEY"]
    payload = {
        "query": query,
        "numResults": MAX_RESULTS,
        "contents": {"text": {"maxCharacters": MAX_CHARS_PER_RESULT}},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            response.raise_for_status()
            data = await response.json()
    parts = []
    total = 0
    for result in data.get("results", []):
        text = result.get("text") or ""
        chunk = f"Title: {result.get('title', 'N/A')}\nURL: {result.get('url', 'N/A')}\n{text}"
        if total + len(chunk) > MAX_CHARS_TOTAL:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(parts) if parts else "No results found."


if __name__ == "__main__":
    mcp.run(transport="stdio")
