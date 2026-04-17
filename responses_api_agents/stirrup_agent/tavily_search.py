# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tavily web search tool for Stirrup agents.

Provides a ``ToolProvider`` that returns ``web_search`` and ``web_fetch``
tools backed by the Tavily Search API (https://tavily.com).  Drop-in
replacement for Stirrup's built-in ``WebToolProvider`` (Brave).

Set ``TAVILY_API_KEY`` in the environment to enable.
"""
from __future__ import annotations

import os
from html import escape
from types import TracebackType
from typing import Annotated, Any

import httpx
from pydantic import BaseModel, Field

from stirrup.core.models import Tool, ToolProvider, ToolResult, ToolUseCountMetadata
from stirrup.utils.text import truncate_msg

MAX_LENGTH = 40_000
TIMEOUT = 60 * 3


# ---------------------------------------------------------------------------
# web_search (Tavily Search API)
# ---------------------------------------------------------------------------


class _SearchParams(BaseModel):
    query: Annotated[str, Field(description="Natural language search query.")]


async def _search_executor(
    params: _SearchParams,
    *,
    api_key: str,
    client: httpx.AsyncClient,
) -> ToolResult[ToolUseCountMetadata]:
    try:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"query": params.query, "max_results": 5, "include_answer": True},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        parts: list[str] = []
        if data.get("answer"):
            parts.append(f"<answer>{escape(data['answer'])}</answer>")

        results = data.get("results", [])
        results_xml = "\n".join(
            f"<result>\n<title>{escape(r.get('title', ''))}</title>"
            f"\n<url>{escape(r.get('url', ''))}</url>"
            f"\n<content>{escape(r.get('content', ''))}</content>\n</result>"
            for r in results
        )
        parts.append(f"<results>\n{results_xml}\n</results>")

        return ToolResult(
            content=truncate_msg("\n".join(parts), MAX_LENGTH),
            metadata=ToolUseCountMetadata(),
        )
    except httpx.HTTPError as exc:
        return ToolResult(
            content=f"<error>{escape(str(exc))}</error>",
            success=False,
            metadata=ToolUseCountMetadata(),
        )


# ---------------------------------------------------------------------------
# web_fetch (plain HTTP GET + trafilatura extraction, same as Stirrup's)
# ---------------------------------------------------------------------------


class _FetchParams(BaseModel):
    url: Annotated[str, Field(description="Full HTTP or HTTPS URL of the web page to fetch.")]


async def _fetch_executor(
    params: _FetchParams,
    *,
    client: httpx.AsyncClient,
) -> ToolResult[ToolUseCountMetadata]:
    try:
        resp = await client.get(
            params.url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        resp.raise_for_status()

        import trafilatura

        body_md = trafilatura.extract(resp.text, output_format="markdown") or ""
        return ToolResult(
            content=f"<web_fetch><url>{params.url}</url><body>"
            f"{truncate_msg(body_md, MAX_LENGTH)}</body></web_fetch>",
            metadata=ToolUseCountMetadata(),
        )
    except httpx.HTTPError as exc:
        return ToolResult(
            content=f"<web_fetch><url>{params.url}</url><error>{escape(str(exc))}</error></web_fetch>",
            success=False,
            metadata=ToolUseCountMetadata(),
        )


# ---------------------------------------------------------------------------
# TavilyToolProvider
# ---------------------------------------------------------------------------


class TavilyToolProvider(ToolProvider):
    """Provides ``web_search`` and ``fetch_web_page`` tools via Tavily API.

    Usage::

        tools = [TavilyToolProvider(), LocalCodeExecToolProvider()]
        agent = Agent(client=client, name="agent", tools=tools)
    """

    def __init__(self, *, api_key: str | None = None, timeout: float = TIMEOUT) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> list[Tool[Any, Any]]:
        self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        await self._client.__aenter__()
        return self._get_tools()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def _get_tools(self) -> list[Tool[Any, Any]]:
        assert self._client is not None
        api_key = self._api_key
        client = self._client

        async def search_exec(p: _SearchParams) -> ToolResult[ToolUseCountMetadata]:
            return await _search_executor(p, api_key=api_key, client=client)

        async def fetch_exec(p: _FetchParams) -> ToolResult[ToolUseCountMetadata]:
            return await _fetch_executor(p, client=client)

        search_tool = Tool[_SearchParams, ToolUseCountMetadata](
            name="web_search",
            description="Search the web using Tavily. Returns top results with content snippets.",
            parameters=_SearchParams,
            executor=search_exec,
        )

        fetch_tool = Tool[_FetchParams, ToolUseCountMetadata](
            name="fetch_web_page",
            description="Fetch and extract the main content from a web page as markdown.",
            parameters=_FetchParams,
            executor=fetch_exec,
        )

        return [search_tool, fetch_tool]
