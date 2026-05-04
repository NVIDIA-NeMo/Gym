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

Configure via the agent's ``tavily_api_key`` config field (preferred) or
``TAVILY_API_KEY`` env var (fallback). Both accept either a single key
or a comma-separated list (with or without surrounding ``[...]`` brackets,
to match the format EFB injects). When multiple keys are provided, the
provider rotates round-robin per call AND retries on key-specific
failures (401, 403, 429) with the next key, up to ``len(keys)`` total
attempts per call.
"""

from __future__ import annotations

import os
from html import escape
from types import TracebackType
from typing import Annotated, Any, List, Optional, Union

import httpx
from pydantic import BaseModel, Field
from stirrup.core.models import Tool, ToolProvider, ToolResult, ToolUseCountMetadata
from stirrup.utils.text import truncate_msg


MAX_LENGTH = 40_000
TIMEOUT = 60 * 3

# HTTP statuses that indicate the *current key* is the problem (auth or
# quota). On these, we rotate to the next key and retry.
KEY_ROTATION_RETRY_STATUSES = frozenset({401, 403, 429})


def _should_rotate_on_status(status: int) -> bool:
    """Whether to rotate to the next key after seeing this HTTP status.

    Rotates on key-specific 4xx (auth/quota) AND any 5xx. The 5xx rationale
    is conservative: even though all keys hit the same upstream
    (api.tavily.com), Tavily fronts multiple backends and occasional 5xx
    are observed for individual requests; a free retry with the next key
    costs nothing and often succeeds. Non-rotating failures (404, 4xx that
    aren't auth/quota, malformed-query) bail after one attempt — rotating
    wouldn't help and would mask the real issue.
    """
    return status in KEY_ROTATION_RETRY_STATUSES or 500 <= status < 600


# ---------------------------------------------------------------------------
# web_search (Tavily Search API)
# ---------------------------------------------------------------------------


class _SearchParams(BaseModel):
    query: Annotated[str, Field(description="Natural language search query.")]


async def _search_executor(
    params: _SearchParams,
    *,
    provider: TavilyToolProvider,
    client: httpx.AsyncClient,
) -> ToolResult[ToolUseCountMetadata]:
    last_status: Optional[int] = None
    n_keys = max(1, len(provider._api_keys))
    n_attempts = provider._max_sweeps * n_keys
    for _ in range(n_attempts):
        api_key = provider._next_key()
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"query": params.query, "max_results": 5, "include_answer": True},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            # Network-level failure — not key-specific. Bail with this error.
            return ToolResult(
                content=f"<error>{escape(str(exc))}</error>",
                success=False,
                metadata=ToolUseCountMetadata(),
            )

        if _should_rotate_on_status(resp.status_code):
            last_status = resp.status_code
            continue  # rotate to next key

        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolResult(
                content=f"<error>{escape(str(exc))}</error>",
                success=False,
                metadata=ToolUseCountMetadata(),
            )

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

    # All attempts exhausted on retryable errors (auth, quota, or 5xx).
    return ToolResult(
        content=(
            f"<error>Tavily exhausted {n_attempts} attempt(s) "
            f"({n_keys} key(s) × {provider._max_sweeps} sweep(s)) on retryable errors "
            f"(last status={last_status}). Refresh keys or check upstream.</error>"
        ),
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
            content=f"<web_fetch><url>{params.url}</url><body>{truncate_msg(body_md, MAX_LENGTH)}</body></web_fetch>",
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

    Accepts a single key, a list of keys, or a comma-separated string
    (with or without surrounding ``[...]`` brackets — the format EFB
    injects via ``host:TAVILY_API_KEY``). When multiple keys are present,
    rotates round-robin and retries auth/quota failures with the next key.

    Usage::

        tools = [TavilyToolProvider(api_keys=["k1", "k2"]), LocalCodeExecToolProvider()]
        agent = Agent(client=client, name="agent", tools=tools)
    """

    def __init__(
        self,
        *,
        api_keys: Optional[Union[str, List[str]]] = None,
        timeout: float = TIMEOUT,
        max_sweeps: int = 1,
    ) -> None:
        if api_keys is None:
            api_keys = os.getenv("TAVILY_API_KEY", "")
        if max_sweeps < 1:
            raise ValueError(f"max_sweeps must be >= 1, got {max_sweeps}")
        self._api_keys: List[str] = self._parse_keys(api_keys)
        self._key_idx: int = 0
        self._timeout = timeout
        # Total attempts per tool call = ``max_sweeps × len(api_keys)``. With
        # the default 1, a single sweep through the keys exits gracefully on
        # full exhaustion — the model is the natural backoff for the next
        # tool call. Bump to 2-3 if the upstream is flaky in short bursts and
        # an extra wallclock minute or two is acceptable.
        self._max_sweeps = max_sweeps
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _parse_keys(value: Union[str, List[str]]) -> List[str]:
        """Normalise ``value`` into a deduplicated, non-empty list of keys.

        Accepts:
        - ``List[str]`` directly,
        - ``"k1,k2,k3"`` (comma-separated string),
        - ``"[k1,k2,k3]"`` (EFB env-var format with surrounding brackets),
        - ``""`` or ``None`` → empty list.
        """
        if isinstance(value, list):
            keys = value
        else:
            s = (value or "").strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            keys = s.split(",") if s else []
        out: List[str] = []
        seen: set[str] = set()
        for k in keys:
            ks = k.strip()
            if ks and ks not in seen:
                seen.add(ks)
                out.append(ks)
        return out

    def _next_key(self) -> str:
        """Return the next key in round-robin order. Empty list → ``""``."""
        if not self._api_keys:
            return ""
        k = self._api_keys[self._key_idx % len(self._api_keys)]
        self._key_idx += 1
        return k

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
        provider = self
        client = self._client

        async def search_exec(p: _SearchParams) -> ToolResult[ToolUseCountMetadata]:
            return await _search_executor(p, provider=provider, client=client)

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
