# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``web_search`` / ``fetch_web_page`` tools backed by the Exa API.

Same tool names, parameter schemas and result XML as
``responses_api_agents/stirrup_agent/tavily_search.TavilyToolProvider`` (what
the GDPVal eval harness gives the model), with Exa (https://api.exa.ai) as the
search backend: the keys available for training are Exa keys (UUID format),
not Tavily keys. Only the tool description's backend name differs.

Keys come from a list, a comma-separated string, or ``api_keys_file`` (one key
per line, blanks and ``#`` comments ignored). Several keys rotate round-robin
and an auth/quota/5xx reply moves to the next key; the raw keys are never
logged or included in tool output.
"""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, List, Optional, Union

import httpx
from pydantic import BaseModel, Field
from stirrup.core.models import Tool, ToolProvider, ToolResult, ToolUseCountMetadata
from stirrup.utils.text import truncate_msg

logger = logging.getLogger(__name__)

EXA_SEARCH_URL = "https://api.exa.ai/search"
MAX_LENGTH = 20_000  # same cap as the Tavily tools
TIMEOUT = 60.0
NUM_RESULTS = 5
SNIPPET_CHARS = 2000
KEY_ROTATION_RETRY_STATUSES = {401, 402, 403, 429}


def _should_rotate_on_status(status: int) -> bool:
    return status in KEY_ROTATION_RETRY_STATUSES or 500 <= status < 600


def parse_keys(value: Union[str, List[str], None]) -> List[str]:
    """Normalise a list / ``"k1,k2"`` / ``"[k1,k2]"`` into a deduplicated key list."""
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
        ks = str(k).strip()
        if ks and ks not in seen:
            seen.add(ks)
            out.append(ks)
    return out


def load_keys_file(path: Union[str, Path]) -> List[str]:
    """One key per line; blank lines and ``#`` comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return parse_keys([line.split("#", 1)[0].strip() for line in lines])


class _SearchParams(BaseModel):
    query: Annotated[str, Field(description="Natural language search query.")]


class _FetchParams(BaseModel):
    url: Annotated[str, Field(description="Full HTTP or HTTPS URL of the web page to fetch.")]


def format_search_results(data: dict[str, Any]) -> str:
    results = data.get("results") or []
    results_xml = "\n".join(
        f"<result>\n<title>{escape(str(r.get('title') or ''))}</title>"
        f"\n<url>{escape(str(r.get('url') or ''))}</url>"
        f"\n<content>{escape(str(r.get('text') or r.get('highlights') or r.get('summary') or ''))}</content>\n</result>"
        for r in results
    )
    return truncate_msg(f"<results>\n{results_xml}\n</results>", MAX_LENGTH)


class ExaToolProvider(ToolProvider):
    """Provides ``web_search`` (Exa) and ``fetch_web_page`` (direct fetch + trafilatura)."""

    def __init__(
        self,
        *,
        api_keys: Optional[Union[str, List[str]]] = None,
        api_keys_file: Optional[Union[str, Path]] = None,
        timeout: float = TIMEOUT,
        max_sweeps: int = 1,
        num_results: int = NUM_RESULTS,
        snippet_chars: int = SNIPPET_CHARS,
    ) -> None:
        if max_sweeps < 1:
            raise ValueError(f"max_sweeps must be >= 1, got {max_sweeps}")
        keys = parse_keys(api_keys)
        if api_keys_file:
            keys = parse_keys(keys + load_keys_file(api_keys_file))
        if not keys:
            raise ValueError("ExaToolProvider needs at least one API key (api_keys or api_keys_file)")
        self._api_keys = keys
        self._key_idx = 0
        self._timeout = timeout
        self._max_sweeps = max_sweeps
        self._num_results = num_results
        self._snippet_chars = snippet_chars
        self._client: httpx.AsyncClient | None = None
        self.searches = 0
        self.fetches = 0
        self.key_rotations = 0

    @property
    def num_keys(self) -> int:
        return len(self._api_keys)

    def _next_key(self) -> str:
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

    # ------------------------------------------------------------------

    async def search(self, query: str) -> ToolResult[ToolUseCountMetadata]:
        assert self._client is not None
        self.searches += 1
        n_attempts = self._max_sweeps * len(self._api_keys)
        last_status: Optional[int] = None
        for _ in range(n_attempts):
            api_key = self._next_key()
            try:
                resp = await self._client.post(
                    EXA_SEARCH_URL,
                    json={
                        "query": query,
                        "numResults": self._num_results,
                        "type": "auto",
                        "contents": {"text": {"maxCharacters": self._snippet_chars}},
                    },
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                return ToolResult(content=f"<error>{escape(str(exc))}</error>", success=False, metadata=ToolUseCountMetadata())
            if _should_rotate_on_status(resp.status_code):
                last_status = resp.status_code
                self.key_rotations += 1
                continue
            try:
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                return ToolResult(content=f"<error>{escape(str(exc))}</error>", success=False, metadata=ToolUseCountMetadata())
            try:
                data = resp.json()
            except ValueError:
                return ToolResult(content="<error>Exa returned a non-JSON response</error>", success=False, metadata=ToolUseCountMetadata())
            return ToolResult(content=format_search_results(data), metadata=ToolUseCountMetadata())
        return ToolResult(
            content=(
                f"<error>Web search exhausted {n_attempts} attempt(s) ({len(self._api_keys)} key(s) x "
                f"{self._max_sweeps} sweep(s)) on retryable errors (last status={last_status}).</error>"
            ),
            success=False,
            metadata=ToolUseCountMetadata(),
        )

    async def fetch(self, url: str) -> ToolResult[ToolUseCountMetadata]:
        assert self._client is not None
        self.fetches += 1
        try:
            resp = await self._client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            resp.raise_for_status()
            import trafilatura

            body_md = trafilatura.extract(resp.text, output_format="markdown") or ""
            return ToolResult(
                content=f"<web_fetch><url>{escape(url)}</url><body>{truncate_msg(body_md, MAX_LENGTH)}</body></web_fetch>",
                metadata=ToolUseCountMetadata(),
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                content=f"<web_fetch><url>{escape(url)}</url><error>{escape(str(exc))}</error></web_fetch>",
                success=False,
                metadata=ToolUseCountMetadata(),
            )

    def _get_tools(self) -> list[Tool[Any, Any]]:
        provider = self

        async def search_exec(p: _SearchParams) -> ToolResult[ToolUseCountMetadata]:
            return await provider.search(p.query)

        async def fetch_exec(p: _FetchParams) -> ToolResult[ToolUseCountMetadata]:
            return await provider.fetch(p.url)

        return [
            Tool[_SearchParams, ToolUseCountMetadata](
                name="web_search",
                description="Search the web using Exa. Returns top results with content snippets.",
                parameters=_SearchParams,
                executor=search_exec,
            ),
            Tool[_FetchParams, ToolUseCountMetadata](
                name="fetch_web_page",
                description="Fetch and extract the main content from a web page as markdown.",
                parameters=_FetchParams,
                executor=fetch_exec,
            ),
        ]
