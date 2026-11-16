# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``web_search`` / ``fetch_web_page`` for the OpenHands harness, as a stdio MCP server.

Runs inside the GDPVal sandbox (uploaded to ``/runner`` next to ``run_openhands.py``)
and is registered with OpenHands as an MCP stdio server, so the model sees the same two
tools, with the same parameter names and result XML, that the GDPVal eval harness and
the Stirrup training agent provide (``gdpval_stirrup_agent/exa_search.py``). Backend:
Exa (https://api.exa.ai). Keys come from the ``GDPVAL_WEB_KEYS`` environment variable
(comma-separated), rotate round-robin, and are never written to tool output.

Importable without ``fastmcp`` (the tool functions are plain coroutines) so the formatting
and key rotation are unit-testable outside the sandbox; ``main()`` needs fastmcp, which the
OpenHands venv ships.
"""

from __future__ import annotations

import asyncio
import os
import re
from html import escape, unescape
from typing import Any, Optional

from aiohttp import ClientError, ClientTimeout

from nemo_gym.client import request


EXA_SEARCH_URL = "https://api.exa.ai/search"
MAX_LENGTH = 20_000
TIMEOUT = 60.0
MAX_FETCH_BYTES = 10 * 1024 * 1024
NUM_RESULTS = 5
SNIPPET_CHARS = 2000
KEY_ROTATION_RETRY_STATUSES = {401, 402, 403, 429}


def truncate_msg(text: str, max_length: int = MAX_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... [truncated {len(text) - max_length} characters]"


def parse_keys(value: Optional[str]) -> list[str]:
    out: list[str] = []
    for key in (value or "").replace("[", "").replace("]", "").split(","):
        key = key.strip()
        if key and key not in out:
            out.append(key)
    return out


def format_search_results(data: dict[str, Any]) -> str:
    results = data.get("results") or []
    results_xml = "\n".join(
        f"<result>\n<title>{escape(str(r.get('title') or ''))}</title>"
        f"\n<url>{escape(str(r.get('url') or ''))}</url>"
        f"\n<content>{escape(str(r.get('text') or r.get('highlights') or r.get('summary') or ''))}</content>\n</result>"
        for r in results
    )
    return truncate_msg(f"<results>\n{results_xml}\n</results>")


def html_to_text(html: str) -> str:
    """Markdown-ish main text when trafilatura is unavailable in the sandbox venv."""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(html, output_format="markdown")
        if extracted:
            return extracted
    except Exception:
        pass
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h[1-6]>|</li>|</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


class WebTools:
    def __init__(self, keys: list[str], *, max_sweeps: int = 1) -> None:
        if not keys:
            raise ValueError("web tools need at least one API key (GDPVAL_WEB_KEYS)")
        self._keys = keys
        self._idx = 0
        self._max_sweeps = max(1, max_sweeps)
        self.searches = 0
        self.fetches = 0
        self.key_rotations = 0

    def _next_key(self) -> str:
        key = self._keys[self._idx % len(self._keys)]
        self._idx += 1
        return key

    async def search(self, query: str) -> str:
        self.searches += 1
        attempts = self._max_sweeps * len(self._keys)
        last_status = None
        for _ in range(attempts):
            try:
                resp = await request(
                    "POST",
                    EXA_SEARCH_URL,
                    json={
                        "query": query,
                        "numResults": NUM_RESULTS,
                        "type": "auto",
                        "contents": {"text": {"maxCharacters": SNIPPET_CHARS}},
                    },
                    headers={"x-api-key": self._next_key(), "Content-Type": "application/json"},
                    timeout=ClientTimeout(total=TIMEOUT),
                    _max_num_tries=1,
                )
            except (ClientError, asyncio.TimeoutError) as exc:
                return f"<error>{escape(str(exc))}</error>"
            async with resp:
                if resp.status in KEY_ROTATION_RETRY_STATUSES or 500 <= resp.status < 600:
                    last_status = resp.status
                    self.key_rotations += 1
                    continue
                if resp.status >= 400:
                    return f"<error>HTTP {resp.status} from the search backend</error>"
                try:
                    data = await resp.json()
                except (ValueError, ClientError, asyncio.TimeoutError):
                    return "<error>Exa returned a non-JSON response</error>"
                if (
                    not isinstance(data, dict)
                    or not isinstance(data.get("results", []), list)
                    or any(not isinstance(item, dict) for item in data.get("results", []))
                ):
                    return "<error>Exa returned an invalid search response</error>"
                return format_search_results(data)
        return (
            f"<error>Web search exhausted {attempts} attempt(s) ({len(self._keys)} key(s) x "
            f"{self._max_sweeps} sweep(s)) on retryable errors (last status={last_status}).</error>"
        )

    async def fetch(self, url: str) -> str:
        self.fetches += 1
        try:
            resp = await request(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=ClientTimeout(total=TIMEOUT),
                _max_num_tries=1,
            )
            async with resp:
                resp.raise_for_status()
                # Bounded and leniently decoded: a huge page or a non-UTF-8 body must not end the tool call.
                raw = await resp.content.read(MAX_FETCH_BYTES)
                try:
                    text = raw.decode(resp.charset or "utf-8", errors="replace")
                except LookupError:  # an unknown charset label
                    text = raw.decode("utf-8", errors="replace")
                body = html_to_text(text)
        except (ClientError, asyncio.TimeoutError) as exc:
            return f"<web_fetch><url>{escape(url)}</url><error>{escape(str(exc))}</error></web_fetch>"
        return f"<web_fetch><url>{escape(url)}</url><body>{truncate_msg(body)}</body></web_fetch>"


def main() -> None:
    from fastmcp import FastMCP  # OpenHands' own dependency

    tools = WebTools(
        parse_keys(os.environ.get("GDPVAL_WEB_KEYS")), max_sweeps=int(os.environ.get("GDPVAL_WEB_MAX_SWEEPS", "1"))
    )
    server = FastMCP("gdpval-web")

    @server.tool(name="web_search", description="Search the web using Exa. Returns top results with content snippets.")
    async def web_search(query: str) -> str:
        """query: Natural language search query."""
        return await tools.search(query)

    @server.tool(name="fetch_web_page", description="Fetch and extract the main content from a web page as markdown.")
    async def fetch_web_page(url: str) -> str:
        """url: Full HTTP or HTTPS URL of the web page to fetch."""
        return await tools.fetch(url)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
