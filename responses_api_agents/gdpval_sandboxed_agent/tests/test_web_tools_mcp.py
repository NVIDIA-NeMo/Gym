# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Exa web tools behind the OpenHands MCP server: formatting, key rotation, fetch fallback."""

import asyncio
import json

import httpx

from responses_api_agents.gdpval_sandboxed_agent import web_tools_mcp as wt


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_search_formats_results_like_the_eval_tool():
    seen = []

    def handler(request):
        seen.append(dict(request.headers))
        body = json.loads(request.content)
        assert body["query"] == "gdp of france" and body["numResults"] == 5
        return httpx.Response(200, json={"results": [{"title": "A & B", "url": "https://x/y", "text": "snippet <1>"}]})

    tools = wt.WebTools(["k1"], client=_client(handler))
    out = asyncio.run(tools.search("gdp of france"))
    assert out.startswith("<results>") and "<title>A &amp; B</title>" in out and "snippet &lt;1&gt;" in out
    assert seen[0]["x-api-key"] == "k1" and tools.searches == 1


def test_keys_rotate_on_auth_or_quota_and_exhaust_cleanly():
    calls = []

    def handler(request):
        calls.append(request.headers["x-api-key"])
        return httpx.Response(429, json={})

    tools = wt.WebTools(["k1", "k2"], client=_client(handler), max_sweeps=1)
    out = asyncio.run(tools.search("q"))
    assert calls == ["k1", "k2"] and tools.key_rotations == 2
    assert out.startswith("<error>Web search exhausted 2 attempt(s)")
    assert "k1" not in out and "k2" not in out  # keys never leak into tool output


def test_fetch_extracts_text_and_reports_http_errors():
    def handler(request):
        if request.url.host == "bad":
            return httpx.Response(404, text="nope")
        return httpx.Response(200, text="<html><body><h1>Title</h1><p>Hello <b>world</b></p><script>x()</script></body></html>")

    tools = wt.WebTools(["k"], client=_client(handler))
    ok = asyncio.run(tools.fetch("https://good/page"))
    assert ok.startswith("<web_fetch><url>https://good/page</url><body>") and "Hello" in ok and "x()" not in ok
    bad = asyncio.run(tools.fetch("https://bad/page"))
    assert "<error>" in bad and "404" in bad


def test_parse_keys_and_truncation():
    assert wt.parse_keys(" a, b ,a,[c]") == ["a", "b", "c"]
    long = wt.truncate_msg("x" * (wt.MAX_LENGTH + 10))
    assert len(long) < wt.MAX_LENGTH + 100 and "truncated 10 characters" in long
