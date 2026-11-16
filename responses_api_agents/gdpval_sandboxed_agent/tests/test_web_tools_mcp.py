# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exa response handling and key rotation through Gym's HTTP transport."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientError

from responses_api_agents.gdpval_sandboxed_agent import web_tools_mcp as wt


class Content:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.requested = None

    async def read(self, n=-1):
        self.requested = n
        return self.raw if n < 0 else self.raw[:n]


class Response:
    def __init__(self, status=200, *, data=None, text="", raw=None, charset=None):
        self.status, self.data, self.body = status, data, text
        self.content = Content(raw if raw is not None else text.encode())
        self.charset = charset
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    async def text(self):
        return self.body

    def raise_for_status(self):
        if self.status >= 400:
            raise ClientError(f"HTTP {self.status}")


async def test_search_formats_results_and_releases_response(monkeypatch):
    response = Response(data={"results": [{"title": "A & B", "url": "https://x/y", "text": "snippet <1>"}]})
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(wt, "request", request)
    tools = wt.WebTools(["k1"])
    out = await tools.search("gdp of france")
    assert out.startswith("<results>") and "<title>A &amp; B</title>" in out and "snippet &lt;1&gt;" in out
    assert request.call_args.kwargs["json"]["query"] == "gdp of france"
    assert request.call_args.kwargs["json"]["numResults"] == 5
    assert request.call_args.kwargs["headers"]["x-api-key"] == "k1"
    assert response.closed and tools.searches == 1


async def test_keys_rotate_on_quota_and_exhaust_without_exposing_keys(monkeypatch):
    responses = [Response(429), Response(429)]
    request = AsyncMock(side_effect=responses)
    monkeypatch.setattr(wt, "request", request)
    tools = wt.WebTools(["k1", "k2"])
    out = await tools.search("q")
    assert [c.kwargs["headers"]["x-api-key"] for c in request.call_args_list] == ["k1", "k2"]
    assert tools.key_rotations == 2 and all(r.closed for r in responses)
    assert out.startswith("<error>Web search exhausted 2 attempt(s)")
    assert "k1" not in out and "k2" not in out


async def test_fetch_extracts_text_and_reports_http_errors(monkeypatch):
    good = Response(text="<html><body><h1>Title</h1><p>Hello <b>world</b></p><script>x()</script></body></html>")
    bad = Response(404)
    monkeypatch.setattr(wt, "request", AsyncMock(side_effect=[good, bad]))
    tools = wt.WebTools(["k"])
    ok = await tools.fetch("https://good/page")
    assert ok.startswith("<web_fetch><url>https://good/page</url><body>") and "Hello" in ok and "x()" not in ok
    assert "404" in await tools.fetch("https://bad/page")
    assert good.closed and bad.closed


async def test_fetch_is_bounded_and_decodes_invalid_bytes(monkeypatch):
    page = Response(raw=b"<p>caf\xe9 ok</p>" + b"x" * 100, charset="no-such-charset")
    monkeypatch.setattr(wt, "request", AsyncMock(return_value=page))
    monkeypatch.setattr(wt, "MAX_FETCH_BYTES", 40)
    out = await wt.WebTools(["k"]).fetch("https://latin1/page")
    assert "<body>" in out and "caf\ufffd ok" in out
    assert page.content.requested == 40


@pytest.mark.parametrize("response", [Response(data=ValueError("bad json")), Response(data=None)])
async def test_invalid_search_payload_is_tool_error(monkeypatch, response):
    monkeypatch.setattr(wt, "request", AsyncMock(return_value=response))
    out = await wt.WebTools(["k"]).search("q")
    assert out.startswith("<error>")
    assert response.closed


def test_parse_keys_and_truncation():
    assert wt.parse_keys(" a, b ,a,[c]") == ["a", "b", "c"]
    long = wt.truncate_msg("x" * (wt.MAX_LENGTH + 10))
    assert len(long) < wt.MAX_LENGTH + 100 and "truncated 10 characters" in long
