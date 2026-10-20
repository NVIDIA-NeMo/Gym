# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exa-backed web_search / fetch_web_page tools: schema parity with the Tavily provider, key rotation."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from responses_api_agents.gdpval_stirrup_agent.exa_search import (
    ExaToolProvider,
    format_search_results,
    load_keys_file,
    parse_keys,
)


def test_parse_keys_and_keys_file(tmp_path):
    assert parse_keys("[a, b,a ,]") == ["a", "b"]
    assert parse_keys(["x", " x", "y"]) == ["x", "y"]
    f = tmp_path / "keys.txt"
    f.write_text("# comment\nk1\n\nk2  # trailing\nk1\n")
    assert load_keys_file(f) == ["k1", "k2"]
    p = ExaToolProvider(api_keys=["z"], api_keys_file=f)
    assert p.num_keys == 3
    with pytest.raises(ValueError):
        ExaToolProvider(api_keys=[])


def test_tool_schema_matches_tavily_provider():
    from responses_api_agents.stirrup_agent.tavily_search import TavilyToolProvider

    exa = ExaToolProvider(api_keys=["k"])
    tav = TavilyToolProvider(api_keys=["k"])
    exa._client = tav._client = object()  # _get_tools only needs a client to exist
    exa_tools = {t.name: t for t in exa._get_tools()}
    tav_tools = {t.name: t for t in tav._get_tools()}
    assert set(exa_tools) == set(tav_tools) == {"web_search", "fetch_web_page"}
    for name in exa_tools:
        assert exa_tools[name].parameters.model_json_schema() == tav_tools[name].parameters.model_json_schema()
    assert exa_tools["fetch_web_page"].description == tav_tools["fetch_web_page"].description


def test_format_search_results_escapes_and_uses_text():
    xml = format_search_results({"results": [{"title": "A & B", "url": "https://x/y?a=1&b=2", "text": "<p>hi</p>"}]})
    assert "<title>A &amp; B</title>" in xml and "&lt;p&gt;hi&lt;/p&gt;" in xml and "a=1&amp;b=2" in xml


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=httpx.Request("POST", "https://api.exa.ai/search"), response=httpx.Response(self.status_code))

    def json(self):
        return self._payload


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        return self.replies.pop(0)

    async def get(self, url, headers=None):
        self.calls.append((url, None, headers))
        return self.replies.pop(0)


def test_search_rotates_keys_on_auth_and_quota_errors():
    provider = ExaToolProvider(api_keys=["k1", "k2", "k3"], max_sweeps=1)
    provider._client = _Client([_Resp(401), _Resp(429), _Resp(200, {"results": [{"title": "T", "url": "u", "text": "body"}]})])
    result = asyncio.run(provider.search("q"))
    assert result.success and "<title>T</title>" in result.content
    used = [c[2]["x-api-key"] for c in provider._client.calls]
    assert used == ["k1", "k2", "k3"]
    assert provider.key_rotations == 2 and provider.searches == 1
    body = provider._client.calls[0][1]
    assert body["query"] == "q" and body["numResults"] == 5 and body["contents"]["text"]["maxCharacters"] == 2000
    # keys never leak into tool output
    assert "k1" not in result.content and "k2" not in result.content


def test_search_exhaustion_and_non_retryable_error():
    provider = ExaToolProvider(api_keys=["k1"], max_sweeps=2)
    provider._client = _Client([_Resp(503), _Resp(503)])
    result = asyncio.run(provider.search("q"))
    assert not result.success and "exhausted 2 attempt(s)" in result.content and "k1" not in result.content
    provider._client = _Client([_Resp(400)])
    result = asyncio.run(provider.search("q"))
    assert not result.success and "<error>" in result.content


def test_fetch_extracts_markdown():
    provider = ExaToolProvider(api_keys=["k1"])
    provider._client = _Client([_Resp(200, text="<html><body><article><h1>Title</h1><p>Hello world paragraph with enough text to extract.</p></article></body></html>")])
    result = asyncio.run(provider.fetch("https://example.com/a"))
    assert result.success and "<web_fetch><url>https://example.com/a</url>" in result.content
    assert "Hello world" in result.content
