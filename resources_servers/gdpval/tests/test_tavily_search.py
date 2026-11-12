# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Tavily key-list parsing, rotation, and rotation-on-retry."""

from __future__ import annotations

import json

import httpx
import pytest

from resources_servers.gdpval.tavily_search import (
    KEY_ROTATION_RETRY_STATUSES,
    TavilySearch,
    _should_rotate_on_status,
)


class TestParseKeys:
    def test_single_string_key(self):
        assert TavilySearch._parse_keys("tvly-prod-abc") == ["tvly-prod-abc"]

    def test_python_list(self):
        assert TavilySearch._parse_keys(["k1", "k2", "k3"]) == ["k1", "k2", "k3"]

    def test_comma_separated_string(self):
        assert TavilySearch._parse_keys("k1,k2,k3") == ["k1", "k2", "k3"]

    def test_efb_bracket_list_format(self):
        # The actual format the EFB harness injects via host:TAVILY_API_KEY.
        raw = "[tvly-prod-aaa,tvly-prod-bbb,tvly-prod-ccc]"
        assert TavilySearch._parse_keys(raw) == [
            "tvly-prod-aaa",
            "tvly-prod-bbb",
            "tvly-prod-ccc",
        ]

    def test_strips_whitespace(self):
        assert TavilySearch._parse_keys("  k1 , k2 ,  k3  ") == ["k1", "k2", "k3"]

    def test_dedupes_preserving_order(self):
        assert TavilySearch._parse_keys("k1,k2,k1,k3,k2") == ["k1", "k2", "k3"]

    def test_empty_string(self):
        assert TavilySearch._parse_keys("") == []

    def test_empty_list(self):
        assert TavilySearch._parse_keys([]) == []

    def test_only_brackets(self):
        # No keys inside the brackets — corner case, must not blow up.
        assert TavilySearch._parse_keys("[]") == []

    def test_skips_blank_items(self):
        assert TavilySearch._parse_keys("k1,,k2,  ,k3") == ["k1", "k2", "k3"]


class TestRotation:
    def test_empty_keys_rejected(self):
        with pytest.raises(ValueError, match="at least one Tavily API key"):
            TavilySearch(api_keys="[]")


class TestShouldRotateOnStatus:
    @pytest.mark.parametrize("status", [401, 403, 429])
    def test_auth_quota_rotates(self, status):
        assert _should_rotate_on_status(status) is True

    @pytest.mark.parametrize("status", [500, 501, 502, 503, 504, 599])
    def test_5xx_rotates(self, status):
        assert _should_rotate_on_status(status) is True

    @pytest.mark.parametrize("status", [200, 201, 301, 400, 404, 405, 410, 600])
    def test_other_does_not_rotate(self, status):
        assert _should_rotate_on_status(status) is False


async def _rotating_search(
    keys: list[str], respond, *, max_sweeps: int = 1, calls: int = 1
) -> tuple[list[str], list[str]]:
    """Run ``calls`` searches; return each result and the bearer keys the upstream saw, in order."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return respond(len(seen), request.headers["Authorization"])

    provider = TavilySearch(api_keys=keys, max_sweeps=max_sweeps)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = [await provider.search("x", client) for _ in range(calls)]
    return results, seen


def _ok(*_args) -> httpx.Response:
    return httpx.Response(200, json={"answer": "ok", "results": []})


class TestSearchExecutorRotation:
    async def test_first_key_succeeds_no_rotation(self):
        results, seen = await _rotating_search(["k1", "k2", "k3"], _ok)

        assert not results[0].startswith("<error>")
        assert seen == ["Bearer k1"]

    async def test_successive_calls_rotate_round_robin(self):
        _, seen = await _rotating_search(["k1", "k2", "k3"], _ok, calls=4)

        assert seen == ["Bearer k1", "Bearer k2", "Bearer k3", "Bearer k1"]

    async def test_rotates_on_401_then_succeeds_on_second_key(self):
        results, seen = await _rotating_search(
            ["bad", "good", "unused"], lambda _n, key: httpx.Response(401) if key == "Bearer bad" else _ok()
        )

        assert not results[0].startswith("<error>")
        assert seen == ["Bearer bad", "Bearer good"]

    @pytest.mark.parametrize("status", sorted(KEY_ROTATION_RETRY_STATUSES) + [500, 502, 503, 504, 599])
    async def test_rotates_on_key_specific_and_server_statuses(self, status):
        results, seen = await _rotating_search(
            ["first", "second"], lambda _n, key: httpx.Response(status) if key == "Bearer first" else _ok()
        )

        assert not results[0].startswith("<error>")
        assert seen == ["Bearer first", "Bearer second"]

    async def test_all_keys_fail_returns_error(self):
        results, seen = await _rotating_search(["k1", "k2", "k3"], lambda *_: httpx.Response(401))

        assert seen == ["Bearer k1", "Bearer k2", "Bearer k3"]
        assert results[0] == (
            "<error>Tavily exhausted 3 attempt(s) (3 key(s) × 1 sweep(s)) on retryable errors "
            "(last status=401). Refresh keys or check upstream.</error>"
        )

    @pytest.mark.parametrize("status", [400, 404])
    async def test_other_statuses_do_not_trigger_rotation(self, status):
        results, seen = await _rotating_search(["k1", "k2"], lambda *_: httpx.Response(status))

        assert results[0].startswith("<error>")
        assert str(status) in results[0]
        assert seen == ["Bearer k1"]

    async def test_max_sweeps_2_tries_each_key_twice(self):
        results, seen = await _rotating_search(["k1", "k2"], lambda *_: httpx.Response(429), max_sweeps=2)

        assert seen == ["Bearer k1", "Bearer k2", "Bearer k1", "Bearer k2"]
        assert "exhausted 4 attempt(s)" in results[0]
        assert "2 key(s) × 2 sweep(s)" in results[0]

    async def test_max_sweeps_3_succeeds_on_third_sweep(self):
        results, seen = await _rotating_search(
            ["only"], lambda n, _key: httpx.Response(503) if n < 3 else _ok(), max_sweeps=3
        )

        assert not results[0].startswith("<error>")
        assert len(seen) == 3

    @pytest.mark.parametrize("max_sweeps", [0, -1])
    def test_max_sweeps_below_one_is_rejected(self, max_sweeps):
        with pytest.raises(ValueError, match="max_sweeps must be >= 1"):
            TavilySearch(api_keys=["k1"], max_sweeps=max_sweeps)

    async def test_a_200_that_is_not_json_is_an_error_not_an_exception(self):
        results, _ = await _rotating_search(["k1"], lambda *_: httpx.Response(200, text="<html>"))

        assert results[0] == "<error>Tavily returned a response that is not JSON</error>"

    async def test_network_error_does_not_trigger_rotation(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["Authorization"])
            raise httpx.ConnectError("dns failed")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await TavilySearch(api_keys=["k1", "k2"]).search("x", client)

        assert result.startswith("<error>") and "dns failed" in result
        assert seen == ["Bearer k1"]


async def _search(reply: dict, query: str = "q") -> tuple[str, httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await TavilySearch(api_keys="k1").search(query, client)
    return result, requests[0]


async def test_search_sends_the_query_and_formats_the_escaped_reply():
    reply = {"answer": "A & B", "results": [{"title": "<T>", "url": "https://x.test/?a=1&b=2", "content": "c"}]}

    result, request = await _search(reply, query="what is <x>?")

    assert result == (
        "<answer>A &amp; B</answer>\n<results>\n<result>\n<title>&lt;T&gt;</title>\n"
        "<url>https://x.test/?a=1&amp;b=2</url>\n<content>c</content>\n</result>\n</results>"
    )
    assert str(request.url) == "https://api.tavily.com/search"
    assert request.headers["Authorization"] == "Bearer k1"
    assert json.loads(request.content) == {"query": "what is <x>?", "max_results": 5, "include_answer": True}


async def test_search_without_an_answer_returns_only_the_results():
    result, _ = await _search({"results": []})

    assert result == "<results>\n\n</results>"


async def test_long_replies_keep_their_start_and_end_around_a_truncation_marker():
    result, _ = await _search({"results": [{"title": "t", "url": "u", "content": "a" * 30_000 + "b" * 30_000}]})

    assert len(result) < 40_200
    assert result.startswith("<results>\n<result>\n<title>t</title>") and result.endswith(
        "b</content>\n</result>\n</results>"
    )
    assert "This content has been truncated from an original" in result
