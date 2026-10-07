# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Tavily key-list parsing, rotation, and rotation-on-retry."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

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
    def test_round_robin(self):
        p = TavilySearch(api_keys=["k1", "k2", "k3"])
        assert [p._next_key() for _ in range(7)] == ["k1", "k2", "k3", "k1", "k2", "k3", "k1"]

    def test_single_key_keeps_returning_same(self):
        p = TavilySearch(api_keys=["only_one"])
        assert [p._next_key() for _ in range(3)] == ["only_one"] * 3

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


# ---------------------------------------------------------------------------
# Rotation-on-retry: send key-specific failures (401/403/429), verify the
# executor rotates to the next key, and verify it stops after len(keys).
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, json_body: dict[str, Any] | None = None) -> AsyncMock:
    resp = AsyncMock()
    resp.status_code = status_code
    resp.json = AsyncMock(return_value=json_body or {})
    if status_code >= 400:

        def _raise():
            raise httpx.HTTPStatusError(
                f"{status_code}", request=AsyncMock(), response=AsyncMock(status_code=status_code)
            )

        resp.raise_for_status = _raise
    else:
        resp.raise_for_status = lambda: None
    # The executor calls .json() synchronously — wrap accordingly.
    if json_body is not None:
        resp.json = lambda: json_body
    return resp


class TestSearchExecutorRotation:
    async def test_first_key_succeeds_no_rotation(self):
        provider = TavilySearch(api_keys=["k1", "k2", "k3"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            return _mock_response(200, {"answer": "hi", "results": []})

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert not result.startswith("<error>")
        assert seen_keys == ["Bearer k1"]
        assert provider._key_idx == 1

    async def test_rotates_on_401_then_succeeds_on_second_key(self):
        provider = TavilySearch(api_keys=["bad", "good", "unused"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            if "bad" in kwargs["headers"]["Authorization"]:
                return _mock_response(401)
            return _mock_response(200, {"answer": "hi", "results": []})

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert not result.startswith("<error>")
        # First key tried, got 401; rotated to second.
        assert seen_keys == ["Bearer bad", "Bearer good"]

    @pytest.mark.parametrize("status", sorted(KEY_ROTATION_RETRY_STATUSES))
    async def test_rotates_on_each_key_specific_status(self, status):
        provider = TavilySearch(api_keys=["first", "second"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            if "first" in kwargs["headers"]["Authorization"]:
                return _mock_response(status)
            return _mock_response(200, {"answer": "ok", "results": []})

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert not result.startswith("<error>")
        assert seen_keys == ["Bearer first", "Bearer second"]

    async def test_all_keys_fail_returns_error(self):
        provider = TavilySearch(api_keys=["k1", "k2", "k3"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            return _mock_response(401)

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert result.startswith("<error>")
        # Tried each key exactly once.
        assert seen_keys == ["Bearer k1", "Bearer k2", "Bearer k3"]
        assert "exhausted 3 attempt(s)" in result
        assert "3 key(s) × 1 sweep(s)" in result
        assert "last status=401" in result
        assert "retryable error" in result

    async def test_404_does_NOT_trigger_rotation(self):
        # 404 isn't a rotation-worthy status — should fail through after one attempt.
        provider = TavilySearch(api_keys=["k1", "k2"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            return _mock_response(404)

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert result.startswith("<error>")
        # Only ONE key attempted — 404 doesn't trigger rotation.
        assert seen_keys == ["Bearer k1"]

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
    async def test_rotates_on_5xx(self, status):
        # 5xx is also key-rotation-worthy: free retry costs nothing and often
        # the next attempt lands on a healthy backend.
        provider = TavilySearch(api_keys=["k1", "k2"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            if "k1" in kwargs["headers"]["Authorization"]:
                return _mock_response(status)
            return _mock_response(200, {"answer": "ok", "results": []})

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert not result.startswith("<error>")
        assert seen_keys == ["Bearer k1", "Bearer k2"]

    async def test_max_sweeps_2_tries_each_key_twice(self):
        # max_sweeps=2 should give us 2 × len(keys) = 4 attempts before giving up.
        provider = TavilySearch(api_keys=["k1", "k2"], max_sweeps=2)
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            return _mock_response(429)

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert result.startswith("<error>")
        # Each key attempted twice in round-robin.
        assert seen_keys == ["Bearer k1", "Bearer k2", "Bearer k1", "Bearer k2"]
        assert "exhausted 4 attempt(s)" in result
        assert "2 key(s) × 2 sweep(s)" in result

    async def test_max_sweeps_3_succeeds_on_third_sweep(self):
        # Single key, fails twice, succeeds on third attempt (third sweep).
        provider = TavilySearch(api_keys=["only"], max_sweeps=3)
        call_count = {"n": 0}

        async def fake_post(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return _mock_response(503)
            return _mock_response(200, {"answer": "ok", "results": []})

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert not result.startswith("<error>")
        assert call_count["n"] == 3

    def test_max_sweeps_default_is_1(self):
        provider = TavilySearch(api_keys=["k1", "k2", "k3"])
        assert provider._max_sweeps == 1

    def test_max_sweeps_zero_rejected(self):
        with pytest.raises(ValueError, match="max_sweeps must be >= 1"):
            TavilySearch(api_keys=["k1"], max_sweeps=0)

    def test_max_sweeps_negative_rejected(self):
        with pytest.raises(ValueError, match="max_sweeps must be >= 1"):
            TavilySearch(api_keys=["k1"], max_sweeps=-1)

    async def test_4xx_400_does_NOT_trigger_rotation(self):
        # 400 (e.g. malformed query) isn't fixable by changing keys.
        provider = TavilySearch(api_keys=["k1", "k2"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            return _mock_response(400)

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert result.startswith("<error>")
        assert seen_keys == ["Bearer k1"]

    async def test_network_error_does_NOT_trigger_rotation(self):
        provider = TavilySearch(api_keys=["k1", "k2"])
        seen_keys: list[str] = []

        async def fake_post(url, **kwargs):
            seen_keys.append(kwargs["headers"]["Authorization"])
            raise httpx.ConnectError("dns failed")

        client = AsyncMock()
        client.post = fake_post
        result = await provider.search("x", client)
        assert result.startswith("<error>")
        # Only one attempt — network errors aren't key-specific.
        assert seen_keys == ["Bearer k1"]
        assert "dns failed" in result


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
