# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Exa provider behind web_search, find_in_page and scroll_page, and the switch that selects it."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from resources_servers.tavily_search import app as module
from resources_servers.tavily_search.app import (
    FindInPageRequest,
    ScrollPageRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


@pytest.fixture
def exclusions(tmp_path):
    path = tmp_path / "excluded.json"
    path.write_text(
        json.dumps(
            {
                "notices": [
                    {
                        "properties": [
                            {"type": "domain", "value": "blocked.test"},
                            {"type": "url_substring", "value": "/answers"},
                        ]
                    }
                ]
            }
        )
    )
    return path


def _server(exclusions, **config) -> TavilySearchResourcesServer:
    cfg = TavilySearchResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="exa_search",
        exclude_domains_file_path=str(exclusions),
        use_judge=False,
        **config,
    )
    return TavilySearchResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def server(exclusions):
    return _server(exclusions, search_provider="exa", exa_api_key="exa-one, exa-two", exa_base_url="https://exa.test/")


def tool_request():
    return SimpleNamespace(session={SESSION_ID_KEY: "test-session"})


def http_response(status, payload=None):
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value=payload)
    return response


@pytest.fixture
def post(monkeypatch):
    post = AsyncMock(return_value=http_response(200, {"results": []}))
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    return post


def sent(call) -> dict:
    return json.loads(call.kwargs["data"])


def test_tavily_stays_the_default_and_needs_no_exa_key(exclusions):
    server = _server(exclusions, tavily_api_key="tavily-key")
    assert server.config.search_provider == "tavily"
    assert len(server._async_tavily_clients) == 1
    assert server._exa_clients is None


def test_exa_needs_its_own_key_and_no_tavily_key(exclusions, server):
    assert server.config.tavily_api_key is None
    assert server._async_tavily_clients is None
    assert len(server._exa_clients) == 2
    with pytest.raises(ValueError, match="At least one Exa API key"):
        _server(exclusions, search_provider="exa", tavily_api_key="tavily-key")
    with pytest.raises(ValueError, match="At least one Tavily API key"):
        _server(exclusions, exa_api_key="exa-key")


async def test_web_search_asks_exa_and_keeps_the_exclusion_policy(server, post):
    post.return_value = http_response(
        200,
        {
            "results": [
                {"url": "https://blocked.test/a", "title": "blocked", "highlights": ["blocked secret"]},
                {"url": "https://ok.test/answers", "title": "blocked path", "highlights": ["answer key"]},
                {"url": "https://ok.test/page", "title": None, "highlights": ["first passage", "", "second passage"]},
                {"title": "no url", "highlights": ["unplaceable passage"]},
                "not a result",
            ]
        },
    )
    result = await server.web_search(tool_request(), TavilySearchRequest(query="what is exa"))

    assert "first passage ... second passage" in result.results_string
    assert "[1]  (ok.test)" in result.results_string
    assert all(text not in result.results_string for text in ("blocked secret", "answer key", "unplaceable"))
    (call,) = post.await_args_list
    assert call.kwargs["url"] == "https://exa.test/search"
    assert call.kwargs["headers"]["x-api-key"] == "exa-one"
    assert "authorization" not in call.kwargs["headers"]
    assert call.kwargs["headers"]["content-type"] == "application/json"
    assert sent(call) == {
        "query": "what is exa",
        "numResults": 10,
        "contents": {"highlights": True},
        "excludeDomains": ["blocked.test"],
    }
    assert server._session_id_to_metrics["test-session"].async_tavily_calls[0].function == "search"


async def test_web_search_asks_exa_for_the_configured_number_of_results(exclusions, post):
    server = _server(exclusions, search_provider="exa", exa_api_key="exa-key", max_results=3)
    await server.web_search(tool_request(), TavilySearchRequest(query="q"))
    assert sent(post.await_args_list[0])["numResults"] == 3


async def test_no_provider_exclusions_sends_no_exclude_domains(exclusions, post):
    server = _server(exclusions, search_provider="exa", exa_api_key="exa-key", tavily_exclude_domains_limit=0)
    await server.web_search(tool_request(), TavilySearchRequest(query="q"))
    assert "excludeDomains" not in sent(post.await_args_list[0])


async def test_find_in_page_returns_the_passages_matching_the_query(server, post):
    post.return_value = http_response(
        200, {"results": [{"url": "https://ok.test/page", "text": "whole page", "highlights": ["one", "two"]}]}
    )
    result = await server.find_in_page(tool_request(), FindInPageRequest(url="https://ok.test/page", query="needle"))

    assert "L0: one\nL1: \nL2: two" in result.results_string
    assert "whole page" not in result.results_string
    (call,) = post.await_args_list
    assert call.kwargs["url"] == "https://exa.test/contents"
    assert sent(call) == {"urls": ["https://ok.test/page"], "text": True, "highlights": {"query": "needle"}}


async def test_find_in_page_falls_back_to_the_page_text(server, post):
    post.return_value = http_response(200, {"results": [{"url": "https://ok.test/page", "text": "whole page"}]})
    result = await server.find_in_page(tool_request(), FindInPageRequest(url="https://ok.test/page", query="needle"))
    assert "L0: whole page" in result.results_string


async def test_scroll_page_reads_and_caches_the_page_text(server, post):
    post.return_value = http_response(200, {"results": [{"url": "https://ok.test/page", "text": "a b c d"}]})
    for _ in range(2):
        result = await server.scroll_page(tool_request(), ScrollPageRequest(url="https://ok.test/page", n=2))
        assert result.total_words == 4
        assert "L0: a b" in result.results_string
    (call,) = post.await_args_list
    assert sent(call) == {"urls": ["https://ok.test/page"], "text": True}


async def test_scroll_page_does_not_cache_a_page_exa_failed_to_fetch(server, post):
    failed = {"results": [], "statuses": [{"id": "https://ok.test/page", "status": "error"}]}
    post.side_effect = [
        http_response(200, failed),
        http_response(200, {"results": [{"url": "https://ok.test/page", "text": "a b c d"}]}),
    ]
    first = await server.scroll_page(tool_request(), ScrollPageRequest(url="https://ok.test/page"))
    assert first.total_words == 0
    assert "https://ok.test/page" not in server._page_cache
    second = await server.scroll_page(tool_request(), ScrollPageRequest(url="https://ok.test/page"))
    assert second.total_words == 4
    assert len(post.await_args_list) == 2


@pytest.mark.parametrize("tool", ["find_in_page", "scroll_page"])
async def test_a_page_exa_resolves_to_an_excluded_url_is_dropped(server, post, tool):
    post.return_value = http_response(200, {"results": [{"url": "https://blocked.test/key", "text": "answer key"}]})
    args = {"url": "https://ok.test/redirect"}
    body = FindInPageRequest(**args, query="answer") if tool == "find_in_page" else ScrollPageRequest(**args)
    result = await getattr(server, tool)(tool_request(), body)
    assert "answer key" not in result.results_string
    assert all("answer key" not in content for content in server._page_cache.values())


async def test_exa_keys_rotate_on_rate_limits_and_rejections(server, post):
    post.side_effect = [http_response(429), http_response(401), http_response(200, {"results": []})]
    await server.web_search(tool_request(), TavilySearchRequest(query="q"))
    assert [c.kwargs["headers"]["x-api-key"] for c in post.await_args_list] == ["exa-one", "exa-two", "exa-one"]
    assert all(sent(c)["query"] == "q" for c in post.await_args_list)


async def test_an_exa_key_out_of_credits_falls_over_to_the_next_key(exclusions, post):
    server = _server(exclusions, search_provider="exa", exa_api_key="exa-one,exa-two,exa-three")
    post.side_effect = [http_response(402), http_response(402), http_response(200, {"results": []})]
    await server.web_search(tool_request(), TavilySearchRequest(query="q"))
    assert [c.kwargs["headers"]["x-api-key"] for c in post.await_args_list] == ["exa-one", "exa-two", "exa-three"]


async def test_an_exa_402_on_every_key_fails_without_reusing_a_spent_key(server, post):
    post.side_effect = [http_response(402), http_response(402)]
    with pytest.raises(RuntimeError, match="Exa HTTP 402 after 2 attempts"):
        await server.web_search(tool_request(), TavilySearchRequest(query="q"))


@pytest.mark.parametrize("decode_error", [ValueError("Expecting value"), None])
async def test_an_unreadable_exa_reply_names_the_provider_without_its_body_or_key(server, post, decode_error):
    response = http_response(200, ["private-body"])
    if decode_error is not None:
        response.json.side_effect = decode_error
    post.return_value = response
    with pytest.raises(RuntimeError, match="Exa HTTP 200 with an unreadable body") as error:
        await server.web_search(tool_request(), TavilySearchRequest(query="q"))
    assert all(secret not in repr(error.value) for secret in ("private-body", "exa-one"))
    assert error.value.__context__ is None
    response.release.assert_called_once()


async def test_exa_transport_failures_name_the_provider(server, post):
    post.side_effect = ClientConnectionError("connection reset")
    with pytest.raises(RuntimeError, match="Exa transport failure after 3 attempts"):
        await server.web_search(tool_request(), TavilySearchRequest(query="q"))


async def test_exa_failures_name_the_provider_without_its_reply(server, post):
    post.return_value = http_response(400, {"error": "private-key-details"})
    with pytest.raises(RuntimeError, match="Exa HTTP 400 after 1 attempts") as error:
        await server.web_search(tool_request(), TavilySearchRequest(query="q"))
    assert "private-key" not in str(error.value)
    post.return_value.json.assert_not_awaited()
