# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.tavily_search import app as module
from resources_servers.tavily_search.app import (
    FindInPageRequest,
    ScrollPageRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    URLExclusionPolicy,
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
                            {"type": "url_substring", "value": "/hle/answers"},
                        ]
                    }
                ]
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "url",
    [
        "https://blocked.test/a",
        "https://sub.blocked.test/a",
        "https://BLOCKED.test./a",
        "https://%62locked.test/a",
        "https://ok.test/%68le/answers",
        "https://user@ok.test/",
        "https://ok.test/%252568le/answers",
        "file:///etc/passwd",
        "not a url",
    ],
)
def test_url_bypasses_blocked(exclusions, url):
    assert URLExclusionPolicy(exclusions).blocked(url)


def test_domain_boundaries(exclusions):
    policy = URLExclusionPolicy(exclusions)
    assert not policy.blocked("https://notblocked.test/")
    assert not policy.blocked("https://blocked.test.example/")


def test_invalid_policy_domain_fails_closed_with_actionable_error(tmp_path):
    path = tmp_path / "invalid-policy.json"
    domain = "a" * 64 + ".test"
    path.write_text(json.dumps({"notices": [{"properties": [{"type": "domain", "value": domain}]}]}))
    with pytest.raises(ValueError, match="Invalid exclusion domain") as error:
        URLExclusionPolicy(path)
    assert str(path) in str(error.value)
    assert domain in str(error.value)


@pytest.fixture
def server(exclusions):
    cfg = TavilySearchResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="tavily",
        tavily_api_key="key-one, key-two,",
        exclude_domains_file_path=str(exclusions),
        use_judge=False,
    )
    return TavilySearchResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


def tool_request():
    return SimpleNamespace(session={SESSION_ID_KEY: "test-session"})


def http_response(status, payload=None):
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value=payload)
    return response


async def test_sdk_requests_rotate_keys_on_rate_limit_and_filter_results(server, monkeypatch):
    limited = http_response(429)
    success = http_response(
        200,
        {
            "answer": "unattributed answer",
            "results": [
                {"url": "https://blocked.test/answer", "title": "blocked", "content": "blocked secret"},
                {"url": "https://ok.test/hle/answers", "title": "blocked path", "content": "answer key"},
                {"url": "https://ok.test/", "title": "allowed", "content": "allowed content"},
            ],
        },
    )
    post = AsyncMock(side_effect=[limited, success])
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    result = await server.web_search(tool_request(), TavilySearchRequest(query="query"))
    assert "allowed content" in result.results_string
    assert all(text not in result.results_string for text in ("blocked secret", "answer key", "unattributed answer"))
    assert [call.kwargs["headers"]["authorization"] for call in post.await_args_list] == [
        "Bearer key-one",
        "Bearer key-two",
    ]
    payload = json.loads(post.await_args_list[0].kwargs["data"])
    assert payload["include_answer"] is False
    assert payload["exclude_domains"] == ["blocked.test"]
    assert payload["search_depth"] == "advanced"
    assert payload["max_results"] == 10
    assert post.await_args_list[0].kwargs["timeout"].total == 60
    limited.release.assert_called_once()
    success.release.assert_called_once()


@pytest.mark.parametrize("status, attempts", [(429, 3), (503, 3), (401, 1)])
async def test_retry_budget_and_redacted_errors(server, monkeypatch, status, attempts):
    response = http_response(status, {"error": "private-key-or-provider-details"})
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    with pytest.raises(RuntimeError, match=f"Tavily HTTP {status} after {attempts} attempts") as error:
        await server.web_search(tool_request(), TavilySearchRequest(query="query"))
    assert "private-key" not in str(error.value)
    assert post.await_count == attempts
    assert response.release.call_count == attempts
    response.json.assert_not_awaited()


@pytest.mark.parametrize("tool", ["find_in_page", "scroll_page"])
async def test_excluded_page_is_not_requested(server, monkeypatch, tool):
    backend = AsyncMock()
    monkeypatch.setattr(server, "_async_tavily_clients", [backend])
    args = {"url": "https://ok.test/hle/answers"}
    body = FindInPageRequest(**args, query="answer") if tool == "find_in_page" else ScrollPageRequest(**args)
    result = await getattr(server, tool)(tool_request(), body)
    assert "excluded" in result.results_string
    backend.extract.assert_not_awaited()


@pytest.mark.parametrize("tool", ["find_in_page", "scroll_page"])
async def test_reported_extract_redirect_is_filtered(server, monkeypatch, tool):
    response = http_response(200, {"results": [{"url": "https://blocked.test/key", "raw_content": "answer key"}]})
    monkeypatch.setattr(module, "request", AsyncMock(return_value=response))
    args = {"url": "https://allowed.test/redirect"}
    body = FindInPageRequest(**args, query="answer") if tool == "find_in_page" else ScrollPageRequest(**args)
    result = await getattr(server, tool)(tool_request(), body)
    assert "answer key" not in result.results_string
    assert all("answer key" not in content for content in server._page_cache.values())


async def test_cache_and_output_limits_are_configurable(server):
    server.config.max_cached_pages = 1
    server.config.max_cached_page_chars = 8
    server.config.max_scroll_words = 2
    backend = MagicMock()
    backend.extract = AsyncMock(side_effect=lambda urls: {"results": [{"url": urls, "raw_content": "a b c d e f"}]})
    server._async_tavily_clients = [backend]
    first = await server.scroll_page(tool_request(), ScrollPageRequest(url="https://one.test/", n=100))
    assert "L0: a b" in first.results_string
    assert "c d" not in first.results_string
    assert first.total_words == 4
    await server.scroll_page(tool_request(), ScrollPageRequest(url="https://one.test/"))
    assert backend.extract.await_count == 1
    await server.scroll_page(tool_request(), ScrollPageRequest(url="https://two.test/"))
    assert list(server._page_cache) == ["https://two.test/"]
    await server.scroll_page(tool_request(), ScrollPageRequest(url="https://one.test/"))
    assert backend.extract.await_count == 3


async def test_disabling_cache_preserves_page_content(server):
    server.config.max_cached_pages = 0
    server.config.max_cached_page_chars = 3
    backend = MagicMock()
    backend.extract = AsyncMock(return_value={"results": [{"url": "https://ok.test/", "raw_content": "a b c d e"}]})
    server._async_tavily_clients = [backend]
    for _ in range(2):
        result = await server.scroll_page(
            tool_request(), ScrollPageRequest(url="https://ok.test/", start_index=3, n=2)
        )
        assert result.total_words == 5
        assert "d e" in result.results_string
    assert backend.extract.await_count == 2
    assert not server._page_cache


def test_mcp_exposes_only_existing_browser_tools(server):
    assert server.mcp_allowed_tools_for_session({}) == ["web_search", "find_in_page", "scroll_page"]


def test_empty_key_pool_is_rejected(server):
    cfg = server.config.model_copy(update={"tavily_api_key": " , "})
    with pytest.raises(ValueError, match="At least one"):
        TavilySearchResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))
