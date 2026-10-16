# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.tavily_search import app as module
from resources_servers.tavily_search.app import (
    DEFAULT_EXCLUDE_DOMAINS_PRIORITY,
    FindInPageRequest,
    ScrollPageRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    URLExclusionPolicy,
    select_provider_domains,
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


@pytest.mark.parametrize("status, attempts", [(429, 3), (503, 3), (401, 2), (432, 2), (433, 2), (400, 1)])
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


@pytest.mark.parametrize("status", [401, 432, 433])
@pytest.mark.parametrize("endpoint", ["/search", "/extract"])
async def test_rejected_key_falls_back_without_changing_request(server, monkeypatch, status, endpoint):
    bad = http_response(status, {"error": "private-key-details"})
    good = http_response(200, {"results": []})
    post = AsyncMock(side_effect=[bad, good])
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    transport = server._async_tavily_clients[0]._client
    result = await transport.post(endpoint, '{"query":"math"}', timeout=60)
    assert result.status_code == 200
    assert [c.kwargs["headers"]["authorization"] for c in post.await_args_list] == ["Bearer key-one", "Bearer key-two"]
    assert all(c.kwargs["data"] == '{"query":"math"}' for c in post.await_args_list)
    assert all(c.kwargs["url"].endswith(endpoint) for c in post.await_args_list)
    bad.release.assert_called_once()
    good.release.assert_called_once()
    bad.json.assert_not_awaited()


async def test_key_failover_shares_budget_and_skips_rejected_key(server, monkeypatch):
    transport = server._async_tavily_clients[0]._client
    post = AsyncMock(side_effect=[http_response(401), http_response(503), http_response(200, {"results": []})])
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    await transport.post("/search", "{}", timeout=60)
    assert [c.kwargs["headers"]["authorization"] for c in post.await_args_list] == [
        "Bearer key-one",
        "Bearer key-two",
        "Bearer key-two",
    ]
    transport.max_attempts = 1
    post.reset_mock(side_effect=True)
    post.return_value = http_response(432)
    with pytest.raises(RuntimeError, match="Tavily HTTP 432 after 1 attempts"):
        await transport.post("/search", "{}", timeout=60)
    assert post.await_count == 1


async def test_single_rejected_key_is_not_retried(server, monkeypatch):
    transport = server._async_tavily_clients[0]._client
    transport.retry_api_keys = ["key-one", "key-one"]
    post = AsyncMock(return_value=http_response(401))
    monkeypatch.setattr(module, "request", post)
    with pytest.raises(RuntimeError, match="Tavily HTTP 401 after 1 attempts"):
        await transport.post("/search", "{}", timeout=60)
    assert post.await_count == 1


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


@pytest.mark.parametrize("failure", [module.ClientConnectionError, module.ClientPayloadError, TimeoutError])
async def test_transport_failures_share_bounded_retry_budget(server, monkeypatch, failure):
    success = http_response(200, {"results": []})
    post = AsyncMock(side_effect=[failure("private-provider-details"), success])
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    await server.web_search(tool_request(), TavilySearchRequest(query="query"))
    assert post.await_count == 2
    assert (
        post.await_args_list[0].kwargs["headers"]["authorization"]
        != post.await_args_list[1].kwargs["headers"]["authorization"]
    )
    success.release.assert_called_once()
    post.reset_mock(side_effect=True)
    post.side_effect = failure("private-provider-details")
    with pytest.raises(RuntimeError, match="transport failure after 3 attempts") as error:
        await server.web_search(tool_request(), TavilySearchRequest(query="query"))
    assert post.await_count == 3
    assert "private-provider-details" not in str(error.value)
    assert error.value.__suppress_context__


async def test_interrupted_response_body_releases_connection_before_retry(server, monkeypatch):
    broken = http_response(200, {})
    broken.json.side_effect = module.ClientPayloadError("truncated body")
    good = http_response(200, {"results": []})
    monkeypatch.setattr(module, "request", AsyncMock(side_effect=[broken, good]))
    monkeypatch.setattr(module, "sleep", AsyncMock())
    await server.web_search(tool_request(), TavilySearchRequest(query="query"))
    broken.release.assert_called_once()
    good.release.assert_called_once()


# Tavily's /search caps exclude_domains at 150 (HTTP 400 beyond it). A 220-domain policy whose priority hosts sit
# at the END of the file: the provider subset must pull them forward and drop the tail, which stays local-only.
FILLERS = [f"filler-{index:03d}.test" for index in range(205)]
PRIORITY_IN_FILE_ORDER = [
    "archive.ph",
    "github.com",
    "gist.github.com",  # not a priority pattern: stays in file order
    "api.scale.com",
    "lastexam.ai",
    "huggingface.co",
    "archive.today",
    "scale.com",
    "safe.ai",
    "githubusercontent.com",
    "agi.safe.ai",
    "hf.co",
    "huggingface.com",  # not a priority pattern
    "answers.test",  # beyond the cap
    "leak.test",  # beyond the cap
]
assert len(FILLERS) + len(PRIORITY_IN_FILE_ORDER) == 220


@pytest.fixture
def capped_policy(tmp_path):
    properties = [{"type": "domain", "value": d} for d in FILLERS + PRIORITY_IN_FILE_ORDER]
    properties.append({"type": "url_substring", "value": "/hle/answers"})
    path = tmp_path / "capped.json"
    path.write_text(json.dumps({"notices": [{"properties": properties}]}))
    return path


def make_server(policy_path, **overrides):
    cfg = TavilySearchResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="tavily",
        tavily_api_key="key-one",
        exclude_domains_file_path=str(policy_path),
        use_judge=False,
        **overrides,
    )
    return TavilySearchResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


SEARCH_RESULTS = {
    "results": [
        {"url": "https://leak.test/key", "title": "beyond the cap", "content": "tail-domain answer key"},
        {"url": "https://sub.answers.test/q", "title": "beyond the cap", "content": "tail-subdomain answer key"},
        {"url": "https://filler-000.test/hle/answers", "title": "substring", "content": "substring answer key"},
        {"url": "https://ok.test/", "title": "allowed", "content": "allowed content"},
    ]
}


async def test_provider_exclude_domains_are_capped_priority_first_and_tail_enforced_locally(
    capped_policy, monkeypatch, caplog
):
    with caplog.at_level("WARNING", logger=module.__name__):
        server = make_server(capped_policy)
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "70 domains are not sent" in warnings[0].getMessage()
    assert str(capped_policy) in warnings[0].getMessage()

    post = AsyncMock(return_value=http_response(200, SEARCH_RESULTS))
    monkeypatch.setattr(module, "request", post)
    for _ in range(2):
        result = await server.web_search(tool_request(), TavilySearchRequest(query="query"))
    assert len([record for record in caplog.records if record.levelname == "WARNING"]) == 1  # once per start

    sent = json.loads(post.await_args_list[0].kwargs["data"])["exclude_domains"]
    assert len(sent) == 150
    assert len(set(sent)) == 150
    # Priority block in pattern order, then file order inside a pattern; then the rest of the file in order.
    assert sent[:11] == [
        "huggingface.co",
        "hf.co",
        "lastexam.ai",
        "agi.safe.ai",
        "safe.ai",
        "scale.com",
        "api.scale.com",
        "github.com",
        "githubusercontent.com",
        "archive.ph",
        "archive.today",
    ]
    assert sent[11:] == FILLERS[:139]
    assert {"gist.github.com", "huggingface.com", "answers.test", "leak.test"}.isdisjoint(sent)

    # Domains beyond the cap and url_substring rules are enforced locally on the returned URLs.
    assert "allowed content" in result.results_string
    assert "answer key" not in result.results_string
    assert server._is_url_excluded("https://leak.test/key")
    assert server._is_url_excluded("https://sub.answers.test/q")


@pytest.mark.parametrize("limit", [0, None])
async def test_disabled_provider_exclusions_keep_local_enforcement(capped_policy, monkeypatch, caplog, limit):
    with caplog.at_level("WARNING", logger=module.__name__):
        server = make_server(capped_policy, tavily_exclude_domains_limit=limit)
    assert not [record for record in caplog.records if record.levelname == "WARNING"]
    post = AsyncMock(return_value=http_response(200, SEARCH_RESULTS))
    monkeypatch.setattr(module, "request", post)
    result = await server.web_search(tool_request(), TavilySearchRequest(query="query"))
    assert "exclude_domains" not in json.loads(post.await_args_list[0].kwargs["data"])
    assert "allowed content" in result.results_string
    assert "answer key" not in result.results_string


def test_select_provider_domains_honours_custom_priority_and_short_policies():
    domains = ["a.test", "b.test", "c.test"]
    assert select_provider_domains(domains, 150, DEFAULT_EXCLUDE_DOMAINS_PRIORITY) == domains
    assert select_provider_domains(domains, 2, ["c.*"]) == ["c.test", "a.test"]
    assert select_provider_domains(domains, 0, ["c.*"]) is None
