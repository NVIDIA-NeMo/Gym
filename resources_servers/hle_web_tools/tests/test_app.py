# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-free tests of the HLE web tools: the Tavily transport is stubbed, every rule uses upstream's blocklist."""

import hashlib
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.hle_web_tools import app as module
from resources_servers.hle_web_tools.app import (
    UPSTREAM_BLOCKLIST_SHA256,
    HLEWebToolsResourcesServer,
    HLEWebToolsResourcesServerConfig,
    HLEWebToolsSeedSessionRequest,
    WebFetchRequest,
    WebSearchRequest,
)


BLOCKLIST_PATH = Path(module.__file__).parent / "blocklist.json"
QUESTION = (
    "Consider a compact Riemann surface of genus three with a fixed theta characteristic. "
    "How many odd spin structures admit a nonvanishing holomorphic section under the given constraints?"
)


def make_server(**overrides) -> HLEWebToolsResourcesServer:
    config = HLEWebToolsResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="hle_tools",
        tavily_api_key="key-one, key-two",
        **overrides,
    )
    return HLEWebToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def server() -> HLEWebToolsResourcesServer:
    return make_server()


def tool_request(session_id: str = "session-a"):
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


def stub_backend(server, *, results=None, page=None, final_url=None) -> MagicMock:
    """Replace the Tavily transport: /search answers ``results``; /extract answers one page at the requested URL."""
    backend = MagicMock()
    backend.search = AsyncMock(return_value={"results": results or []})
    backend.extract = AsyncMock(
        side_effect=lambda url, **_: {"results": [{"url": final_url or url, "raw_content": page}]}
        if page is not None
        else {"results": [], "failed_results": [{"url": url, "error": "not reachable"}]}
    )
    server._backend = backend
    return backend


def result(url: str, title: str = "A page", snippet: str = "Some text") -> dict:
    return {"url": url, "title": title, "content": snippet}


def http_response(status: int, payload=None) -> MagicMock:
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value=payload)
    return response


async def search(server, query: str, session_id: str = "session-a"):
    return await server.web_search(tool_request(session_id), WebSearchRequest(query=query))


async def fetch(server, url: str, session_id: str = "session-a"):
    return await server.web_fetch(tool_request(session_id), WebFetchRequest(url=url))


async def seed(server, body: dict, session_id: str = "session-a"):
    await server.seed_session(tool_request(session_id), HLEWebToolsSeedSessionRequest(**body))


async def test_domains_rule_drops_results_from_blocked_hosts_and_their_subdomains(server, caplog):
    stub_backend(
        server,
        results=[
            result("https://hf.co/datasets/someone/puzzles"),
            result("https://datasets.huggingface.co/someone/puzzles"),
            result("https://example.com/ok", snippet="kept"),
        ],
    )
    with caplog.at_level(logging.INFO, logger=module.__name__):
        response = await search(server, "riemann surfaces")
    assert response.result_count == 1
    assert "example.com/ok" in response.text
    assert "hf.co" not in response.text and "huggingface" not in response.text
    assert "family=domains rule='hf.co'" in caplog.text
    assert "family=domains rule='huggingface.co'" in caplog.text


async def test_url_prefix_rule_ignores_case_scheme_and_leading_www(server, caplog):
    backend = stub_backend(server, page="A page body")
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        refused = await fetch(server, "HTTP://WWW.Nature.com/articles/S41586-025-09962-4?utm=1")
    assert refused.refused and "blocked page" in refused.text
    backend.extract.assert_not_awaited()
    assert "family=url_prefixes rule='nature.com/articles/s41586-025-09962-4'" in caplog.text
    allowed = await fetch(server, "https://www.nature.com/articles/s41586-025-00000-0")
    assert not allowed.refused
    backend.extract.assert_awaited_once()


@pytest.mark.parametrize(
    "url, blocked",
    [
        ("https://www.google.co.uk/search?q=riemann+surfaces", True),
        ("https://scholar.google.de/scholar?q=spin+structures", True),
        ("https://html.duckduckgo.com/html?q=theta", True),
        ("https://www.google.com/maps/place/Berlin", False),
        ("https://duckduckgo.com/about", False),
    ],
)
async def test_serp_rule_refuses_search_engine_result_pages(server, url, blocked):
    backend = stub_backend(server, page="A page body")
    response = await fetch(server, url)
    assert response.refused is blocked
    assert backend.extract.await_count == (0 if blocked else 1)


async def test_url_patterns_rule_applies_to_queries_result_titles_and_snippets(server, caplog):
    backend = stub_backend(
        server,
        results=[
            result("https://example.com/a", snippet="A write-up of Humanity's Last Exam questions"),
            result("https://example.com/b", title="Mirror of cais/hle"),
            result("https://example.com/c", snippet="Spin structures on Riemann surfaces"),
        ],
    )
    with caplog.at_level(logging.INFO, logger=module.__name__):
        response = await search(server, "odd spin structures genus three")
        refused = await search(server, "humanity's last exam answer key")
    assert response.result_count == 1 and "example.com/c" in response.text
    assert refused.refused and "blocked pattern" in refused.text
    assert backend.search.await_count == 1
    assert "family=url_patterns" in caplog.text
    for where in ("result snippet", "result title", "query"):
        assert f"where={where}" in caplog.text


@pytest.mark.parametrize(
    "body, refused",
    [
        ("Worked solutions for the HLE exam, question by question", True),
        ("worked solutions for the hle exam, question by question", False),
    ],
)
async def test_case_sensitive_patterns_apply_to_fetched_content_as_written(server, body, refused):
    stub_backend(server, page=body)
    response = await fetch(server, "https://example.com/notes")
    assert response.refused is refused


@pytest.mark.parametrize(
    "url, title, kept",
    [
        ("https://example.com/p", "Crossword puzzle answers", False),
        ("https://en.wikipedia.org/wiki/Crossword", "Crossword puzzle answers", True),
        ("https://example.com/q", "Daily quiz answers | Example Times", False),
        ("https://example.com/r", "Answers to the Riemann hypothesis critics", True),
    ],
)
async def test_answer_title_rule_exempts_allowed_hosts_and_strips_trailing_site_names(server, url, title, kept):
    stub_backend(server, results=[result(url, title=title)])
    response = await search(server, "crossword history")
    assert (response.result_count == 1) is kept


async def test_query_restricting_search_to_a_blocked_site_is_refused(server, caplog):
    backend = stub_backend(server, results=[result("https://example.com/ok")])
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        refused = await search(server, "spin structures site:huggingface.co")
    assert refused.refused and "blocked site" in refused.text
    backend.search.assert_not_awaited()
    assert "family=domains rule='huggingface.co' where=site: operator" in caplog.text
    allowed = await search(server, "spin structures site:https://www.arxiv.org/abs")
    assert not allowed.refused


@pytest.mark.parametrize(
    "seed_body",
    [
        {"question": QUESTION, "responses_create_params": {"input": []}},
        {"responses_create_params": {"input": [{"role": "user", "content": QUESTION + "\n\nAnswer: {answer}"}]}},
    ],
)
async def test_query_repeating_a_verbatim_question_span_is_refused_per_session(server, caplog, seed_body):
    backend = stub_backend(server, results=[result("https://example.com/ok")])
    await seed(server, seed_body, "session-a")
    await seed(server, {"question": "What is the boiling point of water at sea level in kelvin?"}, "session-b")
    eight = "Theta characteristic: how many odd spin structures admit"  # case and punctuation differ from the question
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        refused = await search(server, f"{eight} paper", "session-a")
    assert refused.refused and "verbatim" in refused.text
    assert "family=question_span rule='8 consecutive words of the question'" in caplog.text
    assert "theta" not in caplog.text.lower()  # the question's words never reach the log
    seven = "theta characteristic how many odd spin structures"
    assert not (await search(server, f"{seven} paper", "session-a")).refused
    assert not (await search(server, f"{eight} paper", "session-b")).refused  # another rollout's question
    assert backend.search.await_count == 2


@pytest.mark.parametrize(
    "url, rule",
    [
        (
            "https://web.archive.org/web/20240101000000/https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fsomeone%2Fpuzzles",
            "huggingface.co",
        ),
        ("https://web.archive.org/web/2024/hf.co/datasets/someone/puzzles", "hf.co"),
        ("https://webcache.googleusercontent.com/search?q=cache:scale.com/leaderboard/exam&strip=1", "scale.com"),
        ("https://proxy.example/fetch?url=https%253A%252F%252Fwww.safe.ai%252Fexam", "safe.ai"),  # double-encoded
        ("https://reader.example/https://openreview.net/pdf?id=46UGfq8kMI", "openreview.net/pdf?id=46UGfq8kMI"),
    ],
)
async def test_archive_and_proxy_wrappers_are_decoded_before_the_url_check(server, caplog, url, rule):
    backend = stub_backend(server, page="A page body")
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        response = await fetch(server, url)
    assert response.refused
    backend.extract.assert_not_awaited()
    assert f"rule={rule!r}" in caplog.text


async def test_fetched_markdown_is_capped_at_60000_characters_with_a_marker(server):
    stub_backend(server, page="x" * 70_000)
    response = await fetch(server, "https://example.com/long")
    assert response.truncated
    assert "x" * 60_000 in response.text and "x" * 60_001 not in response.text
    assert response.text.endswith("[Content truncated at 60,000 characters; 10,000 characters omitted]")
    stub_backend(server, page="y" * 60_000)
    response = await fetch(server, "https://example.com/exact")
    assert not response.truncated and "truncated" not in response.text


async def test_fetch_renders_title_and_final_url_and_strips_data_uris(server):
    page = "# Spin structures\n\nIntro ![figure](data:image/png;base64,iVBORw0KGgo=) end"
    stub_backend(server, page=page, final_url="https://example.com/final")
    response = await fetch(server, "https://example.com/start")
    assert response.text.startswith("Title: Spin structures\nURL: https://example.com/final\n\n")
    assert "base64" not in response.text and "[data URI removed]" in response.text


async def test_redirect_to_a_blocked_host_is_refused_after_the_fetch(server, caplog):
    stub_backend(server, page="content", final_url="https://scale.com/leaderboard")
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        response = await fetch(server, "https://short.example/abc")
    assert response.refused
    assert "family=domains rule='scale.com' where=fetched URL" in caplog.text
    assert not server._page_cache


async def test_cached_pages_are_rechecked_on_every_fetch(server):
    backend = stub_backend(server, page="Lecture notes on theta characteristics")
    assert not (await fetch(server, "https://example.com/notes")).refused
    server._blocklist.patterns_case_sensitive.append(re.compile(r"\bLecture notes\b"))
    second = await fetch(server, "https://example.com/notes")
    assert second.refused
    assert backend.extract.await_count == 1


def test_exactly_web_search_and_web_fetch_are_offered_over_mcp():
    pytest.importorskip("mcp")
    from nemo_gym.mcp_auto_exposure import maybe_auto_expose

    server = make_server(expose_tools_over_mcp=True)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    rpc = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        seeded = client.post("/seed_session", json={"question": QUESTION, "responses_create_params": {"input": []}})
        headers = rpc | {"X-NeMo-Gym-Session-Token": seeded.json()["mcp"]["headers"]["X-NeMo-Gym-Session-Token"]}
        client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"},
                },
            },
        )
        client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).json()
    tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
    assert sorted(tools) == ["web_fetch", "web_search"]
    assert "60,000" in tools["web_fetch"]["description"] and "20" in tools["web_search"]["description"]
    assert server.mcp_allowed_tools_for_session({}) == ["web_search", "web_fetch"]


async def test_search_asks_tavily_for_twenty_results_and_returns_at_most_twenty(server, monkeypatch):
    response = http_response(
        200, {"results": [result(f"https://example.com/{i}", title=f"Result {i}") for i in range(25)]}
    )
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(module, "request", post)
    out = await search(server, "spin structures")
    assert out.result_count == 20
    assert "[20] Result 19" in out.text and "Result 20" not in out.text
    payload = json.loads(post.await_args.kwargs["data"])
    assert payload["max_results"] == 20
    assert payload["include_answer"] is False and payload["include_raw_content"] is False
    assert post.await_args.kwargs["url"] == "https://api.tavily.com/search"
    assert post.await_args.kwargs["headers"]["authorization"] == "Bearer key-one"
    response.release.assert_called_once()
    with pytest.raises(ValidationError, match="max_results"):
        make_server(max_results=21)


async def test_rejected_key_fails_over_and_provider_bodies_stay_private(server, monkeypatch):
    bad = http_response(401, {"detail": "secret-key-details"})
    good = http_response(200, {"results": []})
    post = AsyncMock(side_effect=[bad, good])
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    assert (await search(server, "spin structures")).text == "No results."
    assert [c.kwargs["headers"]["authorization"] for c in post.await_args_list] == ["Bearer key-one", "Bearer key-two"]
    bad.json.assert_not_awaited()
    post.reset_mock(side_effect=True)
    post.return_value = http_response(400, {"detail": "secret-key-details"})
    with pytest.raises(RuntimeError, match="Tavily HTTP 400") as error:
        await search(server, "spin structures")
    assert "secret" not in str(error.value)


def test_shipped_blocklist_is_the_pinned_upstream_snapshot(tmp_path):
    assert hashlib.sha256(BLOCKLIST_PATH.read_bytes()).hexdigest() == UPSTREAM_BLOCKLIST_SHA256
    data = json.loads(BLOCKLIST_PATH.read_text())
    data["domains"].remove("huggingface.co")
    tampered = tmp_path / "blocklist.json"
    tampered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not the pinned upstream snapshot"):
        make_server(blocklist_file_path=str(tampered))
    custom = make_server(blocklist_file_path=str(tampered), blocklist_sha256=None)
    assert "huggingface.co" not in custom._blocklist.domains
