# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-free tests of the HLE web tools: the Tavily transport is stubbed, every rule uses upstream's blocklist."""

import asyncio
import hashlib
import json
import logging
import re
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ContentTypeError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract
from nemo_gym.server_runtime.base_server import refuse_unsupported_num_workers
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


def make_server(tavily_api_key: str = "key-one, key-two", **overrides) -> HLEWebToolsResourcesServer:
    config = HLEWebToolsResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="hle_tools",
        tavily_api_key=tavily_api_key,
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


NEUTRAL_QUESTION = "What is the boiling point of water at sea level in kelvin?"


async def search(server, query: str, session_id: str = "session-a"):
    # web_search fails closed on an unseeded session; tests of the other rules get a question unrelated to their queries.
    if session_id not in server._session_spans:
        await seed(server, {"question": NEUTRAL_QUESTION}, session_id)
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
        pytest.param("https://example.com/s", "Quiz answers" + " " * 5_000 + "| Site", False, id="padded-separator"),
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
    await seed(server, {"question": NEUTRAL_QUESTION}, "session-b")
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
    "query, family",
    [
        # Spellings search engines fold back to the question's words.
        (
            "ｈｏｗ ｍａｎｙ ｏｄｄ ｓｐｉｎ ｓｔｒｕｃｔｕｒｅｓ ａｄｍｉｔ ａ ｎｏｎｖａｎｉｓｈｉｎｇ",  # full-width
            "question_span",
        ),
        ("how many odd sp\u200bin structures admit a nonvanishing", "question_span"),  # zero-width space
        ("how\u2060many\u2060odd\u2060spin\u2060structures\u2060admit\u2060a\u2060nonvanishing", "question_span"),
        ("how many odd spi\u0301n structures admit a nonvanishing", "question_span"),  # combining mark
        ("how_many_odd_spin_structures_admit_a_nonvanishing", "question_span"),  # underscore-joined
        ("humanity\u00ads last exam answers", "url_patterns"),  # soft hyphen inside a blocked phrase
        # A zero-width space between words and a soft hyphen inside one, read as word segmentation reads them.
        ("How\u200bmany odd spin structures admit a no\u00adnvanishing", "question_span"),
        ("H\u00adLE\u200bleaderboard", "patterns_case_sensitive"),
        # A Hangul filler (default-ignorable, category Lo) inside a word.
        ("How many odd spin structures admit a no\u3164nvanishing", "question_span"),
        ("Hum\u3164anity's Last Exam", "url_patterns"),
    ],
    ids=[
        "full-width",
        "zero-width-space",
        "word-joiner-separated",
        "combining-mark",
        "underscore-joined",
        "soft-hyphen-pattern",
        "mixed-format-characters",
        "mixed-format-pattern",
        "hangul-filler",
        "hangul-filler-pattern",
    ],
)
async def test_presentation_variants_of_a_blocked_query_are_refused(server, caplog, query, family):
    backend = stub_backend(server, results=[result("https://example.com/ok")])
    await seed(server, {"question": QUESTION})
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        response = await search(server, query)
    assert response.refused and f"family={family} " in caplog.text
    backend.search.assert_not_awaited()


async def test_reseeding_a_seeded_session_with_another_question_is_refused(server, caplog):
    """A session keeps the question it was seeded with: a retried seed (same question) is served again, a seed
    that would swap or drop the question answers 409, so the span check cannot be replaced from the plain route
    by a caller holding the session. A fresh session seeded by the same caller is not caught here (README)."""
    backend = stub_backend(server, results=[result("https://example.com/ok")])
    await seed(server, {"question": QUESTION}, "session-a")
    await seed(server, {"question": QUESTION.upper() + "  "}, "session-a")  # a retry: same words, other spelling
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        with pytest.raises(module.ResourcesSessionConflictError) as swapped:
            await seed(server, {"question": NEUTRAL_QUESTION}, "session-a")
        with pytest.raises(module.ResourcesSessionConflictError):
            await seed(server, {"responses_create_params": {"input": []}}, "session-a")
    assert swapped.value.status_code == 409
    assert "refused to re-seed session session-a with a different question" in caplog.text
    assert "refused to re-seed session session-a with no question" in caplog.text
    eight = "Theta characteristic: how many odd spin structures admit paper"
    assert (await search(server, eight, "session-a")).refused  # the original question still governs the session
    assert not (await search(server, "boiling point of water", "session-a")).refused
    assert backend.search.await_count == 1


async def test_search_from_a_session_without_a_seeded_question_is_refused(caplog):
    """The plain POST route stays served next to /mcp: a call without the rollout's cookie or token lands in a
    fresh session with no question, which must fail closed rather than skip the verbatim-span rule."""
    server = make_server()
    backend = stub_backend(server, results=[result("https://example.com/ok")])
    app = server.setup_webserver()
    eight = "Theta characteristic: how many odd spin structures admit paper"
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        client.post("/seed_session", json={"question": QUESTION, "responses_create_params": {"input": []}})
        assert client.post("/web_search", json={"query": eight}).json()["refused"]  # the rollout's own session
        client.cookies.clear()
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            bare = client.post("/web_search", json={"query": eight}).json()
    assert bare["refused"] and "seeded with its question" in bare["text"]
    assert "family=unseeded_session rule='require_seeded_question' where=session" in caplog.text
    # A question with no words cannot drive the span rule, so its seed leaves the session unseeded, whether the
    # question comes from the row's field or from the last user message.
    await seed(server, {"question": "?!"}, "wordless")
    await seed(server, {"responses_create_params": {"input": [{"role": "user", "content": "?!"}]}}, "wordless-message")
    for session_id in ("wordless", "wordless-message"):
        assert (await server.web_search(tool_request(session_id), WebSearchRequest(query=eight))).refused
    backend.search.assert_not_awaited()
    # Opting out restores the previous behaviour: an unseeded session is served without the span rule.
    lenient = make_server(require_seeded_question=False)
    stub_backend(lenient, results=[result("https://example.com/ok")])
    served = await lenient.web_search(tool_request("never-seeded"), WebSearchRequest(query=eight))
    assert not served.refused


def test_typed_sessions_follow_the_resources_session_contract():
    """An Environment Server's typed seed binds the session to its episode and task, and a close fences the ID
    against a delayed seed of the same session for the retry window. A seed of the same ID for another task, or
    a close naming another episode, leaves an open session, question and binding, untouched."""
    server = make_server()
    app = server.setup_webserver()
    check_resources_session_contract(
        app,
        ResourcesSeedSessionRequest(
            resources_session_id="typed-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="hle_diamond", task_id="0"),
            task_data={"question": QUESTION},
        ),
        keeps_state=True,
        holds_state=lambda session_id: session_id in server._session_spans or session_id in server._session_identity,
    )
    live = ResourcesSeedSessionRequest(
        resources_session_id="live-session",
        episode_id=EpisodeId(rollout_id="rollout-a", attempt=0),
        task_id=TaskId(taskset="hle_diamond", task_id="1"),
        task_data={"question": QUESTION},
    )
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        assert client.post("/seed_session", json=live.model_dump(mode="json")).status_code == 200
        spans, identity = server._session_spans["live-session"].digest, server._session_identity["live-session"]
        other_task = live.model_copy(update={"task_id": TaskId(taskset="hle_diamond", task_id="2")})
        assert client.post("/seed_session", json=other_task.model_dump(mode="json")).status_code == 409
        assert server._session_spans["live-session"].digest == spans
        assert server._session_identity["live-session"] == identity
        other = {"resources_session_id": "live-session", "episode_id": {"rollout_id": "rollout-b", "attempt": 0}}
        assert client.post("/close_session", json=other).status_code == 409
    assert "live-session" in server._session_spans and "live-session" in server._session_identity


async def test_sessions_past_the_cap_are_forgotten_with_a_warning(server, monkeypatch, caplog):
    """Past the cap the least recently seeded session loses its question and its episode binding together, the
    loss is logged, since its later searches are refused as unseeded, and its typed ID is fenced."""
    monkeypatch.setattr(module, "MAX_SESSION_QUESTIONS", 1)
    for session_id in ("first", "second"):
        typed = {
            "resources_session_id": session_id,
            "episode_id": {"rollout_id": session_id},
            "task_id": {"taskset": "hle_diamond", "task_id": session_id},
            "task_data": {"question": QUESTION},
        }
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            await seed(server, typed, session_id)
    assert "forgot the question of session first" in caplog.text
    assert list(server._session_spans) == list(server._session_identity) == ["second"]
    # The forgotten ID is fenced: a seed binding it to another episode answers 409 rather than binding it.
    with pytest.raises(module.ResourcesSessionConflictError):
        await seed(
            server,
            {
                "resources_session_id": "first",
                "episode_id": {"rollout_id": "another"},
                "task_id": {"taskset": "hle_diamond", "task_id": "first"},
                "task_data": {"question": QUESTION},
            },
            "first",
        )


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


OVERFLOW_DECOYS = "&".join(f"u{i}=https://example.org/{i}" for i in range(63))
# Four timestamped Wayback layers around a percent-encoded page: the deepest nesting MAX_URL_DECODE_ROUNDS allows.
NESTED_WAYBACK = "".join(f"https://web.archive.org/web/2024010100000{i}/" for i in range(4)) + "https://example.org/x%20y"


@pytest.mark.parametrize(
    "url, family",
    [
        # Expansion that stops with candidates unexplored fails closed.
        (f"https://reader.example/fetch?{OVERFLOW_DECOYS}&url=https://scale.com/leaderboard", "malformed_url"),
        ("https://example.com/%2525252Famphtml/", "url_patterns"),  # the fourth decoded form is checked
        ("https://example.com/%252525252Famphtml/", "malformed_url"),  # a fifth is never certified
        # Shapes fetchers read differently from urlsplit.
        ("https://scale.com\\.example.org/leaderboard", "malformed_url"),
        ("https://arxiv.org\t/abs/2605.02442", "malformed_url"),
        ("https://@arxiv.org/abs/2605.02442", "malformed_url"),
        ("https://reader.example/?u=https://scale.com%5C.example.org/x", "domains"),
        ("https://ok.example/go?u=https://[::1]@scale.com/leaderboard", "malformed_url"),  # unparseable inside
        # url_prefixes and serp_url_patterns on the canonical form.
        ("https://arxiv.org:443/abs/2605.02442", "url_prefixes"),
        ("https://arxiv.org./abs/2605.02442", "url_prefixes"),
        ("https://arxiv.org/x/../abs/2605.02442", "url_prefixes"),
        ("https://arxiv.org//abs/2605.02442", "url_prefixes"),
        ("https://arxiv.org/abs/x//%2e%2e/2605.02442", "url_prefixes"),  # slash runs merged before dots (nginx)
        ("https://www.google.com:443/search?q=x", "serp_url_patterns"),
        ("https://openreview.net/pdf?&id=46UGfq8kMI", "url_prefixes"),  # query pairs in any order
        ("https://openreview.net/pdf?x=1&id=46UGfq8kMI", "url_prefixes"),
        # Hosts are mapped as UTS #46 fetchers map them: U+1F130 is "a" to a browser and to yarl.
        ("https://sc\U0001f130le.com/leaderboard", "domains"),
        # Every archive.org host is a wrapper, and https:/ or https:/// in the remainder still names the target.
        ("https://web.archive.org/web/2024/https:/scale.com/leaderboard", "domains"),
        ("https://web.archive.org/web/2024/https:///scale.com/leaderboard", "domains"),
        ("https://wayback.archive.org/web/2024/scale.com/leaderboard", "domains"),
        # A Wayback path with no timestamp names its target in the first segment.
        ("https://web.archive.org/web/scale.com/leaderboard", "domains"),
        ("https://web.archive.org/web/scale.com", "domains"),
        # A wrapper's remainder is a URL however its host is spelled; a query value may end its host with a dot.
        ("https://web.archive.org/web/2024/scale.com./leaderboard", "domains"),
        ("https://web.archive.org/web/2024/x@scale.com/leaderboard", "domains"),
        ("https://ok.example/?u=scale.com./leaderboard", "domains"),
        # Ordinary URLs, redirects and query strings stay allowed.
        ("https://www.google.com/url?q=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSpin_structure&sa=D&ust=17", None),
        ("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSpin_structure&rut=ab", None),
        ("https://example.org/a/./b/../c?next=%2Fhome%3Fq%3Da%2520b&utm_source=x#top", None),
        ("https://web.archive.org/web/2024/https://en.wikipedia.org/wiki/Spin_structure", None),
        (NESTED_WAYBACK, None),  # a timestamp is read as one only, so it spends no candidates or decode rounds
        ("https://medium.com/@someone/notes-on-spin-structures-123", None),
        ("http://example.org:8080/path?x=1", None),
    ],
)
async def test_url_spellings_of_a_blocked_page_are_refused_before_the_fetch(server, caplog, url, family):
    backend = stub_backend(server, page="A page body")
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        response = await fetch(server, url)
    assert response.refused is (family is not None)
    assert backend.extract.await_count == (0 if family else 1)
    if family:
        assert f"family={family} " in caplog.text


@pytest.mark.parametrize(
    "query, family",
    [
        ("(site:example.org OR site:scale.com) leaderboard", "domains"),
        ("site:example.org|site:scale.com leaderboard", "domains"),
        ("site:scale.com, leaderboard", "domains"),
        ("site:.huggingface.co puzzles", "domains"),
        ("site:example..org puzzles", "malformed_url"),  # no hostname: refused, not skipped
        ("site:huggingface.co! puzzles", "domains"),  # trailing punctuation dropped
        ("site:exa_mple!.org puzzles", "malformed_url"),  # a character no hostname has
        ("site:“scale.com” leaderboard", "domains"),  # typographic quotes read like ASCII ones
        ("site:scale.com” leaderboard", "domains"),
        ("site:((scale.com)) leaderboard", "domains"),  # stacked openers
    ],
)
async def test_site_operands_are_read_as_hostnames(server, caplog, query, family):
    backend = stub_backend(server, results=[result("https://example.com/ok")])
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        response = await search(server, query)
    assert response.refused
    assert f"family={family} " in caplog.text and "where=site: operator" in caplog.text
    backend.search.assert_not_awaited()


@pytest.mark.parametrize(
    "url, page",
    [
        # 3,898 characters: under the cap, 190 distinct embedded URLs, more candidates than are checked
        ("https://a.example/" + "".join(f"https://e{i}.example/" for i in range(190)), None),
        ("https://a.example/" + "https://" * 1_250, None),  # 10,018 characters: over the cap
        ("https://example.com/heading", "# a" + " " * 100_000 + "b\n\nbody"),  # no provider title: heading parsed
    ],
    ids=["url-under-cap", "url-over-cap", "heading-100k"],
)
async def test_long_urls_and_headings_are_handled_in_linear_time(server, url, page):
    await warm_page_checks(server)
    backend = stub_backend(server, page=page)
    started = time.perf_counter()
    response = await fetch(server, url)
    assert time.perf_counter() - started < 1.0
    assert response.refused is (page is None)
    assert backend.extract.await_count == (0 if page is None else 1)
    if page is not None:
        assert response.text.startswith("Title: a ")


@pytest.mark.parametrize("title", ["x" + " " * 20_000 + "y", "a | " * 5_000], ids=["blank-run", "separators"])
async def test_long_result_titles_are_checked_in_linear_time(server, title):
    stub_backend(server, results=[result("https://example.com/ok", title=title)])
    started = time.perf_counter()
    response = await search(server, "spin structures")
    assert time.perf_counter() - started < 1.0
    assert response.result_count == 1 and len(response.text) < 2_000  # the title is shown cut


async def warm_page_checks(server) -> None:
    """One small fetch, so a timed fetch after it does not include starting the page-check processes."""
    stub_backend(server, page="warm-up")
    await fetch(server, "https://example.com/warm-up")


async def test_page_checks_do_not_stall_the_event_loop(server):
    """Python's re keeps the GIL for a whole search, so a slow scan of an adversarial page (about 1.2 s here) in a
    thread would hold up every other request; the page work must leave the event loop free."""
    await warm_page_checks(server)
    stub_backend(server, page="抱抱脸" * 100_000)
    gaps = []

    async def heartbeat():
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    response = await fetch(server, "https://example.com/slow")
    beat.cancel()
    assert not response.refused and gaps
    assert max(gaps) < 0.5


async def test_fetched_markdown_is_capped_at_60000_characters_with_a_marker(server):
    stub_backend(server, page="x" * 70_000)
    response = await fetch(server, "https://example.com/long")
    assert response.truncated
    assert "x" * 60_000 in response.text and "x" * 60_001 not in response.text
    assert response.text.endswith("[Content truncated at 60,000 characters; 10,000 characters omitted]")
    stub_backend(server, page="# " + "t" * 100_000 + "\n\nbody")
    titled = await fetch(server, "https://example.com/titled")
    assert titled.text.startswith("Title: " + "t" * 1_000 + "…\nURL: ")  # the title is cut too
    assert len(titled.text) < 60_000 + 1_200
    stub_backend(server, page="y" * 60_000)
    response = await fetch(server, "https://example.com/exact")
    assert not response.truncated and "truncated" not in response.text


async def test_fetch_renders_title_and_final_url_and_strips_data_uris(server):
    page = (
        "# Spin structures #\n\nIntro ![figure](data:image/png;base64,iVBORw0KGgo=) end "
        "![svg](data:image/svg+xml,%3Csvg%3E%3C/svg%3E) "
        "![param](data:image/png;charset=utf-8;base64,AAAA) "
        "![untyped](data:;base64,iVBORw0KGgo=) "
        "bare data:image/png;base64,AAAA followed by prose ![link](https://example.com/a.png)"
    )
    backend = stub_backend(server, page=page, final_url="https://example.com/final")
    fresh = await fetch(server, "https://example.com/start")
    cached = await fetch(server, "https://example.com/start")
    assert backend.extract.await_count == 1 and cached.text == fresh.text
    assert fresh.text == (
        "Title: Spin structures\nURL: https://example.com/final\n\n# Spin structures #\n\n"
        "Intro ![figure]([data URI removed]) end ![svg]([data URI removed]) ![param]([data URI removed]) "
        "![untyped]([data URI removed]) bare [data URI removed] followed by prose ![link](https://example.com/a.png)"
    )


async def test_fetched_content_is_checked_whole_before_it_is_cut():
    """The content families see the provider's whole page: a blocked span past every cut still refuses it.
    A page above max_page_chars is refused unread, and one above max_cached_page_chars is served whole."""
    server = make_server(max_page_chars=600_000, max_cached_page_chars=100)
    stub_backend(server, page="x" * 500_001 + " HLE answers")
    refused = await fetch(server, "https://example.com/tail")
    assert refused.refused and "blocked pattern" in refused.text and not server._page_cache
    stub_backend(server, page="x" * 600_001)
    oversized = await fetch(server, "https://example.com/huge")
    assert oversized.refused and "larger than the policy can check" in oversized.text
    backend = stub_backend(server, page="y" * 200)
    for _ in range(2):
        served = await fetch(server, "https://example.com/clean")
        assert not served.truncated and "y" * 200 in served.text
    assert backend.extract.await_count == 2 and not server._page_cache  # served whole, not cached


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
    from nemo_gym.server_runtime import maybe_auto_expose

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


@pytest.mark.parametrize("status, backoff", [(401, []), (429, [(1,)])])
async def test_rejected_key_fails_over_and_provider_bodies_stay_private(server, monkeypatch, status, backoff):
    """A rejected key (401) moves to the next key at once, a transient failure (429) after a backoff; every
    response is released, whatever its status."""
    bad = http_response(status, {"detail": "secret-key-details"})
    good = http_response(200, {"results": []})
    post = AsyncMock(side_effect=[bad, good])
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    assert (await search(server, "spin structures")).text == "No results."
    assert [c.kwargs["headers"]["authorization"] for c in post.await_args_list] == ["Bearer key-one", "Bearer key-two"]
    assert [c.args for c in module.sleep.await_args_list] == backoff
    bad.release.assert_called_once()
    good.release.assert_called_once()
    bad.json.assert_not_awaited()
    post.reset_mock(side_effect=True)
    post.return_value = http_response(400, {"detail": "secret-key-details"})
    with pytest.raises(RuntimeError, match="Tavily HTTP 400") as error:
        await search(server, "spin structures")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "payload, search_text",
    [
        ([], "Search failed: the search provider returned a malformed response."),
        ({"results": None}, "Search failed: the search provider returned a malformed response."),
        ({"results": [1, None, "x"]}, "No results."),
        (
            {"results": [{"url": "https://example.org/a", "title": 5, "content": {"x": 1}, "raw_content": ["x"]}]},
            "[1] \n    URL: https://example.org/a\n    Snippet: ",
        ),
        # A 200 whose body does not decode: a gateway's HTML page, a cut-off JSON body.
        (ContentTypeError(MagicMock(), ()), "Search failed: the search provider returned a malformed response."),
        (
            json.JSONDecodeError("Expecting value", "{", 1),
            "Search failed: the search provider returned a malformed response.",
        ),
    ],
    ids=["body-not-object", "results-null", "items-not-objects", "fields-not-strings", "body-html", "body-cut-off"],
)
async def test_malformed_tavily_payloads_become_tool_errors(server, monkeypatch, payload, search_text):
    if isinstance(payload, Exception):  # a 200 whose body does not decode, through the real transport
        response = http_response(200)
        response.json = AsyncMock(side_effect=payload)
        monkeypatch.setattr(module, "request", AsyncMock(return_value=response))
    else:
        server._backend = MagicMock(search=AsyncMock(return_value=payload), extract=AsyncMock(return_value=payload))
    found = await search(server, "spin structures")
    assert (found.text, found.refused) == (search_text, False)
    fetched = await fetch(server, "https://example.org/a")
    assert (fetched.text, fetched.refused) == ("Fetch failed for https://example.org/a", False)


class KeyPoolSpin(BaseException):
    """Raised by the alarm; a BaseException so the transport's ``except TimeoutError`` cannot swallow it."""


@pytest.fixture
def key_pool_deadline():
    # An asyncio timeout cannot interrupt a synchronous spin in key selection; SIGALRM can, so the test fails.
    def expire(*_):
        raise KeyPoolSpin("key selection did not terminate")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(5)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(
    "keys, status, tried",
    [
        ("key-one", 401, ["key-one"]),
        ("key-one", 432, ["key-one"]),
        ("key-one", 433, ["key-one"]),
        ("key-one, key-one", 401, ["key-one"]),
        ("key-one, key-two", 401, ["key-one", "key-two"]),
    ],
)
async def test_every_key_rejected_ends_the_call_without_backoff(monkeypatch, key_pool_deadline, keys, status, tried):
    post = AsyncMock(side_effect=lambda *a, **kw: http_response(status, {"detail": "secret-key-details"}))
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    with pytest.raises(RuntimeError, match=f"Tavily HTTP {status} on /search: every configured key") as error:
        await search(make_server(tavily_api_key=keys), "spin structures")
    assert "secret" not in str(error.value)
    assert [c.kwargs["headers"]["authorization"] for c in post.await_args_list] == [f"Bearer {k}" for k in tried]
    module.sleep.assert_not_awaited()


async def test_rejected_keys_do_not_spend_the_retry_budget(monkeypatch, key_pool_deadline):
    """With two attempts, a rejected key and then a transient failure on the only accepted key still leave one
    retry, on that same key: the rejection spends no attempt and key selection wraps round to the current key."""
    post = AsyncMock(side_effect=[http_response(401), http_response(429), http_response(200, {"results": []})])
    monkeypatch.setattr(module, "request", post)
    monkeypatch.setattr(module, "sleep", AsyncMock())
    server = make_server(tavily_api_key="key-one, key-two", max_http_attempts=2)
    assert (await search(server, "spin structures")).text == "No results."
    tried = [c.kwargs["headers"]["authorization"] for c in post.await_args_list]
    assert tried == ["Bearer key-one", "Bearer key-two", "Bearer key-two"]
    module.sleep.assert_awaited_once()


def test_a_launch_with_several_workers_is_refused():
    with pytest.raises(ValueError, match="hle_tools needs num_workers=1: seeded questions"):
        refuse_unsupported_num_workers(make_server(num_workers=2))


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
