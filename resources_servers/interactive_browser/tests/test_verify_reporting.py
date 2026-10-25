# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""`verify` must say when it could not measure, instead of reporting a zero.

A rollout whose browser was lost is not a policy that solved nothing, and the two
are indistinguishable once both arrive as `reward=0.0`. These tests build and
serialize the real response, because the failure they guard against — a request
field colliding with a response field — only shows up at construction time.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from app import (
    BrowserVerifyRequest,
    FinishRequest,
    InteractiveBrowserConfig,
    InteractiveBrowserResourcesServer,
    _SessionState,
)

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


class _DeadBackend:
    """A browser that died mid-episode: every read raises, and close() still works."""

    def __init__(self):
        self.closed = 0

    async def current_url(self):
        raise RuntimeError("Target page, context or browser has been closed")

    async def close(self):
        self.closed += 1


class _LiveBackend:
    """A browser the episode can keep driving after it says it is done."""

    def __init__(self, url: str, text: str = ""):
        self._url = url
        self._text = text
        self.closed = 0

    def navigate(self, url: str, text: str = "") -> None:
        self._url, self._text = url, text

    async def current_url(self):
        return self._url

    async def text(self):
        return self._text

    async def observe(self, max_elements: int = 50):
        return SimpleNamespace(title="")

    async def close(self):
        self.closed += 1


def _server() -> InteractiveBrowserResourcesServer:
    return InteractiveBrowserResourcesServer(
        config=InteractiveBrowserConfig(name="interactive_browser", host="0.0.0.0", port=8080, entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )


def _request(session_id: str = "rollout-1"):
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


def _body(**extra) -> BrowserVerifyRequest:
    # A real response, not `model_construct`: `verify` spreads `body.model_dump()`, so the
    # response round-trips through validation exactly as it does in production.
    return BrowserVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
        response=NeMoGymResponse(
            id="resp-1",
            created_at=0,
            model="test-model",
            object="response",
            output=[],
            tool_choice="auto",
            parallel_tool_calls=True,
            tools=[],
        ),
        **extra,
    )


def test_a_scored_rollout_reports_no_failure():
    server = _server()
    backend = _LiveBackend("https://example.com/done")
    server._session_id_to_state["rollout-1"] = _SessionState(
        backend=backend, gt={"final_url": "https://example.com/done"}
    )

    dumped = asyncio.run(server.verify(_request(), _body())).model_dump()

    assert dumped["reward"] == 1.0
    assert dumped["failure_reason"] is None
    assert backend.closed == 1


def test_a_missing_session_is_reported_rather_than_scored_zero():
    dumped = asyncio.run(_server().verify(_request(), _body())).model_dump()

    assert dumped["reward"] == 0.0
    assert dumped["failure_reason"] is not None


def test_a_browser_that_died_before_scoring_does_not_abort_the_run():
    """Only JudgeError becomes a routed row, so raising here would end collection."""
    server = _server()
    backend = _DeadBackend()
    server._session_id_to_state["rollout-1"] = _SessionState(
        backend=backend, gt={"final_url": "https://example.com/done"}
    )

    dumped = asyncio.run(server.verify(_request(), _body())).model_dump()

    assert dumped["reward"] == 0.0
    assert "browser unreachable while scoring" in dumped["failure_reason"]
    assert backend.closed == 1
    assert "rollout-1" not in server._session_id_to_state


def test_an_unsupported_scoring_key_still_fails_loudly():
    """A dataset typo is a configuration error, not an infrastructure failure."""
    server = _server()
    server._session_id_to_state["rollout-1"] = _SessionState(backend=_LiveBackend("x"), gt={"finl_url": "typo"})

    with pytest.raises(ValueError, match="no supported scoring key"):
        asyncio.run(server.verify(_request(), _body()))


def test_a_request_carrying_a_response_field_does_not_break_construction():
    """`BrowserVerifyRequest` allows extras, so a caller can send `failure_reason`."""
    server = _server()
    server._session_id_to_state["rollout-1"] = _SessionState(
        backend=_LiveBackend("https://example.com/done"), gt={"final_url": "https://example.com/done"}
    )

    dumped = asyncio.run(
        server.verify(_request(), _body(failure_reason="sent by the caller", reward=99.0))
    ).model_dump()

    assert dumped["reward"] == 1.0
    assert dumped["failure_reason"] is None


def _finish(server, request, answer=""):
    return asyncio.run(server.browser_finish(request, FinishRequest(answer=answer)))


def test_a_rollout_that_keeps_browsing_after_finishing_keeps_its_reward():
    """`done` is a hint the agent loop does not enforce, so the episode runs on.

    Grading the live page would score whatever the model wandered onto after it
    committed, turning a solved task into a zero.
    """
    server = _server()
    backend = _LiveBackend("https://example.com/about.html")
    server._session_id_to_state["rollout-1"] = _SessionState(backend=backend, gt={"url_contains": "about.html"})
    request = _request()

    _finish(server, request)
    backend.navigate("https://example.com/index.html")  # the model double-checks

    dumped = asyncio.run(server.verify(request, _body())).model_dump()

    assert dumped["reward"] == 1.0


def test_an_episode_that_never_finishes_is_graded_on_the_live_page():
    server = _server()
    server._session_id_to_state["rollout-1"] = _SessionState(
        backend=_LiveBackend("https://example.com/about.html"), gt={"url_contains": "about.html"}
    )

    dumped = asyncio.run(server.verify(_request(), _body())).model_dump()

    assert dumped["reward"] == 1.0


def test_only_the_first_finish_is_graded():
    """A second finish must not let the model revise a committed answer."""
    server = _server()
    backend = _LiveBackend("https://example.com/about.html")
    server._session_id_to_state["rollout-1"] = _SessionState(backend=backend, gt={"url_contains": "about.html"})
    request = _request()

    _finish(server, request, answer="first")
    backend.navigate("https://example.com/index.html")
    _finish(server, request, answer="second")

    state = server._session_id_to_state["rollout-1"]
    assert state.answer == "first"
    assert "about.html" in state.finished_url


def _url_contains_reward(url: str, expected: str) -> float:
    server = _server()
    server._session_id_to_state["rollout-1"] = _SessionState(backend=_LiveBackend(url), gt={"url_contains": expected})
    return asyncio.run(server.verify(_request(), _body())).reward


@pytest.mark.parametrize(
    ("expected", "url"),
    [
        ("about.html", "data:text/html,about.html"),
        ("about.html", "data:text/html,<p>never visited</p>#about.html"),
        ("about.html", "about:blank#about.html"),
        ("about.html", "file:///srv/site/index.html#about.html"),
        ("about.html", "file:///srv/site/index.html?about.html"),
        ("about.html", "https://example.com/index.html?next=about.html"),
        ("iana.org", "https://example.com/iana.org"),
        ("iana.org", "https://example.com/?q=iana.org"),
        ("iana.org", "https://iana.org.example.com/"),
        ("iana.org", "file://iana.org/"),
        ("about.html", "https://about.html/"),
        ("example-domains", "https://example-domains/"),
        ("about.html", "https://example.com/not%2Fabout.html"),
    ],
)
def test_url_contains_is_not_met_by_a_page_the_target_does_not_name(expected, url):
    """`browser_navigate` takes any URL, so the target text alone does not show the page was reached."""
    assert _url_contains_reward(url, expected) == 0.0


@pytest.mark.parametrize(
    ("expected", "url"),
    [
        # The shipped rows (data/example.jsonl, data/example_remote.jsonl) on their target pages.
        ("about.html", "file:///srv/site/about.html"),
        ("form.html", "file:///srv/site/form.html"),
        ("index.html", "file:///srv/site/index.html"),
        ("iana.org", "https://www.iana.org/help/example-domains"),
        ("example-domains", "https://www.iana.org/help/example-domains"),
        # The same pages in other forms a browser can report.
        ("about.html", "https://example.com/about.html?ref=nav#team"),
        ("about.html", "HTTP://example.com/about.html"),
        ("about.html", "https://example.com/%61bout.html"),
        ("iana.org", "https://iana.org/"),
        ("iana.org", "https://www.iana.org./"),
        ("www.iana.org", "https://www.iana.org/"),
        ("example-domains", "https://www.iana.org/help/example-domains/"),
    ],
)
def test_url_contains_still_accepts_the_target_page(expected, url):
    assert _url_contains_reward(url, expected) == 1.0


@pytest.mark.parametrize(
    ("expected", "url", "reward"),
    [
        # A value that names part of a URL keeps whole-URL substring matching.
        ("search?q=shoes", "https://shop.example/search?q=shoes#results", 1.0),
        ("q=shoes", "https://shop.example/search?q=shoes", 1.0),
        ("#/settings", "https://app.example/#/settings", 1.0),
        ("/checkout", "https://shop.example/checkout-step-2", 1.0),
        ("https://www.iana.org/help", "https://www.iana.org/help/example-domains", 1.0),
        # A bare name no longer matches inside a query, a fragment or part of a segment.
        # Each needs the URL syntax it relies on: "q=shoes", "#/settings", "/checkout",
        # "iana.org" and "/data.csv".
        ("shoes", "https://shop.example/search?q=shoes", 0.0),
        ("settings", "https://app.example/#/settings", 0.0),
        ("checkout", "https://shop.example/checkout-step-2", 0.0),
        ("iana", "https://www.iana.org/", 0.0),
        ("data.csv", "https://files.example/data.csv", 0.0),
    ],
)
def test_url_contains_specs_written_for_whole_url_matching(expected, url, reward):
    """`url_contains` used to be a substring of the whole URL; these pin what changed."""
    assert _url_contains_reward(url, expected) == reward
