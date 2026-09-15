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
