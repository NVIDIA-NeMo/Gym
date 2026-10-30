# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Sessions an Environment Server seeds and closes under its own resources_session_id.

The Environment Server assigns the id and registers the close before it seeds, so the
case these tests care most about is the one that motivated that design: a close that
arrives while, or before, the seed that opens the browser.
"""

import asyncio
import json
from base64 import b64decode
from itertools import count
from types import SimpleNamespace
from unittest.mock import MagicMock

import app as app_module
import pytest
from app import InteractiveBrowserConfig, InteractiveBrowserResourcesServer
from fastapi import HTTPException
from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.core import SESSION_LOST, EpisodeId, TaskId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract


class _Browser:
    """Stands in for a real browser: records what it was opened on and how often it was closed."""

    def __init__(self, open_delay_s: float = 0.0):
        self.url = None
        self.closed = 0
        self._open_delay_s = open_delay_s

    async def open(self, initial_url: str) -> None:
        if self._open_delay_s:
            await asyncio.sleep(self._open_delay_s)
        self.url = initial_url

    async def observe(self, max_elements: int) -> SimpleNamespace:
        return SimpleNamespace(url=self.url, title="")

    async def current_url(self) -> str:
        return self.url

    async def text(self) -> str:
        return ""

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture
def browsers(monkeypatch):
    """Every browser the server creates, in order. `open_delay_s` slows the next one's open."""
    created: list[_Browser] = []
    settings = {"open_delay_s": 0.0}

    def _create_backend(config, session_metadata=None):
        browser = _Browser(open_delay_s=settings["open_delay_s"])
        browser.session_metadata = session_metadata
        created.append(browser)
        return browser

    monkeypatch.setattr(app_module, "create_backend", _create_backend)
    created_settings = SimpleNamespace(created=created, settings=settings)
    return created_settings


def _server(**config) -> InteractiveBrowserResourcesServer:
    return InteractiveBrowserResourcesServer(
        config=InteractiveBrowserConfig(
            name="interactive_browser", host="0.0.0.0", port=8080, entrypoint="app.py", **config
        ),
        server_client=MagicMock(spec=ServerClient),
    )


_EPISODE = {"rollout_id": "rollout", "attempt": 0}
_TASK = {"taskset": "browser", "task_id": "task-1"}
_TASK_DATA = {"initial_url": "https://example.com/start", "verifier_metadata": {"url_contains": "done"}}


def _seed_json(session_id: str = "resources-session", episode=_EPISODE) -> dict:
    return {"resources_session_id": session_id, "episode_id": episode, "task_id": _TASK, "task_data": _TASK_DATA}


def _close_json(session_id: str = "resources-session", episode=_EPISODE) -> dict:
    return {"resources_session_id": session_id, "episode_id": episode}


class TestTypedSessionsOverHttp:
    def test_a_typed_seed_opens_the_browser_under_the_callers_id(self, browsers) -> None:
        server = _server()
        client = TestClient(server.setup_webserver())

        response = client.post("/seed_session", json=_seed_json())

        assert (response.status_code, response.json()["resources_session_id"]) == (200, "resources-session")
        assert len(browsers.created) == 1
        assert browsers.created[0].url == "https://example.com/start"
        assert server._session_id_to_state["resources-session"].gt == {"url_contains": "done"}

    def test_the_tool_calls_that_follow_find_the_same_browser(self, browsers) -> None:
        """The seed binds the cookie, so the agent's tool calls reach the browser the seed opened."""
        server = _server()
        client = TestClient(server.setup_webserver())
        client.post("/seed_session", json=_seed_json())

        finished = client.post("/browser_finish", json={"answer": "42"})

        assert finished.status_code == 200
        assert server._session_id_to_state["resources-session"].answer == "42"

    def test_a_retried_seed_for_the_same_episode_does_not_open_a_second_browser(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())

        first = client.post("/seed_session", json=_seed_json())
        second = client.post("/seed_session", json=_seed_json())

        assert first.json() == second.json()
        assert len(browsers.created) == 1

    def test_one_id_cannot_be_seeded_for_two_episodes(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json=_seed_json())

        refused = client.post("/seed_session", json=_seed_json(episode={"rollout_id": "other", "attempt": 0}))

        assert refused.status_code == 409, "a conflict no retry can fix is refused as final"
        assert "already bound" in refused.json()["detail"]
        assert len(browsers.created) == 1

    def test_closing_an_episode_that_never_reached_verify_releases_its_browser(self, browsers) -> None:
        """The case the Environment Server's cleanup exists for: the episode ended before verify."""
        server = _server()
        client = TestClient(server.setup_webserver())
        client.post("/seed_session", json=_seed_json())

        response = client.post("/close_session", json=_close_json())

        assert (response.status_code, response.json()) == (200, {"resources_session_id": "resources-session"})
        assert browsers.created[0].closed == 1
        assert "resources-session" not in server._session_id_to_state

    def test_closing_twice_closes_the_browser_once(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json=_seed_json())

        client.post("/close_session", json=_close_json())
        again = client.post("/close_session", json=_close_json())

        assert again.status_code == 200
        assert browsers.created[0].closed == 1

    @pytest.mark.parametrize(
        "tool,arguments",
        [
            ("browser_navigate", {"url": "https://example.com"}),
            ("browser_click", {"element_id": 0}),
            ("browser_type", {"element_id": 0, "text": "hi"}),
            ("browser_observe", None),
            ("browser_finish", {"answer": "42"}),
        ],
    )
    def test_a_tool_call_without_a_browser_is_a_lost_session(self, browsers, tool, arguments) -> None:
        """A tool call with no browser gets 404 session_lost, so the trajectory records it as a failed call."""
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json=_seed_json())
        client.post("/close_session", json=_close_json())

        response = client.post(f"/{tool}", json=arguments)

        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == SESSION_LOST

    def test_a_close_for_another_episode_is_refused(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json=_seed_json())

        refused = client.post("/close_session", json=_close_json(episode={"rollout_id": "other", "attempt": 0}))

        assert refused.status_code == 409
        assert "does not match" in refused.json()["detail"]
        assert browsers.created[0].closed == 0

    def test_a_close_for_another_episode_is_refused_after_the_close_too(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json=_seed_json())
        client.post("/close_session", json=_close_json())

        refused = client.post("/close_session", json=_close_json(episode={"rollout_id": "other", "attempt": 0}))

        assert refused.status_code == 409
        assert "does not match the closed" in refused.json()["detail"]

    def test_task_data_that_is_not_a_browser_task_is_refused_as_final(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())
        body = _seed_json()
        body["task_data"] = {"initial_url": 5}

        refused = client.post("/seed_session", json=body)

        assert refused.status_code == 422, "a task no retry can fix is refused as final"
        assert "task_data is not an interactive_browser task" in refused.json()["detail"]
        assert browsers.created == []

    def test_a_refused_seed_leaves_the_clients_cookie_alone(self, browsers, monkeypatch: MonkeyPatch) -> None:
        """A refused seed must not rebind the client to the session it was refused, which another episode owns."""
        # Session middleware may renew the signature without changing the client's session.
        timestamps = count(1_000_000)
        monkeypatch.setattr(TimestampSigner, "get_timestamp", lambda self: next(timestamps))
        server = _server()
        app = server.setup_webserver()
        owner, other = TestClient(app), TestClient(app)
        assert owner.post("/seed_session", json=_seed_json("owner")).status_code == 200
        assert (
            other.post(
                "/seed_session", json=_seed_json("other", episode={"rollout_id": "other", "attempt": 0})
            ).status_code
            == 200
        )
        cookie_key = server.get_session_middleware_key()
        signer = TimestampSigner(cookie_key)
        bad_task = _seed_json("third", episode={"rollout_id": "other", "attempt": 0})
        bad_task["task_data"] = {"initial_url": 5}

        for body, status_code in (
            (_seed_json("owner", episode={"rollout_id": "other", "attempt": 0}), 409),
            (bad_task, 422),
        ):
            response = other.post("/seed_session", json=body)
            assert response.status_code == status_code
            session = json.loads(b64decode(signer.unsign(other.cookies[cookie_key])))
            assert session == {SESSION_ID_KEY: "other"}

        # The tool call still reaches this client's own browser, not the owner's.
        finished = other.post("/browser_finish", json={"answer": "42"})
        assert finished.status_code == 200
        assert server._session_id_to_state["other"].answer == "42"
        assert server._session_id_to_state["owner"].answer is None

    def test_a_malformed_close_is_a_validation_error_not_a_silent_success(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())

        response = client.post("/close_session", json={"resources_session_id": "resources-session"})

        assert response.status_code == 422


class TestLegacySessionsStillWork:
    def test_an_agent_run_seed_still_gets_the_empty_response(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())

        response = client.post("/seed_session", json={"initial_url": "https://example.com/"})

        assert (response.status_code, response.json()) == (200, {})
        assert len(browsers.created) == 1

    def test_an_empty_close_releases_the_cookie_session(self, browsers) -> None:
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json={"initial_url": "https://example.com/"})

        response = client.post("/close_session")

        assert (response.status_code, response.json()) == (200, {"closed": True})
        assert browsers.created[0].closed == 1


def _request() -> SimpleNamespace:
    return SimpleNamespace(session={})


def _typed_seed(session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(**_EPISODE),
        task_id=TaskId(**_TASK),
        task_data=_TASK_DATA,
    )


def _typed_close(session_id: str = "resources-session") -> dict:
    return ResourcesCloseSessionRequest(resources_session_id=session_id, episode_id=EpisodeId(**_EPISODE)).model_dump(
        mode="json"
    )


class TestCloseRacingTheSeed:
    """The Environment Server may close while the seed is still opening the browser, or before it lands."""

    def test_a_close_during_a_slow_seed_waits_and_then_releases_that_browser(self, browsers) -> None:
        browsers.settings["open_delay_s"] = 0.2
        server = _server()

        async def run():
            seed = asyncio.create_task(server.seed_session(_request(), _typed_seed()))
            await asyncio.sleep(0.05)  # the browser is still opening
            await server.close_resources_session(_request(), _typed_close())
            await seed

        asyncio.run(run())

        assert len(browsers.created) == 1
        assert browsers.created[0].closed == 1
        assert "resources-session" not in server._session_id_to_state

    def test_a_seed_that_lands_after_its_close_opens_nothing(self, browsers) -> None:
        """The client gave up and closed; the late seed must not open a browser nobody will close."""
        server = _server()

        async def run():
            await server.close_resources_session(_request(), _typed_close())
            with pytest.raises(HTTPException, match="already closed") as refused:
                await server.seed_session(_request(), _typed_seed())
            assert refused.value.status_code == 409

        asyncio.run(run())

        assert browsers.created == []


def test_typed_sessions_follow_the_session_contract(browsers) -> None:
    from test_verify_reporting import _body

    server = _server()
    check_resources_session_contract(
        server.setup_webserver(),
        _typed_seed("contract-session"),
        keeps_state=True,
        holds_state=lambda resources_session_id: (
            resources_session_id in server._session_id_to_state or resources_session_id in server._typed_identity
        ),
        verify_body=_body().model_dump(mode="json"),
    )
    assert all(browser.closed == 1 for browser in browsers.created)


def test_more_than_one_worker_is_refused() -> None:
    """Browsers live in this process; a second worker would get tool calls for browsers it lacks."""
    with pytest.raises(ValueError, match="num_workers=1"):
        _server(num_workers=2)


def test_verify_after_close_is_a_masked_lost_session(browsers) -> None:
    """A verify that arrives after the close has nothing to score, so the sample is masked rather than scored 0."""
    from test_verify_reporting import _body

    server = _server()

    async def run():
        request = _request()
        await server.seed_session(request, _typed_seed())
        await server.close_resources_session(request, _typed_close())
        return await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "resources-session"}), _body())

    response = asyncio.run(run())

    assert (response.reward, response.mask_sample, response.failure_kind) == (0.0, True, SESSION_LOST)
    assert response.failure_reason == "browser session not found at verify"


def test_a_lost_verify_ignores_response_fields_the_caller_sent(browsers) -> None:
    """The request allows extra fields; a caller's reward or mask must not override the lost-session verdict."""
    from test_verify_reporting import _body

    sent = {"reward": 1.0, "mask_sample": False, "failure_kind": "judge_error", "failure_reason": "sent by caller"}
    response = asyncio.run(
        _server().verify(SimpleNamespace(session={SESSION_ID_KEY: "resources-session"}), _body(**sent))
    )

    assert (response.reward, response.mask_sample, response.failure_kind) == (0.0, True, SESSION_LOST)
    assert response.failure_reason == "browser session not found at verify"


def test_mcp_tool_calls_reach_the_browser_opened_under_the_callers_id(browsers) -> None:
    """With MCP exposure on, the token minted at seed must name the caller's id, not a fresh one."""
    from nemo_gym.mcp_auto_exposure import NEMO_GYM_MCP_SESSION_TOKEN_HEADER, maybe_auto_expose

    server = _server(expose_tools_over_mcp=True)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    rpc_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    with TestClient(app) as client:
        seed = client.post("/seed_session", json=_seed_json())
        token = seed.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        client.post(
            "/mcp",
            headers=rpc_headers,
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
        client.post("/mcp", headers=rpc_headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        called = client.post(
            "/mcp",
            headers={**rpc_headers, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "browser_finish", "arguments": {"answer": "42"}},
            },
        ).json()

    assert called["result"].get("isError") is not True, called
    assert len(browsers.created) == 1
    assert server._session_id_to_state["resources-session"].answer == "42"


def test_the_browser_is_tagged_with_the_episode_the_training_side_records(browsers) -> None:
    """The resources_session_id is minted inside the Environment Server; only the episode
    reaches the training record, so that is what a provider-side session must carry."""
    client = TestClient(_server().setup_webserver())

    client.post("/seed_session", json=_seed_json(episode={"rollout_id": "rollout-7", "attempt": 2}))

    assert browsers.created[0].session_metadata == {
        "rollout_session_id": "resources-session",
        "rollout_id": "rollout-7",
        "attempt": "2",
    }


def test_an_agent_run_seed_has_no_episode_to_tag(browsers) -> None:
    client = TestClient(_server().setup_webserver())

    client.post("/seed_session", json={"initial_url": "https://example.com/"})

    assert set(browsers.created[0].session_metadata) == {"rollout_session_id"}
