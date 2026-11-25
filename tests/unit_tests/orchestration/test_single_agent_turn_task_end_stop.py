# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""end_on_task_state for an agent that runs its own loop: once the task has ended, the run is stopped as completed,
and the call that ended the task is released only after the run's activation has answered."""

import asyncio
from types import SimpleNamespace
from typing import Any

import orjson
import pytest
from aiohttp import ClientResponseError
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

import environment_servers.single_agent_turn.app as single_agent_turn_app
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.config.global_config import ROLLOUT_STOP_AGENT_TYPES
from nemo_gym.server_runtime.base_resources_server import (
    RESOURCES_TASK_STATE_PATH,
    RESOURCES_TASK_STATE_RELEASE_PATH,
)
from tests.helpers.single_agent_turn import _request


ACTIVATION_PATH = "/ng-rollout/rollout-a2/v1/responses"
STOP_PATH = "/rollouts/rollout-a2/stop"
SESSION_LOST_BODY = orjson.dumps({"detail": {"failure_kind": "session_lost", "message": "This session has ended."}})
WATCH = {"watch": "true"}
WATCH_AND_WAIT = {"watch": "true", "wait_s": "30"}


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Response:
    ok = True
    status = 200

    def __init__(self, body: dict, cookies: dict[str, str] | None = None) -> None:
        self.body = orjson.dumps(body)
        self.headers: dict[str, str] = {}
        self.cookies = {name: _Cookie(value) for name, value in (cookies or {}).items()}

    async def read(self) -> bytes:
        return self.body


class _ErrorResponse:
    """A reply with an HTTP error status, which raises as aiohttp's does."""

    ok = False

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        self.headers: dict[str, str] = {}
        self.cookies: dict[str, _Cookie] = {}

    async def read(self) -> bytes:
        return self.body

    def raise_for_status(self) -> None:
        url = "http://server:8000/route"
        request_info = SimpleNamespace(url=url, method="POST", headers={}, real_url=url)
        raise ClientResponseError(request_info, (), status=self.status, message="error", headers={})


def _reply(status: str = "completed") -> dict[str, Any]:
    """The agent's answer to the episode's one activation: a run that called complete_task."""
    return {
        "id": "response",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "tools": [],
        "status": status,
        "metadata": {"ng_rollout_stop": '{"reason":"task_ended","completed":true}'},
        "output": [
            {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call1",
                "name": "company_user_complete_task",
                "arguments": "{}",
                "status": "completed",
            }
        ],
    }


class _Client(ServerClient):
    """The resources server and an agent whose run goes on until a stop ends it (or ends alone).

    The task has ended from the waiting read numbered ``ends_at_read`` on; ``waiting_reply`` answers the waiting reads
    instead, and ``first_reply`` the first read, made before the run. The stop answers ``stop_status`` (with
    ``stopped: stop_stopped``) after ``stop_delay`` seconds, and ends the run unless ``stop_ends_run`` is false; with
    ``release_ends_run`` the release ends it. ``events`` keeps the order of the calls and of the activation's answer,
    ``control`` the calls sent over the reserved control pool, and ``verified`` the bodies ``/verify`` got.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    events: list[str]
    reads: list[dict[str, Any]]
    stops: list[dict[str, Any]]
    ends_at_read: int | None = 1
    run_ends_alone: bool = False
    stop_status: int = 200
    stop_ends_run: bool = True
    release_ends_run: bool = False
    waiting_reply: Any = None
    first_reply: Any = None
    stop_stopped: bool = True
    stop_delay: float = 0.0
    run_ended: Any = None
    read_times: list[float] = []
    control: list[str] = []
    verified: list[dict[str, Any]] = []

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response | _ErrorResponse:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = body.model_dump(mode="json")
        self.events.append(f"POST {url_path}")
        if kwargs.get("_control"):
            self.control.append(f"POST {url_path}")
        if url_path == "/seed_session":
            return _Response({"resources_session_id": body["resources_session_id"]}, {"resources": "seeded"})
        if url_path == "/v1/agent_sessions":
            return _Response({"agent_session_id": body["agent_session_id"]}, {"agent": "seeded"})
        if url_path == ACTIVATION_PATH:
            if not self.run_ends_alone:
                await self.run_ended.wait()
            self.events.append("activation answered")
            return _Response(_reply())
        if url_path == STOP_PATH:
            self.stops.append(body)
            if self.stop_delay:
                await asyncio.sleep(self.stop_delay)
                self.events.append("stop answered")
            if self.stop_status != 200:
                if self.stop_status == 404:
                    self.run_ended.set()
                return _ErrorResponse(self.stop_status, b'{"detail": "no"}')
            if self.stop_ends_run:
                self.run_ended.set()
            return _Response({"rollout_id": "rollout-a2", "stopped": self.stop_stopped})
        if url_path == RESOURCES_TASK_STATE_RELEASE_PATH:
            assert kwargs.get("cookies") == {"resources": "seeded"}
            if self.release_ends_run:
                self.run_ended.set()
            return _Response({"released": True})
        if url_path == "/v1/agent_sessions/close":
            return _Response({"agent_session_id": body["agent_session_id"]})
        if url_path == "/verify":
            self.verified.append(body)
            return _Response({**body, "reward": 1.0})
        if url_path == "/close_session":
            return _Response({"resources_session_id": body["resources_session_id"]})
        raise AssertionError(f"unexpected POST {server_name}{url_path}")

    async def get(self, server_name: str, url_path: str, **kwargs) -> _Response | _ErrorResponse:
        assert (server_name, url_path) == ("resources", RESOURCES_TASK_STATE_PATH)
        assert kwargs.get("cookies") == {"resources": "seeded"}
        params = kwargs.get("params") or {}
        self.reads.append(params)
        self.read_times.append(asyncio.get_running_loop().time())
        self.events.append(f"GET {url_path} {'wait' if 'wait_s' in params else 'watch'}")
        if "wait_s" not in params:
            return self.first_reply or _Response({"task_ended": False})
        if self.waiting_reply is not None:
            return self.waiting_reply
        waiting_reads = sum(1 for read in self.reads if "wait_s" in read)
        return _Response({"task_ended": self.ends_at_read is not None and waiting_reads >= self.ends_at_read})

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _server(agent_type: str = "codex_agent", **client_fields: Any) -> tuple[SingleAgentTurnEnvironmentServer, _Client]:
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"officeenv": {"host": "resources", "port": 8000}}},
            "agent": {"responses_api_agents": {agent_type: {"host": "agent", "port": 8001, "entrypoint": "app.py"}}},
        }
    )
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        events=[],
        reads=[],
        stops=[],
        read_times=[],
        control=[],
        verified=[],
        run_ended=asyncio.Event(),
        **client_fields,
    )
    config = SingleAgentTurnEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        agent_server={"type": "responses_api_agents", "name": "agent"},
        cleanup_timeout_seconds=10,
        resources_tool_transports=["direct_http"],
        end_on_task_state=True,
    )
    return SingleAgentTurnEnvironmentServer(config=config, server_client=client), client


@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(single_agent_turn_app, "TASK_STATE_POLL_SECONDS", 0.01)


class _TimedEvents(list):
    """The events, with the time each was first seen."""

    def __init__(self) -> None:
        super().__init__()
        self.times: dict[str, float] = {}

    def append(self, event: str) -> None:
        self.times.setdefault(event, asyncio.get_running_loop().time())
        super().append(event)


def _index(events: list[str], event: str) -> int:
    return events.index(event)


async def test_the_run_is_stopped_once_its_task_has_ended_and_the_call_released_after_it_answered() -> None:
    server, client = _server()

    result = await server.run_request(_request())

    assert result.failure is None and result.result.reward == 1.0
    # The session is watched before the agent runs, then read waiting for the end.
    assert client.reads == [WATCH, WATCH_AND_WAIT]
    assert _index(client.events, f"GET {RESOURCES_TASK_STATE_PATH} watch") < _index(
        client.events, f"POST {ACTIVATION_PATH}"
    )
    # Stopped as completed, its task having ended; the call that ended it is answered only once the run is over.
    assert client.stops == [{"reason": "task_ended", "completed": True}]
    stop = _index(client.events, f"POST {STOP_PATH}")
    answered = _index(client.events, "activation answered")
    release = _index(client.events, f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}")
    assert stop < answered < release
    assert client.events.count(f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}") == 1
    assert client.events[-3:] == ["POST /v1/agent_sessions/close", "POST /verify", "POST /close_session"]
    # The stop and the release never wait for a connection that the running activation holds.
    assert client.control == [f"POST {STOP_PATH}", f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}"]
    assert "ng_task_end_stop" not in client.verified[0]["response"]["metadata"]


@pytest.mark.parametrize(
    "first_reply",
    [_Response({"task_ended": True}), _ErrorResponse(404, SESSION_LOST_BODY)],
    ids=["task_ended", "session_lost"],
)
async def test_a_task_over_before_the_run_starts_no_run_and_goes_on_to_verification(first_reply: Any) -> None:
    server, client = _server(first_reply=first_reply)

    result = await server.run_request(_request())

    assert result.failure is None
    assert client.reads == [WATCH]
    assert f"POST {ACTIVATION_PATH}" not in client.events
    assert client.stops == []
    assert client.events[-4:] == [
        f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}",
        "POST /v1/agent_sessions/close",
        "POST /verify",
        "POST /close_session",
    ]
    response = client.verified[0]["response"]
    assert (response["output"], response["status"]) == ([], "completed")
    assert response["metadata"] == {"ng_termination_reason": "task_ended_before_run"}


async def test_a_run_that_ends_before_its_task_is_not_stopped_and_the_watch_is_released() -> None:
    server, client = _server(ends_at_read=None, run_ends_alone=True)

    result = await server.run_request(_request())

    assert result.failure is None
    assert client.stops == []
    assert _index(client.events, "activation answered") < _index(
        client.events, f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}"
    )


async def test_a_lost_resources_session_stops_the_run_and_goes_on_to_verification() -> None:
    server, client = _server(waiting_reply=_ErrorResponse(404, SESSION_LOST_BODY))

    result = await server.run_request(_request())

    assert result.failure is None
    assert client.stops == [{"reason": "task_ended", "completed": True}]
    assert "POST /verify" in client.events


@pytest.mark.parametrize(("status", "terminal"), [(500, False), (404, True)])
async def test_a_failed_watching_read_fails_the_episode_and_releases_the_session(status: int, terminal: bool) -> None:
    server, client = _server(waiting_reply=_ErrorResponse(status, b'{"detail": "boom"}'))

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("agent", terminal)
    assert client.stops == []
    assert "activation answered" not in client.events  # the run's activation was cancelled
    assert f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}" in client.events
    assert "POST /verify" not in client.events


@pytest.mark.parametrize(
    ("status", "terminal"),
    # 501: the sandbox's provider cannot stop a running harness, so every attempt would fail the same way.
    [(500, False), (501, True), (502, False)],
)
async def test_a_failed_stop_fails_the_episode_and_releases_the_call(status: int, terminal: bool) -> None:
    server, client = _server(stop_status=status)

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("agent", terminal)
    assert "activation answered" not in client.events
    assert client.events.count(f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}") == 1
    assert "POST /verify" not in client.events


async def test_a_stop_that_finds_no_run_releases_the_call_at_once() -> None:
    # 404: no run of the rollout is open on the agent server any more, so there is nothing to wait for.
    server, client = _server(stop_status=404)

    result = await server.run_request(_request())

    assert result.failure is None
    assert client.events.count(f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}") == 1
    assert "POST /verify" in client.events


async def test_the_call_is_released_after_the_stop_bound_when_the_run_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(single_agent_turn_app, "RUN_STOP_WAIT_SECONDS", 0.05)
    server, client = _server(stop_ends_run=False, release_ends_run=True)

    result = await server.run_request(_request())

    assert result.failure is None
    # The release came first: the run ended only once the call that ended its task was answered.
    assert _index(client.events, f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}") < _index(
        client.events, "activation answered"
    )
    assert client.events.count(f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}") == 1
    # The run may have gone on after its task ended, and its reply says so.
    assert client.verified[0]["response"]["metadata"]["ng_task_end_stop"] == "unconfirmed"


async def test_the_stop_bound_counts_from_the_time_the_stop_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stop that answers only after the bound leaves no more time to wait for the run.
    monkeypatch.setattr(single_agent_turn_app, "RUN_STOP_WAIT_SECONDS", 0.05)
    server, client = _server(stop_ends_run=False, release_ends_run=True, stop_delay=0.2)
    events = _TimedEvents()
    object.__setattr__(client, "events", events)

    result = await server.run_request(_request())

    assert result.failure is None
    assert events.times[f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}"] - events.times["stop answered"] < 0.03


async def test_a_run_the_stop_did_not_stop_is_still_waited_for_before_the_release() -> None:
    # stopped: false, such as for a run a stop already ends: the call stays held until the activation answers.
    server, client = _server(stop_stopped=False)

    result = await server.run_request(_request())

    assert result.failure is None
    assert _index(client.events, "activation answered") < _index(
        client.events, f"POST {RESOURCES_TASK_STATE_RELEASE_PATH}"
    )


async def test_a_server_that_answers_at_once_is_read_no_sooner_than_the_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(single_agent_turn_app, "TASK_STATE_POLL_SECONDS", 0.05)
    server, client = _server(ends_at_read=3)

    result = await server.run_request(_request())

    assert result.failure is None
    waiting = [time for time, read in zip(client.read_times, client.reads) if "wait_s" in read]
    assert len(waiting) == 3
    assert all(later - earlier >= 0.04 for earlier, later in zip(waiting, waiting[1:]))


@pytest.mark.parametrize("agent_type", ["simple_agent", "hermes_agent"])
def test_an_agent_whose_run_cannot_be_stopped_stops_the_server_at_startup(agent_type: str) -> None:
    assert agent_type not in ROLLOUT_STOP_AGENT_TYPES
    with pytest.raises(ValueError, match="cannot stop a run in flight"):
        _server(agent_type)


def test_end_on_task_state_without_max_episode_steps_watches_an_agent_that_runs_its_own_loop() -> None:
    server, _ = _server("terminus_2_agent")

    assert (server.config.end_on_task_state, server.config.max_episode_steps) == (True, None)
    # The run is the agent's one activation, so its checkpoints stay the agent's own.
    assert server.checkpoint_mode_in_effect() == "exported"
