# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server whose tools are a toolset over each session's state."""

from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.server_runtime.episode_tools import ToolResult, episode_tool
from nemo_gym.server_runtime.tool_access import END_ACTIVATION_HEADER
from nemo_gym.server_runtime.toolset_resources_server import ResourcesToolset, ToolsetResourcesServer


class CounterState(BaseModel):
    count: int


class AddArguments(BaseModel):
    amount: int


class NoArguments(BaseModel):
    pass


class CounterSeed(BaseSeedSessionRequest):
    start: int


class CounterTools(ResourcesToolset):
    state: CounterState

    @episode_tool
    async def add(self, arguments: AddArguments) -> CounterState:
        """Add to the counter."""
        self.state.count += arguments.amount
        return self.state

    @episode_tool
    async def submit(self, arguments: NoArguments) -> ToolResult:
        """Submit the counter."""
        return ToolResult(output={"submitted": self.state.count}, ends_activation=True)


class CounterServer(ToolsetResourcesServer):
    seed_request_model = CounterSeed
    session_state_model = CounterState
    toolset = CounterTools

    async def seed_state(self, body: CounterSeed) -> CounterState:
        return CounterState(count=body.start)

    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        state = self.session_state(request.session[SESSION_ID_KEY])
        return BaseVerifyResponse(**body.model_dump(), reward=float(state is not None and state.count == 7))


class EchoTools(ResourcesToolset):
    @episode_tool
    async def echo(self, arguments: AddArguments) -> dict[str, int]:
        """Echo the amount."""
        return {"amount": arguments.amount}


class EchoServer(ToolsetResourcesServer):
    toolset = EchoTools

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


def _server(server_class: type[ToolsetResourcesServer]) -> ToolsetResourcesServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="resources")
    return server_class(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(client: TestClient, session_id: str = "resources-session-1", start: int = 3) -> None:
    seed = {
        "resources_session_id": session_id,
        "episode_id": {"rollout_id": "rollout"},
        "task_id": {"taskset": "counter", "task_id": "0"},
        "task_data": {"start": start, "expected": 7},
    }
    assert client.post("/seed_session", json=seed).status_code == 200


class TestTools:
    def test_the_seed_builds_the_state_and_each_tool_changes_it_at_its_route(self) -> None:
        server = _server(CounterServer)
        client = TestClient(server.setup_webserver())
        _seed(client)

        assert client.post("/add", json={"amount": 4}).json() == {"count": 7}
        assert server.session_state("resources-session-1") == CounterState(count=7)

    def test_a_tool_call_for_a_session_the_server_does_not_hold_answers_404(self) -> None:
        client = TestClient(_server(CounterServer).setup_webserver())
        assert client.post("/add", json={"amount": 1}).status_code == 404

    def test_arguments_the_tool_does_not_accept_answer_422(self) -> None:
        client = TestClient(_server(CounterServer).setup_webserver())
        _seed(client)
        assert client.post("/add", json={"amount": "many"}).status_code == 422

    def test_a_tool_that_ends_the_activation_says_so_in_its_reply(self) -> None:
        client = TestClient(_server(CounterServer).setup_webserver())
        _seed(client)

        reply = client.post("/submit", json={})

        assert reply.json() == {"submitted": 3}
        assert reply.headers[END_ACTIVATION_HEADER] == "true"
        assert END_ACTIVATION_HEADER not in client.post("/add", json={"amount": 1}).headers

    def test_close_releases_the_state(self) -> None:
        server = _server(CounterServer)
        client = TestClient(server.setup_webserver())
        _seed(client)

        close = {"resources_session_id": "resources-session-1", "episode_id": {"rollout_id": "rollout"}}
        assert client.post("/close_session", json=close).status_code == 200

        assert server.session_state("resources-session-1") is None
        assert client.post("/add", json={"amount": 1}).status_code == 404

    def test_a_stateless_servers_tools_need_no_seed(self) -> None:
        client = TestClient(_server(EchoServer).setup_webserver())
        assert client.post("/echo", json={"amount": 5}).json() == {"amount": 5}


class TestStateMove:
    def test_the_state_moves_out_and_back_before_verify_reads_it(self) -> None:
        server = _server(CounterServer)
        client = TestClient(server.setup_webserver())
        _seed(client)

        assert client.post("/export_session_state", json={}).json() == {"state": {"count": 3}}
        assert client.post("/restore_session_state", json={"state": {"count": 7}}).status_code == 200

        assert server.session_state("resources-session-1") == CounterState(count=7)
        verify = {"responses_create_params": {"input": "count"}, "response": _response()}
        assert client.post("/verify", json=verify).json()["reward"] == 1.0

    def test_a_state_that_is_not_the_state_model_answers_422(self) -> None:
        server = _server(CounterServer)
        client = TestClient(server.setup_webserver())
        _seed(client)

        assert client.post("/restore_session_state", json={"state": {"count": "seven"}}).status_code == 422
        assert server.session_state("resources-session-1") == CounterState(count=3)

    def test_a_session_the_server_does_not_hold_has_no_state_to_move(self) -> None:
        server = _server(CounterServer)
        client = TestClient(server.setup_webserver())

        assert client.post("/export_session_state", json={}).status_code == 404
        # A closed session's state does not bring the session back.
        assert client.post("/restore_session_state", json={"state": {"count": 7}}).status_code == 404
        assert server.session_state("resources-session-1") is None

    def test_a_stateless_server_moves_no_state(self) -> None:
        client = TestClient(_server(EchoServer).setup_webserver())
        assert client.post("/export_session_state", json={}).json() == {"state": None}
        assert client.post("/restore_session_state", json={"state": None}).status_code == 200


class TestCheckpoints:
    def test_the_checkpoint_mode_follows_the_state_model(self) -> None:
        assert CounterServer.checkpoint_mode == "exported"
        assert EchoServer.checkpoint_mode == "stateless"

    def test_a_checkpoint_mode_the_class_declares_wins(self) -> None:
        class RestartOnly(CounterServer):
            checkpoint_mode = "restart_only"

        assert RestartOnly.checkpoint_mode == "restart_only"

    async def test_the_checkpoint_hooks_export_and_restore_the_state_models(self) -> None:
        server = _server(CounterServer)
        await server.initialize_session_state("live", CounterSeed(start=3))

        assert await server.export_session_states(["live", "gone"], checkpoint_id="c") == {"live": {"count": 3}}
        with pytest.raises(ValueError):
            await server.restore_session_states({"a": {"count": 1}, "b": {"count": "x"}})
        assert server.session_state("a") is None

        await server.restore_session_states({"a": {"count": 1}})
        assert server.session_state("a") == CounterState(count=1)

    @pytest.mark.parametrize("seed", ["cookie", "typed"])
    async def test_the_replacement_attempts_seed_continues_a_restored_session(self, seed: str) -> None:
        # The crashed process's client holds the session's cookie; another copy of the server signs the same one.
        crashed = TestClient(_server(CounterServer).setup_webserver())
        _seed(crashed)
        server = _server(CounterServer)
        await server.restore_session_states({"resources-session-1": {"count": 9}})
        assert not server.has_session_state("resources-session-1")
        client = TestClient(server.setup_webserver())

        if seed == "typed":
            _seed(client, start=0)
        else:
            client.cookies = crashed.cookies
            assert client.post("/seed_session", json={"start": 0}).status_code == 200

        # The seed claimed the restored counter instead of starting a new one at 0.
        assert client.post("/add", json={"amount": 1}).json() == {"count": 10}
        assert server.has_session_state("resources-session-1")

    async def test_a_restored_session_that_is_closed_unclaimed_is_released(self) -> None:
        server = _server(CounterServer)
        await server.restore_session_states({"resources-session-1": {"count": 9}})
        client = TestClient(server.setup_webserver())

        close = {"resources_session_id": "resources-session-1", "episode_id": {"rollout_id": "rollout"}}
        assert client.post("/close_session", json=close).status_code == 200

        assert server.session_state("resources-session-1") is None


def _response() -> dict[str, object]:
    return {
        "id": "response",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
