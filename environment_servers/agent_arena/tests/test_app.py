# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent_arena Environment Server: one episode per row, its tools served by this server behind a per-episode
secret, and the turn scored in process exactly as the agentic_turn_simulation resources server's /verify scores it."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientConnectionError
from omegaconf import OmegaConf

from environment_servers.agent_arena.app import (
    EPISODE_SECRET_HEADER,
    AgentArenaEnvironmentServer,
    AgentArenaEnvironmentServerConfig,
)
from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.core import (
    MODEL_TRANSPORT_ERROR,
    EpisodeId,
    MaterializedTask,
    NeMoGymResponse,
    TaskId,
)
from nemo_gym.server_runtime import SimpleServer, SingleAgentTurnRequest, SingleAgentTurnTaskInput
from resources_servers.agentic_turn_simulation.schemas import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationVerifyRequest,
    SimulateToolCallRequest,
)
from resources_servers.agentic_turn_simulation.tests.support import VERDICT, chat, http, judge_reply, make_server, req


AGENT = "arena_agent"
RESOURCES = "arena_resources"
ENVIRONMENT = "arena_environment"
SETTINGS = {
    "tool_simulator_model_server": {"type": "responses_api_models", "name": "simulator"},
    "tool_simulator_chat_completions_create_params": {"messages": [], "max_tokens": 1000, "temperature": 1.0},
    "judges": [{"model_server": {"type": "responses_api_models", "name": "judge_0"}}],
    "judge_chat_completions_create_params": {"messages": [], "max_tokens": 2000, "temperature": 1.0},
    # Failed attempts are retried at once.
    "retry_backoff_max_seconds": 0,
}
TOOLS = [
    {"type": "function", "name": n, "description": f"the {n} tool", "parameters": {"type": "object"}, "strict": False}
    for n in ("bash", "read_file", "write_file", "ask_user")
]
HISTORY = [
    {"type": "message", "role": "system", "content": "You are the sandbox agent."},
    {"type": "message", "role": "user", "content": "Build a page in /home/user/site"},
    {
        "type": "function_call",
        "call_id": "h1",
        "name": "write_file",
        "arguments": json.dumps({"path": "/home/user/site/index.html", "content": "<h1>Hi</h1>"}),
    },
    {"type": "function_call_output", "call_id": "h1", "output": json.dumps({"hash": "x", "status": "success"})},
    {"type": "message", "role": "assistant", "content": "Done: /home/user/site/index.html"},
    {"type": "message", "role": "user", "content": "Make the heading blue."},
]
ROW = {
    "responses_create_params": {"input": HISTORY, "tools": TOOLS},
    "verifier_metadata": {"harness": "sandbox", "categories": ["edit"]},
}
# The policy's turn: it reads the file the history wrote (answered exactly, from the replayed workspace), runs one
# command (answered by the simulator LLM), then reports.
READ = {"call_id": "p1", "name": "read_file", "arguments": json.dumps({"path": "/home/user/site/index.html"})}
BASH = {"call_id": "p2", "name": "bash", "arguments": json.dumps({"command": "npm run build"})}
BASH_OUT = '{"duration_ms": 3, "exit_code": 0, "status": "completed", "stderr": "", "stdout": "built in 2s"}'


def _function_call(call: dict[str, str]) -> dict[str, Any]:
    return {"type": "function_call", "id": f"fc_{call['call_id']}", "status": "completed", **call}


def _agent_reply(outputs: dict[str, str], *, status: str | None = None, metadata: dict | None = None) -> dict:
    output = []
    for call in (READ, BASH):
        output += [
            _function_call(call),
            {"type": "function_call_output", "call_id": call["call_id"], "output": outputs[call["call_id"]]},
        ]
    output.append(
        {
            "type": "message",
            "id": "m1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "The heading is blue now.", "annotations": []}],
        }
    )
    reply = NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=output,
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        metadata={"stop_reason": "final_message", "steps": "3", **(metadata or {})},
    ).model_dump(mode="json")
    if status is not None:
        reply["status"] = status
    return reply


class World:
    """The episode's other servers: the agent (which plays the turn through its grant, against the server under test),
    the tool simulator and the judge. Every post is recorded."""

    def __init__(self, server: AgentArenaEnvironmentServer, **agent: Any) -> None:
        self.server = server
        self.calls: list[dict[str, Any]] = []
        self.grant = None
        self.harness = None
        self.outputs: dict[str, str] = {}
        self.agent_reply = agent.get("reply")
        self.judge = agent.get("judge", [judge_reply(VERDICT)])
        self.simulator = agent.get("simulator", [chat(BASH_OUT)])

    async def post(self, *, server_name: str, url_path: str, json: Any = None, **kwargs: Any) -> MagicMock:
        body = json.model_dump(mode="json") if hasattr(json, "model_dump") else json
        self.calls.append({"server_name": server_name, "url_path": url_path, "json": body, **kwargs})
        if server_name == "simulator":
            return self.simulator.pop(0)
        if server_name == "judge_0":
            reply = self.judge.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        assert server_name == AGENT, server_name
        if url_path == "/v1/agent_sessions":
            (self.grant,) = AgentSeedSessionRequest.model_validate(body).tool_accesses
            return http({"agent_session_id": body["agent_session_id"]})
        if url_path == "/v1/agent_sessions/close":
            return http({"agent_session_id": body["agent_session_id"]})
        # /v1/responses: the agent's turn, through its grant.
        request = SimpleNamespace(headers=dict(self.grant.headers))
        self.harness = await self.server.describe_harness(request)
        turn: list[dict[str, Any]] = []
        for call in (READ, BASH):
            turn.append(_function_call(call))
            sim = SimulateToolCallRequest(turn=turn, **call)
            self.outputs[call["call_id"]] = (await self.server.simulate_tool_call(request, sim)).output
            turn.append(
                {"type": "function_call_output", "call_id": call["call_id"], "output": self.outputs[call["call_id"]]}
            )
        return http(self.agent_reply or _agent_reply(self.outputs))


def _global_config(**settings: Any) -> Any:
    return OmegaConf.create(
        {
            RESOURCES: {
                "resources_servers": {
                    "agentic_turn_simulation": {"entrypoint": "app.py", "domain": "agent", **SETTINGS, **settings}
                }
            },
            AGENT: {"responses_api_agents": {"agentic_turn_simulation_agent": {"entrypoint": "app.py"}}},
        }
    )


def _server(**settings: Any) -> AgentArenaEnvironmentServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = _global_config(**settings)
    client._resolve_base_url = lambda name: f"http://{name}:8000/"
    client.post = AsyncMock(side_effect=AssertionError("unexpected call"))
    config = AgentArenaEnvironmentServerConfig(
        name=ENVIRONMENT,
        host="0.0.0.0",
        port=8000,
        entrypoint="app.py",
        agent_server=AgentServerRef(type="responses_api_agents", name=AGENT),
        resources_server=ResourcesServerRef(type="resources_servers", name=RESOURCES),
    )
    return AgentArenaEnvironmentServer(config=config, server_client=client)


def _world(server: AgentArenaEnvironmentServer, **agent: Any) -> World:
    world = World(server, **agent)
    server.server_client.post = AsyncMock(side_effect=world.post)
    return world


def _request(row: dict[str, Any] | None = None) -> SingleAgentTurnRequest:
    return SingleAgentTurnRequest(
        episode_id=EpisodeId(rollout_id="7-0", attempt=0),
        task=MaterializedTask(
            task_id=TaskId(taskset=f"{AGENT}:agent_arena", task_id="7"),
            task_input=SingleAgentTurnTaskInput.model_validate(row or ROW),
        ),
    )


def _paths(world: World) -> list[tuple[str, str]]:
    return [(c["server_name"], c["url_path"]) for c in world.calls]


def test_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "app.py"), run_name="environment_servers.agent_arena.w")
    assert namespace["app"] is worker_app


class TestEpisode:
    async def test_the_episode_serves_the_turns_tools_and_scores_it(self) -> None:
        server = _server()
        world = _world(server)

        response = await server.run_request(_request())

        assert response.failure is None, response.failure
        result = response.result.model_dump(mode="json")
        assert (result["reward"], result["correct_ladder"], result["harness"]) == (1.0, "yes", "sandbox")
        # The read was answered from the history replayed at setup; the command by the simulator.
        assert result["simulated_calls"] == {"exact": 1, "llm": 1}
        assert "<h1>Hi</h1>" in world.outputs["p1"] and "built in 2s" in world.outputs["p2"]
        assert (world.harness.stop_tools, world.harness.protocol) == (["ask_user"], "native")
        assert _paths(world) == [
            (AGENT, "/v1/agent_sessions"),
            (AGENT, "/ng-rollout/7-0/v1/responses"),
            ("simulator", "/v1/chat/completions"),
            (AGENT, "/v1/agent_sessions/close"),
            ("judge_0", "/v1/chat/completions"),
        ]
        # The agent gets the row's prompt only, and one grant: this server's URL, with the episode's secret.
        prompt = world.calls[1]["json"]
        assert len(prompt["input"]) == len(HISTORY) and prompt["input"][-1]["content"] == "Make the heading blue."
        assert "verifier_metadata" not in prompt
        assert str(world.grant.base_url).rstrip("/") == f"http://{ENVIRONMENT}:8000"
        assert world.grant.name == f"{ENVIRONMENT}.direct_http" and world.grant.cookies == {}
        assert len(world.grant.headers[EPISODE_SECRET_HEADER]) >= 32
        # The session is released with the episode.
        assert server._sessions == {}

    async def test_the_settings_are_the_resources_servers(self) -> None:
        server = _server(score_source="reference_turn", masked_stop_reasons=["time_cap", "max_steps"])
        assert server._turns.config.score_source == "reference_turn"
        assert server._turns.config.masked_stop_reasons == ["time_cap", "max_steps"]
        assert server._turns.config.judges[0].model_server.name == "judge_0"

    async def test_a_failed_agent_reply_is_the_episodes_failure_by_the_outcome_policy(self) -> None:
        server = _server()
        outcome = json.dumps({"failure_kind": MODEL_TRANSPORT_ERROR})
        reply = _agent_reply({"p1": "", "p2": ""}, status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: outcome})
        world = _world(server, reply=reply)

        response = await server.run_request(_request())

        assert response.result is None
        assert (response.failure.stage, response.failure.terminal) == ("agent", False)
        assert response.failure.failure_kind == MODEL_TRANSPORT_ERROR
        assert ("judge_0", "/v1/chat/completions") not in _paths(world) and server._sessions == {}

    async def test_a_judge_that_cannot_be_reached_is_a_retryable_verification_failure(self) -> None:
        server = _server(judge_max_attempts=1)
        world = _world(server, judge=[ClientConnectionError("judge down")])

        response = await server.run_request(_request())

        assert response.result is None
        assert (response.failure.stage, response.failure.terminal) == ("verification", False)
        assert response.failure.partial_response.id == "response"
        assert world.calls[-1]["server_name"] == "judge_0" and server._sessions == {}

    async def test_an_agent_that_cannot_be_seeded_leaves_no_session(self) -> None:
        server = _server()
        server.server_client.post = AsyncMock(side_effect=ClientConnectionError("agent down"))

        response = await server.run_request(_request())

        assert (response.failure.stage, response.failure.terminal) == ("agent", False)
        assert server._sessions == {}


class TestToolRoutes:
    """The agent reaches /harness and /simulate_tool_call with its episode's secret; nothing else does."""

    @staticmethod
    def _client(server: AgentArenaEnvironmentServer) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://env")

    async def test_only_a_live_episodes_secret_is_served(self) -> None:
        server = _server()
        server._sessions["secret"] = server._turns.new_session(AgenticTurnSimulationRunRequest.model_validate(ROW))
        sim = SimulateToolCallRequest(**READ, turn=[_function_call(READ)]).model_dump(mode="json")
        async with self._client(server) as client:
            served = await client.post("/harness", headers={EPISODE_SECRET_HEADER: "secret"})
            # With rollout checkpointing on, the agent names its attempt in the path.
            prefixed = await client.post(
                "/ng-rollout/7-0/simulate_tool_call", headers={EPISODE_SECRET_HEADER: "secret"}, json=sim
            )
            missing = await client.post("/harness")
            wrong = await client.post("/simulate_tool_call", headers={EPISODE_SECRET_HEADER: "guess"}, json=sim)
        assert served.status_code == 200 and served.json() == {"stop_tools": ["ask_user"], "protocol": "native"}
        assert prefixed.status_code == 200 and prefixed.json()["source"] == "exact"
        # 403, not 404: the agent would read a 404 from /harness as a bare request without a session.
        assert (missing.status_code, wrong.status_code) == (403, 403)

    async def test_the_secret_stops_working_when_the_episode_ends(self) -> None:
        server = _server()
        world = _world(server)
        await server.run_request(_request())
        async with self._client(server) as client:
            late = await client.post("/harness", headers=dict(world.grant.headers))
        assert late.status_code == 403

    def test_one_worker_only(self) -> None:
        # A tool call must reach the worker that holds its episode's session.
        assert "worker" in _server().multiple_workers_unsupported_reason()


class TestScoringParity:
    """The episode's result is the resources server's /verify response for the same row, turn and simulated calls."""

    async def test_the_result_matches_the_resources_servers_verify(self) -> None:
        server = _server()
        world = _world(server)
        episode = (await server.run_request(_request())).result.model_dump(mode="json")

        resources = make_server(**SETTINGS)
        resources.server_client.post = AsyncMock(side_effect=[chat(BASH_OUT), judge_reply(VERDICT)])
        run_request = AgenticTurnSimulationRunRequest.model_validate(ROW)
        await resources.seed_session(req(), run_request)
        turn: list[dict[str, Any]] = []
        for call in (READ, BASH):
            turn.append(_function_call(call))
            output = (await resources.simulate_tool_call(req(), SimulateToolCallRequest(turn=turn, **call))).output
            assert output == world.outputs[call["call_id"]]
            turn.append({"type": "function_call_output", "call_id": call["call_id"], "output": output})
        verify_request = AgenticTurnSimulationVerifyRequest.model_validate(
            ROW | {"response": _agent_reply(world.outputs)}
        )
        verified = (await resources.verify(req(), verify_request)).model_dump(mode="json")

        assert {k: v for k, v in episode.items() if not k.startswith("ng_")} == verified | {"failure_report": None}

    async def test_a_masked_turn_is_masked_alike(self) -> None:
        server = _server()
        reply = _agent_reply({"p1": "x", "p2": "y"}, metadata={"stop_reason": "time_cap"})
        _world(server, reply=reply)
        episode = (await server.run_request(_request())).result.model_dump(mode="json")
        verified = await make_server(**SETTINGS).verify(
            req(None), AgenticTurnSimulationVerifyRequest.model_validate(ROW | {"response": reply})
        )
        assert episode["mask_sample"] is True and episode["failure_kind"] == verified.failure_kind
        assert episode["failure_kind"] == "agentic_turn_simulation:time_cap"

    async def test_the_metrics_are_the_resources_servers(self) -> None:
        rows = [
            {"reward": 1.0, "stop_reason": "final_message", "verifier_metadata": {"harness": "sandbox"}},
            {"reward": 0.0, "code_gate": "unfinished_turn", "verifier_metadata": {"harness": "cline"}},
        ]
        body = AggregateMetricsRequest(
            verify_responses=[{"_ng_task_index": i, "_ng_rollout_index": 0, **row} for i, row in enumerate(rows)]
        )
        ours = (await _server().aggregate_metrics(body)).model_dump(mode="json")
        theirs = (await make_server(**SETTINGS).aggregate_metrics(body)).model_dump(mode="json")
        assert ours["key_metrics"] == theirs["key_metrics"] and ours["key_metrics"]["turn_score"] == 0.5
        assert {k: v for k, v in ours["agent_metrics"].items() if k in theirs["agent_metrics"]} == theirs[
            "agent_metrics"
        ]


class TestFlatRows:
    """A benchmark's flat row gets a flat result, as single_agent_turn_legacy answers it."""

    def _row(self) -> dict[str, Any]:
        return ROW | {"_ng_task_index": 7, "_ng_rollout_index": 0, "agent_ref": {"name": AGENT}}

    async def test_a_flat_row_gets_a_flat_result(self) -> None:
        server = _server()
        _world(server)
        result = await server.run_legacy(self._row())
        assert (result["reward"], result["agent_ref"]) == (1.0, {"name": AGENT})
        assert "episode_id" not in result and result["verifier_metadata"] == ROW["verifier_metadata"]

    async def test_a_failure_is_a_flat_failure_row(self) -> None:
        server = _server(judge_max_attempts=1)
        _world(server, judge=[ClientConnectionError("judge down")])
        result = await server.run_legacy(self._row())
        assert (result["_ng_failure_stage"], result["_ng_failure_terminal"]) == ("verification", False)
        assert result["agent_ref"] == {"name": AGENT} and "reward" not in result

    async def test_a_row_of_another_agent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not match configured agent server"):
            await _server().run_legacy(self._row() | {"agent_ref": {"name": "other"}})
