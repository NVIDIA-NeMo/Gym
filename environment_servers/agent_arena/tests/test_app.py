# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent_arena Environment Server: one episode per row, the row's tools served by this server for the episode
(BaseEnvironmentServerWithTools), and the turn scored in process exactly as the agentic_turn_simulation resources
server's /verify scores it."""

import json
import runpy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientConnectionError
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.agent_arena.app import AgentArenaEnvironmentServer, AgentArenaEnvironmentServerConfig
from environment_servers.agent_arena.toolset import TurnToolset
from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.core import (
    MODEL_SAMPLING_NULL,
    MODEL_TRANSPORT_ERROR,
    EpisodeId,
    MaterializedTask,
    NeMoGymResponse,
    TaskId,
    current_model_sampling_segment,
)
from nemo_gym.server_runtime import (
    TOOL_CALL_ID_HEADER,
    TOOL_ERROR_HEADER,
    SimpleServer,
    SingleAgentTurnRequest,
    SingleAgentTurnTaskInput,
    ToolCallRecord,
)
from nemo_gym.server_runtime.episode_tokens import EPISODE_TOKEN_HEADER
from resources_servers.agentic_turn_simulation.cline_xml import KNOWN_TOOLS
from resources_servers.agentic_turn_simulation.schemas import (
    PROTOCOL_METADATA_KEY,
    STOP_TOOLS_METADATA_KEY,
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


def _agent_reply(
    outputs: dict[str, str],
    *,
    calls: tuple[dict[str, str], ...] = (READ, BASH),
    status: str | None = None,
    metadata: dict | None = None,
) -> dict:
    output = []
    for call in calls:
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
    """The episode's other servers: the agent (which plays the turn through its grant, against the server under test's
    routes), the tool simulator and the judge. Every post is recorded."""

    def __init__(self, server: AgentArenaEnvironmentServer, **agent: Any) -> None:
        self.server = server
        self.app = server.setup_webserver()
        self.calls: list[dict[str, Any]] = []
        self.grant = None
        self.metadata: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.statuses: dict[str, int] = {}
        self.tool_errors: dict[str, bool] = {}
        self.agent_calls = agent.get("calls", (READ, BASH))
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
        # /v1/responses: the agent's turn, each call through its grant to the tool's route.
        self.metadata = body["metadata"]
        async with _client(self.app) as client:
            for call in self.agent_calls:
                reply = await client.post(
                    f"/{call['name']}",
                    json=_arguments(call),
                    headers=dict(self.grant.headers) | {TOOL_CALL_ID_HEADER: call["call_id"]},
                )
                self.statuses[call["call_id"]] = reply.status_code
                self.tool_errors[call["call_id"]] = reply.headers.get(TOOL_ERROR_HEADER) == "true"
                self.outputs[call["call_id"]] = _text(reply.json())
        return http(self.agent_reply or _agent_reply(self.outputs, calls=self.agent_calls))


def _arguments(call: dict[str, str]) -> Any:
    return json.loads(call["arguments"])


def _text(output: Any) -> str:
    """A tool reply's body as the agent passes it to the model."""
    return output if isinstance(output, str) else json.dumps(output)


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://env")


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
        sandbox_provider={"local": {}},
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
        assert _paths(world) == [
            (AGENT, "/v1/agent_sessions"),
            (AGENT, "/ng-rollout/7-0/v1/responses"),
            ("simulator", "/v1/chat/completions"),
            (AGENT, "/v1/agent_sessions/close"),
            ("judge_0", "/v1/chat/completions"),
        ]
        # The agent gets the row's prompt only, with the row's harness in its metadata, and one grant: this server's
        # URL, with the episode's token.
        prompt = world.calls[1]["json"]
        assert len(prompt["input"]) == len(HISTORY) and prompt["input"][-1]["content"] == "Make the heading blue."
        assert "verifier_metadata" not in prompt
        assert world.metadata == {STOP_TOOLS_METADATA_KEY: '["ask_user"]', PROTOCOL_METADATA_KEY: "native"}
        assert str(world.grant.base_url).rstrip("/") == f"http://{ENVIRONMENT}:8000"
        assert world.grant.name == f"{ENVIRONMENT}.direct_http" and world.grant.required
        assert EPISODE_TOKEN_HEADER in world.grant.headers
        # The episode's tools stop with it.
        assert server._open_episodes == {}

    @pytest.mark.parametrize("training", [False, True])
    async def test_its_own_model_calls_keep_their_sampling_in_a_training_run(self, training: bool) -> None:
        # As a resources server's: `null` in a run that sets model_call_sampling, so a trained agent's Model Server,
        # which refuses a call without a segment, takes them. The agent's calls are not this server's to mark.
        server = _server()
        world = _world(server)
        if training:
            server.server_client.global_config_dict = OmegaConf.merge(
                server.server_client.global_config_dict,
                {
                    "policy_model": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py"}}},
                    "model_call_sampling": {"required_model_servers": ["policy_model"]},
                },
            )
        seen: list[tuple[str, str | None]] = []

        async def post(**kwargs: Any) -> Any:
            seen.append((kwargs["server_name"], current_model_sampling_segment()))
            return await world.post(**kwargs)

        server.server_client.post = AsyncMock(side_effect=post)
        response = await server.run_request(_request())

        assert response.failure is None, response.failure
        own = MODEL_SAMPLING_NULL if training else None
        assert seen == [(AGENT, None), (AGENT, None), ("simulator", own), (AGENT, None), ("judge_0", own)]

    async def test_the_simulator_sees_the_earlier_calls_of_the_episode(self) -> None:
        server = _server()
        world = _world(server)

        await server.run_request(_request())

        (simulated,) = [c for c in world.calls if c["server_name"] == "simulator"]
        prompt = simulated["json"]["messages"][1]["content"]
        # The read before it, its result and the call to simulate, after the row's history.
        assert prompt.index("Make the heading blue.") < prompt.rindex("<h1>Hi</h1>") < prompt.rindex("npm run build")

    async def test_a_call_to_a_tool_the_row_lacks_is_refused_and_gated(self) -> None:
        server = _server()
        missing = {"call_id": "p3", "name": "browse", "arguments": json.dumps({"url": "https://example.com"})}
        world = _world(server, calls=(READ, missing))

        response = await server.run_request(_request())
        assert response.failure is None, response.failure

        # The server refuses the call, as the scoring's gate refuses the turn: the same set of tools.
        assert world.statuses == {"p1": 200, "p3": 404} and world.tool_errors["p3"]
        result = response.result.model_dump(mode="json")
        assert (result["reward"], result["code_gate"]) == (0.0, "nonexistent_tool")

    async def test_the_settings_are_the_resources_servers(self) -> None:
        server = _server(score_source="reference_turn", masked_stop_reasons=["time_cap", "max_steps"])
        assert server._turns.config.score_source == "reference_turn"
        assert server._turns.config.masked_stop_reasons == ["time_cap", "max_steps"]
        assert server._turns.config.judges[0].model_server.name == "judge_0"

    async def test_an_outcome_policy_override_is_the_resources_servers(self) -> None:
        policy = {"by_stage": {"agent": {"by_failure_kind": {"agentic_turn_simulation:time_cap": "fail"}}}}
        server = _server(outcome_policy=policy)
        reply = _agent_reply({"p1": "x", "p2": "y"}, metadata={"stop_reason": "time_cap"})
        _world(server, reply=reply)
        episode = (await server.run_request(_request())).result.model_dump(mode="json")
        assert (episode["failure_outcome_action"], episode["failure_outcome_attribution"]) == ("fail", "evaluation")

    async def test_a_failed_agent_reply_is_the_episodes_failure_by_the_outcome_policy(self) -> None:
        server = _server()
        outcome = json.dumps({"failure_kind": MODEL_TRANSPORT_ERROR})
        reply = _agent_reply({"p1": "", "p2": ""}, status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: outcome})
        world = _world(server, reply=reply)

        response = await server.run_request(_request())

        assert response.result is None
        assert (response.failure.stage, response.failure.terminal) == ("agent", False)
        assert response.failure.failure_kind == MODEL_TRANSPORT_ERROR
        assert ("judge_0", "/v1/chat/completions") not in _paths(world) and server._open_episodes == {}

    async def test_a_judge_that_cannot_be_reached_is_a_retryable_verification_failure(self) -> None:
        server = _server(judge_max_attempts=1)
        world = _world(server, judge=[ClientConnectionError("judge down")])

        response = await server.run_request(_request())

        assert response.result is None
        assert (response.failure.stage, response.failure.terminal) == ("verification", False)
        assert response.failure.partial_response.id == "response"
        assert world.calls[-1]["server_name"] == "judge_0" and server._open_episodes == {}

    async def test_an_agent_that_cannot_be_seeded_leaves_no_session(self) -> None:
        server = _server()
        server.server_client.post = AsyncMock(side_effect=ClientConnectionError("agent down"))

        response = await server.run_request(_request())

        assert (response.failure.stage, response.failure.terminal) == ("agent", False)
        assert server._open_episodes == {}


class TestToolRoutes:
    """The agent reaches the row's tools with its episode's token; nothing else does."""

    async def test_only_the_episodes_token_reaches_its_tools(self) -> None:
        server = _server()
        world = _world(server)
        await server.run_request(_request())
        async with _client(world.app) as client:
            late = await client.post("/read_file", json=_arguments(READ), headers=dict(world.grant.headers))
            bare = await client.post("/read_file", json=_arguments(READ))
            # A token reaches no route but the tools.
            run = await client.post("/run", json={}, headers=dict(world.grant.headers))
        assert (late.status_code, bare.status_code, run.status_code) == (410, 401, 403)

    async def test_a_tool_error_is_marked_for_the_agent(self) -> None:
        server = _server(tool_simulator_max_attempts=1)
        world = _world(server, simulator=[http({"error": "down"}, status=500)])
        await server.run_request(_request())
        assert world.statuses["p2"] == 200 and world.tool_errors["p2"] and not world.tool_errors["p1"]
        assert "[tool simulator failed" in world.outputs["p2"]

    async def test_flat_and_typed_runs_share_the_run_route(self) -> None:
        app = _server().setup_webserver()
        posts = [route for route in app.router.routes if "POST" in getattr(route, "methods", ())]
        paths = [route.path for route in posts]
        # /run takes both shapes, ahead of the tool route.
        assert paths.count("/run") == 1 and paths.index("/run") < paths.index("/{tool_name}")
        assert next(route for route in posts if route.path == "/run").endpoint.__name__ == "run_legacy"


class TestToolset:
    """The tools an episode serves are the ones the scoring's nonexistent_tool gate accepts."""

    @staticmethod
    def _toolset(row: dict[str, Any]) -> TurnToolset:
        server = _server()
        request = AgenticTurnSimulationRunRequest.model_validate(row)
        return TurnToolset(server._turns, server._turns.new_session(request), request)

    def test_a_native_row_serves_its_own_tools(self) -> None:
        specs = self._toolset(ROW).specs()
        assert [(spec.name, spec.description) for spec in specs] == [
            (t["name"], t["description"]) for t in sorted(TOOLS, key=lambda t: t["name"])
        ]

    def test_a_cline_row_serves_clines_tools(self) -> None:
        # Cline rows list no function tools: a call may name any of Cline's.
        row = {
            "responses_create_params": {"input": [{"type": "message", "role": "user", "content": "Fix a.py"}]},
            "verifier_metadata": {"harness": "cline", "protocol": "cline_xml"},
        }
        specs = self._toolset(row).specs()
        assert {spec.name for spec in specs} == set(KNOWN_TOOLS)
        assert all(spec.parameters == {"type": "object"} for spec in specs)

    def test_a_tool_name_no_grant_can_carry_fails_the_setup(self) -> None:
        tool = TOOLS[0] | {"name": "files/read"}
        with pytest.raises(ValidationError, match="pattern"):
            self._toolset(ROW | {"responses_create_params": {"input": HISTORY, "tools": [tool]}})

    async def test_the_simulator_sees_the_calls_before_this_one_and_their_results(self) -> None:
        toolset = self._toolset(ROW)
        toolset.bind(
            [
                ToolCallRecord(
                    name="read_file",
                    arguments={"path": "a"},
                    output="A",
                    is_error=False,
                    duration_seconds=0,
                    call_id="p1",
                )
            ]
        )
        turn = toolset.turn_so_far("bash", {"command": "ls"})
        assert [(item["type"], item.get("name"), item.get("output")) for item in turn] == [
            ("function_call", "read_file", None),
            ("function_call_output", None, "A"),
            ("function_call", "bash", None),
        ]
        assert turn[0]["call_id"] == turn[1]["call_id"] == "p1" and json.loads(turn[2]["arguments"]) == {
            "command": "ls"
        }

    async def test_the_simulator_sees_only_the_results_the_policy_saw(self) -> None:
        def record(name: str, call_id: str, output: Any, is_error: bool = False) -> ToolCallRecord:
            return ToolCallRecord(
                name=name, arguments={}, output=output, is_error=is_error, duration_seconds=0, call_id=call_id
            )

        toolset = self._toolset(ROW)
        toolset.bind(
            [
                # A failed attempt the agent retried under the same call id, then its answer.
                record("read_file", "p1", {"detail": "Tool 'read_file' failed: RuntimeError"}, is_error=True),
                record("read_file", "p1", "A"),
                # A tool the row lacks: the agent answered it with its own refusal.
                record("browse", "p2", {"detail": "Episode 0-0 has no tool 'browse'"}, is_error=True),
                # A failed simulation is an answer the policy saw.
                record("bash", "p3", "[tool simulator failed] HTTP 503", is_error=True),
                # This call's own failed first attempt.
                record("bash", "p4", {"detail": "Tool 'bash' did not finish within 60 seconds"}, is_error=True),
            ]
        )
        turn = toolset.turn_so_far("bash", {"command": "ls"})
        assert [(item["call_id"], item.get("output")) for item in turn] == [
            ("p1", None),
            ("p1", "A"),
            ("p3", None),
            ("p3", "[tool simulator failed] HTTP 503"),
            ("ng_current_call", None),
        ]


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
        # The simulator saw the same call in both, the turn so far rebuilt from the episode's calls here.
        sims = [c for c in world.calls if c["server_name"] == "simulator"]
        theirs = resources.server_client.post.call_args_list[0].kwargs["json"]["messages"]
        ours = json.loads(json.dumps(sims[0]["json"]["messages"]).replace("ng_current_call", "p2"))
        assert ours == theirs
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
        # The outcome policy's decision is saved with the row, as the resources server saves it.
        assert (episode["failure_outcome_action"], episode["failure_outcome_attribution"]) == ("retry", "evaluation")
        assert episode["_ng_failure_stage"] == verified.model_extra["_ng_failure_stage"] == "agent"

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
