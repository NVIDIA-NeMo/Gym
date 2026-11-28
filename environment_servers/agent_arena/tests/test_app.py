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
from aiohttp import ClientConnectionError, ClientResponseError
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
from nemo_gym.core.failure_kinds import ENVIRONMENT_PROTOCOL_VIOLATION
from nemo_gym.server_runtime import (
    TOOL_CALL_ID_HEADER,
    TOOL_ERROR_HEADER,
    SimpleServer,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnTaskInput,
    ToolCallRecord,
)
from nemo_gym.server_runtime.episode_tokens import EPISODE_TOKEN_HEADER
from resources_servers.agentic_turn_simulation.cline_xml import KNOWN_TOOLS
from resources_servers.agentic_turn_simulation.schemas import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationVerifyRequest,
    SimulateToolCallRequest,
)
from resources_servers.agentic_turn_simulation.scripts.validate_rows import MCP_ROUTE_NAME, SERVER_ROUTE_NAMES
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
        self.agent_context = None
        self.metadata: dict[str, str] | None = {}
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
            seed = AgentSeedSessionRequest.model_validate(body)
            (self.grant,) = seed.tool_accesses
            self.agent_context = seed.agent_context
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


def _global_config(agent: dict[str, Any] | None = None, **settings: Any) -> Any:
    return OmegaConf.create(
        {
            RESOURCES: {
                "resources_servers": {
                    "agentic_turn_simulation": {"entrypoint": "app.py", "domain": "agent", **SETTINGS, **settings}
                }
            },
            AGENT: {
                "responses_api_agents": {"agentic_turn_simulation_agent": {"entrypoint": "app.py", **(agent or {})}}
            },
        }
    )


def _server(
    *, agent: dict[str, Any] | None = None, config: dict[str, Any] | None = None, **settings: Any
) -> AgentArenaEnvironmentServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = _global_config(agent, **settings)
    client._resolve_base_url = lambda name: f"http://{name}:8000/"
    client.post = AsyncMock(side_effect=AssertionError("unexpected call"))
    server_config = AgentArenaEnvironmentServerConfig(
        name=ENVIRONMENT,
        host="0.0.0.0",
        port=8000,
        entrypoint="app.py",
        agent_server=AgentServerRef(type="responses_api_agents", name=AGENT),
        resources_server=ResourcesServerRef(type="resources_servers", name=RESOURCES),
        sandbox_provider={"local": {}},
        **(config or {}),
    )
    return AgentArenaEnvironmentServer(config=server_config, server_client=client)


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
        # The agent gets the row's prompt only, as the row has it; the row's harness (stop tools, protocol, the tools
        # this server answers) comes with the seed's context. One grant: this server's URL, with the episode's token.
        prompt = world.calls[1]["json"]
        assert len(prompt["input"]) == len(HISTORY) and prompt["input"][-1]["content"] == "Make the heading blue."
        assert "verifier_metadata" not in prompt and not world.metadata
        assert world.agent_context.values == {
            "stop_tools": ["ask_user"],
            "protocol": "native",
            "tool_names": sorted(tool["name"] for tool in TOOLS),
        }
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

    async def test_a_refused_grant_fails_the_episode_terminally(self) -> None:
        # The agent classifies a grant that refuses the row's tools; every attempt would meet the same refusal.
        server = _server()
        outcome = json.dumps({"failure_kind": ENVIRONMENT_PROTOCOL_VIOLATION})
        reply = _agent_reply({"p1": "", "p2": ""}, status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: outcome})
        _world(server, reply=reply)

        response = await server.run_request(_request())

        assert (response.failure.stage, response.failure.terminal) == ("agent", True)
        assert response.failure.failure_kind == ENVIRONMENT_PROTOCOL_VIOLATION

    @pytest.mark.parametrize(
        ("tool", "reason"),
        [
            # No grant can carry the name, or the parameters are no object schema (ToolSpec).
            ({"name": "files/read"}, "cannot be set up as an episode"),
            ({"parameters": {"type": "array"}}, "cannot be set up as an episode"),
            # Named like one of this server's routes (serve_tools).
            ({"name": "run"}, "cannot take the names of the server's routes: run"),
        ],
    )
    async def test_a_row_tool_no_episode_can_serve_is_a_terminal_seed_failure(
        self, tool: dict[str, Any], reason: str
    ) -> None:
        server = _server()
        world = _world(server)
        row = ROW | {"responses_create_params": {"input": HISTORY, "tools": [TOOLS[0] | tool]}}

        response = await server.run_request(_request(row))

        assert response.result is None
        assert (response.failure.stage, response.failure.terminal) == ("seed", True)
        assert reason in response.failure.failure_reason
        assert world.calls == [] and server._open_episodes == {}

    def test_validate_rows_refuses_the_servers_route_names(self) -> None:
        # scripts/validate_rows.py checks a row's tools against these names before any episode runs; mcp only with --mcp.
        names = {}
        for transports in (("direct_http",), ("direct_http", "mcp")):
            server = _server(config={"tool_transports": list(transports)})
            server.setup_webserver()
            names[transports] = server._reserved_tool_names
        assert names[("direct_http",)] == SERVER_ROUTE_NAMES
        assert names[("direct_http", "mcp")] == SERVER_ROUTE_NAMES | {MCP_ROUTE_NAME}

    @pytest.mark.parametrize(
        "judge_error",
        [
            ClientConnectionError("judge down"),
            # A conversation too long for the judge: a 4xx no retry would pass.
            ClientResponseError(
                request_info=MagicMock(real_url="http://judge_0"),
                history=(),
                status=400,
                message="maximum context length",
            ),
        ],
    )
    async def test_a_judge_that_gives_no_verdict_masks_the_turn_as_verify_does(self, judge_error: Exception) -> None:
        # The resources server's /verify (judge_failsafe) answers it with a masked judge_failed row, which collection
        # keeps and `gym eval reverify --judge-failed-only` re-judges: the policy's turn is not played again.
        server = _server(judge_max_attempts=1)
        world = _world(server, judge=[judge_error])

        response = await server.run_request(_request())

        assert response.failure is None, response.failure
        result = response.result.model_dump(mode="json")
        assert (result["reward"], result["mask_sample"], result["failure_kind"]) == (0.0, True, "judge_failed")
        assert result["_ng_failure_class"] == "judge_failed" and result["response"]["id"] == "response"
        # Rollout collection accepts an episode result's `_ng_` keys only as these judge-failure markers.
        judge_markers = {"_ng_failure_class", "_ng_failure_judge_error", "_ng_failure_events", "_ng_failure_id"}
        assert {key for key in result if key.startswith("_ng_")} <= judge_markers | {"_ng_failure_events_omitted"}
        assert [path for path in _paths(world) if path[1].endswith("/v1/responses")] == [
            (AGENT, "/ng-rollout/7-0/v1/responses")
        ]
        assert world.calls[-1]["server_name"] == "judge_0" and server._open_episodes == {}

    async def test_a_task_context_that_sets_the_harness_is_refused_before_the_seed(self) -> None:
        # Never overridden: the episode would run without the context it was given.
        server = _server()
        world = _world(server)
        row = ROW | {"agent_context": {"agent": {"values": {"protocol": "cline_xml"}}}}

        response = await server.run_request(_request(row))

        assert response.result is None and response.failure.terminal
        assert "sets ['protocol']" in response.failure.failure_reason
        assert world.calls == [] and server._open_episodes == {}

    async def test_an_agent_that_cannot_be_seeded_leaves_no_session(self) -> None:
        server = _server()
        server.server_client.post = AsyncMock(side_effect=ClientConnectionError("agent down"))

        response = await server.run_request(_request())

        assert (response.failure.stage, response.failure.terminal) == ("agent", False)
        assert server._open_episodes == {}

    @pytest.mark.parametrize("context", [None, {"agent": {"values": {"tag": "x"}}}])
    async def test_the_tasks_agent_context_reaches_the_seed(self, context: dict[str, Any] | None) -> None:
        # Beside the row's harness; the agent validates or refuses it. It reaches neither the prompt nor the scoring.
        server = _server()
        world = _world(server)
        await server.run_request(_request(ROW | ({"agent_context": context} if context else {})))
        seed = world.calls[0]
        assert seed["url_path"] == "/v1/agent_sessions"
        harness = {
            "stop_tools": ["ask_user"],
            "protocol": "native",
            "tool_names": sorted(tool["name"] for tool in TOOLS),
        }
        assert seed["json"]["agent_context"]["values"] == ({"tag": "x"} if context else {}) | harness
        assert "agent_context" not in world.calls[1]["json"] and not world.metadata

    def test_obstacle_injection_is_refused_at_startup(self) -> None:
        # An obstacle's decision needs the reply a call belongs to, which a call through the grant does not carry.
        with pytest.raises(ValueError, match="does not support obstacle_injection"):
            _server(obstacle_injection={"enabled": True})


class TestSkipVerification:
    """With skip_verification (this server's, else the agent's, which carries the global flag), the turn gets the fixed
    reward, unscored, as the relay's /run returned it."""

    async def test_the_turn_gets_the_fixed_reward_without_scoring(self) -> None:
        server = _server(agent={"skip_verification": True, "skip_verification_reward": 0.25})
        world = _world(server, judge=[AssertionError("the judge is not called")])

        response = await server.run_request(_request())

        assert response.failure is None, response.failure
        result = response.result.model_dump(mode="json")
        assert (result["reward"], result["verification_skipped"]) == (0.25, True)
        assert "judge_0" not in [call["server_name"] for call in world.calls]
        assert _paths(world)[-1] == (AGENT, "/v1/agent_sessions/close") and server._open_episodes == {}

    @pytest.mark.parametrize(
        ("agent", "config", "reward"),
        [
            ({}, {}, None),
            ({"skip_verification": "false"}, {}, None),
            ({"skip_verification": True}, {"skip_verification": False}, None),
            ({"skip_verification": True}, {}, 0.0),
            ({}, {"skip_verification": True, "skip_verification_reward": 0.5}, 0.5),
        ],
    )
    def test_the_decision(self, agent: dict[str, Any], config: dict[str, Any], reward: float | None) -> None:
        assert _server(agent=agent, config=config)._skipped_verification_reward() == reward

    async def test_the_metrics_are_the_performance_metrics_only(self) -> None:
        # A fixed reward is no measured turn score.
        body = AggregateMetricsRequest(
            verify_responses=[
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 0.25, "verification_skipped": True}
            ]
        )
        metrics = (await _server(agent={"skip_verification": True}).aggregate_metrics(body)).model_dump(mode="json")
        assert "turn_score" not in (metrics.get("key_metrics") or {})


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

    async def test_a_call_id_the_model_reuses_later_keeps_both_calls(self) -> None:
        # Some tool-call parsers take ids from the model's text, which may repeat one in a later step; only a resend
        # directly follows its own call.
        def record(name: str, call_id: str, output: str) -> ToolCallRecord:
            return ToolCallRecord(
                name=name, arguments={}, output=output, is_error=False, duration_seconds=0, call_id=call_id
            )

        toolset = self._toolset(ROW)
        toolset.bind(
            [
                record("read_file", "c0", "A"),
                record("bash", "c1", "B"),
                record("read_file", "c0", "C"),
                # The same id on the very next call, for another tool: a reused id, not a resend.
                record("bash", "c0", "D"),
                # The reply to this one was lost on its way back; the agent sent it again.
                record("bash", "c2", "E"),
                record("bash", "c2", "E again"),
            ]
        )
        turn = toolset.turn_so_far("bash", {"command": "ls"})
        assert [item["output"] for item in turn if item["type"] == "function_call_output"] == [
            "A",
            "B",
            "C",
            "D",
            "E again",
        ]

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
        # Rollout collection refuses an episode result that sets one of its own `_ng_` keys; the flat result, read as
        # an agent's /run row, keeps the stage the resources server saves.
        assert verified.model_extra["_ng_failure_stage"] == "agent"
        assert not [key for key in episode if key.startswith("_ng_")]
        flat = await server.run_legacy(
            ROW | {"_ng_task_index": 7, "_ng_rollout_index": 0, "agent_ref": {"name": AGENT}}
        )
        assert (flat["mask_sample"], flat["_ng_failure_stage"]) == (True, "agent")

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
        server = _server()
        outcome = json.dumps({"failure_kind": MODEL_TRANSPORT_ERROR})
        reply = _agent_reply({"p1": "", "p2": ""}, status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: outcome})
        _world(server, reply=reply)
        result = await server.run_legacy(self._row())
        assert (result["_ng_failure_stage"], result["_ng_failure_terminal"]) == ("agent", False)
        assert result["agent_ref"] == {"name": AGENT} and "reward" not in result

    async def test_a_direct_call_without_collector_indices_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The shipped example row as a caller posts it to /run, without rollout collection's indices or a rollout id:
        # each call is an episode of its own, and the row's content names its task.
        example = Path(__file__).parents[3] / "resources_servers/agentic_turn_simulation/data/example.jsonl"
        row = json.loads(example.read_text().splitlines()[0])
        # The row's agent_ref names the shipped config's agent instance; this test's servers take that name.
        monkeypatch.setitem(globals(), "AGENT", row["agent_ref"]["name"])
        server = _server()
        world = _world(server, calls=(), judge=[judge_reply(VERDICT), judge_reply(VERDICT)])
        requests = []
        run_request = AgentArenaEnvironmentServer.run_request

        async def spy(self: AgentArenaEnvironmentServer, request: SingleAgentTurnRequest) -> SingleAgentTurnResponse:
            requests.append(request)
            return await run_request(self, request)

        monkeypatch.setattr(AgentArenaEnvironmentServer, "run_request", spy)
        async with _client(world.app) as client:
            replies = [await client.post("/run", json=row) for _ in range(2)]
        assert [reply.status_code for reply in replies] == [200, 200], replies[0].text
        assert all("reward" in reply.json() for reply in replies)
        first, second = requests
        assert first.episode_id.rollout_id.startswith("direct-") and first.episode_id.attempt == 0
        assert first.episode_id != second.episode_id
        assert first.task.task_id == second.task.task_id and first.task.task_id.task_id.startswith("direct-")

    async def test_a_row_with_one_collector_index_is_a_terminal_422(self) -> None:
        server = _server()
        row = {key: value for key, value in self._row().items() if key != "_ng_rollout_index"}
        async with _client(server.setup_webserver()) as client:
            reply = await client.post("/run", json=row)
        assert reply.status_code == 422 and "needs both" in reply.text
        server.server_client.post.assert_not_awaited()

    @pytest.mark.parametrize(
        ("change", "detail"),
        [
            ({"agent_ref": {"name": "other"}}, "does not match configured agent server"),
            ({"_ng_attempt_index": -1}, "Invalid episode attempt"),
            ({"_ng_skip_token_capture": "yes"}, ""),
        ],
    )
    async def test_an_invalid_row_is_a_terminal_422(self, change: dict[str, Any], detail: str) -> None:
        # Not a 500 that a caller retries; nothing is called.
        server = _server()
        async with _client(server.setup_webserver()) as client:
            reply = await client.post("/run", json=self._row() | change)
        assert reply.status_code == 422 and detail in reply.text
        server.server_client.post.assert_not_awaited()

    async def test_an_invalid_typed_request_is_a_terminal_422(self) -> None:
        server = _server()
        typed = _request().model_dump(mode="json")
        typed["task"]["task_input"]["responses_create_params"] = "not a prompt"
        async with _client(server.setup_webserver()) as client:
            reply = await client.post("/run", json=typed)
        assert reply.status_code == 422
        server.server_client.post.assert_not_awaited()
