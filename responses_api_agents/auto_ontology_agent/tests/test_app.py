# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from langchain_core.runnables.config import ContextThreadPoolExecutor
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from environment_servers.bird_interact import adk_prompts
from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.trajectory import TrajectoryRecord
from responses_api_agents.auto_ontology_agent import c_interact
from responses_api_agents.auto_ontology_agent.app import (
    AutoOntologyAgent,
    AutoOntologyAgentConfig,
    AutoOntologyCatalogConfig,
)
from responses_api_agents.auto_ontology_agent.model_calls import (
    ModelCallRoute,
    ModelCallTransport,
    model_call_route,
)
from responses_api_agents.auto_ontology_agent.runtime import AskUser, SubmitSQL, send_model_calls_through


KNOWLEDGE = [
    {"id": 0, "knowledge": "Whale Trade", "description": "A large trade.", "definition": "amount > 1000"},
    {"id": 1, "knowledge": "Whale Day", "description": "A day with whale trades.", "definition": "count >= 3"},
]
INSTRUCTIONS = adk_prompts.C_INTERACT_INSTRUCTION.format(
    db_name="crypto",
    db_schema="CREATE TABLE trades (\n  id integer,\n  amount numeric\n);",
    external_kg=json.dumps(KNOWLEDGE, indent=2),
    max_turn=3,
)
QUESTION = "Which days had whale trades?\n\nShow the newest first."
PHASE_1 = adk_prompts.PHASE_1_MESSAGE.format(amb_user_query=QUESTION, max_turn=3)
FOLLOW_UP = adk_prompts.FOLLOW_UP_MESSAGE.format(follow_up_query="Now count them by month.")
PASSED_WITH_FOLLOW_UP = (
    adk_prompts.PHASE_1_PASSED_WITH_FOLLOW_UP.format(reward=0.7) + "\nReward: 0.7\nFollow-up question: Now count them."
)
FAILED = adk_prompts.PHASE_FAILED.format(phase=1, message=adk_prompts.WRONG_RESULT)
TOOLS = [
    {"type": "function", "name": "ask_user", "parameters": {}, "strict": False},
    {"type": "function", "name": "submit_sql", "parameters": {}, "strict": False},
]
GRANT = DirectHTTPToolAccess(
    name="bird_interact.direct_http",
    required=True,
    base_url="http://environment:9000/ng-rollout/rollout",
    headers={"X-NeMo-Gym-Episode-Token": "token"},
)
MODEL_URL = "http://policy_model:8000/v1/chat/completions"
CHAT_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "policy",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "trades"}}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
}


class TestCInteractTexts:
    def test_the_instruction_gives_the_database_schema_knowledge_and_budget(self) -> None:
        context = c_interact.read_instructions(INSTRUCTIONS)

        assert (context.database, context.question_budget, context.knowledge) == ("crypto", 3, KNOWLEDGE)
        assert context.schema == "CREATE TABLE trades (\n  id integer,\n  amount numeric\n);"

    def test_the_phase_1_message_gives_the_whole_question(self) -> None:
        assert c_interact.read_question(PHASE_1) == QUESTION

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (FOLLOW_UP, c_interact.NextTurn("follow_up", "Now count them by month.")),
            (
                adk_prompts.DEBUG_EXECUTION_ERROR_MESSAGE.format(error='column "amt" does not exist'),
                c_interact.NextTurn("debug", 'column "amt" does not exist'),
            ),
            (adk_prompts.DEBUG_WRONG_RESULT_MESSAGE, c_interact.NextTurn("debug", None)),
        ],
    )
    def test_a_later_message_names_its_turn(self, message: str, expected: c_interact.NextTurn) -> None:
        assert c_interact.read_next_turn(message) == expected

    @pytest.mark.parametrize(
        "read", [c_interact.read_instructions, c_interact.read_question, c_interact.read_next_turn]
    )
    def test_an_unknown_text_is_a_protocol_error(self, read: Any) -> None:
        with pytest.raises(c_interact.ProtocolError):
            read("Please write SQL for me.")

    def test_the_answer_loses_only_the_note_on_the_clarifications_left(self) -> None:
        note = adk_prompts.CLARIFICATIONS_LEFT_NOTE.format(remaining=2, max_turn=3)

        assert (
            c_interact.user_answer("Only trades above 1000.\n\nIn USD." + note) == "Only trades above 1000.\n\nIn USD."
        )

    @pytest.mark.parametrize(
        ("reply", "phase"),
        [
            (PASSED_WITH_FOLLOW_UP, 1),
            (adk_prompts.PHASE_2_PASSED.format(reward=0.3) + "\nReward: 0.3", 2),
            (FAILED, None),
        ],
    )
    def test_a_submission_reply_gives_the_graders_message_and_phase(self, reply: str, phase: Optional[int]) -> None:
        assert c_interact.submit_result(reply) == {"message": reply.split("\n")[0], "phase_completed": phase}

    def test_the_knowledge_is_one_block_and_children_are_listed_by_name(self) -> None:
        text, children = c_interact.format_knowledge(KNOWLEDGE, {1: [0, 7]})

        assert text == (
            "- Whale Trade\n  Description: A large trade.\n  Definition: amount > 1000\n"
            "- Whale Day\n  Description: A day with whale trades.\n  Definition: count >= 3"
        )
        assert children == {"Whale Day": ["- Whale Trade\n  Description: A large trade.\n  Definition: amount > 1000"]}
        assert c_interact.format_knowledge(KNOWLEDGE)[1] == {}

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            ("SELECT t.id FROM crypto.public.trades AS t", "SELECT t.id FROM public.trades AS t"),
            ("SELECT id FROM public.trades", "SELECT id FROM public.trades"),
            ("SELEC id FORM", "SELEC id FORM"),
        ],
    )
    def test_the_database_name_is_dropped_from_table_names(self, sql: str, expected: str) -> None:
        assert c_interact.strip_catalog_qualifiers(sql) == expected


class FakeRuntime:
    """Auto Ontology's session calls, scripted: ``step`` makes ``model_call``, if any, and returns the next action."""

    def __init__(self, actions: list[Any], model_call: Optional[Callable[[], None]] = None) -> None:
        self.actions = list(actions)
        self.model_call = model_call
        self.calls: list[tuple] = []

    def create_session(self, **kwargs: Any) -> str:
        self.calls.append(("create_session", kwargs))
        return "session"

    def step(self, session: str, *, turn: str, text: Optional[str]) -> Any:
        self.calls.append(("step", turn, text))
        if self.model_call is not None:
            self.model_call()
        return self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]

    def apply_user_answer(self, session: str, answer: str) -> None:
        self.calls.append(("answer", answer))

    def apply_submit_result(self, session: str, result: dict) -> None:
        self.calls.append(("result", result))


def _agent(**config: Any) -> AutoOntologyAgent:
    settings = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "auto_ontology_agent",
        "reasoning_model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
        "reasoning_model": "azure/openai/model",
        "fast_model_server": ModelServerRef(type="responses_api_models", name="auto_ontology_fast_model"),
        "fast_model": "azure/openai/fast",
        "embedding_base_url": "http://embed/v1",
        "embedding_api_key": "key",
        "embedding_model": "embed-model",
        "catalog": AutoOntologyCatalogConfig(host="catalog", user="postgres", password="secret", database="ao"),
        "source_database_url": "postgresql://root@sources:5432/{database}",
        **config,
    }
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    return AutoOntologyAgent(config=AutoOntologyAgentConfig(**settings), server_client=server_client)


def _seed(**fields: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        **{
            "agent_session_id": "agent-session",
            "episode_id": EpisodeId(rollout_id="rollout", attempt=0),
            "task_id": TaskId(taskset="bird_interact:c_interact", task_id="crypto_2"),
            "tool_accesses": [GRANT],
            "continuation": {"observations": ["trajectory", "timing"]},
            **fields,
        }
    )


def _tool_server(monkeypatch: pytest.MonkeyPatch, replies: dict[str, list[Any]]) -> list[tuple[str, dict]]:
    """Answer the agent's tool calls with ``replies`` (by tool name, in order); return the calls made."""
    calls: list[tuple[str, dict]] = []

    async def request(*, method: str, url: str, headers: dict, **kwargs: Any) -> MagicMock:
        name = url.rsplit("/", 1)[-1]
        assert (method, url) == ("POST", f"http://environment:9000/ng-rollout/rollout/{name}")
        assert headers == {"X-NeMo-Gym-Episode-Token": "token"}
        # Each call changes the episode, so the client must not send it again after it may have arrived.
        assert kwargs["_resend_after_disconnect"] is False
        calls.append((name, kwargs["json"]))
        reply = replies[name].pop(0)
        status, body = (reply, {"detail": "boom"}) if isinstance(reply, int) else (200, reply)
        response = MagicMock(status=status, ok=status < 400, cookies={})
        response.read = AsyncMock(return_value=json.dumps(body).encode())
        response.content.read = AsyncMock(return_value=json.dumps(body).encode())
        if status >= 400:
            info = RequestInfo(url=URL(url), method=method, headers=CIMultiDictProxy(CIMultiDict()), real_url=URL(url))
            response.raise_for_status.side_effect = ClientResponseError(info, (), status=status, message="boom")
        return response

    monkeypatch.setattr("responses_api_agents.auto_ontology_agent.app.http_request", request)
    return calls


def _model_server(server_client: Any, reply: dict, delay: float = 0) -> list[tuple[str, str, Optional[str]]]:
    """Answer the model calls sent through ``server_client`` with ``reply``; return each call's server, path and
    ``x-session-id``."""
    sent: list[tuple[str, str, Optional[str]]] = []

    async def post(*, server_name: str, url_path: str, headers: dict, **kwargs: Any) -> MagicMock:
        sent.append((server_name, url_path, headers.get("x-session-id")))
        await asyncio.sleep(delay)
        response = MagicMock(status=200, headers={"Content-Type": "application/json"})
        response.read = AsyncMock(return_value=json.dumps(reply).encode())
        return response

    server_client.post = AsyncMock(side_effect=post)
    return sent


def _activate(client: TestClient, activation: int, messages: list[dict], tools: Optional[list] = None) -> Any:
    body: dict[str, Any] = {"input": messages}
    if tools is not None:
        body["tools"] = tools
    return client.post(
        "/ng-rollout/rollout/v1/responses", json=body, headers={AGENT_ACTIVATION_HEADER: str(activation)}
    )


def _first_turn() -> list[dict]:
    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": PHASE_1}]


class TestSeed:
    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
            (
                {
                    "sandbox_access": SandboxAccess(
                        connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={}),
                        workdir="/app",
                    )
                },
                "no task sandbox",
            ),
            ({"continuation": None}, "continuation"),
            ({"tool_accesses": []}, "got 0"),
            (
                {
                    "tool_accesses": [
                        GRANT,
                        MCPToolAccess(
                            name="bird_interact",
                            required=True,
                            connection=MCPStreamableHTTPConnection(url="http://environment:9000/mcp"),
                        ),
                    ]
                },
                "direct HTTP only",
            ),
        ],
    )
    def test_a_seed_the_agent_cannot_serve_is_refused_for_good(self, fields: dict, reason: str) -> None:
        assert reason in check_agent_seed_rejected(_agent().setup_webserver(), _seed(**fields))


class TestCInteractSession:
    def test_each_turn_runs_auto_ontology_until_it_submits_and_close_returns_the_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent()
        runtime = FakeRuntime(
            [AskUser("Above which amount?"), SubmitSQL("SELECT d FROM crypto.public.trades"), SubmitSQL("SELECT 2")]
        )
        agent._runtime = runtime
        note = adk_prompts.CLARIFICATIONS_LEFT_NOTE.format(remaining=2, max_turn=3)
        tool_calls = _tool_server(
            monkeypatch,
            {
                "ask_user": ["Above 1000." + note],
                "submit_sql": [PASSED_WITH_FOLLOW_UP, "Phase 2 correct! (Reward: 0.3)."],
            },
        )
        client = TestClient(agent.setup_webserver())
        seed = client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))

        first = _activate(client, 1, _first_turn(), TOOLS)
        second = _activate(client, 2, [{"role": "user", "content": FOLLOW_UP}])
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "agent-session", "episode_id": {"rollout_id": "rollout", "attempt": 0}},
        )

        assert seed.status_code == first.status_code == second.status_code == close.status_code == 200
        assert seed.json()["capabilities"]["observations"] == ["trajectory", "timing"]
        created = runtime.calls[0][1]
        assert (created["database"], created["question_budget"], created["knowledge_children"]) == ("crypto", 3, {})
        assert created["knowledge"].startswith("- Whale Trade\n")
        assert runtime.calls[1:] == [
            ("step", "initial", QUESTION),
            ("answer", "Above 1000."),
            ("step", "initial", QUESTION),
            ("result", {"message": PASSED_WITH_FOLLOW_UP.split("\n")[0], "phase_completed": 1}),
            ("step", "follow_up", "Now count them by month."),
            ("result", {"message": "Phase 2 correct! (Reward: 0.3).", "phase_completed": 2}),
        ]
        assert tool_calls == [
            ("ask_user", {"question": "Above which amount?"}),
            ("submit_sql", {"sql": "SELECT d FROM public.trades"}),
            ("submit_sql", {"sql": "SELECT 2"}),
        ]
        output = first.json()["output"]
        assert [(item["type"], item.get("name")) for item in output] == [
            ("function_call", "ask_user"),
            ("function_call_output", None),
            ("function_call", "submit_sql"),
            ("function_call_output", None),
        ]
        assert output[3]["output"] == PASSED_WITH_FOLLOW_UP
        trajectory = TrajectoryRecord.model_validate(close.json()["trajectory"])
        assert [(i.invocation_id, i.activation_id, i.status) for i in trajectory.invocations] == [
            ("root", 1, "completed"),
            ("activation-2", 2, "completed"),
        ]
        assert all(invocation.duration_ms is not None for invocation in trajectory.invocations)
        assert [(call.invocation_id, call.tool_name, call.status) for call in trajectory.tool_calls] == [
            ("root", "ask_user", "completed"),
            ("root", "submit_sql", "completed"),
            ("activation-2", "submit_sql", "completed"),
        ]
        # The activations came on the rollout's route, and the turns made no model call without usage.
        assert close.json()["agent_observations"]["gaps"] == []

    @pytest.mark.parametrize(
        ("message", "turn", "text"),
        [
            (adk_prompts.DEBUG_EXECUTION_ERROR_MESSAGE.format(error="syntax error"), "debug", "syntax error"),
            (adk_prompts.DEBUG_WRONG_RESULT_MESSAGE, "debug", None),
        ],
    )
    def test_a_debug_turn_gives_auto_ontology_the_error(
        self, monkeypatch: pytest.MonkeyPatch, message: str, turn: str, text: Optional[str]
    ) -> None:
        agent = _agent()
        agent._runtime = runtime = FakeRuntime([SubmitSQL("SELECT 1")])
        _tool_server(monkeypatch, {"submit_sql": [FAILED, FAILED]})
        client = TestClient(agent.setup_webserver())
        client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))

        _activate(client, 1, _first_turn(), TOOLS)
        response = _activate(client, 2, [{"role": "user", "content": message}])

        assert response.status_code == 200
        assert runtime.calls[-2] == ("step", turn, text)
        assert runtime.calls[-1] == ("result", {"message": FAILED, "phase_completed": None})

    def test_a_turn_that_never_submits_stops_after_the_budget_and_its_margin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(step_margin=1)
        agent._runtime = runtime = FakeRuntime([AskUser("And?")])
        tool_calls = _tool_server(monkeypatch, {"ask_user": ["No idea."] * 10})
        client = TestClient(agent.setup_webserver())
        client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))

        response = _activate(client, 1, _first_turn(), TOOLS)

        # A budget of 3 questions, a margin of 1, and the step that would submit.
        assert response.status_code == 200
        assert [name for name, _ in tool_calls] == ["ask_user"] * 5
        assert [call[0] for call in runtime.calls].count("step") == 5

    def test_a_failed_tool_call_fails_the_turn_and_stays_in_the_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent()
        agent._runtime = FakeRuntime([SubmitSQL("SELECT 1")])
        _tool_server(monkeypatch, {"submit_sql": [500]})
        client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)
        client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))

        response = _activate(client, 1, _first_turn(), TOOLS)
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "agent-session", "episode_id": {"rollout_id": "rollout", "attempt": 0}},
        )

        assert response.status_code == 500
        trajectory = TrajectoryRecord.model_validate(close.json()["trajectory"])
        assert [(i.status, i.error_type) for i in trajectory.invocations] == [("failed", "ClientResponseError")]
        assert [(call.tool_name, call.status) for call in trajectory.tool_calls] == [("submit_sql", "failed")]

    @pytest.mark.parametrize(
        ("messages", "tools", "detail"),
        [
            (_first_turn(), [{"type": "function", "name": "submit", "parameters": {}, "strict": False}], "declare"),
            (
                [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": PHASE_1}],
                TOOLS,
                "instruction",
            ),
        ],
    )
    def test_a_first_turn_that_is_not_c_interact_is_refused(
        self, messages: list[dict], tools: list, detail: str
    ) -> None:
        agent = _agent()
        agent._runtime = FakeRuntime([SubmitSQL("SELECT 1")])
        client = TestClient(agent.setup_webserver())
        client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))

        response = _activate(client, 1, messages, tools)

        assert response.status_code == 422 and detail in response.json()["detail"]

    def test_an_activation_without_a_session_is_refused(self) -> None:
        client = TestClient(_agent().setup_webserver())

        assert _activate(client, 1, _first_turn(), TOOLS).status_code == 409

    def test_the_children_of_knowledge_entries_are_read_only_when_configured(self, tmp_path: Any) -> None:
        (tmp_path / "crypto").mkdir()
        rows = [{"id": 0, "children_knowledge": -1}, {"id": 1, "children_knowledge": [0]}]
        (tmp_path / "crypto" / "crypto_kb.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

        assert _agent()._children_of("crypto") is None
        assert _agent(knowledge_children_dir=str(tmp_path))._children_of("crypto") == {1: [0]}


async def test_auto_ontology_gets_the_gym_model_servers_and_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(feature_flags={"INTERACTIVE": "true"})
    monkeypatch.setattr(
        AutoOntologyAgent, "resolve_model_base_url", lambda self, name, **kwargs: f"http://{name}:8000/v1"
    )

    environment = agent.runtime()._environment

    assert (environment["REASONING_ENDPOINT"], environment["NON_REASONING_ENDPOINT"]) == (
        "http://policy_model:8000/v1",
        "http://auto_ontology_fast_model:8000/v1",
    )
    assert (environment["REASONING_MODEL"], environment["NON_REASONING_MODEL"]) == (
        "azure/openai/model",
        "azure/openai/fast",
    )
    assert (environment["POSTGRES_HOST"], environment["POSTGRES_PASSWORD"]) == ("catalog", "secret")
    assert (environment["EMBED_ENDPOINT"], environment["INTERACTIVE"]) == ("http://embed/v1", "true")
    assert agent.runtime()._http_client._transport is agent._model_transport
    # Each role's calls go to its own Model Server.
    assert agent._model_transport._model_servers == {
        ("policy_model", 8000): "policy_model",
        ("auto_ontology_fast_model", 8000): "auto_ontology_fast_model",
    }


@pytest.mark.parametrize("field", ["reasoning_model", "fast_model"])
def test_a_model_name_that_selects_another_client_than_chat_openai_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="openai/, azure/, aws/"):
        _agent(**{field: "meta/llama-3.3-70b-instruct"})


class TestModelCalls:
    async def test_each_call_takes_its_activations_route_from_any_thread_that_copies_the_context(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        sent = _model_server(server_client, CHAT_COMPLETION)
        transport = ModelCallTransport(
            server_client, {"http://policy_model:8000/v1": "policy_model"}, loop=asyncio.get_running_loop()
        )
        client = httpx.Client(transport=transport)
        both_running = threading.Barrier(2, timeout=5)

        def step() -> None:
            """As Auto Ontology does: a call from the step's thread, and two from a LangChain executor."""
            both_running.wait()
            client.post(MODEL_URL, json={"model": "openai/policy", "messages": []})
            with ContextThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda _: client.post(MODEL_URL, json={"messages": []}), range(2)))

        async def activation(rollout: str, invocation_id: str) -> ModelCallRoute:
            route = ModelCallRoute(path_prefix=f"/ng-rollout/{rollout}", invocation_id=invocation_id)
            with model_call_route(route):
                await asyncio.to_thread(step)
            return route

        first, second = await asyncio.gather(activation("episode-a", "root"), activation("episode-b", "activation-2"))

        assert sorted(sent) == [
            *[("policy_model", "/ng-rollout/episode-a/v1/chat/completions", "root")] * 3,
            *[("policy_model", "/ng-rollout/episode-b/v1/chat/completions", "activation-2")] * 3,
        ]
        assert (first.usage().input_tokens, first.usage().output_tokens, second.usage().total_tokens) == (12, 3, 15)
        assert transport.unrouted_calls == 0

    async def test_a_call_from_a_pool_that_drops_the_context_reaches_the_model_without_a_rollout(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        sent = _model_server(server_client, CHAT_COMPLETION)
        transport = ModelCallTransport(
            server_client, {"http://policy_model:8000/v1": "policy_model"}, loop=asyncio.get_running_loop()
        )
        client = httpx.Client(transport=transport)

        def step() -> int:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(client.post, MODEL_URL, json={"messages": []}).result().status_code

        route = ModelCallRoute(path_prefix="/ng-rollout/episode-a", invocation_id="root")
        with model_call_route(route):
            status = await asyncio.to_thread(step)

        assert status == 200
        assert sent == [("policy_model", "/v1/chat/completions", None)]
        assert (transport.unrouted_calls, route.usages) == (1, [])

    @pytest.mark.parametrize(
        "reply", [{key: value for key, value in CHAT_COMPLETION.items() if key != "usage"}, {"usage": "unknown"}]
    )
    async def test_a_reply_without_usage_leaves_the_activations_usage_unknown(self, reply: dict) -> None:
        server_client = MagicMock(spec=ServerClient)
        _model_server(server_client, reply)
        transport = ModelCallTransport(
            server_client, {"http://policy_model:8000/v1": "policy_model"}, loop=asyncio.get_running_loop()
        )
        client = httpx.Client(transport=transport)
        route = ModelCallRoute(path_prefix="/ng-rollout/episode-a", invocation_id="root")

        with model_call_route(route):
            await asyncio.to_thread(client.post, MODEL_URL, json={"messages": []})

        assert route.usages == [None] and route.usage() is None

    async def test_a_model_server_that_does_not_reply_in_time_times_the_call_out(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _model_server(server_client, CHAT_COMPLETION, delay=5)
        transport = ModelCallTransport(
            server_client, {"http://policy_model:8000/v1": "policy_model"}, loop=asyncio.get_running_loop()
        )
        client = httpx.Client(transport=transport, timeout=0.05)

        with pytest.raises(httpx.ReadTimeout):
            await asyncio.to_thread(client.post, MODEL_URL, json={"messages": []})

    async def test_a_call_on_the_event_loop_or_to_another_server_is_refused(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        transport = ModelCallTransport(
            server_client, {"http://policy_model:8000/v1": "policy_model"}, loop=asyncio.get_running_loop()
        )
        client = httpx.Client(transport=transport)

        with pytest.raises(RuntimeError, match="in a thread"):
            client.post(MODEL_URL, json={"messages": []})
        with pytest.raises(httpx.ConnectError):
            await asyncio.to_thread(client.post, "http://elsewhere:8000/v1/chat/completions", json={})


class TestAutoOntologyThroughGym:
    @pytest.mark.parametrize(
        ("path", "reply", "usage", "gaps"),
        [
            ("/ng-rollout/rollout/v1/responses", CHAT_COMPLETION, 5, []),
            (
                "/ng-rollout/rollout/v1/responses",
                {key: value for key, value in CHAT_COMPLETION.items() if key != "usage"},
                None,
                ["model_call_usage_unavailable"],
            ),
            ("/v1/responses", CHAT_COMPLETION, 5, ["model_call_ownership_unavailable"]),
        ],
    )
    def test_auto_ontologys_model_calls_take_the_activations_route_and_report_their_usage(
        self, monkeypatch: pytest.MonkeyPatch, path: str, reply: dict, usage: Optional[int], gaps: list[str]
    ) -> None:
        agent = _agent()
        monkeypatch.setattr(
            AutoOntologyAgent, "resolve_model_base_url", lambda self, name, **kwargs: f"http://{name}:8000/v1"
        )
        sent = _model_server(agent.server_client, reply)

        def runtime(environment: dict, source_database_url: str, *, http_client: httpx.Client) -> FakeRuntime:
            return FakeRuntime(
                [SubmitSQL("SELECT 1"), SubmitSQL("SELECT 2")],
                model_call=lambda: http_client.post(MODEL_URL, json={"model": "openai/policy", "messages": []}),
            )

        monkeypatch.setattr("responses_api_agents.auto_ontology_agent.app.AutoOntologyRuntime", runtime)
        _tool_server(monkeypatch, {"submit_sql": [PASSED_WITH_FOLLOW_UP, "Phase 2 correct! (Reward: 0.3)."]})
        # One event loop for all requests, as in a server: the transport sends on the loop that built it.
        with TestClient(agent.setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            first = client.post(
                path, json={"input": _first_turn(), "tools": TOOLS}, headers={AGENT_ACTIVATION_HEADER: "1"}
            )
            second = client.post(
                path, json={"input": [{"role": "user", "content": FOLLOW_UP}]}, headers={AGENT_ACTIVATION_HEADER: "2"}
            )
            close = client.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": "agent-session", "episode_id": {"rollout_id": "rollout", "attempt": 0}},
            )

        prefix = path.removesuffix("/v1/responses")
        assert sent == [
            ("policy_model", f"{prefix}/v1/chat/completions", "root"),
            ("policy_model", f"{prefix}/v1/chat/completions", "activation-2"),
        ]
        assert [(response.json()["usage"] or {}).get("total_tokens") for response in (first, second)] == [usage] * 2
        observations = close.json()["agent_observations"]
        assert [gap["code"] for gap in observations["gaps"]] == gaps

    def test_a_call_outside_an_activation_marks_each_session_that_was_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent()
        monkeypatch.setattr(
            AutoOntologyAgent, "resolve_model_base_url", lambda self, name, **kwargs: f"http://{name}:8000/v1"
        )
        _model_server(agent.server_client, CHAT_COMPLETION)

        def runtime(environment: dict, source_database_url: str, *, http_client: httpx.Client) -> FakeRuntime:
            def model_call_in_a_pool_that_drops_the_context() -> None:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(http_client.post, MODEL_URL, json={"messages": []}).result()

            return FakeRuntime([SubmitSQL("SELECT 1")], model_call=model_call_in_a_pool_that_drops_the_context)

        monkeypatch.setattr("responses_api_agents.auto_ontology_agent.app.AutoOntologyRuntime", runtime)
        _tool_server(monkeypatch, {"submit_sql": [FAILED]})
        with TestClient(agent.setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            _activate(client, 1, _first_turn(), TOOLS)
            close = client.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": "agent-session", "episode_id": {"rollout_id": "rollout", "attempt": 0}},
            )

        assert [gap["code"] for gap in close.json()["agent_observations"]["gaps"]] == [
            "model_call_ownership_unavailable"
        ]


def test_auto_ontology_builds_its_chat_clients_on_the_agents_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the pinned revision: the two attributes the agent sets exist, and a client Auto Ontology builds uses them."""
    from auto_ontology.retrieval.interactive import coordinator
    from auto_ontology.utils import llm_invoke

    # send_model_calls_through replaces both; monkeypatch puts them back after the test.
    monkeypatch.setattr(llm_invoke, "_build_client", llm_invoke._build_client)
    monkeypatch.setattr(coordinator, "ThreadPoolExecutor", coordinator.ThreadPoolExecutor)
    monkeypatch.setattr(llm_invoke, "_API_KEY", "gym-model-server")
    monkeypatch.setattr(llm_invoke, "_BASE_URL", "http://policy_model:8000/v1")
    monkeypatch.setattr(llm_invoke, "_MODEL_NAME", "openai/policy")
    requests: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CHAT_COMPLETION)

    send_model_calls_through(
        httpx.Client(transport=httpx.MockTransport(reply)), llm_invoke=llm_invoke, coordinator=coordinator
    )
    answer = llm_invoke.get_llm_client().invoke("Which table holds the trades?")

    assert answer.content == "trades"
    assert [str(request.url) for request in requests] == [MODEL_URL]
    assert json.loads(requests[0].content)["model"] == "openai/policy"
    assert coordinator.ThreadPoolExecutor is ContextThreadPoolExecutor


def test_a_chat_client_other_than_chat_openai_is_refused() -> None:
    llm_invoke = SimpleNamespace(_build_client=lambda **arguments: object())
    send_model_calls_through(httpx.Client(), llm_invoke=llm_invoke, coordinator=SimpleNamespace())

    with pytest.raises(TypeError, match="only its ChatOpenAI client"):
        llm_invoke._build_client(model="meta/llama")
