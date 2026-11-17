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
from nemo_gym.server_runtime import AgentContext, DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
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
from responses_api_agents.auto_ontology_agent.runtime import (
    AskUser,
    AutoOntologyRuntime,
    Episode,
    SubmitSQL,
    send_model_calls_through,
    send_through,
)


TASK = {
    "question": "What is the highest eligible free rate for K-12 students in Alameda County?",
    "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
    "database": "california_schools",
}
SQL = "SELECT MAX(`Free Meal Count (K-12)` / `Enrollment (K-12)`) FROM frpm WHERE `County Name` = 'Alameda'"
MODEL_URL = "http://policy_model:8000/v1/chat/completions"
CHAT_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "policy",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "frpm"}}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
}
# The benchmark's own prompt, which the agent receives and does not use.
PROMPT = [{"role": "user", "content": "### Question\nWhat is the highest eligible free rate ...?"}]


class FakeRuntime:
    """Auto Ontology's ReAct episode, scripted: it makes ``model_call``, if any, and ends as ``episode``."""

    def __init__(self, episode: Episode, model_call: Optional[Callable[[], None]] = None) -> None:
        self.episode = episode
        self.model_call = model_call
        self.calls: list[dict] = []

    def run_episode(self, **task: str) -> Episode:
        self.calls.append(task)
        if self.model_call is not None:
            self.model_call()
        return self.episode


def _agent(**config: Any) -> AutoOntologyAgent:
    settings = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "auto_ontology_agent",
        "model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
        "model": "openai/policy",
        "embedding_base_url": "http://embed/v1",
        "embedding_api_key": "key",
        "embedding_model": "embed-model",
        "catalog": AutoOntologyCatalogConfig(host="catalog", user="postgres", password="secret", database="ao"),
        "source_database_url": "sqlite:////data/bird/dev_databases/{database}/{database}.sqlite",
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
            "task_id": TaskId(taskset="birdbench", task_id="0"),
            "agent_context": AgentContext(values=TASK),
            **fields,
        }
    )


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


def _activate(client: TestClient, path: str = "/ng-rollout/rollout/v1/responses", activation: int = 1) -> Any:
    return client.post(path, json={"input": PROMPT}, headers={AGENT_ACTIVATION_HEADER: str(activation)})


def _close(client: TestClient) -> Any:
    return client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "agent-session", "episode_id": {"rollout_id": "rollout", "attempt": 0}},
    )


def _with_runtime(monkeypatch: pytest.MonkeyPatch, runtime: Callable[..., FakeRuntime]) -> None:
    monkeypatch.setattr(
        AutoOntologyAgent, "resolve_model_base_url", lambda self, name, **kwargs: f"http://{name}:8000/v1"
    )
    monkeypatch.setattr("responses_api_agents.auto_ontology_agent.app.AutoOntologyRuntime", runtime)


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
            (
                {"tool_accesses": [DirectHTTPToolAccess(name="bird_sql", required=True, base_url="http://r:1")]},
                "its own tools",
            ),
            ({"agent_context": None}, "question: Field required"),
            ({"agent_context": AgentContext(values={**TASK, "schema": "CREATE TABLE t (x)"})}, "schema"),
            ({"agent_context": AgentContext(values={"question": "?"})}, "database"),
            # A seed with continuation is a c-Interact session, which needs the fast model.
            ({"continuation": {"observations": ["trajectory"]}}, "fast model"),
        ],
    )
    def test_a_seed_the_agent_cannot_serve_is_refused_for_good(self, fields: dict, reason: str) -> None:
        assert reason in check_agent_seed_rejected(_agent().setup_webserver(), _seed(**fields))

    def test_an_optional_grant_is_skipped(self) -> None:
        grant = DirectHTTPToolAccess(name="bird_sql", required=False, base_url="http://r:1")
        client = TestClient(_agent().setup_webserver())

        assert client.post("/v1/agent_sessions", json=_seed(tool_accesses=[grant]).model_dump(mode="json")).is_success


@pytest.mark.parametrize("model", ["meta/llama-3.3-70b-instruct", "policy"])
def test_a_model_name_that_selects_another_client_than_chat_openai_is_refused(model: str) -> None:
    with pytest.raises(ValueError, match="openai/, azure/, aws/"):
        _agent(model=model)


def test_the_source_database_url_must_name_the_database() -> None:
    with pytest.raises(ValueError, match="{database}"):
        _agent(source_database_url="sqlite:////data/bird.sqlite")


class TestEpisode:
    def test_the_episode_runs_on_the_task_and_answers_with_its_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = FakeRuntime(Episode(sql=SQL, reason="submitted", tool_calls=4, trajectory=[{"role": "ai"}]))
        _with_runtime(monkeypatch, lambda *args, **kwargs: runtime)
        with TestClient(_agent().setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            response = _activate(client)
            close = _close(client)

        assert runtime.calls == [TASK]
        body = response.json()
        assert body["output"][0]["content"][0]["text"] == f"```sql\n{SQL}\n```"
        assert (body["metadata"]["auto_ontology_end_reason"], body["metadata"]["auto_ontology_tool_calls"]) == (
            "submitted",
            "4",
        )
        observations = close.json()["agent_observations"]
        assert [(record["kind"], record["status"]) for record in observations["records"]] == [
            ("agent_invocation", "completed")
        ]

    def test_the_answer_is_the_query_alone_in_one_fenced_sql_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """bird_sql grades the last fenced sql block of the response's text; comments and line breaks stay."""
        multi_line = "SELECT a\n-- the free rate\nFROM t"
        _with_runtime(monkeypatch, lambda *args, **kwargs: FakeRuntime(Episode(multi_line, "submitted", 1)))
        with TestClient(_agent().setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            text = _activate(client).json()["output"][0]["content"][0]["text"]

        assert text.startswith("```sql\n") and text.endswith("\n```")
        assert text.removeprefix("```sql\n").removesuffix("\n```") == multi_line

    @pytest.mark.parametrize("reason", ["no_submission", "budget_exhausted", "model_error"])
    def test_an_episode_without_a_query_answers_with_no_sql_block(
        self, monkeypatch: pytest.MonkeyPatch, reason: str
    ) -> None:
        _with_runtime(monkeypatch, lambda *args, **kwargs: FakeRuntime(Episode(None, reason, 24)))
        with TestClient(_agent().setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            body = _activate(client).json()

        text = body["output"][0]["content"][0]["text"]
        assert "```" not in text and reason in text
        assert body["metadata"]["auto_ontology_end_reason"] == reason

    def test_an_activation_sent_again_gets_the_same_reply_without_a_second_episode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = FakeRuntime(Episode(SQL, "submitted", 2))
        _with_runtime(monkeypatch, lambda *args, **kwargs: runtime)
        with TestClient(_agent().setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            first = _activate(client)
            again = _activate(client)

        assert again.status_code == 200 and again.json() == first.json()
        assert len(runtime.calls) == 1

    def test_an_activation_for_another_rollout_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = FakeRuntime(Episode(SQL, "submitted", 2))
        _with_runtime(monkeypatch, lambda *args, **kwargs: runtime)
        with TestClient(_agent().setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            response = _activate(client, "/ng-rollout/another/v1/responses")

        assert response.status_code == 409 and runtime.calls == []

    def test_a_failed_episode_fails_the_activation_and_stays_in_the_observations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Broken(FakeRuntime):
            def run_episode(self, **task: str) -> Episode:
                raise RuntimeError("catalog unreachable")

        _with_runtime(monkeypatch, lambda *args, **kwargs: Broken(Episode(None, "", 0)))
        with TestClient(_agent().setup_webserver(), raise_server_exceptions=False) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            response = _activate(client)
            close = _close(client)

        assert response.status_code == 500
        (record,) = close.json()["agent_observations"]["records"]
        assert (record["status"], record["error_type"]) == ("failed", "RuntimeError")

    def test_an_activation_without_a_session_is_refused(self) -> None:
        assert _activate(TestClient(_agent().setup_webserver())).status_code == 409


class TestRuntime:
    """``AutoOntologyRuntime.run_episode`` against Auto Ontology's functions, stubbed."""

    @staticmethod
    def _runtime(result: Any, seen: dict) -> AutoOntologyRuntime:
        runtime = AutoOntologyRuntime({}, "sqlite:////data/{database}/{database}.sqlite", http_client=httpx.Client())

        def run_episode(llm: Any, ctx: Any, **kwargs: Any) -> Any:
            seen.update(llm=llm, ctx=ctx, **kwargs)
            return result

        connector = SimpleNamespace(database_name="california_schools", dialect="sqlite")
        runtime._modules = {
            "create_connector": lambda url: seen.setdefault("urls", []).append(url) or connector,
            "ToolContext": lambda **fields: SimpleNamespace(**fields),
            "run_episode": run_episode,
            "connector_for": lambda ctx: ctx.connectors[0],
            "llm": "reasoning-llm",
            "data_retriever": "data",
            "semantic_retriever": "semantic",
        }
        return runtime

    def test_the_episode_gets_the_task_as_auto_ontologys_react_step_gives_it(self) -> None:
        seen: dict = {}
        submission = SimpleNamespace(sql=SQL, db_response=object())
        result = SimpleNamespace(submission=submission, reason="submitted", tool_calls=3, trajectory=[{"role": "ai"}])
        runtime = self._runtime(result, seen)

        episode = runtime.run_episode(**TASK)
        runtime.run_episode(**TASK)

        assert episode == Episode(sql=SQL, reason="submitted", tool_calls=3, trajectory=[{"role": "ai"}])
        ctx = seen["ctx"]
        assert (ctx.retriever, ctx.data_retriever, ctx.database_name) == ("semantic", "data", "california_schools")
        # A graded benchmark takes SQL only, so the model is offered no information answers.
        assert (ctx.information_answers, ctx.question, ctx.numeric_type_alignment) == (False, TASK["question"], False)
        assert (seen["llm"], seen["question"], seen["evidence"], seen["dialect"]) == (
            "reasoning-llm",
            TASK["question"],
            TASK["evidence"],
            "sqlite",
        )
        # Each database's connector is made once.
        assert seen["urls"] == ["sqlite:////data/california_schools/california_schools.sqlite"]

    @pytest.mark.parametrize(
        "submission", [None, SimpleNamespace(sql="", response="The schools table.", db_response=None)]
    )
    def test_no_query_without_a_submission_that_ran(self, submission: Any) -> None:
        result = SimpleNamespace(submission=submission, reason="no_submission", tool_calls=5, trajectory=[])
        episode = self._runtime(result, {}).run_episode(question="?", evidence="", database="california_schools")
        assert (episode.sql, episode.reason) == (None, "no_submission")

    def test_empty_evidence_is_none_to_auto_ontology(self) -> None:
        seen: dict = {}
        result = SimpleNamespace(submission=None, reason="no_submission", tool_calls=0, trajectory=[])
        self._runtime(result, seen).run_episode(question="?", evidence="", database="california_schools")
        assert seen["evidence"] is None


class TestModelCalls:
    async def test_each_call_takes_its_activations_route_from_any_thread_that_copies_the_context(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        sent = _model_server(server_client, CHAT_COMPLETION)
        transport = ModelCallTransport(
            server_client, {"http://policy_model:8000/v1": "policy_model"}, loop=asyncio.get_running_loop()
        )
        client = httpx.Client(transport=transport)
        both_running = threading.Barrier(2, timeout=5)

        def episode() -> None:
            """A call from the episode's thread, and two from a LangChain executor."""
            both_running.wait()
            client.post(MODEL_URL, json={"model": "openai/policy", "messages": []})
            with ContextThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda _: client.post(MODEL_URL, json={"messages": []}), range(2)))

        async def activation(rollout: str) -> ModelCallRoute:
            route = ModelCallRoute(path_prefix=f"/ng-rollout/{rollout}", invocation_id="root")
            with model_call_route(route):
                await asyncio.to_thread(episode)
            return route

        first, second = await asyncio.gather(activation("episode-a"), activation("episode-b"))

        assert sorted(sent) == [
            *[("policy_model", "/ng-rollout/episode-a/v1/chat/completions", "root")] * 3,
            *[("policy_model", "/ng-rollout/episode-b/v1/chat/completions", "root")] * 3,
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

        def episode() -> int:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(client.post, MODEL_URL, json={"messages": []}).result().status_code

        route = ModelCallRoute(path_prefix="/ng-rollout/episode-a", invocation_id="root")
        with model_call_route(route):
            status = await asyncio.to_thread(episode)

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
        route = ModelCallRoute(path_prefix="/ng-rollout/episode-a", invocation_id="root")

        with model_call_route(route):
            await asyncio.to_thread(httpx.Client(transport=transport).post, MODEL_URL, json={"messages": []})

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
    def test_the_episodes_model_calls_take_the_activations_route_and_report_their_usage(
        self, monkeypatch: pytest.MonkeyPatch, path: str, reply: dict, usage: Optional[int], gaps: list[str]
    ) -> None:
        agent = _agent()
        sent = _model_server(agent.server_client, reply)
        seen: dict = {}

        def runtime(environment: dict, source_database_url: str, *, http_client: httpx.Client, **kwargs: Any) -> Any:
            seen.update(environment=environment, source_database_url=source_database_url)
            return FakeRuntime(
                Episode(SQL, "submitted", 1),
                model_call=lambda: http_client.post(MODEL_URL, json={"model": "openai/policy", "messages": []}),
            )

        _with_runtime(monkeypatch, runtime)
        # One event loop for all requests, as in a server: the transport sends on the loop that built it.
        with TestClient(agent.setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            response = _activate(client, path)
            close = _close(client)

        prefix = path.removesuffix("/v1/responses")
        assert sent == [("policy_model", f"{prefix}/v1/chat/completions", "root")]
        assert (response.json()["usage"] or {}).get("total_tokens") == usage
        assert [gap["code"] for gap in close.json()["agent_observations"]["gaps"]] == gaps
        # Auto Ontology's reasoning client is the policy, on its Model Server.
        environment = seen["environment"]
        assert (environment["REASONING_ENDPOINT"], environment["REASONING_MODEL"]) == (
            "http://policy_model:8000/v1",
            "openai/policy",
        )
        assert (environment["POSTGRES_HOST"], environment["EMBED_MODEL"]) == ("catalog", "embed-model")

    def test_a_call_outside_an_activation_marks_each_session_that_was_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent()
        _model_server(agent.server_client, CHAT_COMPLETION)

        def runtime(environment: dict, source_database_url: str, *, http_client: httpx.Client, **kwargs: Any) -> Any:
            def model_call_in_a_pool_that_drops_the_context() -> None:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(http_client.post, MODEL_URL, json={"messages": []}).result()

            return FakeRuntime(Episode(SQL, "submitted", 1), model_call=model_call_in_a_pool_that_drops_the_context)

        _with_runtime(monkeypatch, runtime)
        with TestClient(agent.setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
            _activate(client)
            close = _close(client)

        assert [gap["code"] for gap in close.json()["agent_observations"]["gaps"]] == [
            "model_call_ownership_unavailable"
        ]


def test_auto_ontologys_reasoning_client_sends_through_the_agents_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the pinned revision: the client Auto Ontology's ReAct step uses can take the agent's HTTP client."""
    from auto_ontology.utils import llm_invoke

    monkeypatch.setattr(llm_invoke, "_API_KEY", "gym-model-server")
    monkeypatch.setattr(llm_invoke, "_BASE_URL", "http://policy_model:8000/v1")
    monkeypatch.setattr(llm_invoke, "_MODEL_NAME", "openai/policy")
    requests: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CHAT_COMPLETION)

    llm = send_through(llm_invoke.get_llm_client(), httpx.Client(transport=httpx.MockTransport(reply)))
    answer = llm.invoke("Which table holds the free meal counts?")

    assert answer.content == "frpm"
    assert [str(request.url) for request in requests] == [MODEL_URL]
    assert json.loads(requests[0].content)["model"] == "openai/policy"


def test_the_react_loop_has_the_signature_the_runtime_calls() -> None:
    """At the pinned revision: the names and parameters ``AutoOntologyRuntime`` uses exist."""
    import inspect

    from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
    from auto_ontology.retrieval.text_to_sql.react.loop import EpisodeResult, Submission, run_episode

    assert list(inspect.signature(run_episode).parameters)[:5] == ["llm", "ctx", "question", "evidence", "dialect"]
    fields = set(inspect.signature(ToolContext).parameters)
    assert {"retriever", "data_retriever", "connectors", "database_name", "information_answers", "question"} <= fields
    assert {"submission", "reason", "tool_calls", "trajectory"} <= set(inspect.signature(EpisodeResult).parameters)
    assert {"sql", "db_response"} <= set(inspect.signature(Submission).parameters)


def test_a_chat_client_other_than_chat_openai_is_refused() -> None:
    with pytest.raises(TypeError, match="only its ChatOpenAI client"):
        send_through(object(), httpx.Client())


# ----- c-Interact ----- #

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


class FakeInteractiveRuntime:
    """Auto Ontology's interactive session calls, scripted: ``step`` makes ``model_call``, if any, and returns the next
    action."""

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


def _c_agent(**config: Any) -> AutoOntologyAgent:
    return _agent(
        fast_model_server=ModelServerRef(type="responses_api_models", name="auto_ontology_fast_model"),
        fast_model="openai/fast",
        **config,
    )


def _c_seed(**fields: Any) -> AgentSeedSessionRequest:
    return _seed(
        **{
            "task_id": TaskId(taskset="bird_interact:c_interact", task_id="crypto_2"),
            "agent_context": None,
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


def _turn(client: TestClient, activation: int, messages: list[dict], tools: Optional[list] = None) -> Any:
    body: dict[str, Any] = {"input": messages}
    if tools is not None:
        body["tools"] = tools
    return client.post(
        "/ng-rollout/rollout/v1/responses", json=body, headers={AGENT_ACTIVATION_HEADER: str(activation)}
    )


def _first_turn() -> list[dict]:
    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": PHASE_1}]


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


class TestCInteractSeed:
    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
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
            ({"agent_context": AgentContext(values=TASK)}, "from its turns"),
        ],
    )
    def test_a_seed_the_agent_cannot_serve_is_refused_for_good(self, fields: dict, reason: str) -> None:
        assert reason in check_agent_seed_rejected(_c_agent().setup_webserver(), _c_seed(**fields))


class TestCInteractSession:
    def test_each_turn_runs_auto_ontology_until_it_submits_and_close_returns_the_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _c_agent()
        runtime = FakeInteractiveRuntime(
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
        seed = client.post("/v1/agent_sessions", json=_c_seed().model_dump(mode="json"))

        first = _turn(client, 1, _first_turn(), TOOLS)
        second = _turn(client, 2, [{"role": "user", "content": FOLLOW_UP}])
        close = _close(client)

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
        agent = _c_agent()
        agent._runtime = runtime = FakeInteractiveRuntime([SubmitSQL("SELECT 1")])
        _tool_server(monkeypatch, {"submit_sql": [FAILED, FAILED]})
        client = TestClient(agent.setup_webserver())
        client.post("/v1/agent_sessions", json=_c_seed().model_dump(mode="json"))

        _turn(client, 1, _first_turn(), TOOLS)
        response = _turn(client, 2, [{"role": "user", "content": message}])

        assert response.status_code == 200
        assert runtime.calls[-2] == ("step", turn, text)
        assert runtime.calls[-1] == ("result", {"message": FAILED, "phase_completed": None})

    def test_a_turn_that_never_submits_stops_after_the_budget_and_its_margin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _c_agent(step_margin=1)
        agent._runtime = runtime = FakeInteractiveRuntime([AskUser("And?")])
        tool_calls = _tool_server(monkeypatch, {"ask_user": ["No idea."] * 10})
        client = TestClient(agent.setup_webserver())
        client.post("/v1/agent_sessions", json=_c_seed().model_dump(mode="json"))

        response = _turn(client, 1, _first_turn(), TOOLS)

        # A budget of 3 questions, a margin of 1, and the step that would submit.
        assert response.status_code == 200
        assert [name for name, _ in tool_calls] == ["ask_user"] * 5
        assert [call[0] for call in runtime.calls].count("step") == 5

    def test_a_failed_tool_call_fails_the_turn_and_stays_in_the_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _c_agent()
        agent._runtime = FakeInteractiveRuntime([SubmitSQL("SELECT 1")])
        _tool_server(monkeypatch, {"submit_sql": [500]})
        client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)
        client.post("/v1/agent_sessions", json=_c_seed().model_dump(mode="json"))

        response = _turn(client, 1, _first_turn(), TOOLS)
        close = _close(client)

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
        agent = _c_agent()
        agent._runtime = FakeInteractiveRuntime([SubmitSQL("SELECT 1")])
        client = TestClient(agent.setup_webserver())
        client.post("/v1/agent_sessions", json=_c_seed().model_dump(mode="json"))

        response = _turn(client, 1, messages, tools)

        assert response.status_code == 422 and detail in response.json()["detail"]

    def test_the_children_of_knowledge_entries_are_read_only_when_configured(self, tmp_path: Any) -> None:
        (tmp_path / "crypto").mkdir()
        rows = [{"id": 0, "children_knowledge": -1}, {"id": 1, "children_knowledge": [0]}]
        (tmp_path / "crypto" / "crypto_kb.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

        assert _c_agent()._children_of("crypto") is None
        assert _c_agent(knowledge_children_dir=str(tmp_path))._children_of("crypto") == {1: [0]}


async def test_c_interact_gets_the_fast_models_server_too(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _c_agent()
    monkeypatch.setattr(
        AutoOntologyAgent, "resolve_model_base_url", lambda self, name, **kwargs: f"http://{name}:8000/v1"
    )

    environment = agent.runtime()._environment

    assert (environment["NON_REASONING_ENDPOINT"], environment["NON_REASONING_MODEL"]) == (
        "http://auto_ontology_fast_model:8000/v1",
        "openai/fast",
    )
    # Each role's calls go to its own Model Server.
    assert agent._model_transport._model_servers == {
        ("policy_model", 8000): "policy_model",
        ("auto_ontology_fast_model", 8000): "auto_ontology_fast_model",
    }


def test_auto_ontology_builds_every_chat_client_on_the_agents_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the pinned revision: the two attributes the agent sets exist, and a client Auto Ontology builds uses them."""
    from auto_ontology.retrieval.interactive import coordinator
    from auto_ontology.utils import llm_invoke

    # send_model_calls_through replaces both; monkeypatch puts them back after the test.
    monkeypatch.setattr(llm_invoke, "_build_client", llm_invoke._build_client)
    monkeypatch.setattr(coordinator, "ThreadPoolExecutor", coordinator.ThreadPoolExecutor)
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_API_KEY", "gym-model-server")
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_BASE_URL", "http://policy_model:8000/v1")
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_MODEL_NAME", "openai/fast")
    requests: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CHAT_COMPLETION)

    send_model_calls_through(
        httpx.Client(transport=httpx.MockTransport(reply)), llm_invoke=llm_invoke, coordinator=coordinator
    )
    answer = llm_invoke.get_non_reasoning_llm_client().invoke("Merge the answer into the question.")

    assert answer.content == "frpm"
    assert [str(request.url) for request in requests] == [MODEL_URL]
    assert json.loads(requests[0].content)["model"] == "openai/fast"
    assert coordinator.ThreadPoolExecutor is ContextThreadPoolExecutor


def test_the_interactive_mode_has_the_calls_the_runtime_makes() -> None:
    """At the pinned revision: the interactive mode's names and parameters that ``AutoOntologyRuntime`` uses exist."""
    import inspect

    from auto_ontology.retrieval import interactive

    assert {
        "session_id",
        "task_id",
        "db_name",
        "db_schema",
        "external_kb",
        "question",
        "data_retriever",
        "semantic_retriever",
        "connectors",
        "max_clarify_turns",
        "external_kb_children_map",
        "skip_evidence_refinement",
        "numeric_type_alignment",
    } <= set(inspect.signature(interactive.create_session).parameters)
    assert {"turn_type", "initial_question", "follow_up_question", "debug_error"} <= set(
        inspect.signature(interactive.step).parameters
    )
    assert {turn.value for turn in interactive.TurnType} >= {"initial", "follow_up", "debug"}
    assert callable(interactive.apply_user_answer) and callable(interactive.apply_submit_result)
    assert interactive.AskUserAction and interactive.SubmitSQLAction
