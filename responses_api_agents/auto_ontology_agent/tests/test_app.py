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
from fastapi.testclient import TestClient
from langchain_core.runnables.config import ContextThreadPoolExecutor

from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import AgentContext, DirectHTTPToolAccess
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
from responses_api_agents.auto_ontology_agent.runtime import AutoOntologyRuntime, Episode, send_through


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
            ({"continuation": {"observations": ["trajectory"]}}, "continuation"),
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
