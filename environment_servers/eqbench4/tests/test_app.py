# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import runpy
from pathlib import Path
from typing import Any

import orjson
import pytest
from aiohttp import ClientConnectionError, ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import ConfigDict
from yarl import URL

from environment_servers.eqbench4.app import (
    PERSONA_FAILED,
    EQBench4Conversation,
    EQBench4EnvironmentServer,
    EQBench4EnvironmentServerConfig,
    EQBench4EpisodeRequest,
    PersonaReply,
    _AgentSession,
    persona_messages,
    policy_stop_reason,
    published_observations,
    scoped_observations,
)
from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest, BaseServerConfig
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    EpisodeId,
    MaterializedTask,
    ModelCallSampling,
    NeMoGymResponse,
    TaskId,
    encode_model_sampling,
)
from nemo_gym.server_runtime import SimpleServer, SingleAgentTurnTaskInput
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    grading_result_binding,
)
from resources_servers.eqbench4.task_data import CONVERSATION_METADATA_KEY


POLICY = "eqbench4_policy_agent"
PERSONA = "eqbench4_persona_agent"
RESOURCES = "eqbench4_resources_server"
CONVERSATION_FIELDS = {"transcript", "conversation_status", "conversation_error", "policy_stop_reason"}
EXAMPLE_ROWS = Path(__file__).parents[3] / "resources_servers/eqbench4/data/example.jsonl"


def persona_text(dialogue: str, *, internal: str = "thinking", status: str = "active") -> str:
    return (
        f"<internal>\n{internal}\n</internal>\n\n<dialogue>\n{dialogue}\n</dialogue>\n\n<status>\n{status}\n</status>"
    )


def model_response(text: str, *, tokens: int = 10) -> dict[str, Any]:
    return {
        "id": f"resp-{abs(hash(text))}",
        "created_at": 1,
        "model": "m",
        "object": "response",
        "output": [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2 * tokens,
        },
    }


def marked_response(*, metadata: dict | None = None, incomplete: str | None = None, refusal: bool = False) -> dict:
    """A policy reply without visible text, marked the way the model servers mark why."""
    reply = model_response("<think>only thoughts, no answer")
    if refusal:
        reply["output"][0]["content"] = [{"type": "refusal", "refusal": "Blocked by the provider."}]
    if incomplete:
        reply |= {"status": "incomplete", "incomplete_details": {"reason": incomplete}}
    if metadata:
        reply["metadata"] = metadata
    return reply


def failed_reply(status: int) -> dict:
    """A failed model call as an agent session reports it: no output, and the cause in its metadata."""
    return model_response("") | {
        "output": [],
        "status": "failed",
        "usage": None,
        "error": {"code": "server_error", "message": f"Model HTTP {status}: unavailable"},
        "metadata": {
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(
                failure_kind=MODEL_REQUEST_ERROR, status_code=status
            ).model_dump_json()
        },
    }


class Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict[str, Any], cookies: dict[str, str] | None = None) -> None:
        self.body = orjson.dumps(payload)
        self.cookies = {name: Cookie(value) for name, value in (cookies or {}).items()}
        self.headers: dict[str, str] = {}

    async def read(self) -> bytes:
        return self.body


class ErrorResponse(FakeResponse):
    """An HTTP error status from a Gym server."""

    ok = False

    def __init__(self, status: int) -> None:
        super().__init__({})
        self.status = status
        url = URL("http://server/v1/responses")
        self.request_info = RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)

        class Content:
            @staticmethod
            async def read() -> bytes:
                return b"unavailable"

        self.content = Content()

    def raise_for_status(self) -> None:
        raise ClientResponseError(self.request_info, (), status=self.status, message="error", headers=None)


def observations() -> dict[str, Any]:
    """A session's observations as simple_agent names them, whichever agent instance runs it."""
    return AgentObservationBundle(
        source="simple_agent", records=[AgentInvocation(invocation_id="root", status="completed")]
    ).model_dump(mode="json")


class FakeServers(ServerClient):
    """Gym's servers as the environment server reaches them: the resources server, and an agent session for each
    participant whose replies come from per-agent scripts (a reply, or an exception or error response to give)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: list[Any]
    persona: list[Any]
    calls: list[tuple[str, str, dict[str, Any]]] = []
    verify_reward: float = 0.75
    fail: dict[tuple[str, str], Any] = {}

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> FakeResponse:
        body = kwargs.get("json")
        payload = body.model_dump(mode="json") if hasattr(body, "model_dump") else body
        self.calls.append((server_name, url_path, {**kwargs, "json": payload}))
        failure = self.fail.get((server_name, url_path))
        if failure is not None:
            if isinstance(failure, Exception):
                raise failure
            return failure
        if url_path == "/seed_session":
            return FakeResponse({"resources_session_id": payload["resources_session_id"]}, {"rs": "seeded"})
        if url_path == "/v1/agent_sessions":
            return FakeResponse({"agent_session_id": payload["agent_session_id"]}, {f"{server_name}-session": "1"})
        if url_path == "/v1/agent_sessions/close":
            return FakeResponse(
                {"agent_session_id": payload["agent_session_id"], "agent_observations": observations()}
            )
        if url_path == "/verify":
            return FakeResponse(
                payload | {"reward": self.verify_reward, "comparisons": [{"opponent": "x", "fraction": 0.75}]}
            )
        if url_path == "/close_session":
            return FakeResponse({})
        if url_path == "/aggregate_metrics":
            return FakeResponse(
                {"agent_metrics": {"eqbench4_elo": 1200.0}, "key_metrics": {}, "group_level_metrics": []}
            )
        script = self.policy if server_name == POLICY else self.persona
        reply = script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, FakeResponse):
            return reply
        if isinstance(reply, str):
            reply = model_response(reply, tokens=10 if server_name == POLICY else 99)
        return FakeResponse(reply, {"model": "cookie"})

    def bodies(self, server_name: str) -> list[dict[str, Any]]:
        return [
            call["json"] for name, path, call in self.calls if name == server_name and path.endswith("/v1/responses")
        ]

    def paths(self, server_name: str) -> list[str]:
        return [path for name, path, _ in self.calls if name == server_name and path.endswith("/v1/responses")]

    def verify_body(self) -> dict[str, Any]:
        return next(call["json"] for _, path, call in self.calls if path == "/verify")

    def sequence(self) -> list[tuple[str, str]]:
        return [(name, "/v1/responses" if path.endswith("/v1/responses") else path) for name, path, _ in self.calls]


def make_server(
    *, policy: list[Any], persona: list[Any], token_capture: bool = False, **overrides: Any
) -> tuple[EQBench4EnvironmentServer, FakeServers]:
    global_config = OmegaConf.create(
        {
            POLICY: {"responses_api_agents": {"simple_agent": {"token_id_capture": token_capture}}},
            PERSONA: {"responses_api_agents": {"simple_agent": {"token_id_capture": token_capture}}},
            "token_id_capture": {"enabled": token_capture},
        }
    )
    servers = FakeServers(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        policy=list(policy),
        persona=list(persona),
        calls=[],
    )
    settings = {
        "name": "eqbench4_environment_server",
        "host": "127.0.0.1",
        "port": 0,
        "entrypoint": "app.py",
        "resources_server": {"type": "resources_servers", "name": RESOURCES},
        "agent_server": {"type": "responses_api_agents", "name": POLICY},
        "persona_agent": {"type": "responses_api_agents", "name": PERSONA},
        "persona_responses_create_params": {"input": [], "temperature": 0.7, "max_output_tokens": 100},
        "persona_correction_message": "Fix it: {problems}",
        "cleanup_timeout_seconds": 10,
    }
    config = EQBench4EnvironmentServerConfig.model_validate(settings | overrides)
    return EQBench4EnvironmentServer(config=config, server_client=servers), servers


def make_row(num_turns: int = 2) -> dict[str, Any]:
    return {
        "responses_create_params": {
            "input": [{"role": "system", "content": "sys 1"}, {"role": "user", "content": "[Presenting issue]"}],
            "temperature": 0.7,
        },
        "num_turns": num_turns,
        "evaluated_system_prompts": [f"sys {turn}" for turn in range(1, num_turns + 1)],
        "persona_system_prompt": "be the persona",
        "scenario_id": "s1",
        "task_id": "s1",
        # A field only the resources server reads passes through to /verify.
        "opponents": [{"model": "x"}],
    }


def make_request(row: dict[str, Any] | None = None, **updates: Any) -> EQBench4EpisodeRequest:
    return EQBench4EpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="eqbench4:example", task_id="s1"),
            task_input=SingleAgentTurnTaskInput.model_validate(row or make_row()),
        ),
        **updates,
    )


def contents(body: dict[str, Any]) -> list[tuple[str, str]]:
    return [(message["role"], message["content"]) for message in body["input"]]


def closed_every_session(servers: FakeServers) -> bool:
    closes = [(name, path) for name, path, _ in servers.calls if path.endswith("close") or path == "/close_session"]
    return set(closes) == {
        (POLICY, "/v1/agent_sessions/close"),
        (PERSONA, "/v1/agent_sessions/close"),
        (RESOURCES, "/close_session"),
    }


def test_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "app.py"), run_name="environment_servers.eqbench4.worker_test"
    )

    assert namespace["app"] is worker_app


async def test_full_conversation_alternates_and_ends_with_the_persona() -> None:
    server, servers = make_server(
        policy=["<think>plan</think>Hello, what's up?", "That sounds hard."],
        persona=[persona_text("Not much.", internal="guarded"), persona_text("Yeah.")],
    )
    response = await server.run_request(make_request())

    assert response.failure is None, response.failure
    # Seed every session, alternate the turns, close the agent sessions, grade, then close the resources session.
    assert servers.sequence() == [
        (RESOURCES, "/seed_session"),
        (POLICY, "/v1/agent_sessions"),
        (PERSONA, "/v1/agent_sessions"),
        (POLICY, "/v1/responses"),
        (PERSONA, "/v1/responses"),
        (POLICY, "/v1/responses"),
        (PERSONA, "/v1/responses"),
        (POLICY, "/v1/agent_sessions/close"),
        (PERSONA, "/v1/agent_sessions/close"),
        (RESOURCES, "/verify"),
        (RESOURCES, "/close_session"),
    ]
    # Each policy activation is the turn's whole input: the turn's system prompt and the dialogue so far.
    policy = servers.bodies(POLICY)
    assert contents(policy[0]) == [("system", "sys 1"), ("user", "[Presenting issue]")]
    assert contents(policy[1]) == [("system", "sys 2"), ("assistant", "Hello, what's up?"), ("user", "Not much.")]
    assert policy[1]["temperature"] == 0.7
    persona = servers.bodies(PERSONA)
    assert contents(persona[0]) == [("system", "be the persona"), ("user", "[Turn 1 of 2]\n\nHello, what's up?")]
    assert contents(persona[1]) == [
        ("system", "be the persona"),
        ("user", "Hello, what's up?"),
        ("assistant", persona_text("Not much.", internal="guarded")),
        ("user", "[Turn 2 of 2]\n\nThat sounds hard."),
    ]
    assert persona[0]["temperature"] == 0.7 and persona[0]["max_output_tokens"] == 100
    # Each activation carries its session's cookies, which the agent's replies may add to.
    activation = next(call for name, path, call in servers.calls if name == POLICY and path.endswith("/v1/responses"))
    assert activation["cookies"][f"{POLICY}-session"] == "1"

    verify = servers.verify_body()
    assert verify["conversation_status"] == "completed" and verify["conversation_error"] is None
    assert [(m["role"], m["content"]) for m in verify["transcript"]] == [
        ("assistant", "Hello, what's up?"),
        ("user", "Not much."),
        ("assistant", "That sounds hard."),
        ("user", "Yeah."),
    ]
    assert verify["transcript"][1]["raw_response"] == persona_text("Not much.", internal="guarded")
    assert verify["response"]["output"][0]["content"][0]["text"] == "That sounds hard."
    # The row's fields reach the resources server as its flat verify body, under the resources session's cookie.
    assert verify["scenario_id"] == "s1" and verify["opponents"] == [{"model": "x"}]
    assert verify["responses_create_params"]["input"][0]["content"] == "sys 1"
    verify_call = next(call for _, path, call in servers.calls if path == "/verify")
    assert verify_call["cookies"] == {"rs": "seeded"}
    # Usage covers both sessions' calls, the invocation tree the observations publish: two policy calls of 10 tokens
    # and two persona calls of 99.
    assert verify["response"]["usage"]["input_tokens"] == 218

    result = response.result
    assert result.reward == 0.75 and result.model_extra["comparisons"] == [{"opponent": "x", "fraction": 0.75}]
    # The result is the verify response itself, so the saved row has the flat verify-response shape; Gym binds the
    # verifier's grading to the response it saw, which the result keeps unchanged.
    graded = {"response": NeMoGymResponse.model_validate(verify["response"]).model_dump(mode="json"), "reward": 0.75}
    assert grading_result_binding(result.model_dump(mode="json")) == grading_result_binding(graded) is not None
    # The response carries the conversation, which a re-judge rebuilt from the task and the saved response reads.
    carried = json.loads(verify["response"]["metadata"][CONVERSATION_METADATA_KEY])
    assert carried == {key: verify[key] for key in carried}
    assert set(carried) == {"transcript", "conversation_status", "conversation_error", "policy_stop_reason"}
    # Both sessions' observations, each invocation named after its agent instance, so neither session's captured
    # calls join the other's invocations.
    dumped = result.model_dump(mode="json")
    invocations = [record["invocation_id"] for record in dumped["ng_agent_observations"]["records"]]
    assert invocations == [f"{POLICY}:root", f"{PERSONA}:root"]


async def test_persona_can_end_the_conversation() -> None:
    server, servers = make_server(policy=["Hi."], persona=[persona_text("I'm leaving.", status=" Ended ")])
    response = await server.run_request(make_request(make_row(num_turns=3)))
    assert response.failure is None
    verify = servers.verify_body()
    assert verify["conversation_status"] == "persona_ended"
    assert len(verify["transcript"]) == 2


async def test_unusable_persona_replies_are_corrected_then_fail_the_episode_for_a_retry() -> None:
    server, servers = make_server(
        policy=["Hi.", "Go on."],
        persona=[
            "<dialogue>no monologue</dialogue>",
            persona_text("Fine."),
            "<internal>x</internal><dialogue>  </dialogue>",
            "<internal>x</internal>",
        ],
        persona_max_attempts=2,
    )
    response = await server.run_request(make_request())

    # The simulator broke the conversation, not the policy: an environment failure that a resume runs again, never a
    # score against the policy.
    failure = response.failure
    assert failure is not None and response.result is None
    assert failure.failure_kind == PERSONA_FAILED and failure.terminal is False and failure.stage == "agent"
    assert "turn 2 after 2 attempt(s): no <dialogue> section" in failure.failure_reason
    persona = servers.bodies(PERSONA)
    assert len(persona) == 4
    assert contents(persona[1])[-2:] == [
        ("assistant", "<dialogue>no monologue</dialogue>"),
        ("user", "Fix it: no <internal> section"),
    ]
    assert not [path for _, path, _ in servers.calls if path == "/verify"]
    assert closed_every_session(servers)
    # The failure keeps what each participant did before it, and no answer: the conversation broke off, so a resumed
    # run runs it again rather than recovery grading it.
    assert failure.partial_response is None
    assert [record.invocation_id for record in failure.agent_observations.records] == [
        f"{POLICY}:root",
        f"{PERSONA}:root",
    ]


@pytest.mark.parametrize(
    ("outage", "terminal"),
    [(ClientConnectionError("persona endpoint down"), False), (ErrorResponse(503), False), (ErrorResponse(400), True)],
)
async def test_a_persona_outage_is_the_environments_failure(outage: Any, terminal: bool) -> None:
    server, servers = make_server(policy=["Hi."], persona=[outage])
    response = await server.run_request(make_request())

    assert response.failure.failure_kind == PERSONA_FAILED
    # A refused connection or a 5xx clears on a later attempt; a 4xx refuses the request itself.
    assert response.failure.terminal is terminal
    assert not [path for _, path, _ in servers.calls if path == "/verify"]
    assert closed_every_session(servers)


@pytest.mark.parametrize(("status", "terminal"), [(503, False), (500, False)])
async def test_a_failed_call_the_persona_session_reports_is_the_environments_failure(
    status: int, terminal: bool
) -> None:
    server, servers = make_server(policy=["Hi."], persona=[failed_reply(status)])
    response = await server.run_request(make_request())
    assert response.failure.failure_kind == PERSONA_FAILED and response.failure.terminal is terminal
    assert f"Model HTTP {status}" in response.failure.failure_reason
    assert not [path for _, path, _ in servers.calls if path == "/verify"]


async def test_a_policy_session_that_cannot_be_seeded_fails_the_episode_for_a_retry() -> None:
    server, servers = make_server(policy=[], persona=[])
    servers.fail = {(POLICY, "/v1/agent_sessions"): ClientConnectionError("down")}
    response = await server.run_request(make_request())
    assert response.failure.stage == "seed" and response.failure.terminal is False
    assert response.failure.failure_kind != PERSONA_FAILED
    assert not servers.bodies(POLICY) and closed_every_session(servers)


async def test_a_persona_session_that_cannot_be_seeded_is_the_environments_failure() -> None:
    server, servers = make_server(policy=[], persona=[])
    servers.fail = {(PERSONA, "/v1/agent_sessions"): ClientConnectionError("down")}
    response = await server.run_request(make_request())
    assert response.failure.failure_kind == PERSONA_FAILED and response.failure.terminal is False
    assert not servers.bodies(POLICY)
    # Every session registered for cleanup is closed, including the one whose seed failed.
    assert closed_every_session(servers)


@pytest.mark.parametrize(
    ("outage", "terminal"),
    [
        (ClientConnectionError("policy endpoint down"), False),
        (ErrorResponse(503), False),
        (FakeResponse(failed_reply(503)), False),
    ],
)
async def test_a_policy_outage_fails_the_episode_so_it_is_retried_not_scored(outage: Any, terminal: bool) -> None:
    server, servers = make_server(policy=["Hi.", outage], persona=[persona_text("Go on.")])
    response = await server.run_request(make_request())

    # Turn 1 is answered; the policy's turn-2 call fails. A policy stop scores 0, so an outage must not become one:
    # the episode fails, a resume runs it again, and nothing is verified.
    assert response.failure is not None and response.failure.terminal is terminal
    assert response.failure.failure_kind != PERSONA_FAILED
    assert response.failure.partial_response is None and response.failure.agent_observations is not None
    assert len(servers.bodies(POLICY)) == 2 and len(servers.bodies(PERSONA)) == 1
    assert not [path for _, path, _ in servers.calls if path == "/verify"]
    assert closed_every_session(servers)


async def test_a_policy_request_the_model_refuses_as_it_stands_is_a_stop_not_asked_again() -> None:
    # A 400 grades what there is (the outcome policy's verify): asking again would be refused the same way.
    server, servers = make_server(policy=["Hello.", FakeResponse(failed_reply(400))], persona=[persona_text("Go on.")])
    response = await server.run_request(make_request())

    assert response.failure is None
    verify = servers.verify_body()
    assert len(servers.bodies(POLICY)) == 2
    assert verify["conversation_status"] == "policy_failed"
    assert verify["conversation_error"] == "no visible policy reply in turn 2 after 1 attempt(s): empty_reply"


async def test_a_policy_that_never_replies_stops_the_conversation_after_seven_attempts() -> None:
    server, servers = make_server(
        policy=["<think>only thoughts</think>", "", "  "] + ["<think>cut off"] * 4, persona=[]
    )
    response = await server.run_request(make_request())
    assert response.failure is None
    verify = servers.verify_body()
    assert verify["conversation_status"] == "policy_failed"
    assert verify["policy_stop_reason"] == "empty_reply"
    assert verify["conversation_error"] == "no visible policy reply in turn 1 after 7 attempt(s): empty_reply"
    assert verify["transcript"] == []
    assert len(servers.bodies(POLICY)) == 7
    assert verify["response"]["usage"]["input_tokens"] == 70
    # The persona never spoke, but its session is closed all the same.
    assert closed_every_session(servers)


async def test_a_context_overflow_is_not_asked_again_and_is_reported() -> None:
    overflow = marked_response(metadata={"nemo_gym_context_window_exceeded": "true"})
    server, servers = make_server(policy=["Hello.", overflow], persona=[persona_text("Go on.")])
    await server.run_request(make_request())
    verify = servers.verify_body()
    assert len(servers.bodies(POLICY)) == 2
    assert (verify["conversation_status"], verify["policy_stop_reason"]) == ("policy_failed", "context_limit")
    assert verify["conversation_error"] == "no visible policy reply in turn 2 after 1 attempt(s): context_limit"
    assert [message["role"] for message in verify["transcript"]] == ["assistant", "user"]


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (
            marked_response(metadata={"nemo_gym_context_window_exceeded": "true"}, incomplete="max_output_tokens"),
            "context_limit",
        ),
        (marked_response(refusal=True), "content_filter"),
        (marked_response(incomplete="content_filter"), "content_filter"),
        (marked_response(incomplete="max_output_tokens"), "output_budget"),
        (marked_response(metadata={"other": "x"}), "empty_reply"),
        (model_response(""), "empty_reply"),
    ],
)
def test_policy_stop_reason(reply: dict, reason: str) -> None:
    assert policy_stop_reason(NeMoGymResponse.model_validate(reply)) == reason


async def test_a_retried_policy_reply_continues_the_conversation_and_reports_no_stop_reason() -> None:
    server, servers = make_server(policy=["", "Hello."], persona=[persona_text("Bye.", status="done")])
    await server.run_request(make_request())
    verify = servers.verify_body()
    assert verify["conversation_status"] == "persona_ended" and verify["policy_stop_reason"] is None
    assert len(servers.bodies(POLICY)) == 2


@pytest.mark.parametrize(
    ("text", "problems", "ended"),
    [
        (persona_text("ok"), [], False),
        (persona_text("ok", status="STOP"), [], True),
        (persona_text("ok", status="paused"), [], False),
        ("<internal> </internal><dialogue>ok</dialogue>", ["no <internal> section"], False),
        ("<internal>x</internal><dialogue>ok <state>{}</state></dialogue>", ["control tags inside <dialogue>"], False),
        ("<internal>x</internal><dialogue>[INTERNAL] leak</dialogue>", ["control tags inside <dialogue>"], False),
        ("", ["no <internal> section", "no <dialogue> section"], False),
    ],
)
def test_persona_reply_validation(text: str, problems: list[str], ended: bool) -> None:
    reply = PersonaReply(text)
    assert reply.problems == problems
    assert reply.ended is ended


def test_persona_messages_show_only_the_latest_reply_in_full() -> None:
    transcript = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u1", "raw_response": "RAW1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u2", "raw_response": "RAW2"},
        {"role": "assistant", "content": "a3"},
    ]
    assert [(m["role"], m["content"]) for m in persona_messages("P", transcript, 3, 8)] == [
        ("system", "P"),
        ("user", "a1"),
        ("assistant", "u1"),
        ("user", "a2"),
        ("assistant", "RAW2"),
        ("user", "[Turn 3 of 8]\n\na3"),
    ]


@pytest.mark.parametrize("token_capture", [False, True])
async def test_a_training_frameworks_sampling_and_capture_reach_the_policy_but_not_the_persona(
    token_capture: bool,
) -> None:
    server, servers = make_server(
        policy=["Hi."], persona=[persona_text("Bye.", status="done")], token_capture=token_capture
    )
    sampling = ModelCallSampling(temperature=0.6, top_p=0.95, max_output_tokens=64)
    response = await server.run_request(make_request(model_sampling=sampling))

    assert response.failure is None, response.failure
    capture = "/training-token-capture" if token_capture else ""
    assert servers.paths(POLICY) == [
        f"/ng-rollout/rollout-a2{capture}/ng-sampling/{encode_model_sampling(sampling)}/v1/responses"
    ]
    # The persona is not trained: it keeps its own sampling and is never captured for training.
    assert servers.paths(PERSONA) == ["/ng-rollout/rollout-a2/ng-sampling/null/v1/responses"]


async def test_gym_only_evaluation_activates_both_sessions_under_the_rollout() -> None:
    server, servers = make_server(policy=["Hi."], persona=[persona_text("Bye.", status="done")])
    await server.run_request(make_request())
    assert servers.paths(POLICY) == servers.paths(PERSONA) == ["/ng-rollout/rollout-a2/v1/responses"]


async def test_a_validation_rollout_skips_the_policys_capture() -> None:
    server, servers = make_server(policy=["Hi."], persona=[persona_text("Bye.", status="done")], token_capture=True)
    await server.run_request(make_request(skip_token_capture=True))
    assert servers.paths(POLICY) == ["/ng-rollout/rollout-a2/v1/responses"]


@pytest.mark.parametrize("silent", ["policy", "persona"])
async def test_usage_is_reported_only_when_every_call_reported_it(silent: str) -> None:
    quiet_policy = model_response("Hi.")
    quiet_persona = model_response(persona_text("Bye.", status="done"))
    del (quiet_policy if silent == "policy" else quiet_persona)["usage"]
    policy = ["", quiet_policy] if silent == "policy" else ["Hi."]
    persona = [quiet_persona] if silent == "persona" else [persona_text("Bye.", status="done")]
    server, servers = make_server(policy=policy, persona=persona)
    await server.run_request(make_request())
    # A sum of some of the calls would read as all of them.
    assert servers.verify_body()["response"]["usage"] is None


def test_a_row_must_give_every_policy_turn_its_system_prompt() -> None:
    with pytest.raises(ValueError, match="must cover every policy turn"):
        EQBench4Conversation.model_validate(make_row(num_turns=2) | {"evaluated_system_prompts": ["sys 1"]})


async def test_a_row_that_is_not_a_scenario_fails_before_any_session_is_seeded() -> None:
    server, servers = make_server(policy=[], persona=[])
    response = await server.run_request(make_request(make_row() | {"evaluated_system_prompts": ["sys 1"]}))
    assert response.failure.terminal is True and response.failure.stage == "seed"
    assert servers.calls == []


async def test_a_verifier_outage_fails_the_episode_with_its_evidence() -> None:
    server, servers = make_server(policy=["Hi."], persona=[persona_text("Bye.", status="done")])
    servers.fail = {(RESOURCES, "/verify"): ErrorResponse(503)}
    response = await server.run_request(make_request())
    failure = response.failure
    assert failure.stage == "verification" and failure.terminal is False
    # The conversation is complete, so the failure keeps the response to grade, conversation included: recovery grades
    # it again rather than running the conversation again.
    assert failure.partial_response.output[0].content[0].text == "Hi."
    carried = json.loads(failure.partial_response.metadata[CONVERSATION_METADATA_KEY])
    assert carried["conversation_status"] == "persona_ended" and len(carried["transcript"]) == 2
    assert len(failure.agent_observations.records) == 2
    assert closed_every_session(servers)


async def test_aggregate_metrics_are_the_resources_servers_with_performance_metrics() -> None:
    server, servers = make_server(policy=[], persona=[])
    metrics = await server.aggregate_metrics(
        AggregateMetricsRequest(verify_responses=[{"reward": 0.5, "ng_perf": {"elapsed_s": 1.0}}])
    )
    assert metrics.agent_metrics["eqbench4_elo"] == 1200.0
    forwarded = next(call["json"] for name, path, call in servers.calls if path == "/aggregate_metrics")
    # The performance summary is reduced here, once, not by the resources server.
    assert forwarded["verify_responses"] == [{"reward": 0.5}]


def test_the_row_fields_the_environment_server_and_the_resources_server_share_do_not_drift() -> None:
    """The prepared row's schema (TaskData) and the models that read the row declare the fields they share alike."""
    from resources_servers.eqbench4.app import EQBench4RunRequest
    from resources_servers.eqbench4.task_data import TaskData

    def declared(model: Any, name: str) -> tuple[Any, ...]:
        field = model.model_fields[name]
        return (field.annotation, field.is_required(), field.get_default(), repr(field.metadata))

    for model in (EQBench4Conversation, EQBench4RunRequest):
        shared = (set(model.model_fields) & set(TaskData.model_fields)) - {"responses_create_params"}
        assert shared, model
        for name in sorted(shared):
            assert declared(model, name) == declared(TaskData, name), (model.__module__, name)
    # Every field the environment server reads from the row is part of the row's schema.
    assert set(EQBench4Conversation.model_fields) <= set(TaskData.model_fields)


async def test_the_verify_body_is_what_the_resources_server_reads_and_a_rejudge_rebuilds() -> None:
    """The conversation sent to /verify validates as the resources server's verify request for each way a
    conversation ends, so a value one side adds alone fails here, not as a 422 in a run. A re-judge, which rebuilds
    the request from the task and the saved response alone, reads the same conversation."""
    from resources_servers.eqbench4.app import EQBench4VerifyRequest

    row = json.loads(EXAMPLE_ROWS.read_text().splitlines()[0]) | {"num_turns": 1}
    endings = {
        "completed": (["Hi."], [persona_text("Ok.")]),
        "persona_ended": (["Hi."], [persona_text("Bye.", status="done")]),
        "policy_failed": ([marked_response(incomplete="max_output_tokens")] * 7, []),
    }
    for status, (policy, persona) in endings.items():
        server, servers = make_server(policy=policy, persona=persona)
        response = await server.run_request(make_request(row))
        assert response.failure is None, response.failure
        body = servers.verify_body()
        verify = EQBench4VerifyRequest.model_validate(body)
        assert verify.conversation_status == status
        rejudged = EQBench4VerifyRequest.model_validate(
            {key: value for key, value in body.items() if key not in CONVERSATION_FIELDS}
            | {"response": body["response"]}
        )
        assert rejudged.transcript == verify.transcript and rejudged.conversation_status == status
    assert verify.policy_stop_reason == "output_budget"


def test_the_example_rows_are_valid_tasks() -> None:
    for line in EXAMPLE_ROWS.read_text().splitlines():
        task_input = SingleAgentTurnTaskInput.model_validate(json.loads(line))
        EQBench4Conversation.model_validate(task_input.task_data)


def test_scoped_observations_name_every_invocation_after_its_agent_instance() -> None:
    reference = ModelCallRef(model_ref={"type": "responses_api_models", "name": "policy_model"}, response_id="r-1")
    bundle = AgentObservationBundle(
        source="simple_agent",
        records=[
            AgentInvocation(invocation_id="root", model_calls=[reference]),
            AgentInvocation(invocation_id="activation-2", parent_invocation_id="root"),
        ],
        gaps=[ObservationGap(code="model_call_reference_unavailable", invocation_id="activation-2")],
    )
    scoped = scoped_observations(bundle, POLICY)
    root, second = scoped.records
    assert (root.invocation_id, second.invocation_id, second.parent_invocation_id) == (
        f"{POLICY}:root",
        f"{POLICY}:activation-2",
        f"{POLICY}:root",
    )
    # The calls stay referenced, so the policy's turns still bind to its invocations.
    assert root.model_calls == [reference]
    assert scoped.gaps[0].invocation_id == f"{POLICY}:activation-2"
    assert scoped_observations(None, POLICY) is None


def test_published_observations_hold_every_session_that_reported() -> None:
    policy, persona = _AgentSession(POLICY), _AgentSession(PERSONA)
    assert published_observations(policy, persona) is None
    persona.observations = scoped_observations(AgentObservationBundle.model_validate(observations()), PERSONA)
    assert [record.invocation_id for record in published_observations(policy, persona).records] == [f"{PERSONA}:root"]
