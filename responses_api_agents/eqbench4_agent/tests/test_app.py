# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.trajectory.trajectory_record import grading_result_binding
from responses_api_agents.eqbench4_agent.app import (
    EQBench4Agent,
    EQBench4AgentConfig,
    EQBench4RunRequest,
    PersonaReply,
    persona_messages,
    policy_stop_reason,
)


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


class FakeResponse:
    ok = True

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.cookies: dict[str, str] = {"session": "s"}

    async def json(self) -> dict[str, Any]:
        return self.payload


class FakeServers:
    """Routes the agent's calls: seed and verify to the resources server, replies from per-server scripts."""

    def __init__(self, policy: list[str], persona: list[str]) -> None:
        self.policy = list(policy)
        self.persona = list(persona)
        self.calls: list[tuple[str, str, Any]] = []

    async def post(self, *, server_name: str, url_path: str, json: Any, **_: Any) -> FakeResponse:
        payload = json.model_dump(mode="json") if hasattr(json, "model_dump") else json
        self.calls.append((server_name, url_path, payload))
        if url_path == "/seed_session":
            return FakeResponse({})
        if url_path == "/verify":
            return FakeResponse(payload | {"reward": 0.75, "comparisons": [{"opponent": "x", "fraction": 0.75}]})
        if server_name == "eqbench4_agent":
            reply = self.policy.pop(0)
            return FakeResponse(reply if isinstance(reply, dict) else model_response(reply))
        return FakeResponse(model_response(self.persona.pop(0), tokens=99))

    def bodies(self, server_name: str) -> list[dict[str, Any]]:
        return [payload for name, path, payload in self.calls if name == server_name and path == "/v1/responses"]

    def verify_body(self) -> dict[str, Any]:
        return next(payload for _, path, payload in self.calls if path == "/verify")


def make_agent(servers: FakeServers, **overrides: Any) -> EQBench4Agent:
    settings = {
        "host": "127.0.0.1",
        "port": 0,
        "entrypoint": "app.py",
        "name": "eqbench4_agent",
        "resources_server": {"type": "resources_servers", "name": "eqbench4"},
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "persona_model_server": {"type": "responses_api_models", "name": "persona"},
        "persona_responses_create_params": {"input": [], "temperature": 0.7, "max_output_tokens": 100},
        "persona_correction_message": "Fix it: {problems}",
    }
    client = MagicMock(spec=ServerClient)
    client.post = servers.post
    return EQBench4Agent(config=EQBench4AgentConfig.model_validate(settings | overrides), server_client=client)


def make_body(num_turns: int = 2) -> EQBench4RunRequest:
    return EQBench4RunRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "system", "content": "sys 1"}, {"role": "user", "content": "[Presenting issue]"}],
                "temperature": 0.7,
            },
            "num_turns": num_turns,
            "evaluated_system_prompts": [f"sys {turn}" for turn in range(1, num_turns + 1)],
            "persona_system_prompt": "be the persona",
            "scenario_id": "s1",
        }
    )


def request() -> MagicMock:
    mock = MagicMock()
    mock.cookies = {}
    return mock


def contents(body: dict[str, Any]) -> list[tuple[str, str]]:
    return [(message["role"], message["content"]) for message in body["input"]]


async def test_full_conversation_alternates_and_ends_with_the_persona() -> None:
    servers = FakeServers(
        policy=["<think>plan</think>Hello, what's up?", "That sounds hard."],
        persona=[persona_text("Not much.", internal="guarded"), persona_text("Yeah.")],
    )
    result = await make_agent(servers).run(request(), make_body())

    policy = servers.bodies("eqbench4_agent")
    assert contents(policy[0]) == [("system", "sys 1"), ("user", "[Presenting issue]")]
    assert contents(policy[1]) == [
        ("system", "sys 2"),
        ("assistant", "Hello, what's up?"),
        ("user", "Not much."),
    ]
    assert policy[1]["temperature"] == 0.7
    persona = servers.bodies("persona")
    assert contents(persona[0]) == [("system", "be the persona"), ("user", "[Turn 1 of 2]\n\nHello, what's up?")]
    assert contents(persona[1]) == [
        ("system", "be the persona"),
        ("user", "Hello, what's up?"),
        ("assistant", persona_text("Not much.", internal="guarded")),
        ("user", "[Turn 2 of 2]\n\nThat sounds hard."),
    ]
    assert persona[0]["temperature"] == 0.7 and persona[0]["max_output_tokens"] == 100

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
    assert verify["scenario_id"] == "s1"
    # Usage covers the whole conversation: two policy calls of 10 tokens and two persona calls of 99.
    assert result.response.usage.input_tokens == 218
    # Gym binds the verifier's grading to the response the verifier saw, so the saved row must carry that response,
    # usage included.
    assert verify["response"]["usage"]["input_tokens"] == 218
    graded = {"response": NeMoGymResponse.model_validate(verify["response"]).model_dump(mode="json"), "reward": 0.75}
    assert grading_result_binding(result) == grading_result_binding(graded) is not None
    # Without model-call capture there is no trajectory record.
    assert "ng_trajectory" not in (result.model_extra or {})
    assert result.reward == 0.75 and result.comparisons == [{"opponent": "x", "fraction": 0.75}]


async def test_persona_can_end_the_conversation() -> None:
    servers = FakeServers(policy=["Hi."], persona=[persona_text("I'm leaving.", status=" Ended ")])
    await make_agent(servers).run(request(), make_body(num_turns=3))
    verify = servers.verify_body()
    assert verify["conversation_status"] == "persona_ended"
    assert len(verify["transcript"]) == 2


async def test_unusable_persona_replies_are_corrected_then_abandoned() -> None:
    servers = FakeServers(
        policy=["Hi.", "Go on."],
        persona=[
            "<dialogue>no monologue</dialogue>",
            persona_text("Fine."),
            "<internal>x</internal><dialogue>  </dialogue>",
            "<internal>x</internal>",
        ],
    )
    await make_agent(servers, persona_max_attempts=2).run(request(), make_body())
    persona = servers.bodies("persona")
    assert len(persona) == 4
    assert contents(persona[1])[-2:] == [
        ("assistant", "<dialogue>no monologue</dialogue>"),
        ("user", "Fix it: no <internal> section"),
    ]
    verify = servers.verify_body()
    assert verify["conversation_status"] == "persona_failed"
    assert verify["conversation_error"] == "persona turn 2: no <dialogue> section"
    assert [m["role"] for m in verify["transcript"]] == ["assistant", "user", "assistant"]


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


async def test_a_policy_that_never_replies_stops_the_conversation_after_seven_attempts() -> None:
    servers = FakeServers(policy=["<think>only thoughts</think>", "", "  "] + ["<think>cut off"] * 4, persona=[])
    result = await make_agent(servers).run(request(), make_body())
    verify = servers.verify_body()
    assert verify["conversation_status"] == "policy_failed"
    assert verify["policy_stop_reason"] == "empty_reply"
    assert verify["conversation_error"] == "no visible policy reply in turn 1 after 7 attempt(s): empty_reply"
    assert verify["transcript"] == []
    assert len(servers.bodies("eqbench4_agent")) == 7
    assert result.response.usage.input_tokens == 70


async def test_a_context_overflow_is_not_asked_again_and_is_reported() -> None:
    overflow = marked_response(metadata={"nemo_gym_context_window_exceeded": "true"})
    servers = FakeServers(policy=["Hello.", overflow], persona=[persona_text("Go on.")])
    await make_agent(servers).run(request(), make_body())
    verify = servers.verify_body()
    assert len(servers.bodies("eqbench4_agent")) == 2
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


async def test_a_completed_conversation_reports_no_stop_reason() -> None:
    servers = FakeServers(policy=["", "Hello."], persona=[persona_text("Bye.", status="done")])
    await make_agent(servers, policy_max_attempts=2).run(request(), make_body())
    assert servers.verify_body()["policy_stop_reason"] is None


async def test_retried_policy_reply_continues_the_conversation() -> None:
    servers = FakeServers(policy=["", "Hello."], persona=[persona_text("Bye.", status="done")])
    await make_agent(servers).run(request(), make_body())
    assert servers.verify_body()["conversation_status"] == "persona_ended"
    assert len(servers.bodies("eqbench4_agent")) == 2


@pytest.fixture
def capturing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model-call capture on, for rollout r-1."""
    monkeypatch.setattr(EQBench4Agent, "_model_call_capture_enabled", lambda self: True)
    monkeypatch.setattr(EQBench4Agent, "rollout_id_from_run", lambda self, body: "r-1")


def captured_body(num_turns: int = 2) -> EQBench4RunRequest:
    return EQBench4RunRequest.model_validate(make_body(num_turns).model_dump() | {"_ng_task_index": 4})


def turn_ids(trajectory: dict[str, Any]) -> list[tuple[str, int]]:
    return [(turn["invocation_id"], turn["turn_no"]) for turn in trajectory["turns"]]


async def test_capture_records_a_turn_per_model_call_owned_by_its_speaker(capturing: None) -> None:
    policy = [model_response("Hello, what's up?"), model_response("That sounds hard.")]
    persona = [persona_text("Not much."), persona_text("Yeah.")]
    servers = FakeServers(policy=policy, persona=persona)
    inbound = request()
    result = await make_agent(servers).run(inbound, captured_body())

    trajectory = result.model_extra["ng_trajectory"]
    assert (trajectory["task_id"], trajectory["rollout_id"]) == ("4", "r-1")
    assert turn_ids(trajectory) == [("root", 1), ("persona", 1), ("root", 2), ("persona", 2)]
    root, persona_invocation = trajectory["invocations"]
    assert (root["invocation_id"], root["status"]) == ("root", "completed")
    assert [ref["response_id"] for ref in root["model_calls"]] == [reply["id"] for reply in policy]
    assert {ref["model_ref"]["name"] for ref in root["model_calls"]} == {"policy_model"}
    assert (persona_invocation["invocation_id"], persona_invocation["status"]) == ("persona", "completed")
    assert {ref["model_ref"]["name"] for ref in persona_invocation["model_calls"]} == {"persona"}
    # Each turn names exactly the call it was, and carries what that call was asked and answered.
    first, persona_first = trajectory["turns"][:2]
    assert first["model_calls"] == [root["model_calls"][0]] and first["resolved"] is None
    assert [(m["role"], m["content"]) for m in first["question"]] == [
        ("system", "sys 1"),
        ("user", "[Presenting issue]"),
    ]
    assert first["answer"][0]["content"][0]["text"] == "Hello, what's up?"
    assert persona_first["question"][0] == {"role": "system", "content": "be the persona"}
    assert [gap["code"] for gap in trajectory["gaps"]] == ["resolution_unavailable"]
    assert trajectory["tool_calls"] == []
    # The record is also published for the failure report.
    assert inbound.state.nemo_gym_agent_trajectory.turns[-1].invocation_id == "persona"


async def test_capture_records_retried_replies_and_a_stopped_policy(capturing: None) -> None:
    servers = FakeServers(policy=["", "Hello.", "", ""], persona=[persona_text("Go on.")])
    result = await make_agent(servers, policy_max_attempts=2).run(request(), captured_body())
    trajectory = result.model_extra["ng_trajectory"]
    # Every policy call is a turn, a retried empty reply too.
    assert turn_ids(trajectory) == [("root", 1), ("root", 2), ("persona", 1), ("root", 3), ("root", 4)]
    root = trajectory["invocations"][0]
    assert (root["status"], root["stop_reason"]) == ("failed", "empty_reply")
    assert trajectory["invocations"][1]["status"] == "completed"


async def test_capture_marks_the_speaker_whose_call_failed(capturing: None) -> None:
    servers = FakeServers(policy=["Hi."], persona=[])

    async def post(*, server_name: str, url_path: str, json: Any, **kwargs: Any) -> FakeResponse:
        if server_name == "persona":
            raise ConnectionError("persona endpoint down")
        return await FakeServers.post(servers, server_name=server_name, url_path=url_path, json=json, **kwargs)

    agent = make_agent(servers)
    agent.server_client.post = post
    inbound = request()
    with pytest.raises(ConnectionError):
        await agent.run(inbound, captured_body())
    record = inbound.state.nemo_gym_agent_trajectory
    root, persona_invocation = record.invocations
    assert (root.status, root.error_type) == ("incomplete", None)
    # The persona's call raised: its invocation carries the failure, though it never answered.
    assert (persona_invocation.status, persona_invocation.error_type) == ("failed", "ConnectionError")
    assert persona_invocation.model_calls == []
    assert [(turn.invocation_id, turn.turn_no) for turn in record.turns] == [("root", 1)]


async def test_a_reply_without_a_response_id_is_reported_as_a_gap(capturing: None) -> None:
    unidentified = model_response("Hi.") | {"id": ""}
    servers = FakeServers(policy=[unidentified], persona=[persona_text("Bye.", status="done")])
    result = await make_agent(servers).run(request(), captured_body())
    trajectory = result.model_extra["ng_trajectory"]
    assert trajectory["turns"][0]["model_calls"] == []
    assert {"code": "model_call_reference_unavailable", "invocation_id": "root", "detail": "turn:1"} in [
        {key: gap.get(key) for key in ("code", "invocation_id", "detail")} for gap in trajectory["gaps"]
    ]


async def test_a_persona_failure_leaves_the_policy_invocation_incomplete(capturing: None) -> None:
    servers = FakeServers(policy=["Hi."], persona=["<dialogue>no monologue</dialogue>"])
    result = await make_agent(servers, persona_max_attempts=1).run(request(), captured_body())
    root, persona_invocation = result.model_extra["ng_trajectory"]["invocations"]
    assert (root["status"], persona_invocation["status"]) == ("incomplete", "failed")


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


async def test_responses_forwards_to_the_policy_model() -> None:
    servers = FakeServers(policy=[], persona=[])
    captured: dict[str, Any] = {}

    async def post(*, server_name: str, url_path: str, json: Any, **_: Any) -> FakeResponse:
        captured.update(server_name=server_name, url_path=url_path)
        return FakeResponse(model_response("hi"))

    agent = make_agent(servers)
    agent.server_client.post = post
    inbound = MagicMock()
    inbound.cookies = {}
    inbound.path_params = {}
    outbound = MagicMock()
    body = EQBench4RunRequest.model_validate(make_body().model_dump()).responses_create_params
    reply = await agent.responses(inbound, outbound, body)
    assert reply.output[0].content[0].text == "hi"
    assert captured == {"server_name": "policy_model", "url_path": "/v1/responses"}
    outbound.set_cookie.assert_called_once_with("session", "s")


async def test_aggregate_metrics_are_computed_by_the_resources_server() -> None:
    import json as json_module

    from nemo_gym.config import AggregateMetricsRequest

    captured: dict[str, Any] = {}

    async def post(*, server_name: str, url_path: str, json: Any, **_: Any) -> MagicMock:
        captured.update(server_name=server_name, url_path=url_path)
        response = MagicMock(ok=True)
        response.read = AsyncMock(return_value=json_module.dumps({"key_metrics": {"eqbench4_elo": 1234.5}}).encode())
        return response

    agent = make_agent(FakeServers(policy=[], persona=[]))
    agent.server_client.post = post
    metrics = await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[{"reward": 0.5}]))
    assert captured == {"server_name": "eqbench4", "url_path": "/aggregate_metrics"}
    assert metrics.key_metrics == {"eqbench4_elo": 1234.5}
