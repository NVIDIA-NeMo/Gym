# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code's run evidence and the outcome ladder's decision for each run event of a sandbox session."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents import AgentOutcome, OutcomeAction, RunEvidence, agent_outcome_policy
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.claude_code_agent import sandbox as adapter
from responses_api_agents.claude_code_agent.parsing import parse_stream_json
from responses_api_agents.claude_code_agent.tests.test_sandbox import agent, seed


REFUSAL = (
    "API Error: Claude Code is unable to respond to this request, which appears to violate our Usage Policy "
    "(https://www.anthropic.com/legal/aup). Try rephrasing the request."
)
# Claude Code 2.1.108's exact texts from a scripted-model smoke through Gym's vLLM model server (propagate flag on).
CONTEXT_OVERFLOW = (
    'API Error: 400 {"error":{"message":"This model\'s maximum context length is 32768 tokens. However, you '
    "requested 40000 tokens (39000 in the messages, 1000 in the completion). Please reduce the length of the "
    'messages or completion.","type":"invalid_request_error","param":null,"code":"context_length_exceeded"}}'
)
AUTH_FAILURE = (
    'Failed to authenticate. API Error: 401 {"error":{"message":"Incorrect API key provided",'
    '"type":"invalid_request_error","param":null,"code":"invalid_api_key"}}'
)


def reply(text: str = "working", **event) -> dict:
    return {
        "type": "assistant",
        "message": {"id": "msg_1", "model": "policy", "content": [{"type": "text", "text": text}]},
        **event,
    }


def api_error(text: str, **event) -> dict:
    """Claude Code 2.1.108's own message for a model request that failed after its retries (not a served reply)."""
    return {
        "type": "assistant",
        "parent_tool_use_id": None,
        "error": "unknown",
        "message": {
            "id": "local-uuid",
            "model": "<synthetic>",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
        **event,
    }


def error_result(text: str) -> dict:
    return {"type": "result", "subtype": "success", "is_error": True, "result": text}


SUCCESS = {"type": "result", "subtype": "success", "is_error": False}


def test_parser_keeps_the_status_of_the_api_error_the_run_ended_on():
    text = 'API Error: 503 {"error":{"message":"overloaded"}}'
    _, metadata = parse_stream_json("\n".join(map(json.dumps, [reply(), api_error(text), error_result(text)])))
    assert metadata["api_error_status"] == 503
    assert metadata["model_replied"] is True
    assert agent()._run_evidence(metadata, returncode=1, timed_out=False) == RunEvidence(
        has_result=False,
        timed_out=False,
        model_replied=True,
        error=agent().classify_outcome({"api_error_status": 503}),
    )


@pytest.mark.parametrize(
    "case", ["served_text", "subagent", "different_result", "recovered", "tool_output", "no_status"]
)
def test_http_text_that_did_not_end_the_run_is_not_its_status(case):
    text = "API Error: 400 bad request"
    events = [api_error(text), error_result(text)]
    if case == "served_text":
        # The model itself wrote the text; Claude Code's error messages are "<synthetic>".
        events[0] = reply(text)
    elif case == "subagent":
        events[0] = api_error(text, parent_tool_use_id="task")
    elif case == "different_result":
        events[1] = error_result("A tool failed")
    elif case == "recovered":
        events = [api_error(text), reply(), SUCCESS]
    elif case == "tool_output":
        events[0] = {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "t", "is_error": True, "content": text}]},
        }
    elif case == "no_status":
        events = [api_error("API Error: Connection error."), error_result("API Error: Connection error.")]
    _, metadata = parse_stream_json("\n".join(map(json.dumps, events)))
    assert "api_error_status" not in metadata
    assert agent().classify_outcome(metadata).failure_kind == "unknown"


@pytest.mark.parametrize(
    ("text", "outcome"),
    [
        (AUTH_FAILURE, AgentOutcome(failure_kind="model_request_error", status_code=401)),
        (CONTEXT_OVERFLOW, AgentOutcome(failure_kind="model_context_limit")),
        (
            "API Error: 400 Request validation failed: malformed message sequence",
            AgentOutcome(failure_kind="model_request_error", status_code=400),
        ),
        (
            "API Error: 503 Service Unavailable · check status.claude.com",
            AgentOutcome(failure_kind="model_request_error", status_code=503),
        ),
        (
            'API Error: 400 ["context_length_exceeded"]',
            AgentOutcome(failure_kind="model_request_error", status_code=400),
        ),
    ],
)
def test_the_status_is_found_after_a_prefix_and_a_raw_context_overflow_body_is_a_context_limit(text, outcome):
    _, metadata = parse_stream_json("\n".join(map(json.dumps, [api_error(text), error_result(text)])))
    assert agent().classify_outcome(metadata) == outcome


def test_a_subagent_reply_counts_as_a_model_reply_even_under_capture():
    raw = "\n".join(map(json.dumps, [reply(parent_tool_use_id="task"), api_error("API Error: 400 x")]))
    assert parse_stream_json(raw, fold_thinking=False)[1]["model_replied"] is True
    assert parse_stream_json(json.dumps(api_error("API Error: 400 x")))[1]["model_replied"] is False


async def run(monkeypatch, events: list[dict], *, timed_out: bool = False, code: int | None = 0):
    instance = agent()
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session",
            workdir="/app",
            artifacts=None,
            cleanup={"timed_out": timed_out, "return_code": code, "cleanup_confirmed": True},
            expired=False,
        ),
    )
    raw = "\n".join(map(json.dumps, events))
    monkeypatch.setattr(
        adapter.ClaudeCodeSandboxSession, "execute", AsyncMock(return_value=adapter.ClaudeCodeArtifacts(raw))
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    return await instance._sandbox_response(state, body, prompt="Question", system=None)


def api_failure(status: int | str, *, after_reply: bool) -> list[dict]:
    text = status if isinstance(status, str) else f'API Error: {status} {{"error":{{"message":"upstream {status}"}}}}'
    return [*([reply()] if after_reply else []), api_error(text), error_result(text)]


# event -> (stream-json events, timed out, exit code, status, ng_agent_outcome, Environment Server action).
# Before the outcome ladder, every failed row below was failed without ng_agent_outcome (graded), and the
# missing-result row was incomplete (graded).
EVENTS = {
    # (a) With the model server's propagate flag on, a context overflow reaches Claude Code as an HTTP 400 whose
    # body keeps the code context_length_exceeded. It is a context limit before or after a reply (decision 1).
    "context_overflow_after_reply": (
        api_failure(CONTEXT_OVERFLOW, after_reply=True),
        False,
        1,
        "incomplete",
        None,
        OutcomeAction.VERIFY,
    ),
    "context_overflow_first_request": (
        api_failure(CONTEXT_OVERFLOW, after_reply=False),
        False,
        1,
        "incomplete",
        None,
        OutcomeAction.VERIFY,
    ),
    "400_after_reply": (
        api_failure(400, after_reply=True),
        False,
        1,
        "failed",
        {"failure_kind": "model_request_error", "status_code": 400},
        OutcomeAction.VERIFY,
    ),
    "deadline_before_reply": ([], True, 143, "incomplete", None, OutcomeAction.VERIFY),
    "deadline_after_reply": ([reply()], True, 143, "incomplete", None, OutcomeAction.VERIFY),
    "no_result": ([reply()], False, 0, "failed", {"failure_kind": "agent_run_error"}, OutcomeAction.RETRY),
    "nonzero_exit": ([reply(), SUCCESS], False, 1, "failed", {"failure_kind": "agent_run_error"}, OutcomeAction.RETRY),
    "persistent_503": (
        api_failure(503, after_reply=True),
        False,
        1,
        "failed",
        {"failure_kind": "model_request_error", "status_code": 503},
        OutcomeAction.RETRY,
    ),
    "persistent_429_before_reply": (
        api_failure(429, after_reply=False),
        False,
        1,
        "failed",
        {"failure_kind": "model_request_error", "status_code": 429},
        OutcomeAction.RETRY,
    ),
    # Claude Code 2.1.108 prefixes a 401's API Error text with "Failed to authenticate. ".
    "401_before_reply": (
        api_failure(AUTH_FAILURE, after_reply=False),
        False,
        1,
        "failed",
        {"failure_kind": "model_request_rejected"},
        OutcomeAction.FAIL,
    ),
    "400_before_reply": (
        api_failure(400, after_reply=False),
        False,
        1,
        "failed",
        {"failure_kind": "model_request_rejected"},
        OutcomeAction.FAIL,
    ),
    "refusal": ([api_error(REFUSAL), error_result(REFUSAL)], False, 1, "completed", None, OutcomeAction.VERIFY),
    "error_without_status": (
        [reply(), error_result("Execution error")],
        False,
        1,
        "failed",
        {"failure_kind": "unknown"},
        OutcomeAction.RETRY,
    ),
    "turn_limit": (
        [reply(), {"type": "result", "subtype": "error_max_turns", "is_error": True}],
        False,
        1,
        "incomplete",
        None,
        OutcomeAction.VERIFY,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("event", EVENTS)
async def test_each_run_event_gets_the_ladder_outcome(monkeypatch, event):
    events, timed_out, code, status, outcome, action = EVENTS[event]
    response = await run(monkeypatch, events, timed_out=timed_out, code=code)
    assert response.status == status
    assert json.loads(response.metadata.get("ng_agent_outcome", "null")) == outcome
    assert agent_outcome_policy(response) == action
    if status == "failed" and events and events[-1].get("is_error"):
        # Claude Code's own text is the failed response's error, not the ladder's generic one.
        assert response.error.message == events[-1]["result"]
    elif status != "failed":
        assert response.error is None
