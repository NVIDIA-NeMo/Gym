# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-judge grading: call_judge replies become dependencies only inside a reviewed grading with a parsed verdict."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from nemo_gym.server_runtime import verifier_observability
from nemo_gym.server_runtime.verifier_observability import observe_llm_judge_grading
from nemo_gym.trajectory.trajectory_record import VerifierInvocation, grading_result_binding
from nemo_gym.verification import judge
from nemo_gym.verification.judge import call_judge, record_parsed_judge_verdict, record_rejected_judge_reply


class JudgeReply(BaseModel):
    id: str | None = None
    verdict: str = ""


def _result(reward):
    return {
        "response": {"id": "policy-response", "output": [{"type": "message", "content": "saved"}]},
        "reward": reward,
    }


@pytest.fixture
def judge_server(monkeypatch):
    """Test stub for the judge model server: the HTTP layer is replaced, call_judge itself runs."""
    replies = []
    client = AsyncMock()

    async def post(**kwargs):
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    client.post = post
    monkeypatch.setattr(judge, "raise_for_status", AsyncMock())
    monkeypatch.setattr(judge, "get_response_json", AsyncMock(side_effect=lambda reply: reply))
    monkeypatch.setattr(judge, "_judge_retry_wait_s", lambda attempt: 0.0)
    return client, replies


async def _grade(client, *, parsed=True, reward=1.0, judged=True):
    async def calculate():
        if judged:
            await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
        if parsed:
            record_parsed_judge_verdict()
        return _result(reward)

    invocation = VerifierInvocation(invocation_id="verify-1")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        value = await observe_llm_judge_grading(calculate)
    finally:
        verifier_observability._active_verifier.reset(token)
    return value, invocation.grading_operations[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [1.0, 0.0])
async def test_a_parsed_verdict_completes_with_the_judge_reply_as_its_only_dependency(judge_server, reward):
    client, replies = judge_server
    replies.append({"id": "judge-response-1", "verdict": "x"})
    value, operation = await _grade(client, reward=reward)
    assert value["reward"] == reward
    assert (operation.kind, operation.outcome) == ("llm_judge", "completed")
    assert operation.result_binding_sha256 == grading_result_binding(value)
    assert [(d.model_ref.name, d.response_id, d.model_call_id) for d in operation.dependencies] == [
        ("judge", "judge-response-1", None)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["fallback_branch", "reply_without_id", "no_judge_call"])
async def test_a_grade_without_an_identified_parsed_verdict_stays_unknown(judge_server, case):
    client, replies = judge_server
    replies.append({"id": None if case == "reply_without_id" else "judge-response-1"})
    value, operation = await _grade(
        client, parsed=case != "fallback_branch", judged=case != "no_judge_call", reward=0.0
    )
    assert value["reward"] == 0.0 and operation.outcome == "unknown"


@pytest.mark.asyncio
async def test_a_failed_judge_call_propagates_unchanged_and_is_not_a_dependency(judge_server):
    client, replies = judge_server
    # A lost connection is tried again; it propagates once every attempt has failed.
    replies.extend(ConnectionError("judge unreachable") for _ in range(judge.JUDGE_CALL_MAX_ATTEMPTS))
    invocation = VerifierInvocation(invocation_id="verify-1")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        with pytest.raises(Exception) as raised:
            await observe_llm_judge_grading(
                lambda: call_judge(
                    client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply
                )
            )
    finally:
        verifier_observability._active_verifier.reset(token)
    operation = invocation.grading_operations[0]
    assert operation.outcome == "failed" and operation.dependencies == []
    assert "judge unreachable" in str(raised.value) or isinstance(raised.value.__cause__, ConnectionError)


@pytest.mark.asyncio
async def test_call_judge_outside_a_reviewed_grading_records_nothing(judge_server):
    client, replies = judge_server
    replies.append({"id": "judge-response-1"})
    record_parsed_judge_verdict()  # no active grading: a no-op
    reply = await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
    assert reply.id == "judge-response-1" and judge._judge_grading.get() is None


@pytest.mark.asyncio
async def test_one_parsed_reply_does_not_cover_a_later_fallback(judge_server):
    client, replies = judge_server
    replies.extend([{"id": "first"}, {"id": "fallback"}])
    invocation = VerifierInvocation(invocation_id="verify-multiple")
    token = verifier_observability._active_verifier.set(invocation)

    async def calculate():
        await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
        record_parsed_judge_verdict()
        record_parsed_judge_verdict()
        await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
        return _result(0.0)

    try:
        value = await observe_llm_judge_grading(calculate)
    finally:
        verifier_observability._active_verifier.reset(token)
    assert value["reward"] == 0.0
    assert invocation.grading_operations[0].outcome == "unknown"
    assert len(invocation.grading_operations[0].dependencies) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_rejected_reply_is_retained_but_requires_an_accepted_verdict(judge_server, accepted):
    client, replies = judge_server
    replies.append({"id": "rejected"})
    if accepted:
        replies.append({"id": "accepted"})
    invocation = VerifierInvocation(invocation_id="verify-retry")
    token = verifier_observability._active_verifier.set(invocation)

    async def calculate():
        await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
        record_rejected_judge_reply()
        if accepted:
            await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
            record_parsed_judge_verdict()
        return _result(1.0 if accepted else 0.0)

    try:
        await observe_llm_judge_grading(calculate)
    finally:
        verifier_observability._active_verifier.reset(token)
    [operation] = invocation.grading_operations
    assert operation.outcome == ("completed" if accepted else "unknown")
    assert [dependency.response_id for dependency in operation.dependencies] == (
        ["rejected", "accepted"] if accepted else ["rejected"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("judged", [False, True], ids=["no judge call", "judge call"])
async def test_a_judge_optional_grading_without_a_judge_call_records_no_operation(judge_server, judged):
    client, replies = judge_server
    replies.append({"id": "judge-response-1"})
    invocation = VerifierInvocation(invocation_id="verify-optional")
    token = verifier_observability._active_verifier.set(invocation)

    async def calculate():
        if judged:
            await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
            record_parsed_judge_verdict()
        return _result(1.0)

    try:
        await observe_llm_judge_grading(calculate, judge_optional=True)
    finally:
        verifier_observability._active_verifier.reset(token)
    if judged:
        assert [operation.outcome for operation in invocation.grading_operations] == ["completed"]
    else:
        # observe_verifier then records the grade as the verifier's own, which a judge-free grade is.
        assert invocation.grading_operations is None


@pytest.mark.asyncio
async def test_a_judge_optional_grading_keeps_its_operation_for_an_unidentified_judge_reply(judge_server):
    client, replies = judge_server
    replies.append({"id": None})
    invocation = VerifierInvocation(invocation_id="verify-optional-unidentified")
    token = verifier_observability._active_verifier.set(invocation)

    async def calculate():
        await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=JudgeReply)
        record_parsed_judge_verdict()
        return _result(1.0)

    try:
        await observe_llm_judge_grading(calculate, judge_optional=True)
    finally:
        verifier_observability._active_verifier.reset(token)
    # The judge was asked, so the grade is not the verifier's own; the reply has no id, so it stays unknown.
    assert [(operation.kind, operation.outcome) for operation in invocation.grading_operations] == [
        ("llm_judge", "unknown")
    ]
