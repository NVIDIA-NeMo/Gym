# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real verify and call_judge record the equivalence judge grading; only the judge HTTP transport is stubbed."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory.trajectory_record import VerifierInvocation, grading_result_binding
from resources_servers.equivalence_llm_judge.app import (
    LLMJudgeResourcesServer,
    LLMJudgeResourcesServerConfig,
    LLMJudgeVerifyRequest,
)


def _response(id: str, text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=id,
        created_at=1.0,
        model="m",
        object="response",
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                role="assistant",
                status="completed",
                type="message",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            )
        ],
    )


async def _verify(judge_texts, *, swap=False, transport_error=None, incomplete=False):
    config = LLMJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=str(
            Path(__file__).resolve().parents[1] / "prompt_templates/equivalence_llm_judge.txt"
        ),
        check_twice_swap=swap,
    )
    client = MagicMock(spec=ServerClient)
    replies = []
    for index, text in enumerate(judge_texts):
        reply = MagicMock(ok=True)
        response = _response(f"judge-{index}", text)
        if incomplete:
            response = response.model_copy(
                update={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
            )
        reply.read = AsyncMock(return_value=response.model_dump_json())
        replies.append(reply)
    client.post = AsyncMock(side_effect=transport_error or replies)
    server = LLMJudgeResourcesServer(config=config, server_client=client)
    request = LLMJudgeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_response("policy", "my answer"),
        expected_answer="answer",
    )
    invocation = VerifierInvocation(invocation_id="verify-eq")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = await server.verify(request)
    except BaseException as error:
        result = error
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation


@pytest.mark.asyncio
@pytest.mark.parametrize("texts,reward", [(["[[A=B]]"], 1.0), (["[[A!=B]]"], 0.0)])
async def test_a_parsed_correct_or_wrong_verdict_completes_and_binds(texts, reward):
    result, invocation = await _verify(texts)
    assert result.reward == reward
    (operation,) = invocation.grading_operations
    assert operation.outcome == "completed"
    assert [d.response_id for d in operation.dependencies] == ["judge-0"]
    assert operation.result_binding_sha256 == grading_result_binding(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [["no verdict here"], ["[[A=B]] then [[A!=B]]"], [""]])
async def test_a_missing_or_conflicting_verdict_stays_unknown(texts):
    result, invocation = await _verify(texts)
    assert result.reward == 0.0
    (operation,) = invocation.grading_operations
    assert operation.outcome == "unknown"


@pytest.mark.asyncio
async def test_a_swap_pass_without_a_verdict_leaves_the_whole_grade_unknown():
    result, invocation = await _verify(["[[A=B]]", "unparseable"], swap=True)
    (operation,) = invocation.grading_operations
    assert [d.response_id for d in operation.dependencies] == ["judge-0", "judge-1"]
    assert operation.outcome == "unknown"


@pytest.mark.asyncio
async def test_a_failed_judge_transport_is_a_failed_grading():
    result, invocation = await _verify([], transport_error=ConnectionError("judge down"))
    assert isinstance(result, BaseException)
    (operation,) = invocation.grading_operations
    assert operation.outcome == "failed"
    assert operation.dependencies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("text,reward", [("[[A=B]]", 1.0), ("[[A!=B]]", 0.0)])
async def test_truncated_reply_with_valid_label_keeps_score_but_not_completed_grading(text, reward):
    result, invocation = await _verify([text], incomplete=True)
    assert result.reward == reward
    (operation,) = invocation.grading_operations
    assert operation.outcome == "unknown"
    assert [d.response_id for d in operation.dependencies] == ["judge-0"]
