# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AA-LCR verify runs its actual judge branch through call_judge; only parsed verdicts establish the grade."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation
from nemo_gym.trajectory.trajectory_record import grading_result_binding
from nemo_gym.verification import judge
from resources_servers.aalcr.app import LEGACY_JUDGE_PROTOCOL, V1_1_JUDGE_PROTOCOL, AalcrResourcesServer
from resources_servers.aalcr.tests.test_app import _config, _request, _response


async def _verify(judge_text, *, candidate="candidate", monkeypatch, protocol=V1_1_JUDGE_PROTOCOL):
    client = MagicMock(spec=ServerClient)
    # Test stub for the judge model server: only the HTTP reply is replaced; call_judge itself runs.
    client.post = AsyncMock(return_value=_response(judge_text).model_dump(mode="json"))
    monkeypatch.setattr(judge, "raise_for_status", AsyncMock())
    monkeypatch.setattr(judge, "get_response_json", AsyncMock(side_effect=lambda reply: reply))
    server = AalcrResourcesServer(config=_config(protocol), server_client=client)
    request = _request(protocol)
    if candidate != "candidate":
        request = request.model_copy(update={"response": _response(candidate)})
    invocation = VerifierInvocation(invocation_id="verify-1")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = await server.verify(request)
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation.grading_operations


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,reward", [("CORRECT", 1.0), ("INCORRECT", 0.0)])
async def test_a_parsed_verdict_keeps_its_reward_and_completes_the_judge_grading(verdict, reward, monkeypatch):
    result, operations = await _verify(f'{{"verdict": "{verdict}"}}', monkeypatch=monkeypatch)
    assert result.reward == reward and result.invalid_judge_response is False
    (operation,) = operations
    assert (operation.kind, operation.outcome) == ("llm_judge", "completed")
    assert [(d.model_ref.name, d.response_id) for d in operation.dependencies] == [("abcd", "response")]
    assert operation.result_binding_sha256 == grading_result_binding(result)


@pytest.mark.asyncio
async def test_a_legacy_invalid_judge_verdict_scores_zero_and_stays_unknown(monkeypatch):
    result, operations = await _verify("not a verdict", monkeypatch=monkeypatch, protocol=LEGACY_JUDGE_PROTOCOL)
    assert result.reward == 0.0 and result.invalid_judge_response is True
    assert [operation.outcome for operation in operations] == ["unknown"]


@pytest.mark.asyncio
async def test_an_unparseable_v1_1_verdict_is_a_judge_failure_and_the_grading_failed(monkeypatch):
    with pytest.raises(judge.JudgeError):
        await _verify("not json", monkeypatch=monkeypatch)


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_judge_grading(monkeypatch):
    result, operations = await _verify('{"verdict": "CORRECT"}', candidate="  ", monkeypatch=monkeypatch)
    assert result.reward == 0.0 and result.invalid_model_response is True and operations is None
