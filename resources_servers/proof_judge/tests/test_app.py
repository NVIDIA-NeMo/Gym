# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, mark, raises

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.proof_judge.app import (
    ProofWithJudgeResourcesServer,
    ProofWithJudgeResourcesServerConfig,
    ProofWithJudgeVerifyRequest,
)


MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
    "parallel_tool_calls": True,
}

# Parses into (proof, self_analysis, s_prime=1.0).
VALID_POLICY_RESPONSE = "thinking</think>\n## Solution\nA rigorous proof.\n## Self Evaluation\nConfident. \\boxed{1}"


def _make_server(**config_overrides) -> ProofWithJudgeResourcesServer:
    params = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )
    params.update(config_overrides)
    config = ProofWithJudgeResourcesServerConfig(**params)
    return ProofWithJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(assistant_text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": assistant_text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_body(response_text: str = VALID_POLICY_RESPONSE) -> ProofWithJudgeVerifyRequest:
    return ProofWithJudgeVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response(response_text),
        problem="Prove P.",
    )


class TestApp:
    def test_sanity(self) -> None:
        _make_server()


class TestFailureReason:
    @mark.parametrize(
        ("policy_text", "judge_reply", "reward", "reason", "judge_calls"),
        [
            # Genuine judged verdict: no failure reason.
            (VALID_POLICY_RESPONSE, "Fine. \\boxed{1}", 1.0, None, 1),
            # Judge answered without a parseable score: named, reward still 0.
            (VALID_POLICY_RESPONSE, "I refuse to answer.", 0.0, "judge_unparseable", 1),
            # Policy broke the format contract: named, judge never consulted.
            ("No solution headers anywhere here.", "Fine. \\boxed{1}", 0.0, "missing_solution_header", 0),
        ],
    )
    async def test_failure_reason_decomposes_zero_rewards(
        self,
        monkeypatch: MonkeyPatch,
        policy_text: str,
        judge_reply: str,
        reward: float,
        reason: str | None,
        judge_calls: int,
    ) -> None:
        judge = AsyncMock(return_value=(judge_reply, 7))
        monkeypatch.setattr(ProofWithJudgeResourcesServer, "_call_judge", judge)

        result = await _make_server().verify(_make_body(response_text=policy_text))

        assert result.reward == reward
        assert result.failure_reason == reason
        assert judge.await_count == judge_calls

    def test_problem_is_required(self) -> None:
        with raises(Exception):
            ProofWithJudgeVerifyRequest(
                responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
                response=_make_response(VALID_POLICY_RESPONSE),
            )
