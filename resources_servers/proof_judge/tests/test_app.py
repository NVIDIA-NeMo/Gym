# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from asyncio import gather
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, mark, raises

from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import JUDGE_UNPARSEABLE
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


class TestFailureContract:
    @mark.parametrize(
        ("policy_text", "judge_reply", "reward", "mask_sample", "failure_kind", "judge_calls"),
        [
            # Genuine judged verdict: a real reward and no failure fields.
            (VALID_POLICY_RESPONSE, "Fine. \\boxed{1}", 1.0, False, None, 1),
            # Genuine judged zero: a valid wrong answer stays unmasked policy evidence.
            (VALID_POLICY_RESPONSE, "Flawed. \\boxed{0}", 0.0, False, None, 1),
            # Judge answered without a parseable score: the zero is not evidence, so it is masked and named.
            (VALID_POLICY_RESPONSE, "I refuse to answer.", 0.0, True, JUDGE_UNPARSEABLE, 1),
            # Policy broke the format contract: a valid wrong answer, judge never consulted.
            ("No solution headers anywhere here.", "Fine. \\boxed{1}", 0.0, False, None, 0),
            # Policy said nothing at all: likewise a valid wrong answer.
            ("", "Fine. \\boxed{1}", 0.0, False, None, 0),
        ],
    )
    async def test_unparseable_verdicts_are_masked_and_named(
        self,
        monkeypatch: MonkeyPatch,
        policy_text: str,
        judge_reply: str,
        reward: float,
        mask_sample: bool,
        failure_kind: str | None,
        judge_calls: int,
    ) -> None:
        judge = AsyncMock(return_value=(judge_reply, 7))
        monkeypatch.setattr(ProofWithJudgeResourcesServer, "_call_judge", judge)

        result = await _make_server().verify(_make_body(response_text=policy_text))

        assert result.reward == reward
        assert result.mask_sample is mask_sample
        assert result.failure_kind == failure_kind
        if failure_kind is None:
            assert result.failure_reason is None
        else:
            assert result.failure_reason == f"verifier replied without a boxed score: {judge_reply!r}"
        assert judge.await_count == judge_calls

    @mark.parametrize(
        ("verifier_reply", "meta_reply", "reward", "mask_sample", "failure_kind", "named_judge"),
        [
            # Both scores parsed: r_y=1, r_meta=1 -> alpha*1 + beta*(1-|1-1|)*1.
            ("\\boxed{1}", "\\boxed{1}", 1.0, False, None, None),
            # Only the meta score is missing: r_y still measured the proof, so the reward is degraded but valid.
            ("\\boxed{1}", "No score here.", 0.5, False, JUDGE_UNPARSEABLE, "meta-verifier"),
            # The primary verdict is missing: nothing measured the proof, so the sample is masked.
            ("No score here.", "\\boxed{1}", 0.0, True, JUDGE_UNPARSEABLE, "verifier"),
        ],
    )
    async def test_meta_verifier_degrades_without_masking(
        self,
        monkeypatch: MonkeyPatch,
        verifier_reply: str,
        meta_reply: str,
        reward: float,
        mask_sample: bool,
        failure_kind: str | None,
        named_judge: str | None,
    ) -> None:
        async def judge(_self, prompt: str) -> tuple[str, int]:
            # Only the meta-verifier prompt carries the policy's self-analysis.
            return (meta_reply, 3) if "Confident." in prompt else (verifier_reply, 7)

        monkeypatch.setattr(ProofWithJudgeResourcesServer, "_call_judge", judge)

        result = await _make_server(alpha=0.5, beta=0.5).verify(_make_body())

        assert result.reward == reward
        assert result.mask_sample is mask_sample
        assert result.failure_kind == failure_kind
        if named_judge is None:
            assert result.failure_reason is None
        else:
            assert result.failure_reason.startswith(f"{named_judge} replied without a boxed score")

    @mark.parametrize(
        ("member_details", "zeroed"),
        [
            # Two measured zeros: the group is all-incorrect.
            ([{"r_y": 0.0}, {"r_y": 0.0}], True),
            # A masked verdict beside a measured zero: the measured evidence still says all-incorrect.
            ([{"r_y": 0.0, "mask_sample": True}, {"r_y": 0.0}], True),
            # A masked verdict beside a measured pass: not all-incorrect.
            ([{"r_y": 0.0, "mask_sample": True}, {"r_y": 1.0}], False),
            # Nothing measured: a masked zero is not evidence, so the group is not declared incorrect.
            ([{"r_y": 0.0, "mask_sample": True}, {"r_y": 0.0, "mask_sample": True}], False),
        ],
        ids=["both_measured_zero", "masked_and_zero", "masked_and_pass", "all_masked"],
    )
    async def test_masked_verdicts_are_not_evidence_for_zeroing_the_group(
        self, member_details: list[dict], zeroed: bool
    ) -> None:
        server = _make_server(zero_reward_incorrect_groups=True, expected_group_size=len(member_details))

        results = await gather(
            *(
                server._maybe_zero_incorrect_group_reward(problem="Prove P.", reward=0.25, details=details)
                for details in member_details
            )
        )

        assert all(details["group_all_r_y_zero"] is zeroed for _, details in results)
        assert all(reward == (0.0 if zeroed else 0.25) for reward, _ in results)
        assert not server._incorrect_group_coordinators

    @mark.parametrize("problem_kwargs", [{}, {"problem": ""}], ids=["missing", "empty"])
    def test_problem_is_required_and_nonempty(self, problem_kwargs) -> None:
        with raises(Exception):
            ProofWithJudgeVerifyRequest(
                responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
                response=_make_response(VALID_POLICY_RESPONSE),
                **problem_kwargs,
            )
