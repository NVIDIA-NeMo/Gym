# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from asyncio import gather
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
from pytest import MonkeyPatch, mark, raises

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import JUDGE_UNPARSEABLE, NeMoGymResponse
from resources_servers.proof_judge.app import (
    ProofWithJudgeResourcesServer,
    ProofWithJudgeResourcesServerConfig,
    ProofWithJudgeVerifyRequest,
)
from resources_servers.proof_judge.task_data import TaskData


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
        ("self_score", "verifier_reply", "meta_reply", "reward", "mask_sample", "failure_kind", "named_judge"),
        [
            # Both scores parsed: r_y=1, r_meta=1 -> alpha*1 + beta*(1-|1-1|)*1.
            (1, "\\boxed{1}", "\\boxed{1}", 1.0, False, None, None),
            # The meta score is missing: the verifier term alone (0.5) is not the reward, so the sample is masked.
            (1, "\\boxed{1}", "No score here.", 0.0, True, JUDGE_UNPARSEABLE, "meta-verifier"),
            # The verifier score is missing: the meta term alone (0.5 at a self-score of 0) is not the reward either.
            (0, "No score here.", "\\boxed{1}", 0.0, True, JUDGE_UNPARSEABLE, "verifier"),
        ],
    )
    async def test_a_missing_score_from_either_judge_masks_the_sample(
        self,
        monkeypatch: MonkeyPatch,
        self_score: int,
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
        policy_text = VALID_POLICY_RESPONSE.replace("\\boxed{1}", f"\\boxed{{{self_score}}}")

        result = await _make_server(alpha=0.5, beta=0.5).verify(_make_body(response_text=policy_text))

        assert result.reward == reward
        assert result.mask_sample is mask_sample
        assert result.failure_kind == failure_kind
        if named_judge is None:
            assert result.failure_reason is None
        else:
            assert result.failure_reason.startswith(f"{named_judge} replied without a boxed score")

    @mark.parametrize(
        ("member_replies", "zeroed"),
        [
            # Two scored zeros: the group is all-incorrect.
            ([("\\boxed{0}", "\\boxed{1}"), ("\\boxed{0}", "\\boxed{1}")], True),
            # An unscored proof beside a scored zero: the scored evidence still says all-incorrect.
            ([("No score here.", "\\boxed{1}"), ("\\boxed{0}", "\\boxed{1}")], True),
            # An unscored proof beside a scored pass: not all-incorrect.
            ([("No score here.", "\\boxed{1}"), ("\\boxed{1}", "\\boxed{1}")], False),
            # No proof was scored, so the group is not declared incorrect.
            ([("No score here.", "\\boxed{1}"), ("No score here.", "\\boxed{1}")], False),
            # A member masked for its meta-verifier reply still has a verifier score: its zero is evidence,
            ([("\\boxed{0}", "No score here."), ("No score here.", "\\boxed{1}")], True),
            # and so is its pass.
            ([("\\boxed{1}", "No score here."), ("\\boxed{0}", "\\boxed{1}")], False),
        ],
        ids=[
            "both_scored_zero",
            "unscored_and_zero",
            "unscored_and_pass",
            "all_unscored",
            "meta_masked_zero",
            "meta_masked_pass",
        ],
    )
    async def test_unscored_proofs_are_not_evidence_for_zeroing_the_group(
        self, monkeypatch: MonkeyPatch, member_replies: list[tuple[str, str]], zeroed: bool
    ) -> None:
        replies: dict[str, str] = {}

        async def judge(_self, prompt: str) -> tuple[str, int]:
            # Only the meta-verifier prompt carries the policy's self-analysis.
            return (replies["meta"], 3) if "Confident." in prompt else (replies["verifier"], 7)

        monkeypatch.setattr(ProofWithJudgeResourcesServer, "_call_judge", judge)
        server = _make_server(
            alpha=0.5, beta=0.5, zero_reward_incorrect_groups=True, expected_group_size=len(member_replies)
        )
        member_details = []
        for verifier_reply, meta_reply in member_replies:
            replies.update(verifier=verifier_reply, meta=meta_reply)
            member_details.append((await server._judge_single("Prove P.", VALID_POLICY_RESPONSE))[1])

        results = await gather(
            *(
                server._maybe_zero_incorrect_group_reward(problem="Prove P.", reward=0.25, details=details)
                for details in member_details
            )
        )

        assert all(details["group_all_r_y_zero"] is zeroed for _, details in results)
        assert all(reward == (0.0 if zeroed else 0.25) for reward, _ in results)
        assert not server._incorrect_group_coordinators

    @mark.parametrize(
        ("problem_kwargs", "error_type"),
        [({}, "missing"), ({"problem": ""}, "string_too_short")],
        ids=["missing", "empty"],
    )
    def test_problem_is_required_and_nonempty(self, problem_kwargs: dict, error_type: str) -> None:
        request_kwargs = {
            "responses_create_params": MINIMAL_RESPONSES_CREATE_PARAMS,
            "response": _make_response(VALID_POLICY_RESPONSE),
        }
        # Dataset validation (TaskData) must reject exactly what /verify rejects.
        for model, kwargs in ((ProofWithJudgeVerifyRequest, request_kwargs), (TaskData, {})):
            with raises(ValidationError) as error:
                model(**kwargs, **problem_kwargs)
            assert [(e["loc"], e["type"]) for e in error.value.errors()] == [(("problem",), error_type)]

    def test_task_data_keeps_other_task_fields(self) -> None:
        assert TaskData(problem="Prove P.", source="textbook").model_extra == {"source": "textbook"}
