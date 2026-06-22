# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from app import (
    MathProofJudgementConfig,
    MathProofJudgementResourcesServer,
    MathProofJudgementVerifyRequest,
    _extract_assistant_text,
    _label_to_bool,
    _per_sample_precision_recall_f1,
    _select_majority_verdict,
    _select_pass_verdict,
    parse_judgement,
)

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [
                    {"annotations": [], "text": text, "type": "output_text"},
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_server(**cfg_overrides) -> MathProofJudgementResourcesServer:
    cfg = MathProofJudgementConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **cfg_overrides)
    return MathProofJudgementResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


def _make_req(text: str, expected: str) -> MathProofJudgementVerifyRequest:
    return MathProofJudgementVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "judge this"}]},
        response=_make_response(text),
        expected_judgement=expected,
    )


class TestParseJudgement:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Judgement: Yes", True),
            ("Judgement: No", False),
            ("judgement: yes", True),
            ("JUDGEMENT: NO", False),
            ("**Judgement**: Yes", True),
            ("**Judgement: Yes**", True),
            ("  Judgement  :  Yes", True),
            ("<summary>Great</summary><judgement>Judgement: Yes</judgement>", True),
            ("I think Judgement: No because the proof skips steps.", False),
            ("Judgement: yesh it's complicated", True),  # startswith("yes") is enough
            ("Judgement: no, the proof omits steps", False),
        ],
    )
    def test_parse(self, text: str, expected: bool) -> None:
        assert parse_judgement(text) is expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            None,
            "No judgement here",
            "Judgement: maybe",
            "This proof has issues.",
        ],
    )
    def test_no_match_returns_none(self, text) -> None:
        assert parse_judgement(text) is None


class TestLabelToBool:
    def test_yes(self) -> None:
        assert _label_to_bool("Yes") is True

    def test_no(self) -> None:
        assert _label_to_bool("No") is False

    def test_none_or_unknown(self) -> None:
        assert _label_to_bool(None) is None
        assert _label_to_bool("Maybe") is None


class TestPerSamplePrecisionRecallF1:
    def test_empty(self) -> None:
        assert _per_sample_precision_recall_f1([], 0) == (0.0, 0.0, 0.0)


class TestExtractAssistantText:
    def test_string_content(self) -> None:
        """Some models return ``content`` as a bare string rather than a list
        of content parts — that path must be extracted too."""
        # Build a NeMoGymResponse with a duck-typed object (bypass pydantic
        # schema, which only allows a list of typed parts) to exercise the
        # str-content branch of _extract_assistant_text.
        from types import SimpleNamespace

        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content="Judgement: Yes",
                )
            ]
        )
        assert _extract_assistant_text(fake_response) == "Judgement: Yes"

    def test_returns_message_text(self) -> None:
        """The message content is concatenated verbatim — no in-server stripping.

        With vLLM's ``--reasoning-parser`` active the CoT is routed to a
        separate reasoning output item that we don't read, so whatever
        lands in ``type="message"`` is the committed answer.
        """
        r = _make_response("Judgement: Yes")
        assert _extract_assistant_text(r) == "Judgement: Yes"

    def test_ignores_non_message_outputs(self) -> None:
        """Reasoning items (or any non-message output) must be skipped."""
        r = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                {
                    "id": "rs_1",
                    "summary": [],
                    "type": "reasoning",
                    "content": [{"text": "lots of CoT speculation Judgement: No", "type": "reasoning_text"}],
                },
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "Judgement: Yes", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        out = _extract_assistant_text(r)
        assert out == "Judgement: Yes"
        assert "speculation" not in out

    def test_empty_response(self) -> None:
        r = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="d",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        assert _extract_assistant_text(r) == ""


class TestVerify:
    async def test_correct_yes(self) -> None:
        server = _make_server()
        res = await server.verify(_make_req("Judgement: Yes", "Judgement: Yes"))
        assert res.reward == 1.0
        assert res.extracted_judgement == "Yes"
        assert res.tp == 1.0
        assert res.fp == res.fn == res.tn == 0.0

    async def test_correct_no(self) -> None:
        server = _make_server()
        res = await server.verify(_make_req("Judgement: No", "Judgement: No"))
        assert res.reward == 1.0
        assert res.extracted_judgement == "No"
        assert res.tn == 1.0

    async def test_false_positive(self) -> None:
        server = _make_server()
        res = await server.verify(_make_req("Judgement: Yes", "Judgement: No"))
        assert res.reward == 0.0
        assert res.fp == 1.0

    async def test_false_negative(self) -> None:
        server = _make_server()
        res = await server.verify(_make_req("Judgement: No", "Judgement: Yes"))
        assert res.reward == 0.0
        assert res.fn == 1.0

    async def test_unparseable(self) -> None:
        server = _make_server()
        res = await server.verify(_make_req("I cannot decide.", "Judgement: Yes"))
        assert res.reward == 0.0
        assert res.extracted_judgement is None
        # All four confusion counters are zero when pred is None
        assert res.tp == res.fp == res.fn == res.tn == 0.0

    async def test_message_content_parsed_verbatim(self) -> None:
        """CoT stripping is the responsibility of vLLM's ``--reasoning-parser``;
        this server only runs the Judgement regex over whatever the model
        server delivered as ``type="message"`` content.
        """
        server = _make_server()
        res = await server.verify(_make_req("Judgement: Yes", "Judgement: Yes"))
        assert res.reward == 1.0
        assert res.extracted_judgement == "Yes"


class TestVerdictSelectors:
    def test_pass_verdict_prefers_gt(self) -> None:
        # gt=True, preds=[False, True] → pick True
        assert _select_pass_verdict([False, True], True) is True

    def test_pass_verdict_falls_back_to_first_non_none(self) -> None:
        assert _select_pass_verdict([None, False, True], None) is False

    def test_pass_verdict_all_none(self) -> None:
        assert _select_pass_verdict([None, None], True) is None

    def test_majority_yes_wins(self) -> None:
        assert _select_majority_verdict([True, True, False]) is True

    def test_majority_no_wins(self) -> None:
        assert _select_majority_verdict([False, False, True]) is False

    def test_majority_tie(self) -> None:
        # Tie → first non-None pred wins
        assert _select_majority_verdict([True, False]) is True
        assert _select_majority_verdict([False, True]) is False

    def test_majority_all_none(self) -> None:
        assert _select_majority_verdict([None, None]) is None


class TestComputeMetrics:
    def test_empty(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {}

    def test_all_empty_task_groups(self) -> None:
        """Every task has zero rollouts — max_k is 0; early-exit path."""
        server = _make_server()
        # compute_pass_majority_metrics always seeds a per_sample_aggregate
        # placeholder; the important thing is that the custom
        # classification block early-exited without crashing.
        m = server.compute_metrics([[]])
        assert "pass@1[avg-of-1]/accuracy" not in m
        assert "total_positives" not in m

    def test_task_with_no_rollouts_is_skipped_for_gold(self) -> None:
        """When a task group is empty it contributes None to gold_per_task."""
        server = _make_server()
        yes_correct = {
            "reward": 1.0,
            "extracted_judgement": "Yes",
            "expected_judgement": "Judgement: Yes",
            "tp": 1.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
        # One normal task + one empty task — empty task is counted as no gold.
        m = server.compute_metrics([[yes_correct], []])
        assert m["total_positives"] == 1.0

    def test_all_correct_yes(self) -> None:
        """Two Yes-gold tasks, 2 rollouts each, all pred=Yes correctly."""
        server = _make_server()
        rollout = {
            "reward": 1.0,
            "extracted_judgement": "Yes",
            "expected_judgement": "Judgement: Yes",
            "tp": 1.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
        tasks = [[rollout, rollout], [rollout, rollout]]
        m = server.compute_metrics(tasks)

        assert m["pass@1[avg-of-2]/accuracy"] == 100.0
        assert m["pass@2/accuracy"] == 100.0
        assert m["pass@1[avg-of-2]/precision"] == 100.0
        assert m["pass@1[avg-of-2]/recall"] == 100.0
        assert m["pass@1[avg-of-2]/f1"] == 100.0
        assert m["total_positives"] == 2.0

    def test_mixed_positives_and_negatives(self) -> None:
        server = _make_server()
        yes_correct = {
            "reward": 1.0,
            "extracted_judgement": "Yes",
            "expected_judgement": "Judgement: Yes",
            "tp": 1.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
        no_correct = {
            "reward": 1.0,
            "extracted_judgement": "No",
            "expected_judgement": "Judgement: No",
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 1.0,
        }
        tasks = [[yes_correct], [no_correct]]
        m = server.compute_metrics(tasks)
        assert m["pass@1[avg-of-1]/accuracy"] == 100.0
        # precision = 1/(1+0)=1 (only one sample, only one positive prediction)
        assert m["pass@1[avg-of-1]/precision"] == 100.0
        assert m["pass@1[avg-of-1]/recall"] == 100.0
        assert m["pass@1[avg-of-1]/f1"] == 100.0
        assert m["total_positives"] == 1.0

    def test_false_positive_hurts_precision(self) -> None:
        server = _make_server()
        fp_rollout = {
            "reward": 0.0,
            "extracted_judgement": "Yes",
            "expected_judgement": "Judgement: No",  # false positive
            "tp": 0.0,
            "fp": 1.0,
            "fn": 0.0,
            "tn": 0.0,
        }
        tp_rollout = {
            "reward": 1.0,
            "extracted_judgement": "Yes",
            "expected_judgement": "Judgement: Yes",
            "tp": 1.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
        tasks = [[tp_rollout], [fp_rollout]]
        m = server.compute_metrics(tasks)
        # 1 TP, 1 FP, 0 FN → precision = 1/2 = 50.0
        assert m["pass@1[avg-of-1]/precision"] == pytest.approx(50.0)
        # recall = 1/1 = 100
        assert m["pass@1[avg-of-1]/recall"] == pytest.approx(100.0)
        # F1 = 2*0.5*1/(0.5+1) = 0.666... → 66.67
        assert m["pass@1[avg-of-1]/f1"] == pytest.approx(200.0 / 3.0, rel=1e-5)


class TestKeyMetrics:
    def test_surfaces_highest_k(self) -> None:
        server = _make_server()
        am = {
            "mean/reward": 0.75,
            "pass@1[avg-of-2]/accuracy": 80.0,
            "pass@1[avg-of-2]/f1": 70.0,
            "pass@1[avg-of-2]/precision": 85.0,
            "pass@1[avg-of-2]/recall": 60.0,
            "pass@1[avg-of-2]/no_answer": 5.0,
            "pass@2/accuracy": 90.0,
            "pass@2/f1": 85.0,
            "majority@2/accuracy": 85.0,
            "majority@2/f1": 80.0,
        }
        key = server.get_key_metrics(am)
        assert key["mean/reward"] == 0.75
        assert key["pass@1[avg-of-2]/accuracy"] == 80.0
        assert key["pass@1[avg-of-2]/f1"] == 70.0
        assert key["pass@2/accuracy"] == 90.0
        assert key["majority@2/f1"] == 80.0
