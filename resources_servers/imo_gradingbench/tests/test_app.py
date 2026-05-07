# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.imo_gradingbench.app import (
    GRADE_TO_BINARY,
    GRADE_TO_SCORE,
    VALID_GRADES,
    ImoGradingBenchConfig,
    ImoGradingBenchResourcesServer,
    ImoGradingBenchVerifyRequest,
    _extract_assistant_text,
    extract_grade,
    normalize_expected_grade,
)


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


def _make_server(**cfg_overrides) -> ImoGradingBenchResourcesServer:
    cfg = ImoGradingBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **cfg_overrides)
    return ImoGradingBenchResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


def _make_req(text: str, expected: str) -> ImoGradingBenchVerifyRequest:
    return ImoGradingBenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "grade this"}]},
        response=_make_response(text),
        expected_answer=expected,
    )


# ---------------------------------------------------------------------------
# Mapping invariants — guard against accidental edits to GRADE_TO_SCORE.
# ---------------------------------------------------------------------------


class TestMappings:
    def test_grade_to_score(self) -> None:
        assert GRADE_TO_SCORE == {
            "correct": 7,
            "almost": 6,
            "partial": 1,
            "incorrect": 0,
        }

    def test_grade_to_binary(self) -> None:
        assert GRADE_TO_BINARY == {
            "correct": "high",
            "almost": "high",
            "partial": "low",
            "incorrect": "low",
        }

    def test_valid_grades(self) -> None:
        assert VALID_GRADES == {"correct", "almost", "partial", "incorrect"}


class TestExtractGrade:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Final answer: correct", "correct"),
            ("So my grade is incorrect", "incorrect"),
            ("almost", "almost"),
            ("partial", "partial"),
            # Markdown / punctuation stripping
            ("**correct**", "correct"),
            ("*partial.*", "partial"),
            ("`incorrect.`", "incorrect"),
            ("...almost...", "almost"),
            # Case-insensitive
            ("CORRECT", "correct"),
            ("Correct", "correct"),
            # Trailing whitespace / multi-line
            ("\n  almost  \n", "almost"),
            # Punctuation soup
            ("[correct]", "correct"),
            ("(incorrect)", "incorrect"),
            ("{partial}", "partial"),
            ("almost,", "almost"),
            ("correct;", "correct"),
            ("incorrect:", "incorrect"),
            ("partial!", "partial"),
            ("correct?", "correct"),
        ],
    )
    def test_valid(self, text: str, expected: str) -> None:
        assert extract_grade(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            None,
            "   ",
            # Extra trailing word that isn't a grade
            "almost done",
            # Made-up grade
            "Final: brilliant",
            # No words at all
            "\n\n\t",
            # Last token is punctuation only — strip yields empty string
            "...",
            # Numeric last token
            "score: 7",
        ],
    )
    def test_invalid_returns_none(self, text) -> None:
        assert extract_grade(text) is None

    def test_non_string_input(self) -> None:
        assert extract_grade(123) is None  # type: ignore[arg-type]
        assert extract_grade([]) is None  # type: ignore[arg-type]


class TestNormalizeExpectedGrade:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("correct", "correct"),
            ("CORRECT", "correct"),
            ("  almost  ", "almost"),
            ("Partial", "partial"),
        ],
    )
    def test_valid(self, raw: str, expected: str) -> None:
        assert normalize_expected_grade(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", None, "perfect", "7", "yes", "  "],
    )
    def test_invalid_returns_none(self, raw) -> None:
        assert normalize_expected_grade(raw) is None


class TestExtractAssistantText:
    def test_string_content(self) -> None:
        """Bare-string content branch (some model servers don't return a list of parts)."""
        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content="my answer is correct",
                )
            ]
        )
        assert _extract_assistant_text(fake_response) == "my answer is correct"

    def test_list_of_parts(self) -> None:
        r = _make_response("the grade is almost")
        assert _extract_assistant_text(r) == "the grade is almost"

    def test_skips_reasoning_items(self) -> None:
        """vLLM's reasoning-parser routes CoT to a separate type=reasoning item."""
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
                    "content": [{"text": "lots of CoT mentioning incorrect", "type": "reasoning_text"}],
                },
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "Final grade: correct", "type": "output_text"}],
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
        assert out == "Final grade: correct"

    def test_empty(self) -> None:
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
    async def test_exact_match(self) -> None:
        server = _make_server()
        res = await server.verify(_make_req("My final grade: correct", "correct"))
        assert res.reward == 1.0
        assert res.extracted_grade == "correct"
        assert res.binary_match is True
        assert res.score_diff == 0.0

    async def test_almost_vs_correct(self) -> None:
        """Pred=almost, gold=correct — high/high bucket match; score_diff = 7 - 6 = 1."""
        server = _make_server()
        res = await server.verify(_make_req("So my answer is almost", "correct"))
        assert res.reward == 0.0  # not exact
        assert res.binary_match is True
        assert res.score_diff == 1.0

    async def test_partial_vs_incorrect(self) -> None:
        """Pred=partial (low), gold=incorrect (low) — bucket match; score_diff = 1."""
        server = _make_server()
        res = await server.verify(_make_req("answer: partial", "incorrect"))
        assert res.reward == 0.0
        assert res.binary_match is True
        assert res.score_diff == 1.0

    async def test_high_vs_low(self) -> None:
        """Pred=correct, gold=partial — bucket mismatch; score_diff = 7 - 1 = 6."""
        server = _make_server()
        res = await server.verify(_make_req("done. correct", "partial"))
        assert res.reward == 0.0
        assert res.binary_match is False
        assert res.score_diff == 6.0

    async def test_unparseable(self) -> None:
        """Model didn't end with a valid grade — score_diff is None, no MAE contribution."""
        server = _make_server()
        res = await server.verify(_make_req("I cannot decide.", "correct"))
        assert res.reward == 0.0
        assert res.extracted_grade is None
        assert res.binary_match is False
        assert res.score_diff is None

    async def test_invalid_expected_answer(self) -> None:
        """Malformed gold value normalizes to None — never crashes."""
        server = _make_server()
        res = await server.verify(_make_req("correct", "perfect"))
        assert res.reward == 0.0
        assert res.extracted_grade == "correct"
        assert res.binary_match is False
        assert res.score_diff is None

    async def test_missing_expected_answer(self) -> None:
        server = _make_server()
        res = await server.verify(
            ImoGradingBenchVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "x"}]},
                response=_make_response("correct"),
                expected_answer=None,
            )
        )
        assert res.reward == 0.0
        assert res.extracted_grade == "correct"
        assert res.binary_match is False


# ---------------------------------------------------------------------------
# Metric aggregation.
# ---------------------------------------------------------------------------


def _rollout(pred_grade: str | None, gold: str, *, exact: bool, binary: bool, sd: float | None) -> dict:
    """Construct a verify-response-shaped rollout dict for compute_metrics."""
    return {
        "reward": 1.0 if exact else 0.0,
        "extracted_grade": pred_grade,
        "expected_answer": gold,
        "binary_match": binary,
        "score_diff": sd,
    }


class TestComputeMetrics:
    def test_empty(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {}

    def test_all_correct_one_rollout(self) -> None:
        """Two tasks, gold=correct, pred=correct."""
        server = _make_server()
        r = _rollout("correct", "correct", exact=True, binary=True, sd=0.0)
        m = server.compute_metrics([[r], [r]])
        assert m["pass@1[avg-of-1]/exact_accuracy"] == 100.0
        assert m["pass@1[avg-of-1]/binarized_accuracy"] == 100.0
        assert m["pass@1[avg-of-1]/mae"] == 0.0
        assert m["pass@1[avg-of-1]/mae_count"] == 2.0
        # Bare mae key (Skills parity)
        assert m["mae"] == 0.0
        assert m["mae_count"] == 2.0

    def test_pass_k_with_mixed_rollouts(self) -> None:
        """Two rollouts per task: one exact, one off-by-one bucket."""
        server = _make_server()
        good = _rollout("correct", "correct", exact=True, binary=True, sd=0.0)
        almost = _rollout("almost", "correct", exact=False, binary=True, sd=1.0)
        m = server.compute_metrics([[good, almost], [almost, good]])

        # pass@1[avg-of-2]/exact_accuracy: each task averages 1/2 exact across 2 rollouts.
        assert m["pass@1[avg-of-2]/exact_accuracy"] == pytest.approx(50.0)
        # pass@2/exact_accuracy: at least one exact in 2 rollouts → both tasks pass.
        assert m["pass@2/exact_accuracy"] == 100.0
        # binarized_accuracy is 100% on both rollouts.
        assert m["pass@1[avg-of-2]/binarized_accuracy"] == 100.0
        # MAE: pool 4 rollouts → (0+1+1+0)/4 = 0.5
        assert m["pass@1[avg-of-2]/mae"] == 0.5
        assert m["pass@1[avg-of-2]/mae_count"] == 4.0
        # pass@2/mae: closest pred per task is exact → 0.0
        assert m["pass@2/mae"] == 0.0
        assert m["pass@2/mae_count"] == 2.0

    def test_unparseable_rollouts_dont_contribute_mae(self) -> None:
        server = _make_server()
        good = _rollout("correct", "correct", exact=True, binary=True, sd=0.0)
        bad = _rollout(None, "correct", exact=False, binary=False, sd=None)
        m = server.compute_metrics([[good, bad]])
        # MAE only counts the valid (pred, gold) pair.
        assert m["pass@1[avg-of-2]/mae"] == 0.0
        assert m["pass@1[avg-of-2]/mae_count"] == 1.0
        # But exact_accuracy averages both rollouts.
        assert m["pass@1[avg-of-2]/exact_accuracy"] == pytest.approx(50.0)
        assert m["mae_count"] == 1.0

    def test_no_valid_pairs_no_mae_keys(self) -> None:
        """All rollouts unparseable → no mae* keys at all."""
        server = _make_server()
        bad = _rollout(None, "correct", exact=False, binary=False, sd=None)
        m = server.compute_metrics([[bad, bad]])
        assert "mae" not in m
        assert "mae_count" not in m
        assert "pass@1[avg-of-2]/mae" not in m

    def test_binarized_higher_than_exact(self) -> None:
        """Pred=almost, gold=correct: exact_accuracy=0%, binarized_accuracy=100%."""
        server = _make_server()
        almost = _rollout("almost", "correct", exact=False, binary=True, sd=1.0)
        m = server.compute_metrics([[almost], [almost], [almost]])
        assert m["pass@1[avg-of-1]/exact_accuracy"] == 0.0
        assert m["pass@1[avg-of-1]/binarized_accuracy"] == 100.0
        assert m["pass@1[avg-of-1]/mae"] == 1.0


class TestKeyMetrics:
    def test_surfaces_highest_k(self) -> None:
        server = _make_server()
        am = {
            "mean/reward": 0.5,
            "pass@1[avg-of-2]/exact_accuracy": 50.0,
            "pass@1[avg-of-2]/binarized_accuracy": 80.0,
            "pass@1[avg-of-2]/mae": 1.5,
            "pass@1[avg-of-2]/no_answer": 5.0,
            "pass@2/exact_accuracy": 75.0,
            "pass@2/binarized_accuracy": 90.0,
            "pass@2/mae": 0.5,
            "majority@2/exact_accuracy": 60.0,
            "majority@2/binarized_accuracy": 85.0,
            "mae": 1.5,
            "mae_count": 4.0,
        }
        key = server.get_key_metrics(am)
        assert key["mean/reward"] == 0.5
        assert key["pass@1[avg-of-2]/exact_accuracy"] == 50.0
        assert key["pass@1[avg-of-2]/binarized_accuracy"] == 80.0
        assert key["pass@1[avg-of-2]/mae"] == 1.5
        assert key["pass@2/exact_accuracy"] == 75.0
        assert key["pass@2/mae"] == 0.5
        assert key["majority@2/binarized_accuracy"] == 85.0
        assert key["mae"] == 1.5
        assert key["mae_count"] == 4.0
