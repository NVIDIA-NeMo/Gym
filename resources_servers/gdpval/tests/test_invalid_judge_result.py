# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed judgement must be flagged, not averaged in as a real score.

The rubric scorers return a populated metadata dict on failure: ``no_valid_scores``
when every structured trial failed to parse, ``truncated_json`` when only a partial
score could be salvaged, and ``no_score_in_response`` when the judge returned
well-formed JSON with no score in it. All three produce 0.0 or a biased-low partial.
Flagging only ``judge_result is None`` marks those rows valid, so a broken judge is
indistinguishable from a bad deliverable.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.gdpval.app import _is_invalid_judge_result
from resources_servers.gdpval.scoring import (
    SCORING_ERROR_KEY,
    score_with_rubric,
    score_with_rubric_structured,
    score_with_rubric_visual,
)


def test_missing_result_is_invalid():
    assert _is_invalid_judge_result(None)


def test_structured_no_valid_scores_is_invalid():
    assert _is_invalid_judge_result({SCORING_ERROR_KEY: "no_valid_scores", "num_trials": 2})


def test_truncated_partial_score_is_invalid():
    assert _is_invalid_judge_result({SCORING_ERROR_KEY: "truncated_json", "partial_score": 0.536})


def test_judge_returning_no_score_is_invalid():
    assert _is_invalid_judge_result({SCORING_ERROR_KEY: "no_score_in_response", "overall_rationale": "unreadable"})


def test_a_real_judgement_is_valid():
    judged = {
        "judge_name": "minimax-m3",
        "overall_score": 0.52,
        "overall_rationale": "Meets most criteria.",
        "criteria_scores": [{"criterion": "Creates the deck", "score": 1.0}],
    }
    assert not _is_invalid_judge_result(judged)


def test_structured_trial_metadata_is_valid():
    assert not _is_invalid_judge_result(
        {"trial_scores": [11.0, 11.0], "max_possible": 40.0, "percentages": [27.5, 27.5]}
    )


def test_a_judges_own_error_field_does_not_flag_the_row():
    """Success-path metadata IS the judge's parsed JSON, so ``error`` may be its own field.

    Keying the predicate on a plain ``error`` would discard a perfectly good
    judgement just because the rubric made the judge mention one.
    """
    assert not _is_invalid_judge_result(
        {"overall_score": 0.6, "error": "candidate made a minor formula error", "judge_name": "minimax-m3"}
    )


def test_empty_marker_value_does_not_flag():
    assert not _is_invalid_judge_result({SCORING_ERROR_KEY: "", "overall_score": 0.4})


# ---------------------------------------------------------------------------
# The scorers must actually set the marker. Testing the predicate alone leaves
# the scorer return statements free to regress with the suite still green.
# ---------------------------------------------------------------------------


def _stub_judge_and_response(text: str):
    judge = type(
        "J",
        (),
        {"name": "stub", "model": "stub", "base_url": "http://stub", "api_key": "k", "create_overrides": {}},
    )()
    message = type("M", (), {"content": text, "tool_calls": None})()
    choice = type("C", (), {"message": message, "finish_reason": "stop"})()
    response = type("R", (), {"choices": [choice]})()
    return judge, response


async def _score(tmp_path, judge_text: str):
    template = tmp_path / "judge.j2"
    template.write_text("{{ task_prompt }} {{ rubric }} {{ deliverable_text }}")
    judge, response = _stub_judge_and_response(judge_text)
    with patch("openai.AsyncOpenAI") as client_cls:
        client_cls.return_value.chat.completions.create = AsyncMock(return_value=response)
        return await score_with_rubric("text", [], "rubric", "prompt", str(template), [judge])


@pytest.mark.asyncio
async def test_score_with_rubric_tags_truncated_json(tmp_path):
    score, meta = await _score(tmp_path, '{"criteria_scores": [{"score": 0.5}, {"score": 0.7}')

    assert meta is not None, "a truncated salvage must not be reported as a clean failure"
    assert meta[SCORING_ERROR_KEY] == "truncated_json"
    assert _is_invalid_judge_result(meta), "the salvage must reach verify() as invalid"
    assert score == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_score_with_rubric_tags_a_response_carrying_no_score(tmp_path):
    score, meta = await _score(tmp_path, json.dumps({"overall_rationale": "the attachment was unreadable"}))

    assert score == 0.0
    assert meta[SCORING_ERROR_KEY] == "no_score_in_response"
    assert _is_invalid_judge_result(meta), "a scoreless 0.0 must not look like a bad deliverable"


@pytest.mark.asyncio
async def test_a_good_judgement_is_not_tagged(tmp_path):
    score, meta = await _score(tmp_path, json.dumps({"overall_score": 0.75, "overall_rationale": "solid"}))

    assert score == pytest.approx(0.75)
    assert SCORING_ERROR_KEY not in meta
    assert not _is_invalid_judge_result(meta)


async def _score_visual(tmp_path, judge_text: str):
    template = tmp_path / "judge.j2"
    template.write_text("{{ task_prompt }} {{ rubric }}")
    judge, response = _stub_judge_and_response(judge_text)
    with patch("openai.AsyncOpenAI") as client_cls:
        client_cls.return_value.chat.completions.create = AsyncMock(return_value=response)
        return await score_with_rubric_visual([], [], "rubric", "prompt", str(template), [judge])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "judge_text,expected",
    [
        ('{"criteria_scores": [{"score": 0.5}, {"score": 0.7}', "truncated_json"),
        ('{"overall_rationale": "the render was blank"}', "no_score_in_response"),
    ],
)
async def test_visual_scorer_tags_its_failures_too(tmp_path, judge_text, expected):
    """The multimodal path is a parallel implementation and regresses independently."""
    _, meta = await _score_visual(tmp_path, judge_text)

    assert meta[SCORING_ERROR_KEY] == expected
    assert _is_invalid_judge_result(meta)


@pytest.mark.asyncio
async def test_structured_scorer_tags_a_run_with_no_parsable_trial(tmp_path):
    """Every trial failing to emit the score tags is the classic silent 0.0."""
    judge, response = _stub_judge_and_response("I could not evaluate this submission.")

    with patch("openai.AsyncOpenAI") as client_cls:
        client_cls.return_value.chat.completions.create = AsyncMock(return_value=response)
        score, meta = await score_with_rubric_structured(
            "text", [], "rubric", "prompt", [judge], num_trials=1, formatting_retries=1
        )

    assert score == 0.0
    assert meta[SCORING_ERROR_KEY] == "no_valid_scores"
    assert _is_invalid_judge_result(meta)
