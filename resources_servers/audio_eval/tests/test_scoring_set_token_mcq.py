# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set-token multiple-choice scoring (MMAR): the decision rule, the evaluator, the metrics and the lookup by name."""

import asyncio
import json
from pathlib import Path

import pytest

from resources_servers.audio_eval._scoring.evaluation.evaluator.set_token_mcq import (
    SetTokenMCQEvaluator,
    official_set_token_match,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.set_token_mcq_metrics import SetTokenMCQMetrics
from resources_servers.audio_eval.score import evaluator_settings, resolve, score_rows


PROFILES = Path(__file__).resolve().parents[3] / "benchmarks" / "mmar" / "profiles.json"


def test_set_token_match_requires_the_answer_tokens_and_no_other_choice_tokens():
    choices = ["dog", "cat", "big dog"]
    assert official_set_token_match("dog", "It is the Dog.", choices) is True
    assert official_set_token_match("dog", "a big dog", choices) is False  # "big" belongs to a wrong choice
    assert official_set_token_match("dog", "dog or cat", choices) is False  # contains a wrong choice's token
    assert official_set_token_match("dog", "a bird", choices) is False  # answer missing
    assert official_set_token_match("dog", "!!!", choices) is False  # no word tokens at all


def test_tokens_shared_with_the_answer_do_not_count_against_the_prediction():
    # "red" is in a wrong choice and in the answer, so only "light" and "stop" are forbidden.
    choices = ["red light", "red", "stop sign"]
    assert official_set_token_match("red light", "red light", choices) is True
    assert official_set_token_match("red light", "red light stop", choices) is False


def test_every_answer_word_is_required_and_an_empty_prediction_never_matches():
    assert official_set_token_match("red light", "red", ["red light", "blue"]) is False
    assert official_set_token_match("red light", "the light is red", ["red light", "blue"]) is True
    assert official_set_token_match("!!", "", ["!!", "?"]) is False  # no tokens on either side


def grade(row: dict) -> dict:
    return asyncio.run(SetTokenMCQEvaluator({}).eval_single(row))


def test_evaluator_grades_one_row_and_keeps_the_prediction():
    right = grade({"expected_answer": "dog", "choices": ["dog", "cat"], "generation": "The dog"})
    assert right == {
        "answer_prediction": "The dog",
        "predicted_answer": "The dog",
        "symbolic_correct": True,
        "match": 1,
    }
    wrong = grade({"expected_answer": "dog", "choices": ["dog", "cat"], "generation": "cat"})
    assert wrong["match"] == 0 and wrong["symbolic_correct"] is False
    missing = grade({"expected_answer": "dog", "choices": ["dog", "cat"], "generation": None})
    assert missing["answer_prediction"] == "" and missing["predicted_answer"] is None and missing["match"] == 0
    blank = grade({"expected_answer": "dog", "choices": ["dog"], "generation": "  "})
    assert blank["answer_prediction"] == "  " and blank["predicted_answer"] is None


def test_evaluator_rejects_malformed_rows_and_settings():
    with pytest.raises(TypeError, match="string expected_answer"):
        grade({"expected_answer": 3, "choices": ["a"], "generation": "a"})
    for choices in ("abc", ["a", 2], None):
        with pytest.raises(TypeError, match="list of string choices"):
            grade({"expected_answer": "a", "choices": choices, "generation": "a"})
    with pytest.raises(ValueError, match="takes no"):
        SetTokenMCQEvaluator({"official_output_key": "model_prediction"})


def test_profile_contract_resolves_through_the_server():
    contract = json.loads(PROFILES.read_text())["profiles"]["mmar"]
    assert evaluator_settings(contract) == ("set-token-mcq", {})
    assert resolve("evaluators", "set-token-mcq") is SetTokenMCQEvaluator
    assert resolve("metrics", contract["skills"]["METRICS_TYPE"]) is SetTokenMCQMetrics


def _row(answer: str, generation, category: str) -> dict:
    return {
        "expected_answer": answer,
        "choices": ["dog", "cat", "bird"],
        "generation": generation,
        "subset_for_metrics": category,
    }


def test_score_rows_reports_accuracy_overall_and_per_category():
    contract = json.loads(PROFILES.read_text())["profiles"]["mmar"]
    rows = [
        _row("dog", "dog", "Signal Layer"),
        _row("dog", "cat", "Signal Layer"),
        _row("cat", "cat", "Semantic Layer"),
        _row("bird", "bird", "Semantic Layer"),
        _row("bird", "", "Semantic Layer"),
    ]
    scored, metrics = score_rows(rows, profile="mmar", contract=contract)
    assert [row["match"] for row in scored] == [1, 0, 1, 1, 0]
    assert metrics["mmar"]["pass@1"]["accuracy"] == pytest.approx(60.0)
    assert metrics["mmar-Signal Layer"]["pass@1"]["accuracy"] == pytest.approx(50.0)
    assert metrics["mmar-Semantic Layer"]["pass@1"]["accuracy"] == pytest.approx(200 / 3)
    assert metrics["mmar"]["pass@1"]["num_entries"] == 5


def test_metrics_count_empty_answers_as_no_answer():
    metrics = SetTokenMCQMetrics()
    metrics.update([{"symbolic_correct": True, "predicted_answer": "dog"}])
    metrics.update([{"symbolic_correct": False, "predicted_answer": None}])
    result = metrics.get_metrics()["pass@1"]
    assert result["no_answer"] == pytest.approx(50.0)
    assert result["accuracy"] == pytest.approx(50.0)
