# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIR-Bench Foundation scoring (parser, evaluator, per-leaf metrics), driven with small hand-counted rows."""

import asyncio

import pytest

from resources_servers.audio_eval._scoring.evaluation.evaluator.air_bench_foundation import (
    AIRBenchFoundationEvaluator,
    evaluate_air_bench_foundation_sample,
    official_foundation_ground_truth,
    official_foundation_prediction,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.air_bench_foundation_metrics import (
    AIRBenchFoundationMetrics,
)


# ---------------------------------------------------------------- AIR-Bench Foundation evaluator


@pytest.mark.parametrize(
    "response,expected",
    [
        ("A", "A"),
        ("  B\nextra", "B"),
        ("D", "D"),
        ("Dog", "D"),
        ("The answer is C.", "C"),
        ("None", None),
        ("", None),
        ("   ", None),
        ("a", None),
        ("xyz", None),
        ("Answer: D", "A"),  # the official parser's first-character rule fires on the "A" of "Answer"
        ("xx D", None),
    ],
)
def test_official_prediction_parser(response, expected):
    assert official_foundation_prediction(response) == expected


def test_official_ground_truth_uses_first_matching_choice_and_rejects_unmatched_answers():
    point = {"answer_gt": "yes", "choice_a": "no", "choice_b": "yes", "choice_c": "yes", "choice_d": "maybe"}
    assert official_foundation_ground_truth(point) == "B"
    with pytest.raises(ValueError, match="does not match any choice"):
        official_foundation_ground_truth({**point, "answer_gt": "other"})


def foundation_row(generation, answer="yes", leaf=("Speech_grounding", "libri")):
    return {
        "generation": generation,
        "answer_gt": answer,
        "choice_a": "no",
        "choice_b": "yes",
        "choice_c": "maybe",
        "choice_d": "never",
        "task_name": leaf[0],
        "dataset_name": leaf[1],
    }


def test_foundation_sample_grades_against_the_mapped_letter():
    right = evaluate_air_bench_foundation_sample(foundation_row("B"))
    assert right["predicted_answer"] == "B" and right["official_ground_truth"] == "B"
    assert right["official_parse_valid"] and right["symbolic_correct"] and right["match"] == 1
    assert right["air_bench_foundation_leaf"] == right["subset_for_metrics"] == "Speech_grounding_libri"
    assert right["response"] == "B"

    wrong = evaluate_air_bench_foundation_sample(foundation_row("A"))
    assert wrong["official_parse_valid"] and not wrong["symbolic_correct"] and wrong["match"] == 0

    unparsable = evaluate_air_bench_foundation_sample(foundation_row(None))
    assert unparsable["response"] == "" and unparsable["predicted_answer"] is None
    assert not unparsable["official_parse_valid"] and unparsable["match"] == 0


def test_evaluator_class_grades_one_row_and_rejects_unmatched_answers():
    evaluator = AIRBenchFoundationEvaluator({})
    result = asyncio.run(evaluator.eval_single(foundation_row("B")))
    assert result["match"] == 1 and result["subset_for_metrics"] == "Speech_grounding_libri"
    with pytest.raises(ValueError, match="does not match any choice"):
        asyncio.run(evaluator.eval_single(foundation_row("B", answer="not a choice")))


# ---------------------------------------------------------------- AIR-Bench Foundation metrics


def foundation_scored(generation, leaf=("Speech_grounding", "libri")):
    return evaluate_air_bench_foundation_sample(foundation_row(generation, leaf=leaf))


def test_single_leaf_reports_official_accuracy_over_parseable_responses_only():
    metrics = AIRBenchFoundationMetrics()
    for generation in ("B", "A", "garbage"):
        metrics.update([foundation_scored(generation)])
    values = metrics.get_metrics()["pass@1"]
    assert (values["total_sum"], values["correct_num"], values["fail_num"], values["num_leaves"]) == (2, 1, 1, 1)
    assert values["accuracy"] == 50.0 and values["num_entries"] == 3


def test_single_leaf_without_any_parseable_response_has_zero_accuracy():
    metrics = AIRBenchFoundationMetrics()
    metrics.update([foundation_scored("garbage")])
    values = metrics.get_metrics()["pass@1"]
    assert values["accuracy"] == 0.0 and values["total_sum"] == 0 and values["fail_num"] == 1


def test_mixed_leaves_report_counts_but_no_invented_aggregate_accuracy():
    metrics = AIRBenchFoundationMetrics()
    metrics.update([foundation_scored("B")])
    metrics.update([foundation_scored("B", leaf=("Music_QA", "mus"))])
    values = metrics.get_metrics()["pass@1"]
    assert (values["total_sum"], values["correct_num"], values["num_leaves"]) == (2, 2, 2)
    assert "accuracy" not in values


def test_leaf_falls_back_to_subset_for_metrics_and_is_required():
    metrics = AIRBenchFoundationMetrics()
    metrics.update([{"subset_for_metrics": "leaf", "official_parse_valid": True, "symbolic_correct": True}])
    assert metrics.get_metrics()["pass@1"]["accuracy"] == 100.0
    with pytest.raises(ValueError, match="missing its official leaf"):
        AIRBenchFoundationMetrics().update([{"official_parse_valid": True}])
    with pytest.raises(ValueError, match="missing its official leaf"):
        AIRBenchFoundationMetrics().update([{"air_bench_foundation_leaf": ""}])


def test_foundation_metrics_reject_multiple_generations():
    with pytest.raises(ValueError, match="one greedy generation"):
        AIRBenchFoundationMetrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        AIRBenchFoundationMetrics().update([{}, {}])
