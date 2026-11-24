# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The exact-label language-ID evaluator and metrics, driven with small hand-counted rows."""

import asyncio

import pytest

from resources_servers.audio_eval._scoring.evaluation.evaluator.language_id import (
    LanguageIDEvaluator,
    evaluate_language_id_sample,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.language_id_metrics import LanguageIDMetrics
from resources_servers.audio_eval.score import resolve


# ---------------------------------------------------------------- evaluator


def test_language_id_sample_is_an_exact_match_after_trimming_whitespace():
    right = evaluate_language_id_sample({"expected_answer": "fr", "generation": "  fr\n"})
    assert right["lid_correct"] and right["is_correct"] and right["predicted_label"] == "fr"
    assert right["label"] == "fr" and right["label_group"] is None and "valid_label" not in right
    wrong = evaluate_language_id_sample({"expected_answer": "fr", "generation": "FR"})
    assert not wrong["lid_correct"]
    assert not evaluate_language_id_sample({"expected_answer": "fr", "generation": None})["lid_correct"]
    # Only outer whitespace is trimmed: an answer with extra words is wrong, not partially right.
    assert not evaluate_language_id_sample({"expected_answer": "fr", "generation": "fr fr"})["lid_correct"]


@pytest.mark.parametrize(
    "extra",
    [
        {"label_group": "romance"},
        {"lang_group": "romance"},
        {"extra_fields": {"label_group": "romance"}},
        {"extra_fields": {"lang_group": "romance"}},
    ],
)
def test_language_id_sample_reads_the_label_group_from_any_supported_field(extra):
    result = evaluate_language_id_sample({"expected_answer": "fr", "generation": "fr", **extra})
    assert result["label_group"] == "romance"


def test_language_id_sample_validates_labels_when_a_label_set_is_given():
    labels = {"fr", "de"}
    result = evaluate_language_id_sample({"expected_answer": "fr", "generation": "xx"}, valid_labels=labels)
    assert result["valid_label"] is False and result["lid_correct"] is False
    assert evaluate_language_id_sample({"expected_answer": "de", "generation": "fr"}, valid_labels=labels)[
        "valid_label"
    ]
    with pytest.raises(ValueError, match="Non-canonical language-ID reference label"):
        evaluate_language_id_sample({"expected_answer": "es", "generation": "es"}, valid_labels=labels)


def test_language_id_evaluator_reads_valid_labels_from_its_config():
    evaluator = LanguageIDEvaluator({"valid_labels": ["fr", "de"]})
    assert evaluator.valid_labels == frozenset({"fr", "de"})
    result = asyncio.run(evaluator.eval_single({"expected_answer": "fr", "generation": "fr"}))
    assert result["valid_label"] is True
    with pytest.raises(ValueError, match="Non-canonical"):
        asyncio.run(evaluator.eval_single({"expected_answer": "es", "generation": "es"}))
    assert LanguageIDEvaluator({}).valid_labels is None
    assert LanguageIDEvaluator(None).valid_labels is None


def test_the_profile_names_resolve_to_these_classes():
    """``++eval_type=language_id`` and ``METRICS_TYPE: language_id`` are found by module name (no registry file)."""
    assert resolve("evaluators", "language_id") is LanguageIDEvaluator
    assert resolve("metrics", "language_id") is LanguageIDMetrics
    with pytest.raises(KeyError):
        resolve("metrics", "language_id_typo")


# ---------------------------------------------------------------- metrics


def lid_row(label, correct, group=None, generation="x"):
    return {
        "expected_answer": label,
        "label": label,
        "label_group": group,
        "lid_correct": correct,
        "generation": generation,
    }


def test_language_id_metrics_report_micro_macro_label_and_group_accuracy():
    metrics = LanguageIDMetrics()
    for row in (
        lid_row("ar", True, "g1"),
        lid_row("ar", False, "g1"),
        lid_row("bg", True, "g1"),
        lid_row("zh", True, "g2"),
    ):
        metrics.update([row])
    values = metrics.get_metrics()["pass@1"]
    assert values["accuracy_micro"] == 75.0  # 3 / 4
    assert values["accuracy_macro"] == 83.33  # mean of 50, 100, 100
    assert values["num_labels"] == 3
    assert (values["accuracy::ar"], values["accuracy::bg"], values["accuracy::zh"]) == (50.0, 100.0, 100.0)
    assert values["accuracy_group::g1"] == 66.67  # 2 / 3 samples
    assert values["accuracy_group_macro::g1"] == 75.0  # mean of the ar and bg label accuracies
    assert values["accuracy_group::g2"] == values["accuracy_group_macro::g2"] == 100.0
    assert "correct" not in values and values["num_entries"] == 4


def test_language_id_metrics_without_groups_and_with_unlabelled_rows():
    metrics = LanguageIDMetrics()
    metrics.update([{"expected_answer": "fr", "lid_correct": False, "generation": ""}])
    values = metrics.get_metrics()["pass@1"]
    assert values["accuracy_micro"] == 0.0 and values["accuracy::fr"] == 0.0 and values["no_answer"] == 100.0
    assert not [key for key in values if key.startswith("accuracy_group")]
    assert values["accuracy_macro"] == 0.0 and values["num_labels"] == 1


def test_language_id_metrics_reject_a_label_in_two_groups_and_bad_arity():
    metrics = LanguageIDMetrics()
    metrics.update([lid_row("fr", True, "g1")])
    with pytest.raises(ValueError, match="appears in multiple groups"):
        metrics.update([lid_row("fr", True, "g2")])
    with pytest.raises(ValueError, match="top-1"):
        LanguageIDMetrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        LanguageIDMetrics().update([])
