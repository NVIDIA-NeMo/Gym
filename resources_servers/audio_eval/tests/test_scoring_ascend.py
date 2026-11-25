# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ASCEND evaluator and metrics, driven with small hand-counted rows, and their lookup by profile name."""

import asyncio

import pytest


pytest.importorskip("jieba")

from resources_servers.audio_eval._scoring.evaluation.evaluator.ascend import (
    ASCENDEvaluator,
    cer_tokens,
    evaluate_ascend_sample,
    mer_tokens,
    normalize_ascend,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.ascend_metrics import ASCENDMetrics
from resources_servers.audio_eval.score import locate, resolve


def test_a_profile_finds_the_ascend_evaluator_and_metrics_by_name():
    assert resolve("evaluators", "ascend") is ASCENDEvaluator
    assert resolve("metrics", "ascend") is ASCENDMetrics
    # Only a variant the module declares may follow the name: a typo does not fall back to the plain class.
    with pytest.raises(KeyError, match="does not declare the variant"):
        locate("metrics", "ascend_mer")


def test_ascend_normalization_and_tokenizers():
    assert normalize_ascend("A,B?") == "ab "
    assert normalize_ascend("“好”，（ok）") == "好ok "
    assert mer_tokens("hello 你好 world") == ["hello", "你", "好", "world"]
    assert cer_tokens("a b 你") == ["a", "b", "你"]


def test_ascend_sample_reports_adapted_and_upstream_literal_protocols():
    result = evaluate_ascend_sample({"expected_answer": "我 hello world", "generation": "我 Hello, world!"})
    # Symmetric normalization removes the casing and punctuation difference ...
    assert result["mer_errors"] == 0 and result["mer_ref_units"] == 3
    assert result["cer_errors"] == 0 and result["cer_ref_chars"] == 11
    assert result["is_correct"] is True and result["mer"] == 0.0 and result["cer"] == 0.0
    # ... but the literal official code tokenizes the raw hypothesis, so "Hello", ",", "world", "!" are not the
    # normalized reference units "hello" and "world" (three edits in total).
    assert result["mer_upstream_literal_errors"] == 3 and result["mer_upstream_literal"] == 1.0
    assert result["cer_upstream_literal_errors"] == 3
    assert result["cer_upstream_literal"] == pytest.approx(3 / 11)
    assert result["is_correct_upstream_literal"] is False
    assert result["text"] == "我 hello world " and result["generation"] == "我 Hello, world!"


def test_ascend_sample_counts_a_mandarin_character_substitution_and_an_english_word_deletion():
    result = evaluate_ascend_sample({"expected_answer": "你好 world today", "generation": "你坏 world"})
    assert (result["mer_errors"], result["mer_ref_units"]) == (2, 4)  # 好 -> 坏, "today" deleted
    assert (result["cer_errors"], result["cer_ref_chars"]) == (6, 12)  # 好 -> 坏 and the five letters of "today"
    assert result["mer"] == 0.5 and result["is_correct"] is False


def test_ascend_sample_with_an_empty_normalized_reference_has_no_rates():
    result = evaluate_ascend_sample({"expected_answer": "，。", "generation": ""})
    assert result["mer"] is None and result["cer"] is None
    assert result["mer_ref_units"] == 0 and result["cer_ref_chars"] == 0
    assert result["mer_upstream_literal"] is None and result["cer_upstream_literal"] is None
    assert result["is_correct"] is True


def test_ascend_mer_requires_the_pinned_jieba_version(monkeypatch):
    import resources_servers.audio_eval._scoring.evaluation.evaluator.ascend as ascend

    monkeypatch.setattr(ascend.jieba, "__version__", "0.0.1")
    with pytest.raises(RuntimeError, match="jieba==0.42.1"):
        mer_tokens("hello")


def test_ascend_evaluator_scores_asynchronously():
    result = asyncio.run(ASCENDEvaluator({}).eval_single({"expected_answer": "你好", "generation": "你坏"}))
    assert result["mer_errors"] == 1 and result["mer_ref_units"] == 2


def two_rows():
    first = evaluate_ascend_sample({"expected_answer": "我 hello world", "generation": "我 Hello, world!"})
    second = evaluate_ascend_sample({"expected_answer": "你好", "generation": "你坏"})
    return first, second


def test_ascend_metrics_pool_errors_over_the_corpus_under_both_protocols():
    metrics = ASCENDMetrics()
    for row in reversed(two_rows()):  # the row with the errors comes first, so keeping only the last row is caught
        metrics.update([row])
    values = metrics.get_metrics()["pass@1"]
    assert (values["mer_errors"], values["mer_ref_units"]) == (1, 5)
    assert (values["cer_errors"], values["cer_ref_chars"]) == (1, 13)
    assert values["mer"] == 20.0 and values["cer"] == 7.69  # pooled, not the mean of 0 % and 50 % (25 %)
    assert (values["mer_upstream_literal_errors"], values["mer_upstream_literal_ref_units"]) == (4, 5)
    assert (values["cer_upstream_literal_errors"], values["cer_upstream_literal_ref_chars"]) == (4, 13)
    assert values["mer_upstream_literal"] == 80.0 and values["cer_upstream_literal"] == 30.77
    assert values["success_rate"] == 50.0 and "correct" not in values
    assert values["no_answer"] == 0.0 and values["num_entries"] == 2


def test_ascend_metrics_report_half_the_empty_answer_share_and_omit_rates_without_reference_units():
    metrics = ASCENDMetrics()
    metrics.update([evaluate_ascend_sample({"expected_answer": "，", "generation": ""})])
    values = metrics.get_metrics()["pass@1"]
    assert values["no_answer"] == 50.0  # 100 % unanswered, halved
    assert "mer" not in values and "cer" not in values and "mer_upstream_literal" not in values
    assert values["mer_ref_units"] == 0 and values["success_rate"] == 100.0


def test_ascend_metrics_reset_clears_the_pooled_counts():
    metrics = ASCENDMetrics()
    for row in two_rows():
        metrics.update([row])
    metrics.reset()
    assert metrics.total == 0
    assert (metrics.mer_errors, metrics.mer_ref_units, metrics.cer_errors, metrics.cer_ref_chars) == (0, 0, 0, 0)
    assert (metrics.upstream_mer_errors, metrics.upstream_cer_errors) == (0, 0)
    metrics.update([two_rows()[1]])
    assert metrics.get_metrics()["pass@1"]["mer"] == 50.0


def test_ascend_metrics_reject_multiple_generations():
    with pytest.raises(ValueError, match="max_k=1"):
        ASCENDMetrics(max_k=4)
    with pytest.raises(ValueError, match="exactly one prediction"):
        ASCENDMetrics().update([])
