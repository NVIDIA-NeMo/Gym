# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contextual Earnings-22 scorer: alignment, keyword TP/FP/FN, normalization modes and corpus metrics."""

import asyncio

import pytest

from resources_servers.audio_eval._scoring.evaluation.evaluator import contextual_earnings22 as ce
from resources_servers.audio_eval._scoring.evaluation.evaluator.contextual_earnings22 import (
    ContextualEarnings22Evaluator,
    _find_subseq_occurrences,
    calculate_wer_with_alignment,
    evaluate_contextual_earnings22_sample,
    normalize_text,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.contextual_earnings22_metrics import (
    ContextualEarnings22Metrics,
)
from resources_servers.audio_eval.score import score_rows


def test_normalize_text():
    assert normalize_text("Q&A session") == "q a session"
    assert normalize_text("Hyoung-il we'll 100% of_it") == "hyoung il we ll 100 of it"


def test_alignment_branches():
    assert calculate_wer_with_alignment([], []) == (0.0, 0, [])
    assert calculate_wer_with_alignment(["a", "b"], []) == (2.0, 2, [])
    wer, errors, align = calculate_wer_with_alignment(["a", "x", "c"], ["a", "b", "c"])
    assert (wer, errors) == (pytest.approx(1 / 3), 1)
    assert align == [("M", 0), ("S", 1), ("M", 2)]
    wer, errors, align = calculate_wer_with_alignment(["a"], ["a", "b"])
    assert (wer, errors, align) == (0.5, 1, [("M", 0), ("D", None)])
    wer, errors, align = calculate_wer_with_alignment(["a", "z", "b"], ["a", "b"])
    assert (wer, errors) == (0.5, 1)
    assert align == [("M", 0), ("M", 2)]
    wer, errors, align = calculate_wer_with_alignment(["z", "a"], ["a"])
    assert (errors, align) == (1, [("M", 1)])


def test_find_subseq_occurrences():
    assert _find_subseq_occurrences(["a", "a", "a"], ["a", "a"]) == [0]
    assert _find_subseq_occurrences(["a"], ["a", "b"]) == []
    assert _find_subseq_occurrences(["a"], []) == []
    assert _find_subseq_occurrences(["x", "a", "b", "a", "b"], ["a", "b"]) == [1, 3]


def test_sample_keyword_tp_fp_fn():
    row = {
        "expected_answer": "we use acme cloud and zeta labs daily",
        "generation": "we use acme cloud and zeta daily acme cloud",
        "keyword_list": ["acme cloud", "zeta labs", "!!!", "missing"],
    }
    out = evaluate_contextual_earnings22_sample(row)
    assert out["wer_ref_words"] == 8
    assert out["wer_errors"] == 3
    assert out["keyword_total_ref"] == 2
    assert out["keyword_tp"] == 1
    assert out["keyword_fn"] == 1
    # The second "acme cloud" in the hypothesis does not align to a reference occurrence.
    assert out["keyword_fp"] == 1
    assert out["is_correct"] is True
    assert out["text"] == "we use acme cloud and zeta labs daily"


def test_sample_deletion_inside_keyword_is_false_negative():
    out = evaluate_contextual_earnings22_sample(
        {"expected_answer": "alpha beta gamma", "generation": "alpha gamma", "keyword_list": ["alpha beta"]}
    )
    assert (out["keyword_tp"], out["keyword_fn"], out["keyword_fp"]) == (0, 1, 0)
    assert out["is_correct"] is False or out["wer"] < 0.5


def test_insertion_inside_keyword_is_false_negative_and_substitution_is_not_a_match():
    inserted = evaluate_contextual_earnings22_sample(
        {"expected_answer": "alpha beta gamma", "generation": "alpha x beta gamma", "keyword_list": ["alpha beta"]}
    )
    assert (inserted["keyword_tp"], inserted["keyword_fn"], inserted["keyword_fp"]) == (0, 1, 0)
    substituted = evaluate_contextual_earnings22_sample(
        {"expected_answer": "alpha beta gamma", "generation": "alpha bata gamma", "keyword_list": ["beta"]}
    )
    assert (substituted["keyword_tp"], substituted["keyword_fn"]) == (0, 1)


def test_open_asr_mode_uses_cased_reference_and_normalizer():
    ce._OPEN_ASR_NORMALIZER = None
    row = {
        "expected_answer": "wrong normalized text",
        "expected_answer_cased": "In 2021 we grew",
        "generation": "in twenty twenty one we grew",
        "keyword_list": ["2021"],
    }
    out = evaluate_contextual_earnings22_sample(row, "open_asr")
    assert out["wer"] == 0.0
    assert out["keyword_tp"] == 1
    assert ce._OPEN_ASR_NORMALIZER is not None
    # The cached normalizer is reused, and a missing cased reference falls back to expected_answer.
    row2 = {"expected_answer": "In 2021", "generation": "in 2021", "keyword_list": []}
    assert evaluate_contextual_earnings22_sample(row2, "open_asr")["wer"] == 0.0
    # Basic mode on the same row would charge the spoken-number spelling.
    assert evaluate_contextual_earnings22_sample(row)["wer"] > 0


def test_unknown_normalization_mode():
    with pytest.raises(ValueError, match="Unknown normalization_mode"):
        evaluate_contextual_earnings22_sample({}, "fancy")


def test_evaluator_class():
    evaluator = ContextualEarnings22Evaluator({"normalization_mode": "basic"})
    out = asyncio.run(evaluator.eval_single({"expected_answer": "a b", "generation": "a b", "keyword_list": ["a b"]}))
    assert out["keyword_tp"] == 1
    with pytest.raises(TypeError):
        ContextualEarnings22Evaluator({"bogus": 1})
    with pytest.raises(ValueError, match="Unknown normalization_mode"):
        ContextualEarnings22Evaluator({"normalization_mode": "fancy"})


def test_corpus_metrics_through_score_rows():
    contract = {
        "skills": {
            "EVAL_ARGS": "++eval_type=contextual_earnings22 ++eval_config.normalization_mode=basic",
            "METRICS_TYPE": "contextual_earnings22",
        }
    }
    rows = [
        {
            "expected_answer": "we use acme cloud and zeta labs daily",
            "generation": "we use acme cloud and zeta daily acme cloud",
            "keyword_list": ["acme cloud", "zeta labs"],
        },
        {"expected_answer": "hello there", "generation": "hello there", "keyword_list": []},
    ]
    _, metrics = score_rows(rows, profile="ce", contract=contract)
    result = metrics["ce"]["pass@1"]
    # 3 errors over 10 reference words.
    assert result["wer"] == 30.0
    assert result["keyword_precision"] == 50.0
    assert result["keyword_recall"] == 50.0
    assert result["keyword_f1"] == 50.0
    assert result["success_rate"] == 100.0
    assert result["num_entries"] == 2


def _pred(**kw):
    base = {
        "generation": "x",
        "is_correct": True,
        "wer_errors": 0,
        "wer_ref_words": 0,
        "keyword_tp": 0,
        "keyword_fp": 0,
        "keyword_fn": 0,
    }
    return {**base, **kw}


def test_metrics_edge_cases():
    with pytest.raises(ValueError, match="max_k=1"):
        ContextualEarnings22Metrics(max_k=2)
    metrics = ContextualEarnings22Metrics()
    with pytest.raises(ValueError, match="exactly 1 generation"):
        metrics.update([_pred(), _pred()])
    # No keywords anywhere: no precision/recall/f1 keys.
    metrics.update([_pred()])
    result = metrics.get_metrics()["pass@1"]
    assert "keyword_precision" not in result and "keyword_f1" not in result and "wer" not in result
    # Only false negatives: recall 0, precision undefined, f1 0.
    metrics.reset()
    metrics.update([_pred(keyword_fn=2, wer_errors=1, wer_ref_words=4)])
    result = metrics.get_metrics()["pass@1"]
    assert result["keyword_recall"] == 0.0
    assert "keyword_precision" not in result
    assert result["keyword_f1"] == 0.0
    assert result["wer"] == 25.0
    # Only false positives.
    metrics.reset()
    metrics.update([_pred(keyword_fp=1)])
    result = metrics.get_metrics()["pass@1"]
    assert result["keyword_precision"] == 0.0
    assert "keyword_recall" not in result
    assert result["keyword_f1"] == 0.0
    # Both defined but zero true positives.
    metrics.reset()
    metrics.update([_pred(keyword_fp=1, keyword_fn=1)])
    assert metrics.get_metrics()["pass@1"]["keyword_f1"] == 0.0


def test_no_answer_is_half_the_empty_share():
    metrics = ContextualEarnings22Metrics()
    metrics.update([_pred(generation="")])
    metrics.update([_pred(generation="hello")])
    assert metrics.get_metrics()["pass@1"]["no_answer"] == 25.0


def test_update_common_metrics_timing():
    metrics = ContextualEarnings22Metrics()
    agg = {}
    metrics.update_common_metrics(agg)
    assert agg["avg_tokens"] == 0
    metrics.max_end_time = 5.5
    metrics.min_start_time = 1.0
    metrics.total = 1
    metrics.avg_tokens = 7
    metrics.update_common_metrics(agg)
    assert agg["gen_seconds"] == 4 and agg["avg_tokens"] == 7
