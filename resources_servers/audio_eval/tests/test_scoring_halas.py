# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HALAS evaluator and metrics, driven with small hand-counted rows, and their lookup by profile name."""

import asyncio

import pytest


pytest.importorskip("whisper_normalizer")

from resources_servers.audio_eval._scoring.evaluation.evaluator.halas import HALASEvaluator, evaluate_halas_sample
from resources_servers.audio_eval._scoring.evaluation.metrics.halas_metrics import HALASMetrics
from resources_servers.audio_eval.score import locate, resolve


def test_a_profile_finds_the_halas_evaluator_and_metrics_by_name():
    assert resolve("evaluators", "halas") is HALASEvaluator
    assert resolve("metrics", "halas") is HALASMetrics
    # Only a variant the module declares may follow the name: a typo does not fall back to the plain class.
    with pytest.raises(KeyError, match="does not declare the variant"):
        locate("metrics", "halas_wer")


def test_halas_sample_counts_insertions_and_speaking_rate():
    result = evaluate_halas_sample(
        {"expected_answer": "the cat sat", "generation": "the big cat sat down", "audio_duration": 6}
    )
    assert (result["wer_errors"], result["wer_insertions"], result["wer_ref_words"]) == (2, 2, 3)
    assert result["pred_words"] == 5
    assert result["insertion_rate"] == pytest.approx(2 / 3)
    assert result["has_insertion"] and result["hallucination_proxy"]
    assert result["pred_words_per_minute"] == pytest.approx(50.0)  # 5 words in 6 s
    assert result["generation"] == "the big cat sat down" and result["audio_duration"] == 6.0


def test_halas_sample_ignores_case_and_punctuation():
    # The standard normalization lowercases and drops punctuation, so these are the same words.
    result = evaluate_halas_sample({"expected_answer": "the cat sat", "generation": "The Cat, sat."})
    assert result["wer_errors"] == 0 and not result["has_insertion"] and result["is_correct"] is True


def test_halas_sample_without_reference_words_has_no_rate():
    result = evaluate_halas_sample({"expected_answer": "", "generation": "hi there", "audio_duration": 3})
    assert result["insertion_rate"] is None and not result["has_insertion"]
    assert result["pred_words"] == 2 and result["pred_words_per_minute"] == pytest.approx(40.0)


def test_halas_sample_without_duration_or_insertions_has_no_speaking_rate():
    result = evaluate_halas_sample({"expected_answer": "x", "generation": "[unclear]"})
    assert result["wer_errors"] == 1 and result["wer_insertions"] == 0
    assert result["insertion_rate"] == 0.0 and not result["has_insertion"]
    assert result["pred_words_per_minute"] is None and result["audio_duration"] == 0.0


def test_halas_evaluator_scores_asynchronously():
    result = asyncio.run(HALASEvaluator({}).eval_single({"expected_answer": "a b", "generation": "a b"}))
    assert result["wer_errors"] == 0 and result["is_correct"] is True


def halas_rows():
    first = evaluate_halas_sample(
        {"expected_answer": "the cat sat", "generation": "the big cat sat down", "audio_duration": 6}
    )
    second = evaluate_halas_sample(
        {"expected_answer": "hello world", "generation": "hello world", "audio_duration": 6}
    )
    first.update(generation_start_time=10, generation_end_time=20, num_generated_tokens=4)
    second.update(generation_start_time=12, generation_end_time=25, num_generated_tokens=8)
    return first, second


def test_halas_metrics_aggregate_corpus_wer_and_hallucination_proxy():
    metrics = HALASMetrics()
    for row in halas_rows():
        metrics.update([row])
    values = metrics.get_metrics()["pass@1"]
    assert values["wer"] == 40.0  # 2 errors / 5 reference words
    assert values["insertion_rate"] == 40.0
    assert values["insertion_sample_rate"] == 50.0 and values["hallucination_proxy_rate"] == 50.0
    assert values["pred_words_per_minute"] == 35.0  # 7 words over 12 s
    assert (values["substitutions"], values["insertions"], values["deletions"], values["ref_words"]) == (0, 2, 0, 5)
    assert values["success_rate"] == values["correct"] == 50.0
    assert values["no_answer"] == 0.0
    assert values["num_entries"] == 2 and values["avg_tokens"] == 6 and values["gen_seconds"] == 15

    metrics.reset()
    assert metrics.total == 0 and metrics.wer_total_ref_words == 0 and metrics.audio_seconds == 0.0


def test_halas_metrics_pool_errors_over_the_corpus_not_per_row():
    # Row one: 1 error in 2 words (50 %); row two: 0 errors in 8 words. The corpus rate is 1/10, not the mean of rows.
    rows = [
        evaluate_halas_sample({"expected_answer": "a b", "generation": "a c", "audio_duration": 1}),
        evaluate_halas_sample(
            {"expected_answer": "a b c d e f g h", "generation": "a b c d e f g h", "audio_duration": 1}
        ),
    ]
    metrics = HALASMetrics()
    for row in rows:
        metrics.update([row])
    assert metrics.get_metrics()["pass@1"]["wer"] == 10.0


def test_halas_metrics_without_references_or_audio_omit_derived_rates():
    metrics = HALASMetrics()
    metrics.update([{"generation": "", "is_correct": False}])
    values = metrics.get_metrics()["pass@1"]
    assert "wer" not in values and "pred_words_per_minute" not in values
    # no_answer is half the share of empty generations, the convention the audio metrics share (see the audio_eval README).
    assert values["insertion_sample_rate"] == 0.0 and values["no_answer"] == 50.0
    assert values["num_entries"] == 1 and values["avg_tokens"] == 0 and "gen_seconds" not in values


def test_halas_metrics_reject_multiple_generations():
    with pytest.raises(ValueError, match="max_k=1"):
        HALASMetrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        HALASMetrics().update([{}, {}])
