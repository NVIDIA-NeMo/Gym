# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard and preference-aware corpus WER for public Preference-ASR."""

import importlib
from pathlib import Path

import pytest


pytest.importorskip("whisper_normalizer")

from resources_servers.audio_eval.profiles import load_profiles
from resources_servers.audio_eval.score import evaluator_settings, resolve


ROOT = Path(__file__).resolve().parents[3]
scorer = importlib.import_module("resources_servers.audio_eval._scoring.dataset.preference-asr-public.scorer")


def sample(category, sub_category, reference, generation, direction="keep", variant="default"):
    return {
        "preference_type": category,
        "sub_category": sub_category,
        "direction": direction,
        "preference_source_dataset": "ami",
        "preference_prompt_variant": variant,
        "expected_answer": reference,
        "generation": generation,
    }


def test_word_error_counts_split_substitutions_insertions_and_deletions():
    counts = scorer.word_error_counts("a b c d".split(), "a x c d e".split())
    assert counts == {"errors": 2, "substitutions": 1, "insertions": 1, "deletions": 0, "ref_words": 4}
    assert scorer.word_error_counts(["a", "b"], ["b"])["deletions"] == 1


@pytest.mark.parametrize(
    "category,sub_category,reference,generation,direction",
    [
        ("case", ["proper_punctuated_capitalized_format"], "Hello.", "hello", "keep"),
        ("normalization", ["numbers"], "twenty two", "22", "keep"),
        ("disfluencies", [], "um hello", "hello", "keep"),
    ],
)
def test_preference_normalizer_keeps_a_distinction_the_standard_normalizer_erases(
    category, sub_category, reference, generation, direction
):
    result = scorer.evaluate_preference_sample(sample(category, sub_category, reference, generation, direction))
    assert result["standard_errors"] == 0
    assert result["preference_errors"] > 0
    exact = scorer.evaluate_preference_sample(sample(category, sub_category, reference, reference, direction))
    assert exact["preference_errors"] == 0 and exact["is_correct"]


def test_case_category_scores_the_requested_case_not_just_the_words():
    lower = sample("case", ["lower_case"], "hello world", "Hello world")
    assert scorer.evaluate_preference_sample(lower)["preference_errors"] == 1
    assert scorer.evaluate_preference_sample(lower)["standard_errors"] == 0
    upper = sample("case", ["upper_case"], "HELLO WORLD", "hello world")
    assert scorer.evaluate_preference_sample(upper)["preference_errors"] == 2
    assert scorer.evaluate_preference_sample(sample("case", ["upper_case"], "HELLO WORLD", "HELLO WORLD"))[
        "is_correct"
    ]


def test_missing_prediction_is_flagged():
    result = scorer.evaluate_preference_sample(sample("standard", [], "hello world", ""))
    assert result["missing_prediction"] and result["standard_deletions"] == 2


def test_unsupported_rows_are_rejected():
    with pytest.raises(ValueError, match="category"):
        scorer.evaluate_preference_sample(sample("other", [], "a", "a"))
    with pytest.raises(ValueError, match="prompt variant"):
        scorer.evaluate_preference_sample(sample("standard", [], "a", "a", variant="other"))
    with pytest.raises(ValueError, match="target_case"):
        scorer.evaluate_preference_sample(sample("case", ["lower_case", "upper_case"], "a", "a"))


def test_corpus_wer_is_micro_averaged_per_type_and_prompt_variant():
    metrics = scorer.PreferenceASRMetrics()
    for row in [
        sample("standard", [], "hello world", "hello world"),
        sample("standard", [], "good morning all", "good"),
    ]:
        metrics.update([{**row, **scorer.evaluate_preference_sample(row)}])
    result = metrics.get_metrics()["pass@1"]
    assert result["standard_wer"] == pytest.approx(100 * 2 / 5, abs=0.01)
    assert result["standard_wer_standard"] == result["standard_wer"]
    assert result["standard_ref_words"] == 5 and result["scored_entries"] == 2
    assert result["prompt_variant"] == "default"

    other = sample("standard", [], "a", "a", variant="instructed")
    with pytest.raises(ValueError, match="summarized separately"):
        metrics.update([{**other, **scorer.evaluate_preference_sample(other)}])


def test_profiles_use_the_public_scorer():
    profiles = load_profiles(ROOT / "benchmarks/preference_asr/profiles.json")["profiles"]
    assert set(profiles) == {"preference-asr-public.default", "preference-asr-public.instructed"}
    for contract in profiles.values():
        evaluator, settings = evaluator_settings(contract)
        assert evaluator == "nemo_skills.dataset.preference-asr-public.scorer::PreferenceASREvaluator"
        assert settings == {"cleanup_generation_loops": True}
        assert resolve("evaluators", evaluator).__name__ == "PreferenceASREvaluator"
        assert resolve("metrics", contract["skills"]["METRICS_TYPE"]) is scorer.PreferenceASRMetrics
    with pytest.raises(KeyError):
        resolve("metrics", "nemo_skills.dataset.preference-asr.scorer::PreferenceASRMetrics")
