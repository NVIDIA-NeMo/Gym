# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases of the Preference-ASR evaluator and metrics: source-dataset validation and metric contracts."""

import asyncio
import importlib

import pytest


pytest.importorskip("whisper_normalizer")

scorer = importlib.import_module("resources_servers.audio_eval._scoring.dataset.preference-asr-public.scorer")


def test_preference_asr_validates_the_source_dataset():
    point = {
        "preference_type": "standard",
        "preference_source_dataset": "unknown",
        "preference_prompt_variant": "default",
        "expected_answer": "hello",
        "generation": "hello",
        "sub_category": [],
        "direction": "keep",
    }
    with pytest.raises(ValueError, match="Unsupported Preference-ASR source dataset: 'unknown'"):
        scorer.evaluate_preference_sample(point)
    point["preference_source_dataset"] = "ami"
    result = asyncio.run(scorer.PreferenceASREvaluator({}).eval_single(point))
    assert result["preference_errors"] == 0 and result["is_correct"] is True


def test_preference_asr_metric_contracts():
    metrics = scorer.PreferenceASRMetrics()
    assert metrics._get_score_dict({"preference_errors": 0}) == {"preference_exact": True}
    assert metrics._get_score_dict({}) == {"preference_exact": False}
    with pytest.raises(ValueError, match="one greedy generation"):
        scorer.PreferenceASRMetrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        metrics.update([])
    with pytest.raises(ValueError, match="no predictions to score"):
        metrics.get_metrics()
    with pytest.raises(ValueError, match="no reference words"):
        scorer.PreferenceASRMetrics._wer({"ref_words": 0, "errors": 0})
    assert scorer.PreferenceASRMetrics._wer({"ref_words": 8, "errors": 1}) == 12.5
