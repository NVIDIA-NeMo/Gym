# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coval STT v1 scorer, driven with small hand-counted rows."""

import asyncio
import importlib
import json
from pathlib import Path

import pytest

from resources_servers.audio_eval.score import aggregate, flatten_metrics, locate


pytest.importorskip("whisper_normalizer")

coval = importlib.import_module("resources_servers.audio_eval._scoring.dataset.coval-stt-v1.scorer")

PROFILES = Path(__file__).resolve().parents[3] / "benchmarks" / "coval_stt_v1" / "profiles.json"


def coval_rows():
    inserted = coval.evaluate_coval_stt_v1_sample(
        {"expected_answer": "Hello world, it's 5 PM", "generation": "hello there world it is 5 pm"}
    )
    missing = coval.evaluate_coval_stt_v1_sample({"expected_answer": "good morning", "generation": ""})
    return inserted, missing


def test_coval_sample_normalizes_both_sides_and_counts_edits():
    inserted, missing = coval_rows()
    assert inserted["text"] == "hello world it is 5 pm" and inserted["pred_text"] == "hello there world it is 5 pm"
    assert (inserted["wer_errors"], inserted["wer_insertions"], inserted["wer_ref_words"]) == (1, 1, 6)
    assert inserted["coval_wer"] == inserted["wer"] == pytest.approx(1 / 6)
    assert not inserted["is_correct"] and not inserted["missing_prediction"]
    assert (missing["wer_errors"], missing["wer_deletions"], missing["wer_ref_words"]) == (2, 2, 2)
    assert missing["wer"] == 1.0 and missing["missing_prediction"] is True


def test_coval_reference_that_normalizes_to_nothing_is_rejected():
    with pytest.raises(ValueError, match="normalized to an empty string"):
        coval.evaluate_coval_stt_v1_sample({"expected_answer": "[noise]", "generation": "x"})


def test_coval_evaluator_scores_asynchronously():
    result = asyncio.run(coval.CovalSTTV1Evaluator({}).eval_single({"expected_answer": "Hi.", "generation": "hi"}))
    assert result["is_correct"] is True and result["wer_errors"] == 0


def test_coval_metrics_report_utterance_macro_wer_and_diagnostic_corpus_wer():
    metrics = coval.CovalSTTV1Metrics()
    for row in coval_rows():
        metrics.update([row])
    values = metrics.get_metrics()["pass@1"]
    assert values["wer"] == 58.33  # mean(1/6, 1) per utterance, not corpus-weighted
    assert values["corpus_wer_diagnostic"] == 37.5  # 3 errors / 8 reference words
    assert values["coverage"] == 50.0 and values["missing_predictions"] == 1 and values["scored_entries"] == 2
    assert (values["wer_errors"], values["wer_ref_words"]) == (3, 8)
    assert (values["wer_substitutions"], values["wer_insertions"], values["wer_deletions"]) == (0, 1, 2)
    assert values["num_entries"] == 2


def test_coval_metrics_contracts():
    assert coval.CovalSTTV1Metrics()._get_score_dict({"wer_errors": 0}) == {"exact": True}
    assert coval.CovalSTTV1Metrics()._get_score_dict({}) == {"exact": False}
    with pytest.raises(ValueError, match="one greedy generation"):
        coval.CovalSTTV1Metrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        coval.CovalSTTV1Metrics().update([])
    with pytest.raises(ValueError, match="no predictions/reference words"):
        coval.CovalSTTV1Metrics().get_metrics()


def test_coval_normalizer_is_pinned_to_its_version(monkeypatch):
    coval._english_text_normalizer.cache_clear()
    monkeypatch.setattr(coval, "version", lambda name: "0.0.1")
    try:
        with pytest.raises(RuntimeError, match=r"whisper-normalizer==0\.1\.12; found 0\.0\.1"):
            coval.normalize_text("x")
    finally:
        coval._english_text_normalizer.cache_clear()


def test_profile_keys_resolve_to_the_dataset_scorer_and_publish_one_namespaced_group():
    contract = json.loads(PROFILES.read_text())["profiles"]["coval-stt-v1"]
    evaluator_key = contract["skills"]["EVAL_ARGS"].split()[0].removeprefix("++eval_type=")
    assert locate("evaluators", evaluator_key)[1] is coval.CovalSTTV1Evaluator
    assert locate("metrics", contract["skills"]["METRICS_TYPE"])[1] is coval.CovalSTTV1Metrics

    flat = flatten_metrics(aggregate(list(coval_rows()), "coval-stt-v1", contract))
    assert flat["audio/coval-stt-v1/pass@1/wer"] == 58.33
    assert flat["audio/coval-stt-v1/pass@1/corpus_wer_diagnostic"] == 37.5
    assert {key.split("/")[1] for key in flat} == {"coval-stt-v1"}
