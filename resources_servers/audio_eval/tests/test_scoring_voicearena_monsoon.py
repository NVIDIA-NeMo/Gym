# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VoiceArena Monsoon (WER / OIWER) scorer, driven with small hand-counted rows.

The Hindi path needs the ``voi-oiwer`` package; most tests substitute a stub for the cached ``_official_oiwer``
loader so the surrounding validation and count arithmetic (the code owned here) is exercised without it.
"""

import asyncio
import importlib
import sys
import types

import pytest


pytest.importorskip("whisper_normalizer")

monsoon = importlib.import_module("resources_servers.audio_eval._scoring.dataset.voicearena-monsoon.scorer")


def test_monsoon_english_uses_the_open_asr_normalizer_and_counts_words():
    exact = monsoon.evaluate_monsoon_sample(
        {"expected_answer": "I have two cats", "generation": "I have 2 cats", "monsoon_language": "en-in"}
    )
    assert exact["monsoon_metric"] == "wer" and exact["text"] == "i have 2 cats"
    assert exact["monsoon_errors"] == 0 and exact["monsoon_ref_words"] == 4 and exact["is_correct"] is True
    assert exact["missing_prediction"] is False and exact["generation"] == "I have 2 cats"

    empty = monsoon.evaluate_monsoon_sample(
        {"expected_answer": "I have two cats", "generation": None, "monsoon_language": "en-in"}
    )
    assert (empty["monsoon_errors"], empty["monsoon_deletions"], empty["monsoon_ref_words"]) == (4, 4, 4)
    assert empty["missing_prediction"] is True and empty["is_correct"] is False and empty["pred_text"] == ""


def test_monsoon_english_reference_must_not_normalize_to_nothing():
    with pytest.raises(ValueError, match="normalized to an empty string"):
        monsoon.evaluate_monsoon_sample({"expected_answer": "[noise]", "generation": "x", "monsoon_language": "en-in"})


def test_monsoon_rejects_unknown_languages():
    for language in (None, "ta-in"):
        with pytest.raises(ValueError, match="Unsupported Monsoon language"):
            monsoon.evaluate_monsoon_sample({"expected_answer": "x", "generation": "x", "monsoon_language": language})


def stub_oiwer(monkeypatch, result):
    calls = []

    def oiwer(generation, lattice, language):
        calls.append((generation, lattice, language))
        return result

    monkeypatch.setattr(monsoon, "_official_oiwer", lambda: oiwer)
    return calls


def hindi_row(generation="नमस्ते दुनिया"):
    return {
        "expected_answer": "नमस्ते दुनिया",
        "generation": generation,
        "monsoon_language": "hi-in",
        "reference_lattice": [["नमस्ते"], ["दुनिया", "दुनीया"]],
    }


def test_monsoon_hindi_converts_oiwer_counts_into_error_fields(monkeypatch):
    # voi-oiwer returns (score %, ., ., ., (insertions, deletions, substitutions), reference words).
    calls = stub_oiwer(monkeypatch, (60.0, None, None, None, (1, 2, 3), 10))
    result = monsoon.evaluate_monsoon_sample(hindi_row(" नमस्ते "))
    assert calls == [("नमस्ते", hindi_row()["reference_lattice"], "hindi")]
    assert result["monsoon_metric"] == "oiwer"
    assert (result["monsoon_insertions"], result["monsoon_deletions"], result["monsoon_substitutions"]) == (1, 2, 3)
    assert result["monsoon_errors"] == 6 and result["monsoon_ref_words"] == 10
    assert result["oiwer_utterance_diagnostic"] == pytest.approx(0.6)
    assert result["text"] == "नमस्ते दुनिया" and result["pred_text"] == "नमस्ते"
    assert result["is_correct"] is False and result["missing_prediction"] is False


def test_monsoon_hindi_without_expected_answer_text_defaults_to_empty(monkeypatch):
    stub_oiwer(monkeypatch, (0.0, None, None, None, (0, 0, 0), 2))
    row = {key: value for key, value in hindi_row().items() if key != "expected_answer"}
    result = monsoon.evaluate_monsoon_sample(row)
    assert result["text"] == "" and result["is_correct"] is True


@pytest.mark.parametrize("lattice", [None, [], "abc", [["a"], []], [["a"], "b"]])
def test_monsoon_hindi_requires_a_nonempty_list_of_nonempty_lists(monkeypatch, lattice):
    stub_oiwer(monkeypatch, (0.0, None, None, None, (0, 0, 0), 2))
    with pytest.raises(ValueError, match="nonempty list-of-lists reference_lattice"):
        monsoon.evaluate_monsoon_sample({**hindi_row(), "reference_lattice": lattice})


@pytest.mark.parametrize(
    "result,message",
    [
        ((0.0, None, None, None, (0, 0), 2), "invalid count/reference contract"),
        ((0.0, None, None, None, (0, 0, 0), 0), "invalid count/reference contract"),
        ((50.0, None, None, None, (0, 0, 1), 4), "score/count invariant failed"),
    ],
)
def test_monsoon_hindi_rejects_inconsistent_oiwer_output(monkeypatch, result, message):
    stub_oiwer(monkeypatch, result)
    with pytest.raises(ValueError, match=message):
        monsoon.evaluate_monsoon_sample(hindi_row())


def test_official_oiwer_is_pinned_to_its_version(monkeypatch):
    monsoon._official_oiwer.cache_clear()
    monkeypatch.setattr(monsoon, "version", lambda name: "0.0.1")
    with pytest.raises(RuntimeError, match=r"voi-oiwer==0\.1\.4; found 0\.0\.1"):
        monsoon._official_oiwer()
    monsoon._official_oiwer.cache_clear()


def test_official_oiwer_returns_the_packages_entry_point_when_the_version_matches(monkeypatch):
    sentinel = object()
    monsoon._official_oiwer.cache_clear()
    monkeypatch.setattr(monsoon, "version", lambda name: monsoon.OIWER_VERSION)
    monkeypatch.setitem(sys.modules, "voi_oiwer", types.SimpleNamespace(oiwer=sentinel))
    try:
        assert monsoon._official_oiwer() is sentinel
    finally:
        monsoon._official_oiwer.cache_clear()


def test_monsoon_evaluator_scores_asynchronously():
    result = asyncio.run(
        monsoon.MonsoonASREvaluator({}).eval_single(
            {"expected_answer": "yes", "generation": "yes", "monsoon_language": "en-in"}
        )
    )
    assert result["is_correct"] is True


def monsoon_english_rows():
    return [
        monsoon.evaluate_monsoon_sample(
            {"expected_answer": "I have two cats", "generation": g, "monsoon_language": "en-in"}
        )
        for g in ("I have 2 cats", "")
    ]


def test_monsoon_metrics_are_corpus_micro_wer_with_count_evidence():
    metrics = monsoon.MonsoonASRMetrics()
    for row in monsoon_english_rows():
        metrics.update([row])
    values = metrics.get_metrics()["pass@1"]
    assert values["wer"] == 50.0 and "oiwer" not in values  # 4 errors / 8 reference words
    assert values["coverage"] == 50.0 and values["missing_predictions"] == 1 and values["scored_entries"] == 2
    assert (values["errors"], values["ref_words"]) == (4, 8)
    assert (values["substitutions"], values["insertions"], values["deletions"]) == (0, 0, 4)
    assert values["num_entries"] == 2
    assert metrics._get_score_dict({"monsoon_errors": 0}) == {"exact": True}
    assert metrics._get_score_dict({}) == {"exact": False}


def test_monsoon_metrics_pool_every_count_across_rows():
    metrics = monsoon.MonsoonASRMetrics()
    for generation in ("", "", "I have two dogs", "I have two cats today"):
        metrics.update(
            [
                monsoon.evaluate_monsoon_sample(
                    {"expected_answer": "I have two cats", "generation": generation, "monsoon_language": "en-in"}
                )
            ]
        )
    values = metrics.get_metrics()["pass@1"]
    assert (values["substitutions"], values["insertions"], values["deletions"]) == (1, 1, 8)
    assert (values["errors"], values["ref_words"], values["wer"]) == (10, 16, 62.5)
    assert values["missing_predictions"] == 2 and values["coverage"] == 50.0


def test_monsoon_metrics_name_the_headline_after_the_row_metric(monkeypatch):
    stub_oiwer(monkeypatch, (60.0, None, None, None, (1, 2, 3), 10))
    metrics = monsoon.MonsoonASRMetrics()
    metrics.update([monsoon.evaluate_monsoon_sample(hindi_row())])
    values = metrics.get_metrics()["pass@1"]
    assert values["oiwer"] == 60.0 and "wer" not in values and values["errors"] == 6


def test_monsoon_metrics_reject_mixed_or_unknown_metrics_and_bad_arity(monkeypatch):
    stub_oiwer(monkeypatch, (0.0, None, None, None, (0, 0, 0), 2))
    metrics = monsoon.MonsoonASRMetrics()
    metrics.update([monsoon.evaluate_monsoon_sample(hindi_row())])
    with pytest.raises(ValueError, match="separate profiles"):
        metrics.update([monsoon_english_rows()[0]])
    with pytest.raises(ValueError, match="Unsupported Monsoon metric"):
        monsoon.MonsoonASRMetrics().update([{"monsoon_metric": "cer"}])
    with pytest.raises(ValueError, match="one greedy generation"):
        monsoon.MonsoonASRMetrics(max_k=3)
    with pytest.raises(ValueError, match="exactly one prediction"):
        monsoon.MonsoonASRMetrics().update([])
    with pytest.raises(ValueError, match="no predictions/reference words"):
        monsoon.MonsoonASRMetrics().get_metrics()


def test_profile_contracts_resolve_to_the_scorer_classes():
    from pathlib import Path

    from benchmarks import audio_manifest
    from resources_servers.audio_eval import score

    root = Path(__file__).resolve().parents[3]
    contracts = audio_manifest.load_profiles(root / "benchmarks" / "voicearena_monsoon" / "profiles.json")["profiles"]
    assert set(contracts) == {"voicearena-monsoon.en-in.public", "voicearena-monsoon.hi-in.public"}
    for contract in contracts.values():
        name, settings = score.evaluator_settings(contract)
        assert score.resolve("evaluators", name) is monsoon.MonsoonASREvaluator
        assert settings == {"cleanup_generation_loops": True}
        assert score.resolve("metrics", contract["skills"]["METRICS_TYPE"]) is monsoon.MonsoonASRMetrics


def test_scoring_a_profile_end_to_end_reports_namespaced_wer():
    from resources_servers.audio_eval import score

    contract = {
        "request": "greedy",
        "skills": {
            "EVAL_ARGS": "++eval_type=nemo_skills.dataset.voicearena-monsoon.scorer::MonsoonASREvaluator",
            "METRICS_TYPE": "nemo_skills.dataset.voicearena-monsoon.scorer::MonsoonASRMetrics",
        },
    }
    rows = [
        {"expected_answer": "I have two cats", "generation": g, "monsoon_language": "en-in"}
        for g in ("I have 2 cats", "I have two dogs")
    ]
    _, metrics = score.score_rows(rows, profile="p", contract=contract)
    assert metrics["p"]["pass@1"]["wer"] == 12.5  # 1 substitution / 8 reference words
    assert score.flatten_metrics(metrics)["audio/p/pass@1/errors"] == 1
