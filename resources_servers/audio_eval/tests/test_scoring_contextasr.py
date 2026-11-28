# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ContextASR-Bench scorer: normalization, entity matching, WER/NE-WER/NE-FNR and corpus metrics."""

import asyncio
from pathlib import Path

import pytest


pytest.importorskip("editdistance")
pytest.importorskip("regex")
pytest.importorskip("contractions")

from resources_servers.audio_eval._scoring.evaluation.evaluator.contextasr import (
    ContextASREvaluator,
    _merge_single_letters,
    calculate_wer,
    evaluate_contextasr_sample,
    extract_entities,
    extract_entities_fuzzy,
    simple_tokenize,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.contextasr_metrics import ContextASRMetrics
from resources_servers.audio_eval.profiles import load_profiles
from resources_servers.audio_eval.score import evaluator_settings, resolve, score_rows


CONTRACT = {
    "skills": {
        "EVAL_ARGS": "++eval_type=contextasr ++eval_config.cleanup_generation_loops=true",
        "METRICS_TYPE": "contextasr",
    }
}


def test_merge_single_letters():
    assert _merge_single_letters("u s a rocks") == "usa rocks"
    assert _merge_single_letters("a b 1 c") == "ab 1 c"
    assert _merge_single_letters("x s y 's") == "xsy 's"
    assert _merge_single_letters("a b word c") == "ab word c"
    assert _merge_single_letters("a b's") == "ab's"


def test_simple_tokenize():
    assert simple_tokenize("HELLO WORLD") == "hello world"
    assert simple_tokenize("I can't go, well-known!") == "i cannot go well known"
    assert simple_tokenize("O' Brien") == "o brien"
    assert simple_tokenize("o' brien") == "o brien"
    assert simple_tokenize("你好world") == "你 好 world"
    assert simple_tokenize("world你好") == "world 你 好"
    assert simple_tokenize("I.B.M. rocks") == "ibm rocks"


def test_extract_entities_exact_with_counts():
    text = "alice met bob and alice"
    assert extract_entities(text, ["alice", "bob"]) == ["alice", "bob", "alice"]
    assert extract_entities(text, ["alice", "bob"], {"alice": 1, "bob": 1}) == ["alice", "bob"]
    assert extract_entities("a b", ["a b c"]) == []


def test_extract_entities_fuzzy():
    assert extract_entities_fuzzy("we met new york city today", ["new york city"]) == ["new york city"]
    # One substituted token out of three is within the allowed distance and returns the hypothesis window.
    assert extract_entities_fuzzy("we met new yark city", ["new york city"]) == ["new yark city"]
    assert extract_entities_fuzzy("nothing relevant", ["zebra crossing"]) == []
    assert extract_entities_fuzzy("anything", [""]) == []
    # Window longer than the entity that contains it exactly gives back the entity and rewinds the scan.
    assert extract_entities_fuzzy("big apple pie", ["apple"]) == ["apple"]
    assert extract_entities_fuzzy("alpha beta gamma delta", ["alpha beta gamma", "beta"]) == [
        "alpha beta gamma",
        "beta",
    ]


def test_calculate_wer():
    assert calculate_wer([], []) == (0.0, 0, 0, 0)
    assert calculate_wer(["a", "b"], []) == (2.0, 2, 0, 0)
    assert calculate_wer(["a", "b"], ["a", "b"]) == (0.0, 0, 0, 0)
    assert calculate_wer(["a", "x"], ["a", "b"]) == (0.5, 0, 0, 1)
    assert calculate_wer(["a"], ["a", "b"]) == (0.5, 0, 1, 0)
    assert calculate_wer(["a", "b", "c"], ["a", "c"]) == (0.5, 1, 0, 0)
    assert calculate_wer([], ["a", "b"]) == (1.0, 0, 2, 0)


def test_sample_with_entities_exact_hit_and_miss():
    row = {
        "expected_answer": "Alice visited New York with Bob.",
        "generation": "  alice visited new york with rob  ",
        "entity_list": ["New York", "Bob", "Not Present"],
    }
    out = evaluate_contextasr_sample(row)
    assert out["text"] == "alice visited new york with bob"
    assert out["pred_text"] == "alice visited new york with rob"
    assert out["wer_errors"] == 1
    assert out["wer_ref_words"] == 6
    assert out["wer"] == pytest.approx(1 / 6)
    assert out["is_correct"] is True
    # Reference entities: "new york" (2 words) and "bob"; hypothesis fuzzily recovers only "new york".
    assert out["ne_fnr_total"] == 2
    assert out["ne_fnr_hits"] == 1
    assert out["ne_fnr"] == pytest.approx(0.5)
    assert out["ne_wer_ref_words"] == 3
    assert out["ne_wer_errors"] == 1
    assert out["ne_wer"] == pytest.approx(1 / 3)


def test_sample_without_entities_and_wrong_generation():
    out = evaluate_contextasr_sample(
        {"expected_answer": "one two", "generation": "three four five", "entity_list": []}
    )
    assert out["is_correct"] is False
    assert out["wer"] == pytest.approx(1.5)
    assert out["wer_errors"] == 3
    assert out["ne_wer_ref_words"] == 0
    assert out["ne_fnr_total"] == 0
    assert out["ne_fnr"] == 0.0


def test_a_clip_at_wer_one_half_is_not_correct():
    out = evaluate_contextasr_sample({"expected_answer": "one two", "generation": "one x", "entity_list": []})
    assert out["wer"] == 0.5 and out["is_correct"] is False
    assert evaluate_contextasr_sample(
        {"expected_answer": "one two three", "generation": "one x three", "entity_list": []}
    )["is_correct"]


def test_an_entity_absent_from_the_reference_is_not_scored_even_if_hallucinated():
    out = evaluate_contextasr_sample(
        {
            "expected_answer": "we met ann",
            "generation": "we met ann at zebra crossing",
            "entity_list": ["ann", "zebra crossing"],
        }
    )
    assert out["ne_fnr_total"] == 1 and out["ne_fnr_hits"] == 1
    assert out["ne_wer_ref_words"] == 1 and out["ne_wer_errors"] == 0 and out["ne_wer"] == 0.0


def test_sample_entity_with_empty_normalization_is_ignored():
    out = evaluate_contextasr_sample({"expected_answer": "hello", "generation": "hello", "entity_list": ["!!!"]})
    assert out["ne_fnr_total"] == 0
    assert out["wer"] == 0.0


def test_evaluator_class_eval_single():
    evaluator = ContextASREvaluator({})
    out = asyncio.run(
        evaluator.eval_single({"expected_answer": "ann bob", "generation": "ann bob", "entity_list": ["ann"]})
    )
    assert out["wer"] == 0.0
    assert out["ne_fnr_hits"] == 1


def test_corpus_metrics_through_score_rows():
    rows = [
        {
            "expected_answer": "alice visited new york with bob",
            "generation": "alice visited new york with rob",
            "entity_list": ["new york", "bob"],
            "subset_for_metrics": "en",
        },
        {
            "expected_answer": "hello world",
            "generation": "hello world",
            "entity_list": [],
            "subset_for_metrics": "en",
        },
    ]
    assert evaluator_settings(CONTRACT)[0] == "contextasr"
    scored, metrics = score_rows(rows, profile="ctx", contract=CONTRACT)
    assert scored[0]["generation_raw"] == "alice visited new york with rob"
    result = metrics["ctx"]["pass@1"]
    # 1 word error out of 6 + 2 reference words.
    assert result["wer"] == 12.5
    assert result["ne_wer"] == round(100 / 3, 2)
    assert result["ne_fnr"] == 50.0
    assert result["success_rate"] == result["correct"] == 100.0
    assert result["num_entries"] == 2
    assert metrics["ctx-en"]["pass@1"]["wer"] == 12.5


PROFILES = Path(__file__).resolve().parents[3] / "benchmarks" / "contextasr_bench" / "profiles.json"


def test_every_benchmark_profile_resolves_to_these_modules():
    contracts = load_profiles(PROFILES)["profiles"]
    assert len(contracts) == 12
    for contract in contracts.values():
        name, settings = evaluator_settings(contract)
        assert name == "contextasr" and settings == {"cleanup_generation_loops": True}
        assert resolve("evaluators", name) is ContextASREvaluator
        assert resolve("metrics", contract["skills"]["METRICS_TYPE"]) is ContextASRMetrics


def test_ne_fnr_is_the_share_of_missed_entities():
    rows = [
        {"expected_answer": "ann bob cy dee", "generation": "ann bob cy dee", "entity_list": ["ann", "bob"]},
        {"expected_answer": "ann bob cy dee", "generation": "ann x y z", "entity_list": ["bob", "cy", "dee"]},
    ]
    _, metrics = score_rows(rows, profile="ctx", contract=CONTRACT)
    # 5 reference entities, 2 found exactly (ann, bob of the first clip are found; bob, cy, dee of the second are not).
    assert metrics["ctx"]["pass@1"]["ne_fnr"] == 60.0


def test_row_without_a_generation_scores_as_empty():
    out = evaluate_contextasr_sample({"expected_answer": "ann bob", "generation": None, "entity_list": ["ann"]})
    assert out["wer"] == 1.0 and out["wer_errors"] == 2 and out["ne_fnr_hits"] == 0 and out["ne_fnr_total"] == 1


def test_no_answer_is_half_the_share_of_empty_generations():
    rows = [
        {"expected_answer": "ann bob", "generation": generation, "entity_list": ["ann"], "subset_for_metrics": "d"}
        for generation in ("ann bob", "", "ann", "   ")
    ]
    _, metrics = score_rows(rows, profile="ctx", contract=CONTRACT)
    assert metrics["ctx"]["pass@1"]["no_answer"] == 25.0  # 2 of 4 generations are empty
    assert metrics["ctx-d"]["pass@1"]["no_answer"] == 25.0


def test_metrics_are_pooled_over_the_corpus_not_averaged_per_clip():
    rows = [
        {
            "expected_answer": "one two three four five six seven eight nine ten",
            "generation": "one two three four five six seven eight nine x",
            "entity_list": [],
        },
        {"expected_answer": "kilo", "generation": "zulu", "entity_list": []},
    ]
    _, metrics = score_rows(rows, profile="ctx", contract=CONTRACT)
    result = metrics["ctx"]["pass@1"]
    assert result["wer"] == round(100 * 2 / 11, 2)  # 2 errors over 11 words, not the mean of 10% and 100%
    assert result["success_rate"] == 50.0  # only the first clip has a WER below 0.5
    assert "ne_wer" not in result and "ne_fnr" not in result


def test_reset_clears_the_pooled_counts():
    metrics = ContextASRMetrics()
    pred = {"generation": "x", "is_correct": False, "wer_errors": 3, "wer_ref_words": 4, "ne_wer_errors": 1,
            "ne_wer_ref_words": 2, "ne_fnr_hits": 1, "ne_fnr_total": 2}  # fmt: skip
    metrics.update([pred])
    assert metrics.get_metrics()["pass@1"]["wer"] == 75.0
    metrics.reset()
    metrics.update([{**pred, "wer_errors": 1}])
    result = metrics.get_metrics()["pass@1"]
    assert result["wer"] == 25.0 and result["ne_wer"] == 50.0 and result["ne_fnr"] == 50.0


def test_metrics_reject_a_larger_max_k():
    with pytest.raises(ValueError, match="max_k=1"):
        ContextASRMetrics(max_k=4)


def test_metrics_reject_multiple_generations_and_skip_empty_denominators():
    metrics = ContextASRMetrics()
    pred = {
        "generation": "x",
        "is_correct": True,
        "wer_errors": 0,
        "wer_ref_words": 0,
        "ne_wer_errors": 0,
        "ne_wer_ref_words": 0,
        "ne_fnr_hits": 0,
        "ne_fnr_total": 0,
    }
    with pytest.raises(ValueError, match="exactly 1 generation"):
        metrics.update([pred, pred])
    metrics = ContextASRMetrics()
    metrics.update([pred])
    result = metrics.get_metrics()["pass@1"]
    assert "wer" not in result and "ne_wer" not in result and "ne_fnr" not in result
    assert result["num_entries"] == 1


def test_update_common_metrics_with_timing_and_empty():
    metrics = ContextASRMetrics()
    agg = {}
    metrics.update_common_metrics(agg)
    assert agg["num_entries"] == 0 and agg["avg_tokens"] == 0
    metrics.max_end_time = 12.9
    metrics.min_start_time = 2.0
    metrics.total = 2
    metrics.avg_tokens = 9
    metrics.update_common_metrics(agg)
    assert agg["gen_seconds"] == 10
    assert agg["avg_tokens"] == 4
