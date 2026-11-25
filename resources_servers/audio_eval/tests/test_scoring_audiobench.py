# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AudioBench scoring: the normalization modes, the Spoken-MQA answer protocol and its metrics."""

import asyncio

import pytest


pytest.importorskip("jiwer")
pytest.importorskip("whisper_normalizer")

from resources_servers.audio_eval._scoring.evaluation.evaluator import audio as A
from resources_servers.audio_eval._scoring.evaluation.evaluator import audiobench as AB
from resources_servers.audio_eval._scoring.evaluation.evaluator.audiobench import AudioBenchEvaluator
from resources_servers.audio_eval._scoring.evaluation.metrics.audiobench_metrics import AudioBenchMetrics
from resources_servers.audio_eval.score import resolve


def run_row(row, **settings):
    return asyncio.run(AudioBenchEvaluator(settings).eval_single(row))


def metrics_of(rows_per_example, **kwargs):
    metrics = AudioBenchMetrics(**kwargs)
    for rows in rows_per_example:
        metrics.update(rows)
    return metrics.get_metrics()


def test_audiobench_scorers_are_found_by_name():
    assert resolve("evaluators", "audiobench") is AudioBenchEvaluator
    assert resolve("metrics", "audiobench") is AudioBenchMetrics


def test_the_audiobench_modes_are_registered_with_the_audio_evaluator():
    assert {"audiobench", "audiobench_earnings"} <= set(A.valid_normalization_modes())
    cfg = A.AudioEvaluatorConfig(normalization_mode="audiobench")
    assert A.resolve_asr_normalization_mode(cfg) == "audiobench"
    cfg = A.AudioEvaluatorConfig(normalization_mode="audiobench", apply_normalization=False)
    assert A.resolve_asr_normalization_mode(cfg) == "none"


def test_preprocess_asr_text_audiobench_modes():
    # The ``audiobench`` mode spells numerals out, expands contractions and drops fillers/brackets.
    out = A.preprocess_asr_text("Um, I'm 1 of [noise] them", mode="audiobench")
    assert out == "i am one of them"
    assert "two" in A.preprocess_asr_text("2 cats", mode="audiobench").split()
    out_e = A.preprocess_asr_text("I kinda cause 2 problems", mode="audiobench_earnings")
    assert "kinda" in out_e and "cause" in out_e
    # The ``audiobench`` mode applies the slang expansion; the Earnings mode must not.
    out_h = A.preprocess_asr_text("I kinda cause 2 problems", mode="audiobench")
    assert "kind of" in out_h


def test_audiobench_helpers():
    assert AB._normalize_digits_to_words("1 and 20 and 100") == "one and twenty and 100"
    assert AB._expand_contractions("he's here and i'm there, don't") == "he is here and i am there, do not"
    assert AB._remove_non_speech_elements("uh yes um no") == " yes  no"


def test_spoken_mqa_canonical_number():
    assert AB._spoken_mqa_canonical_number("3.0") == "3"
    assert AB._spoken_mqa_canonical_number("3.5") == "3.5"
    assert AB._spoken_mqa_canonical_number("1/4") == "0.25"
    assert AB._spoken_mqa_canonical_number("4/2") == "2"
    assert AB._spoken_mqa_canonical_number("1/0") == "1/0"
    assert AB._spoken_mqa_canonical_number("abc") == "abc"
    assert AB._spoken_mqa_canonical_number(None) == "None"


def test_clean_spoken_mqa_prediction():
    assert AB.clean_spoken_mqa_prediction("The answer is: 1,234.") == "1234"
    assert AB.clean_spoken_mqa_prediction("so 3 then 4 then 5") == "5"  # no trigger: last numeral
    assert AB.clean_spoken_mqa_prediction("The answer is 7 or 8") == "7"  # trigger: first numeral after it
    assert AB.clean_spoken_mqa_prediction("#### 12.0") == "12"
    assert AB.clean_spoken_mqa_prediction("so #### 12 and later 15") == "12"  # the marker, not the last numeral
    assert AB.clean_spoken_mqa_prediction("no numbers here") == ""
    assert AB.clean_spoken_mqa_prediction("The answer is 1/2") == "0.5"
    repeated = "The answer is 4\n\nExample: The answer is 9"
    assert AB.clean_spoken_mqa_prediction(repeated) == "4"
    assert AB.clean_spoken_mqa_prediction(None) == ""


def test_spoken_mqa_reference_list_protocols():
    assert AB._spoken_mqa_reference_list("5", "arithmetic") == ["5"]
    assert AB._spoken_mqa_reference_list({"text": ["a", "b"]}, "arithmetic") == ["a", "b"]
    assert AB._spoken_mqa_reference_list("work #### 1,200", "reasoning") == ["1200"]
    assert AB._spoken_mqa_reference_list(["7", "8.0"], "reasoning") == ["7", "8"]
    assert AB._spoken_mqa_reference_list([], "arithmetic") == ["empty"]
    with pytest.raises(ValueError, match="Unknown Spoken-MQA protocol"):
        AB._spoken_mqa_reference_list("1", "bogus")


def test_spoken_mqa_answers_match():
    assert AB._spoken_mqa_answers_match("12", "12") is True
    assert AB._spoken_mqa_answers_match("12", "13") is False
    assert AB._spoken_mqa_answers_match("\\text{12}", "12") is True
    assert AB._spoken_mqa_answers_match("12", "\\text{12}") is True
    assert AB._spoken_mqa_answers_match("the (b) one", "(B)") is True
    assert AB._spoken_mqa_answers_match("(c)", "(b)") is False


def test_evaluate_spoken_mqa_rows():
    ok = AB.evaluate_spoken_mqa("42", "The answer is 42.", "arithmetic")
    assert ok["is_correct"] is True
    assert ok["spoken_mqa_prediction"] == "42" == ok["predicted_answer"]
    assert ok["spoken_mqa_references"] == ["42"]
    assert ok["spoken_mqa_parse_failure"] is False
    assert ok["requires_judge"] is False
    fail = AB.evaluate_spoken_mqa("42", "no idea", "arithmetic")
    assert fail["is_correct"] is False
    assert fail["spoken_mqa_prediction"] == "empty" and fail["spoken_mqa_parse_failure"] is True
    # An empty reference degrades to the "empty" sentinel, which a parse failure matches (upstream quirk).
    assert AB.evaluate_spoken_mqa("", "nothing", "arithmetic")["is_correct"] is True
    assert AB.evaluate_spoken_mqa(["   "], "7", "arithmetic")["spoken_mqa_references"] == ["empty"]
    reasoning = AB.evaluate_spoken_mqa("steps #### 1,200", "The answer is: 1200", "reasoning")
    assert reasoning["is_correct"] is True


def test_eval_single_math_qa_row():
    res = run_row({"task_type": "MathQA", "generation": "The answer is: 12", "expected_answer": "12"})
    assert res["is_correct"] is True and res["spoken_mqa_prediction"] == "12"
    res = run_row(
        {
            "task_type": "MathQA",
            "generation": "so #### 7",
            "expected_answer": "reasoning ... #### 8",
            "spoken_mqa_protocol": "reasoning",
        }
    )
    assert res["is_correct"] is False and res["spoken_mqa_references"] == ["8"]


def test_audiobench_earnings_normalizer_rejects_unexpected_whisper_normalizer(monkeypatch):
    import whisper_normalizer.english as english

    class Drifted:
        replacers = {r"\bkinda\b": "kind-of-different"}

    monkeypatch.setattr(english, "EnglishTextNormalizer", Drifted)
    AB._get_audiobench_earnings_english_normalizer.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="no longer matches the pinned 0.1.12 compatibility contract"):
            AB._get_audiobench_earnings_english_normalizer()
    finally:
        AB._get_audiobench_earnings_english_normalizer.cache_clear()


def test_spoken_mqa_accuracy_alias_and_parse_failure_rate():
    rows = [
        {"task_type": "MathQA", "is_correct": True, "spoken_mqa_parse_failure": False, "generation": "1"},
        {"task_type": "MathQA", "is_correct": False, "spoken_mqa_parse_failure": True, "generation": "?"},
        {"task_type": "MathQA", "is_correct": True, "generation": "3"},  # no diagnostic field: not counted
        {"task_type": "MathQA", "is_correct": False, "spoken_mqa_parse_failure": False, "generation": "4"},
    ]
    out = metrics_of([[r] for r in rows])["pass@1"]
    assert out["accuracy"] == 50.0 and out["success_rate"] == 50.0
    assert out["no_numeric_answer_rate"] == round(100 * 1 / 3, 2)
