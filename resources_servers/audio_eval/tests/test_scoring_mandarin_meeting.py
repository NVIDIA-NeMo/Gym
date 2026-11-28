# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mandarin meeting cpCER scoring (normalization, evaluator, metrics), driven with small hand-counted rows."""

import asyncio

import pytest


pytest.importorskip("kaldialign")

from resources_servers.audio_eval._scoring.dataset.mandarin_meeting_common import (
    cpcer_token_string,
    normalize_aishell4_text,
    normalize_alimeeting_text,
    normalize_mandarin_meeting_text,
)
from resources_servers.audio_eval._scoring.evaluation.evaluator.mandarin_meeting import (
    MandarinMeetingEvaluator,
    evaluate_mandarin_meeting_sample,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.mandarin_meeting_metrics import MandarinMeetingMetrics
from resources_servers.audio_eval.score import evaluator_settings, resolve


# ---------------------------------------------------------------- reference normalization


def test_alimeeting_normalization_strips_markers_punctuation_and_folds_fullwidth_letters():
    assert normalize_alimeeting_text("你好，<sil>ａｂ (SIL)？ c d") == "你好ABCD"
    assert normalize_alimeeting_text("你好。") == "你好"  # no Latin letters: no case folding is needed
    assert normalize_alimeeting_text("hello world") == "HELLOWORLD"


def test_aishell4_normalization_keeps_only_han_characters():
    assert normalize_aishell4_text("你好, 世界! abc 123") == "你好世界"


def test_corpus_dispatch_and_character_tokenization():
    assert normalize_mandarin_meeting_text("你好a", "alimeeting") == "你好A"
    assert normalize_mandarin_meeting_text("你好a", "aishell4") == "你好"
    assert cpcer_token_string("你好a", "alimeeting-sot") == "你 好 A"
    with pytest.raises(ValueError, match="Unsupported Mandarin meeting corpus"):
        normalize_mandarin_meeting_text("x", "other")


# ---------------------------------------------------------------- Mandarin meeting evaluator


def test_swapped_speakers_are_matched_before_counting_character_errors():
    result = evaluate_mandarin_meeting_sample(
        {
            "expected_answer": "<spk:1> 你好吗 <spk:2> 我很好",
            "generation": "<spk:1> 我很好 <spk:2> 你好",
            "mandarin_corpus": "aishell4-sot",
        }
    )
    assert (result["cpcer_errors"], result["cpcer_substitutions"]) == (1, 0)
    assert (result["cpcer_insertions"], result["cpcer_deletions"], result["cpcer_ref_chars"]) == (0, 1, 6)
    assert result["cpcer"] == pytest.approx(1 / 6)
    assert result["reference_streams"] == {"s1": "你 好 吗", "s2": "我 很 好"}
    assert result["hypothesis_streams"] == {"s1": "我 很 好", "s2": "你 好"}
    assert result["text"] == "你 好 吗 我 很 好" and result["pred_text"] == "我 很 好 你 好"
    assert result["speaker_assignment"] == [
        {"reference": "s0", "hypothesis": "s1"},
        {"reference": "s1", "hypothesis": "s0"},
    ]
    assert result["mandarin_corpus"] == "aishell4-sot" and result["is_correct"] is False


def test_exact_session_with_alimeeting_normalization_is_correct():
    result = evaluate_mandarin_meeting_sample(
        {"expected_answer": "<spk:1> 你好，ａｂ", "generation": " <spk:1> 你好AB "},
        normalization="alimeeting-sot",
    )
    assert result["cpcer"] == 0.0 and result["cpcer_errors"] == 0 and result["is_correct"] is True
    assert result["text"] == "你 好 A B" and result["generation"] == "<spk:1> 你好AB"


def test_corpus_comes_from_the_row_unless_the_config_overrides_it():
    base = {"expected_answer": "<spk:1> 你好a", "generation": "<spk:1> 你好a"}
    assert evaluate_mandarin_meeting_sample({**base, "mandarin_corpus": "alimeeting-sot"})["text"] == "你 好 A"
    assert evaluate_mandarin_meeting_sample({**base, "mandarin_corpus": "aishell4-sot"})["text"] == "你 好"
    assert (
        evaluate_mandarin_meeting_sample({**base, "mandarin_corpus": "aishell4-sot"}, normalization="alimeeting-sot")[
            "mandarin_corpus"
        ]
        == "alimeeting-sot"
    )
    with pytest.raises(ValueError, match="missing a recognized corpus"):
        evaluate_mandarin_meeting_sample(base)
    with pytest.raises(ValueError, match="Unsupported Mandarin meeting corpus"):
        evaluate_mandarin_meeting_sample({**base, "mandarin_corpus": "other"})


def test_reference_that_normalizes_to_nothing_is_rejected():
    with pytest.raises(ValueError, match="empty after corpus-specific normalization"):
        evaluate_mandarin_meeting_sample(
            {"expected_answer": "<spk:1> ，。", "generation": "<spk:1> 你好"}, normalization="aishell4-sot"
        )


def test_mandarin_meeting_evaluator_applies_its_config():
    evaluator = MandarinMeetingEvaluator({"speaker_token": "<spk:*>", "normalization": "aishell4-sot"})
    result = asyncio.run(
        evaluator.eval_single({"expected_answer": "<spk:1> 你好吗", "generation": "<spk:1> 你好", "unused": 1})
    )
    assert result["cpcer_deletions"] == 1 and result["mandarin_corpus"] == "aishell4-sot"
    assert evaluator.eval_config.speaker_token == "<spk:*>"


def test_mandarin_meeting_metrics_aggregate_corpus_and_macro_cpcer():
    first = evaluate_mandarin_meeting_sample(
        {
            "expected_answer": "<spk:1> 你好吗 <spk:2> 我很好",
            "generation": "<spk:1> 我很好 <spk:2> 你好",
            "mandarin_corpus": "aishell4-sot",
        }
    )
    second = evaluate_mandarin_meeting_sample(
        {"expected_answer": "<spk:1> 你好吗啊", "generation": "<spk:1> 你好吗啊", "mandarin_corpus": "aishell4-sot"}
    )
    metrics = MandarinMeetingMetrics()
    metrics.update([first])
    metrics.update([second])
    values = metrics.get_metrics()["pass@1"]
    assert values["cpcer"] == 10.0  # 1 error over 6 + 4 reference characters
    assert values["cpcer_macro"] == 8.33  # mean of 1/6 and 0
    assert (values["cpcer_errors"], values["cpcer_deletions"], values["cpcer_ref_chars"]) == (1, 1, 10)
    assert (values["cpcer_substitutions"], values["cpcer_insertions"]) == (0, 0)
    assert values["success_rate"] == 50.0 and "correct" not in values


def test_mandarin_meeting_metrics_reject_multiple_generations():
    with pytest.raises(ValueError, match="one greedy generation"):
        MandarinMeetingMetrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        MandarinMeetingMetrics().update([])


def test_mandarin_meeting_evaluator_rejects_unsupported_settings_at_startup():
    with pytest.raises(ValueError, match="Unsupported Mandarin meeting corpus"):
        MandarinMeetingEvaluator({"normalization": "aishell-4"})
    with pytest.raises(ValueError, match="Unsupported speaker_token"):
        MandarinMeetingEvaluator({"speaker_token": "<speaker>"})


def test_hypothesis_with_extra_and_missing_speakers_is_padded_not_dropped():
    result = evaluate_mandarin_meeting_sample(
        {
            "expected_answer": "<spk:1> 你好 <spk:2> 我很好",
            "generation": "<spk:1> 你好 <spk:2> 我很好 <spk:3> 再见",
            "mandarin_corpus": "aishell4-sot",
        }
    )
    assert (result["cpcer_errors"], result["cpcer_insertions"], result["cpcer_ref_chars"]) == (2, 2, 5)
    merged = evaluate_mandarin_meeting_sample(
        {
            "expected_answer": "<spk:1> 你好 <spk:2> 我很好",
            "generation": "<spk:1> 你好我很好",
            "mandarin_corpus": "aishell4-sot",
        }
    )
    # the single hypothesis stream is matched to the longer reference stream: 2 inserted and 2 deleted characters
    assert (merged["cpcer_errors"], merged["cpcer_insertions"], merged["cpcer_deletions"]) == (4, 2, 2)


def test_empty_generation_is_all_deletions_and_counts_as_no_answer():
    result = evaluate_mandarin_meeting_sample(
        {"expected_answer": "<spk:1> 你好吗", "generation": None, "mandarin_corpus": "aishell4-sot"}
    )
    assert (result["cpcer"], result["cpcer_deletions"], result["generation"]) == (1.0, 3, "")
    metrics = MandarinMeetingMetrics()
    metrics.update([result])
    values = metrics.get_metrics()["pass@1"]
    assert values["no_answer"] == 50.0  # one empty answer out of one: halved, as in the other audio benchmarks
    assert values["cpcer"] == 100.0


def test_mandarin_meeting_metrics_reset_clears_the_counts():
    wrong = evaluate_mandarin_meeting_sample(
        {"expected_answer": "<spk:1> 你好", "generation": "<spk:1> 你", "mandarin_corpus": "aishell4-sot"}
    )
    right = evaluate_mandarin_meeting_sample(
        {"expected_answer": "<spk:1> 你好", "generation": "<spk:1> 你好", "mandarin_corpus": "aishell4-sot"}
    )
    metrics = MandarinMeetingMetrics()
    metrics.update([wrong])
    metrics.reset()
    assert (metrics.errors, metrics.ref_chars, metrics.session_scores) == (0, 0, [])
    metrics.update([right])
    values = metrics.get_metrics()["pass@1"]
    assert (values["cpcer"], values["cpcer_errors"], values["cpcer_ref_chars"]) == (0.0, 0, 2)


def test_aishell4_profile_resolves_through_the_server():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "benchmarks" / "aishell4_sot" / "profiles.json"
    contract = json.loads(path.read_text())["profiles"]["aishell4-sot"]
    assert evaluator_settings(contract) == (
        "mandarin_meeting",
        {"speaker_token": "spk", "normalization": "aishell4-sot", "cleanup_generation_loops": True},
    )
    assert resolve("evaluators", "mandarin_meeting").__name__ == "MandarinMeetingEvaluator"
    assert resolve("metrics", contract["skills"]["METRICS_TYPE"]).__name__ == "MandarinMeetingMetrics"
