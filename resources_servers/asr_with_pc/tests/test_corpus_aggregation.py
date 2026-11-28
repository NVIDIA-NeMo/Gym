# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Corpus pooling across task types and units, and the key space of the breakdown metrics."""

from typing import Any, Dict

import jiwer
import pytest

from resources_servers.asr_with_pc.tests.test_speech_task_types import _server, _verify


def _counted(errors: int, words: int, **extra: Any) -> Dict[str, Any]:
    row = {
        "is_correct": errors == 0,
        "pred_text": "x",
        "wer": errors / words,
        "wer_errors": errors,
        "wer_ref_words": words,
    }
    return row | extra


async def test_mixed_task_types_pool_into_one_corpus_wer() -> None:
    # Two wrong ASR-PC rows (no edit counts) and one correct ASR row (edit counts): neither group may vanish.
    rows = [
        await _verify("ASR-PC", "hello world", "completely different"),
        await _verify("ASR-PC", "good morning", "something else"),
        await _verify("ASR", "see you soon", "see you soon"),
    ]
    metrics = _server().compute_metrics([[row] for row in rows])
    expected = 100.0 * jiwer.wer([r["text"] for r in rows], [r["pred_text"] for r in rows])
    assert metrics["corpus_wer@k=1"] == pytest.approx(expected)
    assert 0.0 < expected < 100.0


async def test_a_reference_that_normalizes_to_empty_is_left_out_of_corpus_wer() -> None:
    # Whisper's normalizer erases fillers such as "Uh.". On main the hypothesis words of such a row counted as
    # insertions; like the Open ASR leaderboard, the row now has no edit counts and is dropped.
    filler = await _verify("ASR", "Uh.", "hello there")
    scored = await _verify("ASR", "hello world", "hello world")
    assert filler["text"] == "" and filler["wer_ref_words"] is None
    metrics = _server().compute_metrics([[filler], [scored]])
    assert metrics["corpus_wer@k=1"] == 0.0


def test_character_scored_rows_stay_out_of_the_word_pool() -> None:
    # English 5/10 words plus Chinese 0/990 characters must not give a headline WER of 0.5.
    english = _counted(5, 10, language="en", error_unit="word", cer_errors=5, cer_ref_words=50)
    chinese = _counted(0, 990, language="zh", error_unit="char", cer_errors=0, cer_ref_words=990)
    metrics = _server("Multilingual-ASR").compute_metrics([[english], [chinese]])
    assert metrics["corpus_wer@k=1"] == pytest.approx(50.0)
    # Each language's own rate stays available in its own unit, and CER pools the characters of both.
    assert metrics["wer::language::zh@k=1"] == 0.0 and metrics["wer::language::en@k=1"] == pytest.approx(50.0)
    assert metrics["cer@k=1"] == pytest.approx(100.0 * 5 / 1040)
    assert metrics["mean_utterance_wer@k=1"] == pytest.approx(50.0)


def test_breakdown_kinds_do_not_share_a_key_space() -> None:
    # A subset and a language that happen to be called "de" keep separate rates.
    rows = [
        [_counted(1, 10, subset_for_metrics="de", language="fr", benchmark_profile="de")],
        [_counted(3, 10, subset_for_metrics="mcv", language="de", benchmark_profile="p")],
    ]
    metrics = _server().compute_metrics(rows)
    assert metrics["wer::subset::de@k=1"] == pytest.approx(10.0)
    assert metrics["wer::language::de@k=1"] == pytest.approx(30.0)
    assert metrics["wer::profile::de@k=1"] == pytest.approx(10.0)
    assert not [key for key in metrics if key.startswith("wer::") and key.count("::") == 1]


def test_the_language_breakdown_is_published() -> None:
    # Dropping "language" from the grouped fields removes these keys.
    rows = [[_counted(1, 10, language="de")], [_counted(2, 10, language="fr")]]
    metrics = _server().compute_metrics(rows)
    assert (metrics["wer::language::de@k=1"], metrics["wer::language::fr@k=1"]) == pytest.approx((10.0, 20.0))


def test_the_multilingual_macro_is_chosen_by_server_config_not_by_a_profile_name() -> None:
    rows = [
        [_counted(0, 1, benchmark_profile="suite", language="de", subset_for_metrics="a")],
        [_counted(100, 100, benchmark_profile="suite", language="de", subset_for_metrics="b")],
    ]
    assert "multilingual_macro_wer@k=1" not in _server().compute_metrics(rows)
    configured = _server(multilingual_macro_profiles=["suite"]).compute_metrics(rows)
    assert configured["multilingual_macro_wer@k=1"] == pytest.approx(50.0)
    assert configured["corpus_wer@k=1"] == pytest.approx(50.0)
    # The hard-coded leaderboard name no longer switches anything on.
    renamed = [[row[0] | {"benchmark_profile": "asr-leaderboard-multilingual"}] for row in rows]
    assert "multilingual_macro_wer@k=1" not in _server().compute_metrics(renamed)


def test_the_utterance_mean_is_not_defined_for_hallucination_rows() -> None:
    # MUSAN rows have no edit counts: the old wer_macro published a constant 0.0 for them.
    rows = [[{"is_correct": True, "pred_text": "short", "text": "", "hallucination_rate": 0.0}]]
    metrics = _server("Hallucination").compute_metrics(rows)
    assert not [key for key in metrics if key.startswith(("mean_utterance", "wer_macro", "corpus_wer"))]
