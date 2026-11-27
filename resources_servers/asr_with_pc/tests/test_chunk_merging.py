# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merging of chunked OpenASR rows into one session at aggregation time."""

from typing import Any, Dict, Optional

import pytest

from resources_servers.asr_with_pc.speech_metrics import normalize_open_asr_text
from resources_servers.asr_with_pc.tests.test_speech_task_types import _server


def _chunk(
    index: Optional[int],
    hypothesis: str,
    *,
    attempt: Optional[int] = None,
    parent: str = "call",
    reference: str = "good morning everyone",
    **extra: Any,
) -> Dict[str, Any]:
    row = {
        "parent_id": parent,
        "chunk_index": index,
        "subset_for_metrics": "earnings",
        "expected_answer": reference,
        # As verify() leaves them: the extracted output, and its normalization.
        "raw_pred_text": hypothesis,
        "pred_text": normalize_open_asr_text(hypothesis),
    }
    if attempt is not None:
        row["_ng_rollout_index"] = attempt
    return row | extra


def test_merge_joins_the_raw_hypotheses_not_their_normalizations() -> None:
    # "twenty" alone normalizes to "20" and "five dollars" to "$5": only the raw text rebuilds "twenty five dollars".
    tasks = [
        [_chunk(0, "twenty", reference="twenty five dollars")],
        [_chunk(1, "five dollars", reference="twenty five dollars")],
    ]
    metrics = _server().compute_metrics(tasks)
    assert metrics["corpus_wer@k=1"] == 0.0


def test_attempts_are_paired_by_rollout_index_not_list_position() -> None:
    # The repeats of chunk 1 arrive in the opposite order. Attempt 0 is "good" + "xyz abc" (two substitutions against
    # "good morning everyone"); pairing by list position would join "good" with attempt 1's "morning everyone".
    tasks = [
        [_chunk(0, "good", attempt=0), _chunk(0, "bad", attempt=1)],
        [_chunk(1, "morning everyone", attempt=1), _chunk(1, "xyz abc", attempt=0)],
    ]
    metrics = _server().compute_metrics(tasks)
    assert metrics["corpus_wer@k=1"] == pytest.approx(100.0 * 2 / 3)
    assert metrics["corpus_wer@k=2"] == pytest.approx(100.0 * 3 / 6)


def test_a_missing_chunk_rollout_skips_that_attempt_instead_of_failing() -> None:
    # Attempt 1 of chunk 1 was masked (and so removed before aggregation).
    tasks = [
        [_chunk(0, "good", attempt=0), _chunk(0, "good", attempt=1)],
        [_chunk(1, "morning everyone", attempt=0)],
    ]
    metrics = _server().compute_metrics(tasks)
    assert metrics["corpus_wer@k=1"] == 0.0
    assert metrics["open_asr_incomplete_sessions"] == 1
    assert _server().get_key_metrics(metrics)["open_asr_incomplete_sessions"] == 1


def test_a_gap_or_a_repeated_chunk_index_leaves_the_session_out() -> None:
    server = _server()
    gap = server.compute_metrics([[_chunk(0, "good")], [_chunk(2, "morning everyone")], [_chunk(0, "x", parent="b")]])
    assert gap["open_asr_incomplete_sessions"] == 1
    twice = server.compute_metrics([[_chunk(0, "good")], [_chunk(0, "morning everyone")]])
    assert "corpus_wer@k=1" not in twice and twice["open_asr_incomplete_sessions"] == 1


def test_only_openasr_rows_are_merged() -> None:
    # A Multilingual-ASR row with a parent_id is scored on its own: merging would re-score it with the English Open
    # ASR normalizer and report 0 errors instead of the 1 it already carries.
    row = {
        "task_type": "Multilingual-ASR",
        "parent_id": "p",
        "chunk_index": 0,
        "language": "de",
        "expected_answer": "hallo welt",
        "raw_pred_text": "hallo welt",
        "text": "hallo welt",
        "pred_text": "hallo welt",
        "is_correct": False,
        "wer": 0.5,
        "wer_errors": 1,
        "wer_ref_words": 2,
    }
    metrics = _server("Multilingual-ASR").compute_metrics([[row]])
    assert metrics["corpus_wer@k=1"] == pytest.approx(50.0)
    assert "open_asr_incomplete_sessions" not in metrics


def test_the_server_default_task_type_gates_the_merge_too() -> None:
    # Rows carry no task_type of their own when the server config supplies it.
    metrics = _server("Multilingual-ASR").compute_metrics([[_chunk(0, "good")], [_chunk(1, "morning everyone")]])
    assert "open_asr_incomplete_sessions" not in metrics


def test_the_loop_guardrail_applies_to_each_chunk_before_the_merge() -> None:
    looped = "good " + "ab" * 40
    tasks = [[_chunk(0, looped, reference="good ab morning")], [_chunk(1, "morning", reference="good ab morning")]]
    assert _server(cleanup_generation_loops=True).compute_metrics(tasks)["corpus_wer@k=1"] == 0.0
    assert _server().compute_metrics(tasks)["corpus_wer@k=1"] > 0.0


def test_the_utterance_mean_comes_from_merged_sessions() -> None:
    # Whole-session reference on each chunk: per row the chunks score 0.67 error against it, merged they score 0.
    tasks = [[_chunk(0, "good")], [_chunk(1, "morning everyone")]]
    metrics = _server().compute_metrics(tasks)
    assert metrics["mean_utterance_wer@k=1"] == 0.0
    assert "wer_macro@k=1" not in metrics
