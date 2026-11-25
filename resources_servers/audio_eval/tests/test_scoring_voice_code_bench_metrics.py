# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VoiceCodeBench corpus metrics: pooled WER, CTEM/TSR from verified rows, and the consistency checks."""

import pytest

from resources_servers.audio_eval._scoring.evaluation.metrics.voice_code_bench_metrics import VoiceCodeBenchMetrics


def vcb_row(**override):
    row = {
        "voice_code_bench_revision": "r1",
        "voice_code_wer_errors": 1,
        "voice_code_wer_ref_words": 10,
        "voice_code_wer_hyp_words": 9,
        "voice_code_wer_substitutions": 1,
        "voice_code_wer_deletions": 0,
        "voice_code_wer_insertions": 0,
        "generation": "text",
        "is_correct": True,
    }
    row.update(override)
    return row


def verified(**override):
    fields = {
        "voice_code_entity_verifier_status": "complete",
        "voice_code_correct_entities": 1,
        "voice_code_total_entities": 2,
        "voice_code_task_success": False,
        "voice_code_verifier_id": "v1",
        "voice_code_verifier_config_digest": "d1",
        "voice_code_correct_by_type": {"phone_number": 1},
        "voice_code_total_by_type": {"phone_number": 2},
    }
    return vcb_row(**{**fields, **override})


def test_voice_code_bench_metrics_report_token_and_generation_time_stats():
    metrics = VoiceCodeBenchMetrics()
    metrics.update([vcb_row(num_generated_tokens=10, generation_start_time=100, generation_end_time=110)])
    metrics.update(
        [vcb_row(num_generated_tokens=20, generation_start_time=105, generation_end_time=130, voice_code_wer_errors=3)]
    )
    values = metrics.get_metrics()["pass@1"]
    assert values["num_entries"] == 2 and values["avg_tokens"] == 15 and values["gen_seconds"] == 30
    assert values["wer"] == 20.0 and values["reference_words"] == 20 and values["hypothesis_words"] == 18
    assert (values["substitutions"], values["deletions"], values["insertions"]) == (2, 0, 0)
    assert values["official_revision"] == "r1" and values["entity_verified_entries"] == 0
    assert "ctem" not in values


def test_voice_code_bench_metrics_without_tokens_or_timing_omit_them():
    metrics = VoiceCodeBenchMetrics()
    metrics.update([vcb_row(voice_code_wer_ref_words=0)])
    values = metrics.get_metrics()["pass@1"]
    assert "avg_tokens" not in values and "gen_seconds" not in values and values["wer"] == 0.0


def test_voice_code_bench_metrics_aggregate_verified_entities_per_type():
    metrics = VoiceCodeBenchMetrics()
    metrics.update([verified()])
    metrics.update([verified(voice_code_task_success=True, voice_code_correct_entities=2)])
    values = metrics.get_metrics()["pass@1"]
    assert values["entity_verified_entries"] == 2
    assert (values["correct_entities"], values["total_entities"], values["successful_tasks"]) == (3, 4, 1)
    assert values["ctem"] == 75.0 and values["tsr"] == 50.0
    assert values["verifier_id"] == "v1" and values["verifier_config_digest"] == "d1"
    assert values["ctem::phone_number"] == 100 * 2 / 4  # 1 + 1 correct by type of 2 + 2 total


def test_voice_code_bench_no_answer_is_the_plain_empty_generation_share():
    # Unlike the audio metrics, which publish half of this share, nothing halves it here.
    metrics = VoiceCodeBenchMetrics()
    metrics.update([vcb_row(generation="")])
    metrics.update([vcb_row()])
    assert metrics.get_metrics()["pass@1"]["no_answer"] == 50.0


def test_voice_code_bench_metrics_reset_starts_a_new_corpus():
    metrics = VoiceCodeBenchMetrics()
    metrics.update([verified()])
    metrics.reset()
    metrics.update([vcb_row(voice_code_bench_revision="r2", voice_code_wer_errors=0)])
    values = metrics.get_metrics()["pass@1"]
    assert values["num_entries"] == 1 and values["wer"] == 0.0 and values["official_revision"] == "r2"
    assert values["entity_verified_entries"] == 0 and "ctem::phone_number" not in values


def test_voice_code_bench_metrics_reject_inconsistent_runs():
    with pytest.raises(ValueError, match="one raw batch transcript"):
        VoiceCodeBenchMetrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        VoiceCodeBenchMetrics().update([vcb_row(), vcb_row()])

    mixed_revisions = VoiceCodeBenchMetrics()
    mixed_revisions.update([vcb_row()])
    mixed_revisions.update([vcb_row(voice_code_bench_revision="r2")])
    with pytest.raises(ValueError, match="mixed revisions"):
        mixed_revisions.get_metrics()

    partial = VoiceCodeBenchMetrics()
    partial.update([verified()])
    partial.update([vcb_row()])
    with pytest.raises(ValueError, match="Partial VoiceCodeBench entity scoring: 1/2"):
        partial.get_metrics()

    mixed_verifiers = VoiceCodeBenchMetrics()
    mixed_verifiers.update([verified()])
    mixed_verifiers.update([verified(voice_code_verifier_id="v2")])
    with pytest.raises(ValueError, match="mixed verifier versions"):
        mixed_verifiers.get_metrics()
