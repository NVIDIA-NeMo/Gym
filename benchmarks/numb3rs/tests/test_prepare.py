# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for benchmarks/numb3rs/prepare.py row formatting.

The Numb3rs HF dataset is large (~10K WAVs, ~5GB) so the integration test
in ``prepare.prepare()`` is left for the cluster-side prepare job. Here we
exercise the row formatter on synthetic dicts to confirm:

* The Gym row carries ``responses_create_params.metadata.audio_path``
  pointed at ``<audio_prefix>/Numb3rs/<CATEGORY>/<filename>.wav``.
* ``expected_answer`` is the spoken form (Skills' default for the neutral
  variant).
* ``text_tn`` and ``text_itn`` ride alongside on the request and
  ``reference_fields`` lists exactly those two field names so the
  ``asr_with_pc`` server's ``ASR_LEADERBOARD`` mode can score against
  both references.
* Short-audio and missing-field rows are dropped.
"""

from benchmarks.numb3rs.prepare import _format_row


def _entry(**overrides):
    base = {
        "original_text": "$100",
        "text": "one hundred dollars",
        "file_name": "MONEY/MONEY_540__21_999.wav",
        "duration": 1.23,
        "category": "MONEY",
    }
    base.update(overrides)
    return base


def test_format_row_neutral_shape():
    row = _format_row(_entry(), audio_prefix="/data/numb3rs")

    assert row is not None
    # Audio sidechannel
    assert row["responses_create_params"]["metadata"] == {
        "audio_path": "/data/numb3rs/Numb3rs/MONEY/MONEY_540__21_999.wav",
    }
    # Neutral variant: expected_answer is the spoken form.
    assert row["expected_answer"] == "one hundred dollars"
    assert row["text_tn"] == "$100"
    assert row["text_itn"] == "one hundred dollars"
    # ASR_LEADERBOARD wiring
    assert row["task_type"] == "ASR_LEADERBOARD"
    assert row["reference_fields"] == ["text_tn", "text_itn"]
    # Misc Skills-parity fields
    assert row["audio_duration"] == 1.23
    assert row["category"] == "MONEY"
    assert row["sample_id"] == "MONEY_540__21_999"
    assert row["subset_for_metrics"] == "numb3rs_MONEY"


def test_format_row_strips_audio_prefix_trailing_slash():
    row = _format_row(_entry(), audio_prefix="/data/numb3rs/")
    assert row is not None
    assert row["responses_create_params"]["metadata"]["audio_path"] == (
        "/data/numb3rs/Numb3rs/MONEY/MONEY_540__21_999.wav"
    )


def test_format_row_drops_short_audio():
    assert _format_row(_entry(duration=0.05), audio_prefix="/data/numb3rs") is None


def test_format_row_drops_missing_fields():
    for missing in ("original_text", "text", "file_name", "category"):
        bad = _entry()
        bad[missing] = ""
        assert _format_row(bad, audio_prefix="/data/numb3rs") is None, f"row should drop when {missing!r} is empty"


def test_format_row_uppercases_category_in_path():
    """Skills uses CATEGORY-uppercase folders; the row must mirror that."""
    row = _format_row(_entry(category="money"), audio_prefix="/data/numb3rs")
    assert row is not None
    assert "/Numb3rs/MONEY/" in row["responses_create_params"]["metadata"]["audio_path"]
    assert row["category"] == "MONEY"
    assert row["subset_for_metrics"] == "numb3rs_MONEY"


def test_format_row_filename_handles_nested_path():
    """``file_name`` arrives like ``MONEY/MONEY_540__21_999.wav`` from HF."""
    row = _format_row(
        _entry(file_name="MONEY/sub/MONEY_540__21_999.wav"),
        audio_prefix="/data/numb3rs",
    )
    assert row is not None
    # Path basename is appended; intermediate dirs are dropped.
    assert row["responses_create_params"]["metadata"]["audio_path"].endswith("/Numb3rs/MONEY/MONEY_540__21_999.wav")
    assert row["sample_id"] == "MONEY_540__21_999"
