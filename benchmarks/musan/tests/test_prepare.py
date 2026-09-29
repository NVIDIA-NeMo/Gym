# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the MUSAN row formatter.

We don't download the 11 GB OpenSLR archive in CI — we just verify the
shape of the row that ``prepare.py`` emits, which is the data-parity
contract with Skills.
"""

from benchmarks.musan.prepare import _build_row


class TestBuildRow:
    def test_row_has_verifier_fields(self) -> None:
        """The verifier reads expected_answer, audio_duration, task_type."""
        row = _build_row(
            category="noise",
            sample_id=42,
            audio_filename="musan_noise_000042.wav",
            audio_path="/data/musan/noise/audio/musan_noise_000042.wav",
            duration=7.5,
        )
        assert row["expected_answer"] == ""
        assert row["audio_duration"] == 7.5
        assert row["task_type"] == "Hallucination"

    def test_row_audio_path_in_metadata(self) -> None:
        """audio_path lives on responses_create_params.metadata for vllm_model."""
        row = _build_row(
            category="music",
            sample_id=0,
            audio_filename="musan_music_000000.wav",
            audio_path="/data/musan/music/audio/musan_music_000000.wav",
            duration=10.0,
        )
        meta = row["responses_create_params"]["metadata"]
        assert meta["audio_path"] == "/data/musan/music/audio/musan_music_000000.wav"
        # responses_create_params.input is left empty: prompt_config materializes
        # the messages at rollout time.
        assert "input" not in row["responses_create_params"]

    def test_subset_for_metrics_is_per_category(self) -> None:
        """Per-row subset_for_metrics enables noise/music/speech breakdown."""
        for category in ("noise", "music", "speech"):
            row = _build_row(
                category=category,
                sample_id=1,
                audio_filename=f"musan_{category}_000001.wav",
                audio_path=f"/data/musan/{category}/audio/musan_{category}_000001.wav",
                duration=5.0,
            )
            assert row["subset_for_metrics"] == f"musan_{category}"
            assert row["category"] == category

    def test_original_label_is_filename_stem(self) -> None:
        row = _build_row(
            category="speech",
            sample_id=3,
            audio_filename="musan_speech_000003.wav",
            audio_path="/data/musan/speech/audio/musan_speech_000003.wav",
            duration=1.0,
        )
        assert row["original_label"] == "musan_speech_000003"
