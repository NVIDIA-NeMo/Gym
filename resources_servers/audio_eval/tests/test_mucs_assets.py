# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MUCS scorer applies the downloaded maps; the download and verification are tested in the core benchmark tests."""

import importlib

from tests.unit_tests.benchmarks.test_mucs_assets import downloads  # noqa: F401  (fixture)


scorer = importlib.import_module("resources_servers.audio_eval._scoring.dataset.mucs2021.scorer")


def test_scorer_applies_the_downloaded_maps_in_the_official_order(downloads):
    row = {
        "extra_fields": {"mucs_language": "hindi-english"},
        "expected_answer": "Slash क",
        "generation": "slash कम",
    }
    result = scorer.evaluate_mucs2021_sample(row)
    # Normal WER sees the punctuation map only; transliterated WER also applies the language map.
    assert result["pred_text"] == "/ कम" and result["transliterated_pred_text"] == "/ cam"
    assert result["wer_errors"] == 2 and result["twer_errors"] == 2
