# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merged multi-speaker parents keep exact per-leaf cpWER counts."""

import pytest


pytest.importorskip("kaldialign")

from resources_servers.audio_eval._scoring.evaluation.metrics.audio_metrics import AudioMetrics


def test_audio_metrics_reports_cpwer_subset_error_counts():
    metrics = AudioMetrics()
    sample = {
        "cpwer": 0.1,
        "cpwer_errors": 1,
        "cpwer_substitutions": 1,
        "cpwer_insertions": 0,
        "cpwer_deletions": 0,
        "cpwer_ref_words": 10,
        "subset_for_metrics": "ami-ihm-test",
        "is_correct": False,
    }
    metrics.update([sample])
    metrics.update([{**sample, "cpwer": 0.2, "cpwer_errors": 2, "cpwer_substitutions": 0, "cpwer_insertions": 2}])
    output = metrics.get_metrics()["pass@1"]
    assert output["cpwer"] == 15.0
    assert output["cpwer_errors"] == 3
    assert output["cpwer::ami-ihm-test"] == 15.0
    assert output["cpwer::ami-ihm-test_errors"] == 3
    assert output["cpwer::ami-ihm-test_substitutions"] == 1
    assert output["cpwer::ami-ihm-test_insertions"] == 2
    assert output["cpwer::ami-ihm-test_deletions"] == 0
    assert output["cpwer::ami-ihm-test_ref_words"] == 20
    assert output["cpwer::ami-ihm-test_num_entries"] == 2
