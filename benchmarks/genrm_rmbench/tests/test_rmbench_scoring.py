# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMBENCH scoring tests."""

import pytest

from benchmarks.genrm_rmbench.metrics import compute_metrics, score


def rollout(ranking: float | None, **metadata: object) -> dict:
    return {"verifier_metadata": metadata, "predicted_ranking": ranking}


def test_rmbench_domain_macro_average_and_difficulty() -> None:
    rows = [
        rollout(1, domain="chat-extra", correct="first", response_1_style_value=2, response_2_style_value=1),
        rollout(6, domain="math", correct="second", response_1_style_value=2, response_2_style_value=1),
        rollout(None, domain="code", correct="first", response_1_style_value=1, response_2_style_value=1),
        rollout(6, domain="safety", correct="first", response_1_style_value=1, response_2_style_value=2),
        rollout(6, domain="safety", correct="first", response_1_style_value=1, response_2_style_value=2),
    ]
    metrics = compute_metrics([[row] for row in rows])
    assert metrics["domain_coverage"] == 1
    assert metrics["safety_count"] == 2
    assert metrics["avg"] == 0.625  # Macro average, not mean over rows.
    assert metrics["easy"] == 0.25
    assert metrics["medium"] == 0.125
    assert metrics["hard"] == 0.25
    assert metrics["no_answer"] == 0.2


@pytest.mark.parametrize(
    ("metadata", "ranking", "expected"), [({"correct": "first"}, None, 0.5), ({"correct": "second"}, 6, 1)]
)
def test_per_row_reward(metadata, ranking, expected) -> None:
    assert score(metadata, ranking) == expected


def test_partial_run_reports_missing_domain_coverage(caplog) -> None:
    row = rollout(1, domain="chat", correct="first", response_1_style_value=0, response_2_style_value=0)
    metrics = compute_metrics([[row]])
    assert metrics["avg"] == 0.25
    assert metrics["domain_coverage"] == 0.25
    assert metrics["chat_count"] == 1
    assert metrics["math_count"] == metrics["code_count"] == metrics["safety_count"] == 0
    assert "Incomplete RM-Bench score" in caplog.text
