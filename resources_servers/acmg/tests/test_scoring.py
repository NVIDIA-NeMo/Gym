# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest

from .. import official_reward
from ..scoring import final_answer_text, metrics, score, summarize


UPSTREAM_SHA256 = "e2a80a9d3d0cc5051d73c9d6ea6245e4e2272ea5ef00cb6375a978b152666f1f"  # pragma: allowlist secret
MARKER = b"# --- upstream file begins below, byte for byte ---\n"
REFERENCE = {
    "label": "LP",
    "label_scoring": "scored",
    "accepted_labels": ["LP"],
    "required": {"PVS1": ["VeryStrong"], "PM2": ["Supporting"]},
    "acceptable": [["PVS1", "VeryStrong"], ["PM2", "Supporting"]],
    "combining": None,
}


def test_vendored_scorer_matches_the_pinned_upstream_file() -> None:
    text = (Path(__file__).resolve().parents[1] / "official_reward.py").read_bytes()
    header, body = text.split(MARKER, 1)
    assert UPSTREAM_SHA256 in header.decode()
    assert hashlib.sha256(body).hexdigest() == UPSTREAM_SHA256


def test_default_scalar_is_the_published_weighting() -> None:
    parts = official_reward.reward("CRITERIA: PVS1 VeryStrong\nCLASSIFICATION: LP", REFERENCE, "without_rules")
    # Label right, PM2 missing (recall 1/2, precision 1, F1 2/3); PVS1 alone is LP on the 2020 point scale (8 points).
    assert parts["label"] == 1.0
    assert parts["criterion_f1"] == pytest.approx(2 / 3)
    assert parts["self_consistent"] == 1.0
    assert official_reward.scalar(parts) == pytest.approx(0.5 + 0.3 * 2 / 3 + 0.2)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>draft</think>final", "final"),
        ("<thinking>\nCLASSIFICATION: P\n</thinking>\n\nCLASSIFICATION: B", "CLASSIFICATION: B"),
        ("no opening tag</think>CLASSIFICATION: VUS", "CLASSIFICATION: VUS"),
        ("<think>never closed", ""),
        ("", ""),
    ],
)
def test_final_answer_text_removes_reasoning(text: str, expected: str) -> None:
    assert final_answer_text(text) == expected


def test_score_counts_only_acceptable_pairs() -> None:
    result = score(
        "CRITERIA: PVS1 Strong; PM2 Supporting; BP4 Supporting\nCLASSIFICATION: VUS", REFERENCE, "without_rules"
    )
    assert result["predicted_criteria"] == {"PVS1": "Strong", "PM2": "Supporting", "BP4": "Supporting"}
    assert (result["returned_n"], result["returned_acceptable"]) == (3, 1)
    assert (result["required_n"], result["required_hit"]) == (2, 1)
    assert result["self_consistency_assessable"]
    assert result["components"]["label"] == 0.0


def test_score_without_a_criteria_list_is_not_assessable() -> None:
    result = score("CLASSIFICATION: LP", REFERENCE, "without_rules")
    assert result["predicted_label"] == "LP"
    assert result["predicted_criteria"] is None
    assert not result["self_consistency_assessable"]
    assert result["components"]["self_consistent"] == 0.0


def test_summarize_pools_criterion_counts_rather_than_averaging() -> None:
    perfect = score("CRITERIA: PVS1 VeryStrong; PM2 Supporting\nCLASSIFICATION: LP", REFERENCE, "without_rules")
    noisy = score(
        "CRITERIA: PVS1 VeryStrong; PP3 Supporting; PP1 Supporting; PM1 Moderate\nCLASSIFICATION: LP",
        REFERENCE,
        "without_rules",
    )
    summary = summarize([perfect, noisy])
    # 2 + 1 acceptable of 2 + 4 returned; 2 + 1 required hits of 2 + 2.
    assert summary["criterion_precision"] == pytest.approx(3 / 6)
    assert summary["criterion_recall"] == pytest.approx(3 / 4)
    assert summary["exact_criteria"] == 0.5
    assert summary["label_accuracy"] == 1.0
    assert summary["responses"] == 2
    assert summarize([]) == {}


def test_metrics_skip_arms_without_rows() -> None:
    row = {
        **score("CRITERIA: PVS1 VeryStrong; PM2 Supporting\nCLASSIFICATION: LP", REFERENCE, "without_rules"),
        "verifier_metadata": {"arm": "without_rules"},
    }
    result = metrics([row])
    assert result["acmg/without_rules/label_accuracy"] == 1.0
    assert not any(key.startswith("acmg/with_rules/") for key in result)
