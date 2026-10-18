# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JUDGEBENCH scoring tests."""

import pytest

from benchmarks.genrm_judgebench.metrics import compute_metrics, score


def rollout(ranking: float | None, *, repeat: int = 0, **metadata: object) -> dict:
    return {"verifier_metadata": metadata, "predicted_ranking": ranking, "_ng_rollout_index": repeat}


def test_judgebench_both_orientations_and_repeats() -> None:
    metrics = compute_metrics(
        [
            [
                rollout(1, pair_id="p", source="mmlu-pro-math", label="A>B"),
                rollout(1, repeat=1, pair_id="p", source="mmlu-pro-math", label="A>B"),
            ],
            [
                rollout(6, pair_id="p-flipped", source="mmlu-pro-math", label="B>A"),
                rollout(1, repeat=1, pair_id="p-flipped", source="mmlu-pro-math", label="B>A"),
            ],
            [rollout(1, pair_id="incomplete", source="livecodebench", label="A>B")],
            [rollout(None, pair_id="q", source="livebench-math", label="A>B")],
            [rollout(6, pair_id="q-flip", source="livebench-math", label="B>A")],
        ],
    )
    assert metrics["metrics"]["overall"] == pytest.approx(2 / 4)
    assert metrics["metrics"]["knowledge"] == 0.5
    assert metrics["metrics"]["math"] == 1
    assert metrics["counts"]["overall"]["total"] == 4
    assert metrics["incomplete_pairs"] == 1
    assert metrics["no_answer"] == 1


@pytest.mark.parametrize(
    ("metadata", "ranking", "expected"),
    [({"label": "A>B"}, 3, 1), ({"label": "B>A"}, 3, 0), ({"label": "A>B"}, 4, 0), ({"label": "B>A"}, 4, 1)],
)
def test_per_row_reward(metadata, ranking, expected) -> None:
    assert score(metadata, ranking) == expected


def test_duplicate_orientations_are_rejected() -> None:
    row = rollout(1, pair_id="p", label="A>B")
    with pytest.raises(ValueError, match="Duplicate"):
        compute_metrics([[row], [row]])


def test_incomplete_pair_counts_as_incorrect() -> None:
    rows = [
        rollout(1, pair_id="p", label="A>B"),
        rollout(6, pair_id="p-flipped", label="B>A"),
        rollout(6, pair_id="q", label="A>B"),
    ]
    metrics = compute_metrics([[row] for row in rows])
    assert metrics["metrics"]["overall"] == 0.5
    assert metrics["counts"]["overall"]["total"] == 2
    assert metrics["incomplete_pairs"] == 1


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (1, 6, 1),
        (1, None, 1),
        (None, 6, 1),
        (1, 1, 0),
        (6, 6, 0),
        (6, 1, 0),
        (6, None, 0),
        (None, 1, 0),
        (None, None, 0),
    ],
)
def test_upstream_two_order_scoring(first, second, expected) -> None:
    metrics = compute_metrics(
        [
            [rollout(first, pair_id="p", label="A>B")],
            [rollout(second, pair_id="p-flipped", label="B>A")],
        ]
    )
    assert metrics["metrics"]["overall"] == expected
    assert metrics["counts"]["overall"]["total"] == 1
    assert metrics["no_answer"] == (first is None) + (second is None)
