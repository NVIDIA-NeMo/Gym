# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HS3 scoring tests."""

import pytest

from resources_servers.genrm_eval.preferences import compute_metrics, score


def rollout(ranking: float | None, **metadata: object) -> dict:
    return {"verifier_metadata": metadata, "predicted_ranking": ranking}


def test_hs3_denominator_and_slices() -> None:
    metrics = compute_metrics(
        [
            [rollout(2, ground_truth_label=2.75, data_type="chat")],
            [rollout(None, ground_truth_label=1, data_type="chat")],
            [rollout(4, ground_truth_label=4.25, data_type="math")],
            [rollout(1)],  # No human label: excluded.
        ],
    )
    assert metrics["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert metrics["overall"]["total"] == 3
    assert metrics["overall"]["no_answer"] == 1
    assert metrics["by_data_type"]["chat"]["accuracy"] == 0.5
    assert metrics["by_data_type"]["math"]["accuracy"] == 1


def test_collection_failure_without_verifier_output() -> None:
    failure = {"verifier_metadata": {"ground_truth_label": 1}, "reward": 0.0}
    metrics = compute_metrics([[failure]])
    assert metrics["overall"]["total"] == 1
    assert metrics["overall"]["no_answer"] == 1


@pytest.mark.parametrize(
    ("metadata", "ranking", "expected"),
    [
        ({"ground_truth_label": 1}, 3, 1),
        ({"ground_truth_label": 6}, 3, 0),
        ({"ground_truth_label": 1}, 4, 0),
        ({"ground_truth_label": 6}, 4, 1),
        ({"ground_truth_label": 3.2}, 3, 1),
        ({"ground_truth_label": 3.2}, 4, 0),
        ({"ground_truth_label": 3.8}, 4, 1),
        ({"ground_truth_label": 3.8}, 3, 0),
        ({}, 1, 0),
        ({"ground_truth_label": 1}, None, 0),
    ],
)
def test_per_row_reward(metadata, ranking, expected) -> None:
    assert score(metadata, ranking) == expected


@pytest.mark.parametrize("data_type", [["chat"], {"category": "chat"}, None])
def test_malformed_saved_data_type_uses_unknown_slice(data_type: object) -> None:
    metrics = compute_metrics([[rollout(2, ground_truth_label=2, data_type=data_type)]])
    assert metrics["overall"]["accuracy"] == 1
    assert metrics["by_data_type"]["unknown"]["total"] == 1
