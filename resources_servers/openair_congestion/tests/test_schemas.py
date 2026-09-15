# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest
from openair_congestion.replay_env import build_trajectory
from openair_congestion.schemas import GlobalObservation, Observation


def _valid_observation_dict() -> dict:
    trajectory, _ = build_trajectory(
        seed=555,
        difficulty=0.95,
        regime_mix={"prb_exhaustion": 1.0},
        tier="replay",
        n_steps=1,
    )
    return trajectory[0].model_dump(by_alias=True)


def _wrong_cell_count(data: dict) -> None:
    data["global"]["n_cells"] += 1


def _wrong_total_ues(data: dict) -> None:
    data["global"]["n_ues_total"] += 1


def _wrong_rrc_count(data: dict) -> None:
    data["cells"][0]["rrc_connected_ues"] += 1


def _duplicate_cell_id(data: dict) -> None:
    data["cells"][1]["cell_id"] = data["cells"][0]["cell_id"]


def _duplicate_ue_id(data: dict) -> None:
    data["cells"][0]["ues"][1]["ue_id"] = data["cells"][0]["ues"][0]["ue_id"]


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_cell_count,
        _wrong_total_ues,
        _wrong_rrc_count,
        _duplicate_cell_id,
        _duplicate_ue_id,
    ],
    ids=[
        "cell-count",
        "ue-count",
        "rrc-count",
        "duplicate-cell",
        "duplicate-ue",
    ],
)
def test_observation_rejects_inconsistent_topology(mutation):
    data = copy.deepcopy(_valid_observation_dict())
    mutation(data)

    with pytest.raises(ValueError):
        Observation.model_validate(data)


@pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
def test_reset_schema_rejects_deferred_tiers(tier):
    with pytest.raises(ValueError):
        GlobalObservation.model_validate({"n_cells": 1, "n_ues_total": 0, "difficulty": 0.5, "tier": tier})
