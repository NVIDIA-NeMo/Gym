# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assigned harnesses preserve distinct-task batches across all execution routes."""

import copy
from types import SimpleNamespace

import pytest
from scripts import prepare_swe_single_agent_turn_recipe as recipe

from tests.unit_tests.tooling.swe_training_fixtures import NEXT_ROW


@pytest.mark.parametrize("route", ["old-run", "single-agent-turn", "harbor"])
def test_assignment_preserves_batch_tasks_and_hidden_verifier_data(route):
    rows = [
        copy.deepcopy(NEXT_ROW) | {"instance_id": f"task-{index}", "harbor_task_id": f"prepared-{index}"}
        for index in range(32)
    ]
    original = copy.deepcopy(rows)
    assignment = ["opencode", "pi"] * 16
    prepared = None
    if route == "harbor":
        prepared = SimpleNamespace(
            tasks=[
                SimpleNamespace(
                    task_id=row["harbor_task_id"],
                    task_data={"instance_id": row["instance_id"]},
                    digest="a" * 64,
                    instruction=row["responses_create_params"]["input"][0]["content"],
                    turns=[],
                    config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=480)),
                )
                for row in rows
            ]
        )
    single = recipe.materialize_training_rows(rows, harnesses=["opencode"], route=route, prepared=prepared)
    mixed = recipe.materialize_training_rows(
        rows, harnesses=["opencode", "pi"], route=route, prepared=prepared, task_harnesses=assignment
    )
    assert len(single) == len(mixed) == 32
    for index, (one, many) in enumerate(zip(single, mixed)):
        if route == "old-run":
            assert one["instance_id"] == many["instance_id"] == rows[index]["instance_id"]
            assert many["agent_ref"]["name"] == recipe.OLD_AGENTS[assignment[index]]
            assert many["responses_create_params"] == rows[index]["responses_create_params"]
            assert many["test_patch"] == rows[index]["test_patch"]
        else:
            assert one["task_id"]["task_id"] == many["task_id"]["task_id"]
            assert many["task_id"]["taskset"] == "swe_next_" + assignment[index]
            if route == "single-agent-turn":
                assert many["task_input"]["task_data"]["test_patch"] == rows[index]["test_patch"]
                assert many["task_input"]["responses_create_params"] == rows[index]["responses_create_params"]
    for start in (0, 16):
        assert assignment[start : start + 16].count("opencode") == 8
        assert assignment[start : start + 16].count("pi") == 8
    assert rows == original
    default = recipe.materialize_training_rows(rows, harnesses=["opencode", "pi"], route=route, prepared=prepared)
    assert len(default) == 64


@pytest.mark.parametrize("assignment", [[], ["pi", "opencode"], ["hermes"]])
def test_invalid_assignment_is_rejected(assignment):
    with pytest.raises(ValueError):
        recipe.materialize_training_rows(
            [NEXT_ROW], harnesses=["opencode", "pi"], route="old-run", task_harnesses=assignment
        )
