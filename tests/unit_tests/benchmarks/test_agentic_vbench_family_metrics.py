# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark's equal-family mean from a rollouts file, grouped by the families the entry's task_data.json name."""

import json
import math
from pathlib import Path

import pytest

from benchmarks.agentic_vbench.family_metrics import family_metrics, row_task_id, task_families


def _entry(tmp_path: Path, families: dict[str, str]) -> Path:
    entry = tmp_path / "entry"
    for task_id, family in families.items():
        (entry / task_id).mkdir(parents=True)
        (entry / task_id / "task_data.json").write_text(json.dumps({"family": family}) + "\n")
    return entry


def _row(task_id: str, reward: float | None, *, masked: bool = False) -> dict:
    return {"_ng_task_id": {"taskset": "agentic_vbench", "task_id": task_id}, "reward": reward, "mask_sample": masked}


def test_family_means_count_masked_rows_apart_and_average_the_families_equally(tmp_path: Path) -> None:
    families = task_families(
        _entry(tmp_path, {"a1": "assembly", "a2": "assembly", "r1": "repair", "s1": "sequencing", "p1": "repurpose"})
    )
    rows = [
        _row("a1", 1.0),
        _row("a2", 0.0),
        _row("r1", 0.5),
        _row("s1", None, masked=True),
        _row("p1", 0.25),
    ]
    metrics = family_metrics(rows, families)
    assert metrics["assembly"] == {"rows": 2, "scored": 2, "masked": 0, "mean": 0.5}
    assert metrics["repair"] == {"rows": 1, "scored": 1, "masked": 0, "mean": 0.5}
    assert metrics["sequencing"]["masked"] == 1 and math.isnan(metrics["sequencing"]["mean"])
    assert metrics["repurpose"]["mean"] == 0.25
    # Three families scored: their unweighted mean, and not the benchmark score.
    assert metrics["equal_family"] == {"mean": pytest.approx((0.5 + 0.5 + 0.25) / 3), "families": 3}


def test_rows_of_unprepared_tasks_and_unscored_rewards_are_refused(tmp_path: Path) -> None:
    families = task_families(_entry(tmp_path, {"a1": "assembly"}))
    with pytest.raises(ValueError, match="no task_data.json"):
        family_metrics([_row("zz", 1.0)], families)
    with pytest.raises(ValueError, match="without a finite reward"):
        family_metrics([_row("a1", None)], families)
    assert row_task_id({"_ng_task_id": "a1"}) == "a1"
    with pytest.raises(ValueError, match="without _ng_task_id"):
        row_task_id({"reward": 1.0})
