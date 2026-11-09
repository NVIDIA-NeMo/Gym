# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the audit exporter preserves the observation unit and rejects bad evidence."""

import json
from pathlib import Path

import pytest
import yaml
from scripts.validate_repeat_planning import export_suite_planning_inputs


def _report() -> dict:
    return {
        "cells": [
            {
                "model": "model",
                "benchmark": "benchmark",
                "eligible": True,
                "baseline": {
                    "coverage": {"status": "verified"},
                    "repeat_scores": [10.0, 12.0, 14.0],
                    "summary": {"sample_variance": 4.0},
                },
                "candidate": {
                    "coverage": {"status": "verified"},
                    "repeat_scores": [20.0, 26.0],
                    "summary": {"sample_variance": 18.0},
                },
            }
        ]
    }


def test_exports_complete_pass_sd_and_frozen_baseline_count(tmp_path: Path) -> None:
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(_report()))

    records = export_suite_planning_inputs(source, tmp_path)

    assert records[0]["baseline"] == {
        "count": 3,
        "mean": 12.0,
        "variance": 4.0,
        "std": 2.0,
        "scores": [10.0, 12.0, 14.0],
    }
    for level in ("benchmark", "suite"):
        manifest = yaml.safe_load((tmp_path / f"cluster_{level}_plan.yaml").read_text())
        assert manifest["aggregation_level"] == level
        assert manifest["baseline"] == {"mode": "existing"}
        assert manifest["cells"] == [
            {
                "model": "model",
                "benchmark": "benchmark",
                "baseline_std_points": 2.0,
                "candidate_std_points": pytest.approx(18**0.5),
                "existing_baseline_repeats": 3,
                "repeat_step": 1,
            }
        ]


def test_exports_each_benchmark_separately_and_one_complete_suite(tmp_path: Path) -> None:
    report = _report()
    second = _report()["cells"][0]
    second["benchmark"] = "another"
    report["cells"].append(second)
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(report))

    export_suite_planning_inputs(source, tmp_path)

    benchmarks = [
        yaml.safe_load((tmp_path / f"cluster_benchmark_{index}_plan.yaml").read_text())["cells"] for index in (1, 2)
    ]
    assert [[cell["benchmark"] for cell in cells] for cells in benchmarks] == [["benchmark"], ["another"]]
    suite = yaml.safe_load((tmp_path / "cluster_suite_plan.yaml").read_text())
    assert [cell["benchmark"] for cell in suite["cells"]] == ["benchmark", "another"]


@pytest.mark.parametrize("defect", ["incomplete", "wrong_variance"])
def test_rejects_incomplete_or_inconsistent_saved_evidence(tmp_path: Path, defect: str) -> None:
    report = _report()
    if defect == "incomplete":
        report["cells"][0]["baseline"]["coverage"]["status"] = "incomplete"
        message = "complete task coverage is not verified"
    else:
        report["cells"][0]["candidate"]["summary"]["sample_variance"] = 9.0
        message = "reference=9"
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(report))

    with pytest.raises(ValueError, match=message):
        export_suite_planning_inputs(source, tmp_path)

    assert not (tmp_path / "cluster_benchmark_plan.yaml").exists()
