# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the audit exporter preserves the observation unit and rejects bad evidence."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.validate_repeat_planning import export_suite_planning_inputs, run_validation


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


@pytest.mark.parametrize("excluded_count", [0, 2])
def test_rejects_suite_without_eligible_cells_before_writing_outputs(tmp_path: Path, excluded_count: int) -> None:
    report = {
        "cells": [
            {"model": "excluded", "benchmark": f"benchmark-{index}", "eligible": False}
            for index in range(excluded_count)
        ]
    }
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(report))
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ValueError, match="no eligible model/benchmark cells"):
        export_suite_planning_inputs(source, output)

    assert list(output.iterdir()) == []


def test_mixed_eligibility_exports_only_validated_eligible_evidence(tmp_path: Path) -> None:
    report = _report()
    excluded = {"model": "excluded", "benchmark": "unscored", "eligible": False}
    report["cells"] = [excluded, *report["cells"], excluded]
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(report))

    records = export_suite_planning_inputs(source, tmp_path)

    assert [(record["model"], record["benchmark"]) for record in records] == [("model", "benchmark")]
    assert records[0]["baseline"]["scores"] == [10.0, 12.0, 14.0]
    assert records[0]["candidate"]["variance"] == 18.0
    assert json.loads((tmp_path / "cluster_numeric_evidence.json").read_text()) == records
    for level in ("benchmark", "suite"):
        manifest = yaml.safe_load((tmp_path / f"cluster_{level}_plan.yaml").read_text())
        assert [(cell["model"], cell["benchmark"]) for cell in manifest["cells"]] == [("model", "benchmark")]
        assert manifest["cells"][0]["baseline_std_points"] == 2.0


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

    assert list(tmp_path.glob("cluster_*")) == []


def _audit_snapshot(output: Path) -> dict[str, tuple[bytes, int]]:
    return {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}


@pytest.fixture
def completed_audit(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(_report()))
    output = tmp_path / "audit"
    run_validation(reference_root=None, output=output, suite_report=source, trials=256)
    (output / "user_notes.txt").write_text("Keep these unrelated notes.")
    return source, output


def _change_baseline_evidence(source: Path) -> None:
    report = _report()
    report["cells"][0]["baseline"]["repeat_scores"] = [20.0, 24.0, 28.0]
    report["cells"][0]["baseline"]["summary"]["sample_variance"] = 16.0
    source.write_text(json.dumps(report))


@pytest.mark.parametrize("failure", ["excluded", "simulation"])
def test_failed_rerun_preserves_previous_successful_audit(completed_audit: tuple[Path, Path], failure: str) -> None:
    source, output = completed_audit
    before = _audit_snapshot(output)
    if failure == "excluded":
        report = _report()
        report["cells"][0]["eligible"] = False
        source.write_text(json.dumps(report))
        trials, message = 256, "no eligible model/benchmark cells"
    else:
        _change_baseline_evidence(source)
        trials, message = 0, "trials"

    with pytest.raises(ValueError, match=message):
        run_validation(reference_root=None, output=output, suite_report=source, trials=trials)

    assert _audit_snapshot(output) == before


def test_successful_rerun_publishes_new_evidence_and_preserves_unrelated_files(
    completed_audit: tuple[Path, Path],
) -> None:
    source, output = completed_audit
    _change_baseline_evidence(source)

    receipt = run_validation(reference_root=None, output=output, suite_report=source, trials=256)

    assert (output / "user_notes.txt").read_text() == "Keep these unrelated notes."
    assert (output / "VALIDATION.md").is_file()
    saved_receipt = json.loads((output / "validation_receipt.json").read_text())
    assert saved_receipt["cluster_input"] == receipt["cluster_input"]
    assert saved_receipt["cluster_input"]["cells"][0]["baseline"]["std"] == 4.0
    evidence = json.loads((output / "cluster_numeric_evidence.json").read_text())
    assert evidence[0]["baseline"]["std"] == 4.0
    plan = yaml.safe_load((output / "cluster_suite_plan.yaml").read_text())
    assert plan["cells"][0]["baseline_std_points"] == 4.0


@pytest.mark.parametrize("completion_file", ["validation_receipt.json", "VALIDATION.md"])
def test_publication_failure_does_not_leave_passed_markers_beside_new_evidence(
    completed_audit: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, completion_file: str
) -> None:
    source, output = completed_audit
    _change_baseline_evidence(source)
    original_replace = Path.replace

    def reject_completion_publication(path: Path, target: Path) -> Path:
        if target == output / completion_file:
            raise OSError("Completion publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", reject_completion_publication)

    with pytest.raises(OSError, match="Completion publication failed"):
        run_validation(reference_root=None, output=output, suite_report=source, trials=256)

    assert not (output / "VALIDATION.md").exists()
    assert not (output / "validation_receipt.json").exists()
    assert (output / "user_notes.txt").read_text() == "Keep these unrelated notes."
    evidence = json.loads((output / "cluster_numeric_evidence.json").read_text())
    assert evidence[0]["baseline"]["std"] == 4.0


def test_rerun_replaces_numbered_benchmark_plans_when_suite_shrinks(tmp_path: Path) -> None:
    report = _report()
    second = _report()["cells"][0]
    second["benchmark"] = "another"
    report["cells"].append(second)
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(report))
    output = tmp_path / "audit"
    run_validation(reference_root=None, output=output, suite_report=source, trials=256)
    assert (output / "cluster_benchmark_1_plan.yaml").is_file()
    assert (output / "cluster_benchmark_2_plan.yaml").is_file()
    source.write_text(json.dumps(_report()))

    run_validation(reference_root=None, output=output, suite_report=source, trials=256)

    assert {path.name for path in output.glob("cluster_benchmark*plan.yaml")} == {"cluster_benchmark_plan.yaml"}
    plan = yaml.safe_load((output / "cluster_suite_plan.yaml").read_text())
    assert [cell["benchmark"] for cell in plan["cells"]] == ["benchmark"]


def test_rerun_removes_dropped_optional_artifacts_and_keeps_similarly_named_user_files(
    completed_audit: tuple[Path, Path],
) -> None:
    _, output = completed_audit
    references = ["reference_benchmark_parity.csv", "reference_suite_parity.csv", "shared_baseline_parity.csv"]
    for name in references:
        (output / name).write_text("previous,reference\n")
    user_files = [
        "cluster_notes_plan.yaml",
        "cluster_benchmark_notes_plan.yaml",
        "cluster_benchmark_1_plan.yaml.backup",
        "reference_notes.csv",
    ]
    for name in user_files:
        (output / name).write_text("Keep this user file.")

    receipt = run_validation(reference_root=None, output=output, suite_report=None, trials=256)

    assert "cluster_input" not in receipt
    assert all(not (output / name).exists() for name in references)
    assert all(
        not (output / name).exists()
        for name in ["cluster_suite_plan.yaml", "cluster_benchmark_plan.yaml", "cluster_numeric_evidence.json"]
    )
    assert all((output / name).read_text() == "Keep this user file." for name in user_files)
    assert (output / "user_notes.txt").read_text() == "Keep these unrelated notes."


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="Requires a non-root user and POSIX directory permissions.",
)
def test_audit_can_publish_into_existing_output_with_unwritable_parent(tmp_path: Path) -> None:
    parent = tmp_path / "readonly"
    output = parent / "audit"
    output.mkdir(parents=True)
    mode = parent.stat().st_mode & 0o777
    parent.chmod(0o555)
    try:
        run_validation(reference_root=None, output=output, suite_report=None, trials=256)
    finally:
        parent.chmod(mode)

    assert (output / "VALIDATION.md").is_file()
    assert (output / "validation_receipt.json").is_file()


def test_audit_records_gym_commit_when_called_outside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[3]
    expected = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    receipt = run_validation(reference_root=None, output=tmp_path / "audit", suite_report=None, trials=256)

    assert receipt["gym_commit"] == expected


@pytest.mark.parametrize("failure_name", ["validation_receipt.json", "reference_suite_parity.csv"])
def test_publication_cleanup_failure_invalidates_both_completion_markers(
    completed_audit: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, failure_name: str
) -> None:
    _, output = completed_audit
    (output / "reference_suite_parity.csv").write_text("previous,reference\n")
    original_unlink = Path.unlink
    failed = False

    def reject_first_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal failed
        if path == output / failure_name and not failed:
            failed = True
            raise OSError("Artifact cleanup failed")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", reject_first_cleanup)

    with pytest.raises(OSError, match="Artifact cleanup failed"):
        run_validation(reference_root=None, output=output, suite_report=None, trials=256)

    assert not (output / "VALIDATION.md").exists()
    assert not (output / "validation_receipt.json").exists()
    assert (output / "user_notes.txt").read_text() == "Keep these unrelated notes."
