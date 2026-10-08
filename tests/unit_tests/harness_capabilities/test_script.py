# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise script command routing, retained payload joins, and report publication."""

import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from nemo_gym.harness_capabilities import cli
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/inspect_harness_conformance.py"


@pytest.fixture
def record():
    return evidence_record()


@pytest.mark.parametrize(
    "relative_path",
    ["rollouts.jsonl", "artifacts/rollouts.jsonl", "evaluator_rollouts.jsonl", "artifacts/evaluator_rollouts.jsonl"],
)
def test_script_discovers_bundle(record, tmp_path, monkeypatch, relative_path):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")
    quality = tmp_path / "quality_summary.json"
    quality.write_text('{"health": "unchanged"}\n')
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--bundle", str(tmp_path), "--output", str(tmp_path / "reports")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exit_info.value.code == 0
    (summary_path,) = (tmp_path / "reports").glob("*/evidence_summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["verdict"] == "fulfilled"
    assert summary["sources"].keys() == {str(path.resolve())}
    assert quality.read_text() == '{"health": "unchanged"}\n'


@pytest.mark.parametrize(
    "profile,code", [("gym-artifacts-p1/v1", 2), ("gym-artifacts-all/v1", 2), ("onboarding-p0/v1", 2)]
)
def test_script_gate_exit_codes(record, tmp_path, monkeypatch, profile, code):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--bundle",
            str(path),
            "--output",
            str(tmp_path / "reports"),
            "--profile",
            profile,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exit_info.value.code == code
    if code == 2:
        assert not (tmp_path / "reports").exists()


def test_rejects_hydra_overrides_before_writing(record, tmp_path, monkeypatch):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--bundle", str(path), "--output", str(tmp_path / "reports"), "+model=other"],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exit_info.value.code == 2
    assert not (tmp_path / "reports").exists()


def test_ambiguous_directory_requires_explicit_file(record, tmp_path):
    for name in ("rollouts.jsonl", "evaluator_rollouts.jsonl"):
        (tmp_path / name).write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        cli.inspect_bundle(tmp_path, output=tmp_path / "reports", profile="gym-p0/v1")
    _, summary = cli.inspect_bundle(tmp_path / "rollouts.jsonl", output=tmp_path / "reports", profile="gym-p0/v1")
    assert summary["verdict"] == "fulfilled"


@pytest.mark.parametrize("sidecar_state", ["complete", "missing", "incomplete", "extra_call"])
def test_sidecars_validate_but_do_not_repair_missing_jsonl_payloads(record, tmp_path, monkeypatch, sidecar_state):
    full = deepcopy(record)
    capture_dir = tmp_path / "model-calls"
    capture_dir.mkdir()
    capture_file = capture_dir / "0-0.capture.jsonl"
    calls = full["ng_model_call_capture"]["calls"]
    if sidecar_state == "extra_call":
        calls.append({**calls[0], "model_call_id": "unexpected-attempt"})
    if sidecar_state != "missing":
        capture_file.write_text("".join(json.dumps(call) + "\n" for call in calls))
    if sidecar_state == "incomplete":
        (capture_dir / "0-0.capture.incomplete").touch()
    for call in record["ng_trajectory"]["model_calls"]:
        call["request"] = call["response"] = None
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--bundle",
            str(path),
            "--output",
            str(tmp_path / "reports"),
            "--capture-dir",
            str(capture_dir),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exit_info.value.code == 1  # Sidecars cannot repair the delivered JSONL surface.
    (summary_file,) = (tmp_path / "reports").glob("*/evidence_summary.json")
    summary = json.loads(summary_file.read_text())
    if sidecar_state != "missing":
        assert str(capture_file.resolve()) in summary["sources"]


def test_changing_input_publishes_no_report(record, tmp_path, monkeypatch):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    original_rows = cli.json_rows

    def mutate_after_read(path):
        yield from original_rows(path)
        path.write_text("{}\n")

    monkeypatch.setattr(cli, "json_rows", mutate_after_read)
    with pytest.raises(ValueError, match="source changed"):
        cli.inspect_bundle(path, output=tmp_path / "reports", profile="gym-p0/v1")
    assert list((tmp_path / "reports").iterdir()) == []


def test_reports_do_not_include_payload_values(record, tmp_path, capsys):
    secret = "PRIVATE-PAYLOAD-MARKER"
    call = record["ng_trajectory"]["model_calls"][0]
    call["request"]["input"] = secret
    call["token_stats"]["total_tokens"] = secret
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert cli.run_inspection(bundle=path, output=tmp_path / "reports") == 1
    for file in (tmp_path / "reports").glob("*/*"):
        assert secret not in file.read_text()
    path.write_text('{"' + secret + '":')
    assert cli.run_inspection(bundle=path, output=tmp_path / "reports") == 2
    assert secret not in capsys.readouterr().out


def test_script_matrix_routes_and_returns_failed_gate(record, tmp_path, monkeypatch):
    passing = tmp_path / "passing.jsonl"
    failing = tmp_path / "failing.jsonl"
    passing.write_text(json.dumps(record) + "\n")
    record["ng_trajectory"]["model_calls"][0]["request"] = None
    failing.write_text(json.dumps(record) + "\n")
    output = tmp_path / "reports"
    args = [
        str(SCRIPT),
        "matrix",
        "--output",
        str(output),
        "--harness",
        f"complete={passing}",
        "--harness",
        f"missing-payload={failing}",
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert error.value.code == 1
    (matrix,) = output.glob("*/harness_evidence.json")
    rows = json.loads(matrix.read_text())["harnesses"]
    assert rows["complete"]["verdict"] == "fulfilled"
    assert rows["missing-payload"]["evidence"]["TE-7"]["verdict"] == "not_fulfilled"
    table = matrix.with_suffix(".md").read_text()
    assert "| complete | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |" in table
    assert "| missing-payload | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL | PASS | PASS | FAIL |" in table
    # Replaying the same matrix is idempotent.
    assert cli.inspect_matrix({"complete": passing, "missing-payload": failing}, output=output)[0] == matrix.parent


def test_matrix_error_cannot_publish_partial_matrix(record, tmp_path):
    bundle = tmp_path / "complete.jsonl"
    bundle.write_text(json.dumps(record) + "\n")
    with pytest.raises(OSError):
        cli.inspect_matrix({"complete": bundle, "missing": tmp_path / "missing"}, output=tmp_path)
    assert not list(tmp_path.glob("*/harness_evidence.json"))
