# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native command routing, retained payload joins, and report publication."""

import json
import sys
from pathlib import Path

import pytest

from nemo_gym.cli.main import main as gym_main
from nemo_gym.harness_capabilities import cli
from nemo_gym.harness_capabilities.reader import hydrate_record


@pytest.fixture
def record():
    return json.loads((Path(__file__).parent / "fixtures/opencode.json").read_text())


@pytest.mark.parametrize(
    "relative_path",
    ["rollouts.jsonl", "artifacts/rollouts.jsonl", "evaluator_rollouts.jsonl", "artifacts/evaluator_rollouts.jsonl"],
)
def test_native_cli_discovers_bundle(record, tmp_path, monkeypatch, relative_path):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")
    quality = tmp_path / "quality_summary.json"
    quality.write_text('{"health": "unchanged"}\n')
    monkeypatch.setattr(
        sys, "argv", ["gym", "eval", "conformance", "--bundle", str(tmp_path), "--output", str(tmp_path / "reports")]
    )
    with pytest.raises(SystemExit) as exit_info:
        gym_main()
    assert exit_info.value.code == 0
    (summary_path,) = (tmp_path / "reports").glob("*/capability_summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["verdict"] == "fulfilled"
    assert summary["sources"].keys() == {str(path.resolve())}
    assert quality.read_text() == '{"health": "unchanged"}\n'


@pytest.mark.parametrize(
    "profile,code", [("gym-artifacts-p1/v1", 1), ("gym-artifacts-all/v1", 1), ("onboarding-p0/v1", 2)]
)
def test_native_cli_gate_exit_codes(record, tmp_path, monkeypatch, profile, code):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            "eval",
            "conformance",
            "--bundle",
            str(path),
            "--output",
            str(tmp_path / "reports"),
            "--profile",
            profile,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        gym_main()
    assert exit_info.value.code == code
    if code == 2:
        assert not (tmp_path / "reports").exists()


def test_rejects_hydra_overrides_before_writing(record, tmp_path, monkeypatch):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gym", "eval", "conformance", "--bundle", str(path), "--output", str(tmp_path / "reports"), "+model=other"],
    )
    with pytest.raises(SystemExit) as exit_info:
        gym_main()
    assert exit_info.value.code == 2
    assert not (tmp_path / "reports").exists()


def test_ambiguous_directory_requires_explicit_file(record, tmp_path):
    for name in ("rollouts.jsonl", "evaluator_rollouts.jsonl"):
        (tmp_path / name).write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        cli.inspect_bundle(tmp_path, output=tmp_path / "reports", profile="gym-artifacts-p0/v1")
    _, summary = cli.inspect_bundle(
        tmp_path / "rollouts.jsonl", output=tmp_path / "reports", profile="gym-artifacts-p0/v1"
    )
    assert summary["verdict"] == "fulfilled"


@pytest.mark.parametrize("sidecar_state", ["complete", "missing", "incomplete", "extra_call"])
def test_native_cli_recovers_payloads_and_validates_sidecars(record, tmp_path, monkeypatch, sidecar_state):
    full = hydrate_record(record)
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
            "gym",
            "eval",
            "conformance",
            "--bundle",
            str(path),
            "--output",
            str(tmp_path / "reports"),
            "--capture-dir",
            str(capture_dir),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        gym_main()
    assert exit_info.value.code == (0 if sidecar_state == "complete" else 1)
    (summary_file,) = (tmp_path / "reports").glob("*/capability_summary.json")
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
        cli.inspect_bundle(path, output=tmp_path / "reports", profile="gym-artifacts-p0/v1")
    assert list((tmp_path / "reports").iterdir()) == []


def test_reports_do_not_include_payload_values(record, tmp_path, capsys):
    secret = "PRIVATE-PAYLOAD-MARKER"
    call = record["ng_trajectory"]["model_calls"][0]
    call["request"]["input"] = secret
    call["response"]["usage"]["total_tokens"] = secret
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert cli.run_inspection(bundle=path, output=tmp_path / "reports") == 1
    for file in (tmp_path / "reports").glob("*/*"):
        assert secret not in file.read_text()
    path.write_text('{"' + secret + '":')
    assert cli.run_inspection(bundle=path, output=tmp_path / "reports") == 2
    assert secret not in capsys.readouterr().out
