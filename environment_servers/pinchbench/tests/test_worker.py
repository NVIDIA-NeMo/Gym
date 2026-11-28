# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run these contracts in Dockerfile.verifier, where the pinned upstream task definitions live."""

import json
import sys
from pathlib import Path

import pytest

from environment_servers.pinchbench import worker


pytestmark = pytest.mark.skipif(not (worker.SKILL / "tasks").is_dir(), reason="requires the PinchBench verifier image")


def test_hybrid_task_stages_public_fixture_and_uses_upstream_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(worker, "STATE", tmp_path / "state")
    monkeypatch.setattr(worker, "WORKSPACE", tmp_path / "workspace")
    worker.WORKSPACE.mkdir()

    def run(action: str) -> None:
        monkeypatch.setattr(sys, "argv", ["worker", action, "task_csv_cities_filter"])
        worker.main()

    run("seed")
    plan = json.loads(capsys.readouterr().out)
    assert plan["grading_type"] == "hybrid"
    fixtures = worker.STATE / "fixtures"
    assert [f.name for f in fixtures.iterdir()] == ["us_cities_top1000.csv"]
    assert (fixtures / "us_cities_top1000.csv").read_bytes() == (
        worker.SKILL / "assets/csvs/us_cities_top1000.csv"
    ).read_bytes()

    import lib_grading

    (worker.STATE / "transcript.json").write_text("[]")
    monkeypatch.setenv("JUDGE_MODEL", "test-model")
    monkeypatch.setattr(
        lib_grading,
        "call_judge_api",
        lambda **_: {"status": "success", "text": '{"total":1,"scores":{"analysis":1},"notes":"recorded"}'},
    )
    run("judge")
    monkeypatch.delenv("JUDGE_MODEL")
    run("automated")
    grade = json.loads((worker.STATE / "grade.json").read_text())
    assert grade["score"] == pytest.approx(0.4)
    assert grade["grading_type"] == "hybrid"
    assert grade["breakdown"]["llm_judge.analysis"] == 1
    assert grade["breakdown"]["automated.report_created"] == 0


def test_failed_judge_remains_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "STATE", tmp_path)
    monkeypatch.setattr(worker, "WORKSPACE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["worker", "judge", "task_csv_cities_filter"])
    monkeypatch.syspath_prepend(str(worker.SKILL / "scripts"))
    import lib_grading

    monkeypatch.setenv("JUDGE_MODEL", "malformed-judge")
    (tmp_path / "transcript.json").write_text("[]")
    monkeypatch.setattr(lib_grading, "call_judge_api", lambda **_: {"status": "success", "text": "unparseable"})
    with pytest.raises(ValueError, match="no parseable score"):
        worker.main()
    assert not (tmp_path / "grade.json").exists()
