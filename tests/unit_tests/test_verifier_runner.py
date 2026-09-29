# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.environment import _verifier_runner
from nemo_gym.environment.onboarding import VerifierReport, VerifierRunSpec


def _request(path: Path) -> VerifierRunSpec:
    spec = VerifierRunSpec(
        name="sample",
        kind="environment",
        resources_server="sample",
        manifest_path="/work/environments/sample/manifest.yaml",
        app_path="/work/resources_servers/sample/app.py",
        component_root="/work",
        server_dir="/work/resources_servers/sample",
        bundled_component=False,
        reward_range=(0.0, 1.0),
        higher_is_better=True,
        determinism="seeded",
    )
    path.write_text(json.dumps({"spec": spec.to_dict(), "update_expected": True}), encoding="utf-8")
    return spec


def test_writes_success_report_without_using_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    expected_spec = _request(request_path)
    report = VerifierReport(
        name="sample",
        kind="environment",
        resources_server="sample",
        manifest_path=expected_spec.manifest_path,
        fixture_path="/work/resources_servers/sample/tests/verifier_cases.jsonl",
        cases=(),
    )

    async def exercise(spec: VerifierRunSpec, *, update_expected: bool = False) -> VerifierReport:
        assert spec == expected_spec
        assert update_expected is True
        return report

    monkeypatch.setattr(_verifier_runner, "exercise_verifier_run", exercise)

    assert _verifier_runner.main(["--request", str(request_path), "--result", str(result_path)]) == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "report": {**report.to_dict(), "cases": []},
    }
    assert capsys.readouterr().out == ""


def test_writes_error_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    _request(request_path)

    async def exercise(spec: VerifierRunSpec, *, update_expected: bool = False) -> VerifierReport:
        raise RuntimeError("fixture failed")

    monkeypatch.setattr(_verifier_runner, "exercise_verifier_run", exercise)

    assert _verifier_runner.main(["--request", str(request_path), "--result", str(result_path)]) == 1
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"ok": False, "error": "fixture failed"}
    assert capsys.readouterr().out == ""
