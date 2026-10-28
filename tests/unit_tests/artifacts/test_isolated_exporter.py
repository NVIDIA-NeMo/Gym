# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.artifacts.exporters.isolated as isolated
from nemo_gym.artifacts.exporters.base import ExportState
from nemo_gym.artifacts.exporters.isolated import (
    ExportPreparationError,
    IsolatedExporter,
    ensure_exporter_environment,
    exporter_venv_path,
)
from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.config.config_types import RunExportConfig


needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required to build the exporter environment")

WORKER = """
from __future__ import annotations
import importlib.util
import os
import sys
from dataclasses import dataclass


@dataclass
class Upload:
    count: int


def run(payload):
    print("noise from the SDK")
    mode = payload["mode"]
    if mode == "paths":
        return {"detail": {"sys_path": sys.path}}
    if mode == "hang":
        import time

        time.sleep(60)
    if mode == "crash":
        os._exit(3)
    if mode == "exit_after_result":
        import atexit

        atexit.register(os._exit, 4)
        return {"detail": {"sent": True}}
    if mode == "dataclass":
        return {"detail": {"count": Upload(1).count}}
    if mode == "which_python":
        import subprocess

        return {"detail": {"prefix": subprocess.check_output(["python", "-c", "import sys; print(sys.prefix)"], text=True).strip()}}
    if mode == "raise":
        raise ValueError("backend refused the upload")
    if mode == "skip":
        return {"state": "skipped", "detail": {"reason": "duplicate"}}
    if mode == "fail":
        return {"state": "failed", "error": "quota exceeded", "detail": {"retry_after": 5}}
    return {
        "detail": {
            "echo": payload["echo"],
            "prefix": sys.prefix,
            "can_import_gym": importlib.util.find_spec("nemo_gym") is not None,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "cwd": os.getcwd(),
        }
    }
"""


class FakeExporter(IsolatedExporter):
    name: ClassVar[str] = "fake"
    config_model: ClassVar[type[RunExportConfig]] = RunExportConfig
    prepared_workdirs: ClassVar[list[Path]] = []
    mode: ClassVar[str] = "ok"
    refuse: ClassVar[bool] = False

    def build_payload(self, artifacts: RunArtifacts, workdir: Path) -> dict[str, Any]:
        if self.refuse:
            raise ExportPreparationError("the run has nothing to export")
        self.prepared_workdirs.append(workdir)
        (workdir / "staged.txt").write_text("staged")
        return {"mode": self.mode, "echo": str(artifacts.rollouts_jsonl_fpath)}


@pytest.fixture
def fake_exporter(tmp_path: Path, monkeypatch: MonkeyPatch) -> FakeExporter:
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    monkeypatch.setattr(FakeExporter, "worker_fpath", worker, raising=False)
    monkeypatch.setattr(FakeExporter, "prepared_workdirs", [])
    config = RunExportConfig(rollouts_jsonl_fpath=tmp_path / "run.jsonl")
    return FakeExporter(config, venv_root=tmp_path / "venvs")


def test_venv_path_is_keyed_by_the_requirements_not_their_order(tmp_path: Path) -> None:
    first = exporter_venv_path(tmp_path, "evalrdl", ("a>=1", "b"))

    assert exporter_venv_path(tmp_path, "evalrdl", ("b", "a>=1")) == first
    assert exporter_venv_path(tmp_path, "evalrdl", ("a>=2", "b")) != first
    assert first.parent.parent == tmp_path / "exporters" / "evalrdl"
    assert first.name == ".venv"


def test_environment_command_installs_the_requirements_with_the_environments_python(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    commands: list[str] = []

    def record(venv: Path, command: str, *, skip_if_ready: bool) -> int:
        commands.append(command)
        assert skip_if_ready
        return 0

    monkeypatch.setattr(isolated, "setup_environment", record)

    python = ensure_exporter_environment(tmp_path, "evalrdl", ("evalrdl>=1.2", "odd name; python_version>'3'"))
    ensure_exporter_environment(tmp_path, "plain", ())

    assert (
        python
        == exporter_venv_path(tmp_path, "evalrdl", ("evalrdl>=1.2", "odd name; python_version>'3'")) / "bin/python"
    )
    assert f"uv venv --allow-existing --python {sys.executable}" in commands[0]
    assert f"uv pip install --python {python} " in commands[0]
    assert "'evalrdl>=1.2'" in commands[0] and "'odd name; python_version>'\"'\"'3'\"'\"''" in commands[0]
    assert "uv pip install" not in commands[1]


@needs_uv
class TestRealEnvironment:
    def test_the_worker_runs_in_its_own_environment_without_gym(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        artifacts = RunArtifacts(tmp_path / "run.jsonl")

        result = fake_exporter.export(artifacts)

        assert result.state is ExportState.SUCCEEDED, result.error
        detail = result.detail
        assert detail["echo"] == str(artifacts.rollouts_jsonl_fpath)
        assert detail["prefix"] == str(exporter_venv_path(tmp_path / "venvs", "fake", ()).parent / ".venv")
        assert detail["prefix"] != sys.prefix
        assert detail["can_import_gym"] is False
        assert detail["pythonpath"] is None

    def test_the_harnesss_own_directory_is_not_on_the_workers_path(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", "paths")

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.SUCCEEDED, result.error
        assert str(Path(isolated.WORKER_HARNESS_FPATH).parent) not in result.detail["sys_path"]

    def test_a_worker_that_outlives_its_time_limit_is_killed_and_fails(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", "hang")
        monkeypatch.setattr(FakeExporter, "worker_timeout_s", lambda self, payload: 1.0)

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.FAILED
        assert result.error == "The fake exporter worker was killed after 1s."

    def test_a_ready_environment_is_reused(self, fake_exporter: FakeExporter, tmp_path: Path) -> None:
        artifacts = RunArtifacts(tmp_path / "run.jsonl")
        fake_exporter.export(artifacts)
        marker = exporter_venv_path(tmp_path / "venvs", "fake", ()) / ".nemo-gym-setup-complete"
        built_at = marker.stat().st_mtime_ns

        assert fake_exporter.export(artifacts).state is ExportState.SUCCEEDED
        assert marker.stat().st_mtime_ns == built_at

    def test_the_staging_directory_is_removed_afterwards(self, fake_exporter: FakeExporter, tmp_path: Path) -> None:
        fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        (workdir,) = fake_exporter.prepared_workdirs
        assert not workdir.exists()

    @pytest.mark.parametrize(
        ("mode", "state", "expected"),
        [
            ("skip", ExportState.SKIPPED, {"reason": "duplicate"}),
            ("fail", ExportState.FAILED, {"retry_after": 5}),
        ],
    )
    def test_the_workers_state_and_detail_come_back(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path, mode: str, state, expected
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", mode)

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is state
        assert result.detail == expected
        if state is ExportState.FAILED:
            assert (result.exit_code, result.error) == (1, "quota exceeded")

    def test_a_worker_exception_is_a_failed_result_naming_the_error(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", "raise")

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.FAILED
        assert result.error == "ValueError: backend refused the upload"

    def test_a_worker_that_dies_without_a_result_is_a_failed_result_with_its_exit_code(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", "crash")

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.FAILED
        assert "exited with code 3 without a result" in result.error

    def test_a_worker_that_exits_non_zero_after_its_result_is_a_failed_result(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", "exit_after_result")

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.FAILED
        assert "reported succeeded but then exited with code 4" in result.error

    def test_a_worker_module_may_use_dataclasses_with_postponed_annotations(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", "dataclass")

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.SUCCEEDED
        assert result.detail == {"count": 1}

    def test_a_bare_python_in_the_worker_is_the_exporters_not_gyms(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(FakeExporter, "mode", "which_python")

        result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.SUCCEEDED
        assert Path(result.detail["prefix"]) == exporter_venv_path(fake_exporter.venv_root, "fake", ()).resolve()

    def test_a_relative_venv_root_and_worker_path_are_anchored_to_the_callers_directory(
        self, fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(FakeExporter, "worker_fpath", Path("worker.py"))
        exporter = FakeExporter(RunExportConfig(rollouts_jsonl_fpath=tmp_path / "run.jsonl"), venv_root=Path("venvs"))

        result = exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

        assert result.state is ExportState.SUCCEEDED, result.error


def test_a_worker_that_cannot_be_launched_is_a_failed_result(
    fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(isolated, "setup_environment", lambda *args, **kwargs: 0)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError(11, "Resource temporarily unavailable")

    monkeypatch.setattr(isolated.subprocess, "run", refuse)

    result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

    assert result.state is ExportState.FAILED
    assert "Could not launch the fake exporter worker" in result.error


def test_a_run_that_cannot_be_prepared_fails_before_any_environment_is_built(
    fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(FakeExporter, "refuse", True)
    built = []
    monkeypatch.setattr(isolated, "setup_environment", lambda *args, **kwargs: built.append(args) or 0)

    result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

    assert result.state is ExportState.FAILED
    assert result.error == "the run has nothing to export"
    assert built == []


def test_a_failed_environment_build_is_a_failed_result(
    fake_exporter: FakeExporter, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(isolated, "setup_environment", lambda *args, **kwargs: 2)

    result = fake_exporter.export(RunArtifacts(tmp_path / "run.jsonl"))

    assert result.state is ExportState.FAILED
    assert "setup exited with 2" in result.error


def test_from_global_config_validates_the_config_and_reads_the_venv_root(tmp_path: Path) -> None:
    exporter = FakeExporter.from_global_config(
        OmegaConf.create({"rollouts_jsonl_fpath": str(tmp_path / "run.jsonl"), "uv_venv_dir": str(tmp_path / "v")})
    )

    assert exporter.venv_root == tmp_path / "v"
    with pytest.raises(Exception, match="rollouts_jsonl_fpath"):
        FakeExporter.from_global_config(OmegaConf.create({"uv_venv_dir": str(tmp_path)}))
