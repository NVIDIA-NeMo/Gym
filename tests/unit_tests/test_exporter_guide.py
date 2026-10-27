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
import errno
import importlib.util
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.exporters.isolated as isolated
from nemo_gym.exporters.base import ExportState
from nemo_gym.run_artifacts import RunArtifacts


REPO_ROOT = Path(__file__).parents[2]
EXAMPLES_DIR = Path(__file__).parent / "exporter_guide"
GUIDE_FPATH = REPO_ROOT / "fern" / "versions" / "latest" / "pages" / "contribute" / "exporters.mdx"


def _load_example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"exporter_guide_{name}", EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(tmp_path: Path) -> RunArtifacts:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text('{"reward": 1.0}\n{"reward": 0.0}\n')
    artifacts = RunArtifacts(rollouts)
    artifacts.aggregate_metrics_json_fpath.write_text('[{"mean/reward": 0.5}]')
    artifacts.resolved_config_yaml_fpath.write_text("model: m\n")
    stamp = rollouts.stat().st_mtime + 10
    os.utime(artifacts.aggregate_metrics_json_fpath, (stamp, stamp))
    return artifacts


def _settings(artifacts: RunArtifacts, tmp_path: Path, **settings: Any) -> Any:
    return OmegaConf.create(
        {
            "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath),
            "uv_venv_dir": str(tmp_path / "venvs"),
            **settings,
        }
    )


class TestSummaryExporter:
    def test_writes_the_summary_and_reports_it(self, tmp_path: Path) -> None:
        summary = _load_example("summary_exporter")
        artifacts = _make_run(tmp_path)
        exporter = summary.SummaryExporter.from_global_config(
            _settings(artifacts, tmp_path, destination_dirpath=str(tmp_path / "out"))
        )

        result = exporter.export(artifacts)

        destination = tmp_path / "out" / "rollouts_summary.json"
        assert result.state is ExportState.SUCCEEDED
        assert result.detail == {"destination": str(destination), "rollouts": 2}
        assert json.loads(destination.read_text()) == {"rollouts": 2, "aggregate_metrics": [{"mean/reward": 0.5}]}

    def test_a_second_export_is_skipped(self, tmp_path: Path) -> None:
        summary = _load_example("summary_exporter")
        artifacts = _make_run(tmp_path)
        exporter = summary.SummaryExporter.from_global_config(
            _settings(artifacts, tmp_path, destination_dirpath=str(tmp_path / "out"))
        )
        exporter.export(artifacts)

        assert exporter.export(artifacts).state is ExportState.SKIPPED

    def test_a_run_without_aggregate_metrics_fails_before_writing(self, tmp_path: Path) -> None:
        summary = _load_example("summary_exporter")
        artifacts = _make_run(tmp_path)
        artifacts.aggregate_metrics_json_fpath.unlink()
        exporter = summary.SummaryExporter.from_global_config(
            _settings(artifacts, tmp_path, destination_dirpath=str(tmp_path / "out"))
        )

        result = exporter.export(artifacts)

        assert result.state is ExportState.FAILED
        assert "Aggregate metrics not found" in result.error
        assert not (tmp_path / "out").exists()

    def test_an_unwritable_destination_is_a_failed_result(self, tmp_path: Path) -> None:
        summary = _load_example("summary_exporter")
        artifacts = _make_run(tmp_path)
        (tmp_path / "blocked").write_text("a file where a directory is needed")
        exporter = summary.SummaryExporter.from_global_config(
            _settings(artifacts, tmp_path, destination_dirpath=str(tmp_path / "blocked" / "out"))
        )

        assert exporter.export(artifacts).state is ExportState.FAILED

    def test_the_destination_setting_is_required(self, tmp_path: Path) -> None:
        summary = _load_example("summary_exporter")
        artifacts = _make_run(tmp_path)

        with pytest.raises(Exception, match="destination_dirpath"):
            summary.SummaryExporter.from_global_config(_settings(artifacts, tmp_path))


# A stand-in for `acme_sdk`, which the worker imports in its own environment.
FAKE_ACME_SDK = """
import os
from pathlib import Path


class UploadRefused(Exception):
    pass


class _Upload:
    id = "up-1"

    def __init__(self, duplicate):
        self.duplicate = duplicate


class Client:
    def upload(self, path, *, project):
        Path(os.environ["FAKE_ACME_LOG"]).write_text(f"{project}:{sorted(p.name for p in Path(path).iterdir())}")
        mode = os.environ["FAKE_ACME_MODE"]
        if mode == "refused":
            raise UploadRefused("project is archived")
        return _Upload(duplicate=mode == "duplicate")
"""


@pytest.fixture
def fake_acme_sdk(tmp_path: Path, monkeypatch: MonkeyPatch) -> dict[str, Any]:
    sdk_root = tmp_path / "fake_sdk"
    sdk_root.mkdir()
    (sdk_root / "acme_sdk.py").write_text(textwrap.dedent(FAKE_ACME_SDK))
    log = tmp_path / "acme_call.txt"
    requested: list[tuple[str, ...]] = []

    def fake_environment(venv_root: Path, name: str, requirements: tuple[str, ...]) -> Path:
        requested.append(requirements)
        return Path(sys.executable)

    monkeypatch.setattr(isolated, "ensure_exporter_environment", fake_environment)
    monkeypatch.setattr(
        isolated, "_worker_environment", lambda: {**os.environ, "PYTHONPATH": str(sdk_root), "FAKE_ACME_LOG": str(log)}
    )
    monkeypatch.setenv("FAKE_ACME_MODE", "uploaded")
    return {"log": log, "requested": requested}


class TestAcmeExporter:
    @staticmethod
    def _export(tmp_path: Path, **settings: Any) -> Any:
        acme = _load_example("acme_exporter")
        artifacts = _make_run(tmp_path)
        exporter = acme.AcmeExporter.from_global_config(_settings(artifacts, tmp_path, project="demo", **settings))
        return exporter.export(artifacts)

    def test_the_worker_uploads_the_staged_metrics_from_its_own_process(
        self, tmp_path: Path, fake_acme_sdk: dict[str, Any]
    ) -> None:
        result = self._export(tmp_path)

        assert result.state is ExportState.SUCCEEDED, result.error
        assert result.detail == {"upload_id": "up-1"}
        assert fake_acme_sdk["log"].read_text() == "demo:['rollouts_aggregate_metrics.json']"
        assert fake_acme_sdk["requested"] == [("acme-sdk",)]

    def test_the_requirement_is_configurable(self, tmp_path: Path, fake_acme_sdk: dict[str, Any]) -> None:
        self._export(tmp_path, requirement="acme-sdk==2.0")

        assert fake_acme_sdk["requested"] == [("acme-sdk==2.0",)]

    def test_a_duplicate_upload_is_skipped(
        self, tmp_path: Path, fake_acme_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ACME_MODE", "duplicate")

        assert self._export(tmp_path).state is ExportState.SKIPPED

    def test_a_refusal_fails_with_its_own_exit_code(
        self, tmp_path: Path, fake_acme_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ACME_MODE", "refused")

        result = self._export(tmp_path)

        assert (result.state, result.exit_code) == (ExportState.FAILED, 4)
        assert result.error == "Acme refused the upload: project is archived"

    def test_an_exporter_that_does_not_read_the_rollouts_does_not_need_their_file(
        self, tmp_path: Path, fake_acme_sdk: dict[str, Any]
    ) -> None:
        acme = _load_example("acme_exporter")
        artifacts = _make_run(tmp_path)
        artifacts.rollouts_jsonl_fpath.unlink()
        exporter = acme.AcmeExporter.from_global_config(_settings(artifacts, tmp_path, project="demo"))

        assert exporter.export(artifacts).state is ExportState.SUCCEEDED

    def test_a_staging_copy_that_fails_is_a_failed_result_before_an_environment_is_built(
        self, tmp_path: Path, fake_acme_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        acme = _load_example("acme_exporter")
        artifacts = _make_run(tmp_path)
        exporter = acme.AcmeExporter.from_global_config(_settings(artifacts, tmp_path, project="demo"))

        def disk_full(source: Path, destination: Path) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(acme.shutil, "copy2", disk_full)

        result = exporter.export(artifacts)

        assert result.state is ExportState.FAILED
        assert "Could not stage the aggregate metrics: [Errno 28] No space left on device" in result.error
        assert fake_acme_sdk["requested"] == []

    def test_a_run_that_cannot_be_exported_fails_before_an_environment_is_built(
        self, tmp_path: Path, fake_acme_sdk: dict[str, Any]
    ) -> None:
        acme = _load_example("acme_exporter")
        artifacts = _make_run(tmp_path)
        artifacts.resolved_config_yaml_fpath.unlink()
        exporter = acme.AcmeExporter.from_global_config(_settings(artifacts, tmp_path, project="demo"))

        result = exporter.export(artifacts)

        assert result.state is ExportState.FAILED
        assert "Saved run config not found" in result.error
        assert fake_acme_sdk["requested"] == []


def _without_license(source: str) -> str:
    lines = source.splitlines()
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).rstrip()


def test_the_code_blocks_of_the_guide_are_the_example_files() -> None:
    page = GUIDE_FPATH.read_text()
    blocks = dict(re.findall(r'```python title="([\w.]+)"\n(.*?)\n```', page, flags=re.DOTALL))

    assert set(blocks) == {"summary_exporter.py", "acme_exporter.py", "acme_worker.py"}
    for name, block in blocks.items():
        assert block.rstrip() == _without_license((EXAMPLES_DIR / name).read_text()), name
