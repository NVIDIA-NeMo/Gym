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
import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf
from pytest import MonkeyPatch

import nemo_gym.exporters.runner as runner
from nemo_gym.config_types import ConfigError, MLflowExportConfig, RunExportConfig, WandbExportConfig
from nemo_gym.exporters.base import ExportResult, ExportState
from nemo_gym.run_artifacts import RunArtifacts


class FakeExporter:
    instances: list["FakeExporter"] = []
    outcome: Any = ExportResult.succeeded(run_id="r-1")
    seen_during_export: Any = None

    def __init__(self, global_config_dict: DictConfig) -> None:
        self.result_fpath = (
            Path(global_config_dict["result_json_fpath"]) if "result_json_fpath" in global_config_dict else None
        )
        FakeExporter.instances.append(self)

    @classmethod
    def from_global_config(cls, global_config_dict: DictConfig) -> "FakeExporter":
        return cls(global_config_dict)

    def export(self, artifacts: RunArtifacts) -> ExportResult:
        if self.result_fpath is not None:
            FakeExporter.seen_during_export = json.loads(self.result_fpath.read_text())
        if isinstance(FakeExporter.outcome, Exception):
            raise FakeExporter.outcome
        return FakeExporter.outcome


@pytest.fixture(autouse=True)
def fake_exporter(monkeypatch: MonkeyPatch) -> None:
    FakeExporter.instances = []
    FakeExporter.outcome = ExportResult.succeeded(run_id="r-1")
    FakeExporter.seen_during_export = None
    monkeypatch.setattr(runner, "load_run_exporter_class", lambda name: FakeExporter)


def _config(tmp_path: Path, *, with_result_file: bool = True) -> DictConfig:
    settings: dict[str, Any] = {"rollouts_jsonl_fpath": str(tmp_path / "rollouts.jsonl")}
    if with_result_file:
        settings["result_json_fpath"] = str(tmp_path / "exports" / "fake.result.json")
    return OmegaConf.create(settings)


def _result(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "exports" / "fake.result.json").read_text())


def test_the_result_reads_running_during_the_export_and_the_outcome_afterwards(tmp_path: Path) -> None:
    result = runner.run_export("fake", _config(tmp_path))

    assert result.state is ExportState.SUCCEEDED
    assert FakeExporter.seen_during_export == {"state": "running"}
    assert _result(tmp_path) == {"state": "succeeded", "exit_code": 0, "error": None, "detail": {"run_id": "r-1"}}


def test_a_failed_export_is_recorded_with_its_exit_code_and_error(tmp_path: Path) -> None:
    FakeExporter.outcome = ExportResult.failed("quota exceeded", exit_code=4, stage="decode")

    result = runner.run_export("fake", _config(tmp_path))

    assert result.state is ExportState.FAILED
    assert _result(tmp_path) == {
        "state": "failed",
        "exit_code": 4,
        "error": "quota exceeded",
        "detail": {"stage": "decode"},
    }


@pytest.mark.parametrize("previous", ["succeeded", "skipped"])
def test_an_export_that_already_succeeded_is_not_repeated_and_its_result_is_kept(
    tmp_path: Path, previous: str
) -> None:
    result_fpath = tmp_path / "exports" / "fake.result.json"
    result_fpath.parent.mkdir()
    result_fpath.write_text(json.dumps({"state": previous, "detail": {"run_id": "earlier"}}))

    result = runner.run_export("fake", _config(tmp_path))

    assert result.state is ExportState.SKIPPED
    assert result.detail == {"reason": "a previous attempt already exported this run", "previous_state": previous}
    assert FakeExporter.instances == []
    assert json.loads(result_fpath.read_text())["detail"] == {"run_id": "earlier"}


@pytest.mark.parametrize("previous", ['{"state": "failed"}', '{"state": "running"}', "not json", "[]"])
def test_a_failed_interrupted_or_unreadable_earlier_result_is_retried(tmp_path: Path, previous: str) -> None:
    result_fpath = tmp_path / "exports" / "fake.result.json"
    result_fpath.parent.mkdir()
    result_fpath.write_text(previous)

    result = runner.run_export("fake", _config(tmp_path))

    assert result.state is ExportState.SUCCEEDED
    assert _result(tmp_path)["state"] == "succeeded"


def test_a_crash_inside_the_export_is_recorded_as_failed_and_still_raised(tmp_path: Path) -> None:
    FakeExporter.outcome = RuntimeError("backend exploded")

    with pytest.raises(RuntimeError, match="backend exploded"):
        runner.run_export("fake", _config(tmp_path))

    assert _result(tmp_path)["state"] == "failed"
    assert _result(tmp_path)["error"] == "RuntimeError: backend exploded"


def test_a_bad_exporter_is_recorded_as_failed_and_still_raised(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    def unknown(name: str) -> type:
        raise ConfigError(f"Unknown exporter {name!r}.")

    monkeypatch.setattr(runner, "load_run_exporter_class", unknown)

    with pytest.raises(ConfigError, match="Unknown exporter 'nope'"):
        runner.run_export("nope", _config(tmp_path))

    assert _result(tmp_path)["state"] == "failed"


def test_without_a_result_file_nothing_is_written_and_nothing_is_skipped(tmp_path: Path) -> None:
    result = runner.run_export("fake", _config(tmp_path, with_result_file=False))

    assert result.state is ExportState.SUCCEEDED
    assert not (tmp_path / "exports").exists()


def test_the_result_is_written_atomically_leaving_no_temporary_file(tmp_path: Path) -> None:
    runner.run_export("fake", _config(tmp_path))

    assert [p.name for p in (tmp_path / "exports").iterdir()] == ["fake.result.json"]


def test_the_rollouts_path_is_required(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="rollouts_jsonl_fpath"):
        runner.run_export("fake", OmegaConf.create({"result_json_fpath": str(tmp_path / "r.json")}))


def test_a_missing_rollouts_path_is_recorded_as_a_failed_result_and_still_raised(tmp_path: Path) -> None:
    result_fpath = tmp_path / "exports" / "fake.result.json"

    with pytest.raises(Exception, match="rollouts_jsonl_fpath"):
        runner.run_export("fake", OmegaConf.create({"result_json_fpath": str(result_fpath)}))

    recorded = json.loads(result_fpath.read_text())
    assert recorded["state"] == "failed"
    assert recorded["exit_code"] != 0
    assert "rollouts_jsonl_fpath" in recorded["error"]
    assert FakeExporter.instances == []


def test_a_succeeded_result_is_kept_even_when_the_settings_are_incomplete(tmp_path: Path) -> None:
    result_fpath = tmp_path / "exports" / "fake.result.json"
    result_fpath.parent.mkdir()
    result_fpath.write_text(json.dumps({"state": "succeeded"}))

    result = runner.run_export("fake", OmegaConf.create({"result_json_fpath": str(result_fpath)}))

    assert result.state is ExportState.SKIPPED
    assert json.loads(result_fpath.read_text()) == {"state": "succeeded"}
    assert FakeExporter.instances == []


@pytest.mark.parametrize("model", [RunExportConfig, MLflowExportConfig, WandbExportConfig])
def test_every_exporters_settings_accept_a_result_file(model: type[RunExportConfig]) -> None:
    assert "result_json_fpath" in model.model_fields
