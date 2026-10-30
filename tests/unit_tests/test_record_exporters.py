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
import os
from pathlib import Path
from typing import Any, ClassVar, Optional
from unittest.mock import MagicMock

import orjson
import pytest
import yaml
from omegaconf import DictConfig, OmegaConf
from pytest import MonkeyPatch

import nemo_gym.exporters as exporters_module
import nemo_gym.exporters.mlflow as mlflow_module
import nemo_gym.exporters.wandb as wandb_module
from nemo_gym.config.config_types import ConfigError
from nemo_gym.exporters import load_run_exporter_class
from nemo_gym.exporters.base import BaseExporter, ExportState, _write_marker_atomically
from nemo_gym.exporters.metrics import flatten_aggregate_metrics
from nemo_gym.exporters.mlflow import MLflowExporter
from nemo_gym.exporters.wandb import WandbExporter
from nemo_gym.run_artifacts import RunArtifacts


AGGREGATE_ENTRIES = [
    {
        "agent_ref": {"name": "simple_agent"},
        "agent_metrics": {"mean/reward": 0.5, "note": "ok", "nested": {"skipped": 1}},
        "key_metrics": {"pass@1": 0.25},
    },
    {"agent_ref": {"name": "other_agent"}, "agent_metrics": {"mean/reward": 1.0}, "key_metrics": {}},
]


class RecordingBackend(BaseExporter):
    name: ClassVar[str] = "recording"
    fail_logging: ClassVar[bool] = False

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.events: list[str] = []
        self.configs: list[DictConfig] = []
        self.metrics: list[dict[str, Any]] = []
        self.rollouts: list[list[dict[str, Any]]] = []
        self.marker_during_logging: Optional[bool] = None

    def setup(self) -> None:
        self.events.append("setup")
        self._record_run_id("backend-run-7")

    def teardown(self) -> None:
        self.events.append("teardown")

    def _log_config(self, config_dict: DictConfig) -> None:
        self.marker_during_logging = self._exported_artifacts.export_done_fpath(self.name).exists()
        self.configs.append(config_dict)

    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self.fail_logging:
            raise RuntimeError("backend rejected the metrics")
        self.metrics.append(metrics)

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        self.rollouts.append(rollouts)


def _make_run(tmp_path: Path, *, saved_config: Optional[dict] = None) -> RunArtifacts:
    rollouts = tmp_path / "run.jsonl"
    rows = [
        {"_ng_task_index": 0, "reward": 1.0, "ng_trajectory": {"big": "x"}, "ng_model_call_capture": {"calls": []}},
        {"_ng_task_index": 1, "reward": 0.0},
        {"_ng_task_index": 2, "_ng_dispatch_drained": True},
    ]
    rollouts.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
    artifacts = RunArtifacts(rollouts)
    artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(AGGREGATE_ENTRIES))
    artifacts.resolved_config_yaml_fpath.write_text(
        yaml.safe_dump(saved_config if saved_config is not None else {"policy_api_key": "****", "model": "m"})
    )
    stamp = rollouts.stat().st_mtime + 10
    os.utime(artifacts.aggregate_metrics_json_fpath, (stamp, stamp))
    return artifacts


def _backend(artifacts: RunArtifacts, **overrides: Any) -> RecordingBackend:
    return RecordingBackend(
        OmegaConf.create({"rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath), **overrides})
    )


def test_flatten_aggregate_metrics_keeps_primitives_under_agent_prefixes() -> None:
    assert flatten_aggregate_metrics(AGGREGATE_ENTRIES) == {
        "simple_agent/mean/reward": 0.5,
        "simple_agent/note": "ok",
        "key_metrics/simple_agent/pass@1": 0.25,
        "other_agent/mean/reward": 1.0,
    }


class TestExport:
    def test_sends_the_saved_config_flattened_metrics_and_sanitized_rollouts(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        backend = _backend(artifacts)

        result = backend.export(artifacts)

        assert result.state is ExportState.SUCCEEDED
        assert result.detail == {"run_id": "backend-run-7", "uploaded_rollouts": True}
        assert OmegaConf.to_container(backend.configs[0]) == {"policy_api_key": "****", "model": "m"}
        assert backend.metrics == [flatten_aggregate_metrics(AGGREGATE_ENTRIES)]
        assert backend.rollouts == [
            [
                {"_ng_task_index": 0, "reward": 1.0, "ng_model_call_capture": {"calls": []}},
                {"_ng_task_index": 1, "reward": 0.0},
            ]
        ]
        assert backend.events == ["setup", "teardown"]

    def test_marker_is_written_after_the_run_is_sent_not_when_the_backend_opens(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        backend = _backend(artifacts)

        backend.export(artifacts)

        assert backend.marker_during_logging is False
        assert artifacts.export_done_fpath("recording").read_text() == "backend-run-7"

    def test_an_already_exported_run_is_skipped_without_opening_the_backend(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.export_done_fpath("recording").write_text("earlier-run")
        backend = _backend(artifacts)

        result = backend.export(artifacts)

        assert result.state is ExportState.SKIPPED
        assert result.detail == {"reason": "already exported", "run_id": "earlier-run"}
        assert backend.events == []

    def test_a_backend_failure_is_reported_leaves_no_marker_and_still_closes_the_backend(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        artifacts = _make_run(tmp_path)
        monkeypatch.setattr(RecordingBackend, "fail_logging", True)
        backend = _backend(artifacts)

        result = backend.export(artifacts)

        assert result.state is ExportState.FAILED
        assert result.exit_code == 1
        assert "backend rejected the metrics" in result.error
        assert not artifacts.export_done_fpath("recording").exists()
        assert backend.events == ["setup", "teardown"]

    def test_a_retry_after_failure_starts_a_fresh_backend_run(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        artifacts = _make_run(tmp_path)
        monkeypatch.setattr(RecordingBackend, "fail_logging", True)
        _backend(artifacts).export(artifacts)
        monkeypatch.setattr(RecordingBackend, "fail_logging", False)

        retry = _backend(artifacts)

        assert retry.export(artifacts).state is ExportState.SUCCEEDED

    def test_live_runs_still_record_the_run_id_when_the_backend_opens(self, tmp_path: Path) -> None:
        output = tmp_path / "live.jsonl"
        backend = RecordingBackend(OmegaConf.create({"output_jsonl_fpath": str(output)}))

        backend.setup()

        assert (tmp_path / "live_recording_run_id").read_text() == "backend-run-7"
        assert backend._recorded_run_id() is None
        resumed = RecordingBackend(OmegaConf.create({"output_jsonl_fpath": str(output), "resume_from_cache": True}))
        assert resumed._recorded_run_id() == "backend-run-7"
        assert not (tmp_path / "live_recording_export_done").exists()

    def test_the_done_marker_is_its_own_file_and_leaves_the_live_run_id_alone(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)

        _backend(artifacts).export(artifacts)

        assert (tmp_path / "run_recording_export_done").read_text() == "backend-run-7"
        assert not (tmp_path / "run_recording_run_id").exists()

    def test_a_live_run_id_does_not_make_the_export_skip_and_is_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        artifacts = _make_run(tmp_path)
        live = tmp_path / "run_recording_run_id"
        live.write_text("live-run-3")
        backend = _backend(artifacts)

        with caplog.at_level("WARNING"):
            result = backend.export(artifacts)

        assert result.state is ExportState.SUCCEEDED
        assert result.detail == {"run_id": "backend-run-7", "uploaded_rollouts": True, "live_run_id": "live-run-3"}
        assert backend.events == ["setup", "teardown"]
        assert "logged" in caplog.text and "live as run live-run-3" in caplog.text
        assert live.read_text() == "live-run-3"
        assert (tmp_path / "run_recording_export_done").read_text() == "backend-run-7"

    def test_an_export_does_not_become_the_run_a_later_live_resume_reopens(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        _backend(artifacts).export(artifacts)

        resumed = RecordingBackend(
            OmegaConf.create({"output_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath), "resume_from_cache": True})
        )

        assert resumed._recorded_run_id() is None

    def test_a_done_marker_in_a_directory_that_does_not_exist_yet_is_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "not" / "there" / "marker"

        _write_marker_atomically(nested, "run-1")

        assert nested.read_text() == "run-1"

    def test_a_marker_write_that_fails_midway_leaves_no_marker_and_a_retry_succeeds(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        artifacts = _make_run(tmp_path)
        real_write_text = Path.write_text

        def disk_full(self: Path, text: str, *args: Any, **kwargs: Any) -> int:
            if self.name.endswith(".tmp") and "_export_done" in self.name:
                real_write_text(self, text[:1], *args, **kwargs)
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_write_text(self, text, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(Path, "write_text", disk_full)
            result = _backend(artifacts).export(artifacts)

        assert result.state is ExportState.FAILED
        assert "No space left on device" in result.error
        assert not artifacts.export_done_fpath("recording").exists()
        assert not list(tmp_path.glob(".*.tmp"))
        assert _backend(artifacts).export(artifacts).state is ExportState.SUCCEEDED
        assert artifacts.export_done_fpath("recording").read_text() == "backend-run-7"


class ForgetfulBackend(RecordingBackend):
    name: ClassVar[str] = "forgetful"

    def setup(self) -> None:
        self.events.append("setup")


class TestBackendThatReportsNoRunId:
    def test_the_export_fails_and_writes_no_marker(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        backend = ForgetfulBackend(OmegaConf.create({"rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath)}))

        result = backend.export(artifacts)

        assert result.state is ExportState.FAILED
        assert "did not report a run id" in result.error
        assert "_record_run_id()" in result.error
        assert not artifacts.export_done_fpath("forgetful").exists()
        assert backend.events == ["setup", "teardown"]


class TestRunMustBeExportable:
    def test_missing_aggregate_metrics(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.aggregate_metrics_json_fpath.unlink()

        result = _backend(artifacts).export(artifacts)

        assert result.state is ExportState.FAILED
        assert "Aggregate metrics not found" in result.error

    def test_aggregate_metrics_older_than_the_rollouts(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        stamp = artifacts.rollouts_jsonl_fpath.stat().st_mtime - 100
        os.utime(artifacts.aggregate_metrics_json_fpath, (stamp, stamp))

        result = _backend(artifacts).export(artifacts)

        assert result.state is ExportState.FAILED
        assert "older than" in result.error

    def test_missing_saved_config(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.resolved_config_yaml_fpath.unlink()

        result = _backend(artifacts).export(artifacts)

        assert result.state is ExportState.FAILED
        assert "Saved run config not found" in result.error

    def test_missing_rollouts_fail_when_they_would_be_uploaded_and_say_how_to_skip_them(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.rollouts_jsonl_fpath.unlink()
        backend = _backend(artifacts)

        result = backend.export(artifacts)

        assert result.state is ExportState.FAILED
        assert f"Rollouts file not found: {artifacts.rollouts_jsonl_fpath}" in result.error
        assert "+upload_rollouts=false" in result.error
        assert backend.events == []
        assert not artifacts.export_done_fpath("recording").exists()

    @pytest.mark.parametrize("where", ["command", "saved config"])
    def test_missing_rollouts_are_fine_when_they_are_not_uploaded(self, tmp_path: Path, where: str) -> None:
        artifacts = _make_run(tmp_path, saved_config={"upload_rollouts": False} if where == "saved config" else None)
        artifacts.rollouts_jsonl_fpath.unlink()
        backend = _backend(artifacts, **({"upload_rollouts": False} if where == "command" else {}))

        result = backend.export(artifacts)

        assert result.state is ExportState.SUCCEEDED, result.error
        assert result.detail["uploaded_rollouts"] is False
        assert backend.metrics == [flatten_aggregate_metrics(AGGREGATE_ENTRIES)]
        assert backend.rollouts == []
        assert artifacts.export_done_fpath("recording").read_text() == "backend-run-7"

    def test_the_freshness_check_needs_the_rollouts_file_to_compare_against(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.rollouts_jsonl_fpath.unlink()
        os.utime(artifacts.aggregate_metrics_json_fpath, (1, 1))

        assert _backend(artifacts, upload_rollouts=False).export(artifacts).state is ExportState.SUCCEEDED

    def test_a_run_with_nothing_left_fails_on_what_every_export_needs(self, tmp_path: Path) -> None:
        artifacts = RunArtifacts(tmp_path / "none.jsonl")

        result = _backend(artifacts, upload_rollouts=False).export(artifacts)

        assert result.state is ExportState.FAILED
        assert "Aggregate metrics not found" in result.error

    def test_a_run_that_cannot_be_exported_does_not_open_the_backend(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.resolved_config_yaml_fpath.unlink()
        backend = _backend(artifacts)

        backend.export(artifacts)

        assert backend.events == []

    def test_an_aggregate_file_with_a_failed_agent_is_refused_until_it_is_repaired(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        healthy = {"agent_ref": {"name": "good"}, "agent_metrics": {"mean/reward": 1.0}, "key_metrics": {}}
        failed = {"agent_ref": {"name": "bad"}, "agent_metrics": {}, "key_metrics": {}, "aggregation_error": {"x": 1}}
        artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps([healthy, failed]))
        os.utime(artifacts.aggregate_metrics_json_fpath, (os.stat(artifacts.rollouts_jsonl_fpath).st_mtime + 10,) * 2)
        backend = _backend(artifacts)

        result = backend.export(artifacts)

        assert result.state is ExportState.FAILED
        assert result.error == "Aggregation failed for agents: bad. Run `gym eval aggregate` again."
        assert backend.events == []
        assert not artifacts.export_done_fpath("recording").exists()

        repaired = {"agent_ref": {"name": "bad"}, "agent_metrics": {"mean/reward": 0.5}, "key_metrics": {}}
        artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps([healthy, repaired]))
        os.utime(artifacts.aggregate_metrics_json_fpath, (os.stat(artifacts.rollouts_jsonl_fpath).st_mtime + 20,) * 2)
        retry = _backend(artifacts)
        assert retry.export(artifacts).state is ExportState.SUCCEEDED
        assert retry.metrics == [{"good/mean/reward": 1.0, "bad/mean/reward": 0.5}]
        assert _backend(artifacts).export(artifacts).state is ExportState.SKIPPED

    def test_every_failed_agent_is_named(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        entries = [{"agent_ref": {"name": name}, "aggregation_error": {}} for name in ("b", "a")]
        artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(entries))
        os.utime(artifacts.aggregate_metrics_json_fpath, (os.stat(artifacts.rollouts_jsonl_fpath).st_mtime + 10,) * 2)

        assert "agents: a, b." in _backend(artifacts).export(artifacts).error

    def test_aggregate_metrics_that_are_not_json(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.aggregate_metrics_json_fpath.write_text("{not json")
        os.utime(artifacts.aggregate_metrics_json_fpath, (os.stat(artifacts.rollouts_jsonl_fpath).st_mtime + 10,) * 2)

        result = _backend(artifacts).export(artifacts)

        assert result.state is ExportState.FAILED
        assert "are not valid JSON" in result.error


class TestUploadIntent:
    def test_the_runs_saved_setting_decides_by_default(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path, saved_config={"upload_rollouts": False})
        backend = _backend(artifacts)

        result = backend.export(artifacts)

        assert result.detail["uploaded_rollouts"] is False
        assert backend.rollouts == []

    def test_an_explicit_override_on_the_export_command_wins_over_the_saved_setting(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path, saved_config={"upload_rollouts": False})
        backend = _backend(artifacts, upload_rollouts=True)

        assert backend.export(artifacts).detail["uploaded_rollouts"] is True
        assert len(backend.rollouts[0]) == 2

    def test_an_explicit_false_skips_the_upload_of_a_run_that_would_upload(self, tmp_path: Path) -> None:
        artifacts = _make_run(tmp_path)
        backend = _backend(artifacts, upload_rollouts=False)

        assert backend.export(artifacts).detail["uploaded_rollouts"] is False

    @pytest.mark.parametrize("send_rows", [False, True])
    def test_an_interpolated_setting_in_the_saved_config_is_resolved(self, tmp_path: Path, send_rows: bool) -> None:
        artifacts = _make_run(tmp_path, saved_config={"send_rows": send_rows, "upload_rollouts": "${send_rows}"})
        backend = _backend(artifacts)

        result = backend.export(artifacts)

        assert result.state is ExportState.SUCCEEDED, result.error
        assert result.detail["uploaded_rollouts"] is send_rows
        assert bool(backend.rollouts) is send_rows

    @pytest.mark.parametrize("send_rows", [False, True])
    def test_an_interpolated_setting_on_the_export_command_is_resolved(self, tmp_path: Path, send_rows: bool) -> None:
        artifacts = _make_run(tmp_path, saved_config={"upload_rollouts": not send_rows})
        backend = _backend(artifacts, send_rows=send_rows, upload_rollouts="${send_rows}")

        result = backend.export(artifacts)

        assert result.detail["uploaded_rollouts"] is send_rows

    def test_the_rest_of_the_saved_config_stays_unresolved(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_VAR_FOR_TEST", raising=False)
        artifacts = _make_run(
            tmp_path, saved_config={"upload_rollouts": False, "other": "${oc.env:UNSET_VAR_FOR_TEST}"}
        )
        backend = _backend(artifacts)

        assert backend.export(artifacts).state is ExportState.SUCCEEDED
        assert OmegaConf.to_container(backend.configs[0])["other"] == "${oc.env:UNSET_VAR_FOR_TEST}"

    def test_a_losing_deprecated_value_is_never_evaluated(self, tmp_path: Path) -> None:
        artifacts = _make_run(
            tmp_path, saved_config={"upload_rollouts": False, "upload_rollouts_to_wandb": "${nowhere}"}
        )

        assert _backend(artifacts).export(artifacts).state is ExportState.SUCCEEDED


class TestRegisteredBackends:
    def test_mlflow_and_wandb_are_post_run_exporters(self) -> None:
        assert load_run_exporter_class("mlflow") is MLflowExporter
        assert load_run_exporter_class("wandb") is WandbExporter

    def test_a_missing_sdk_names_the_extra_to_install(self, monkeypatch: MonkeyPatch) -> None:
        def missing(name: str):
            raise ImportError("No module named 'mlflow'")

        monkeypatch.setattr(exporters_module, "import_module", missing)

        with pytest.raises(ConfigError, match=r"pip install nemo-gym\[mlflow\]"):
            load_run_exporter_class("mlflow")

    def test_the_rollouts_path_is_required(self) -> None:
        with pytest.raises(Exception, match="rollouts_jsonl_fpath"):
            MLflowExporter.from_global_config(OmegaConf.create({"mlflow_tracking_uri": "http://x"}))

    def test_mlflow_export_creates_one_run_with_the_saved_config_and_records_its_id(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        artifacts = _make_run(tmp_path)
        client = MagicMock()
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
        client.create_run.return_value = MagicMock(info=MagicMock(run_id="mlflow-run-1"))
        monkeypatch.setattr(mlflow_module, "MlflowClient", MagicMock(return_value=client))
        config = OmegaConf.create(
            {
                "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath),
                "mlflow_tracking_uri": "https://tracking.example",
                "mlflow_experiment_name": "gym",
                "mlflow_run_name": "nightly",
            }
        )

        result = MLflowExporter.from_global_config(config).export(artifacts)

        assert result.state is ExportState.SUCCEEDED
        client.create_run.assert_called_once()
        client.log_dict.assert_called_once()
        assert client.log_dict.call_args.args[1] == {"policy_api_key": "****", "model": "m"}
        client.set_terminated.assert_called_once_with("mlflow-run-1", status="FINISHED")
        assert artifacts.export_done_fpath("mlflow").read_text() == "mlflow-run-1"
        assert MLflowExporter.from_global_config(config).export(artifacts).state is ExportState.SKIPPED
        assert client.create_run.call_count == 1

    def test_wandb_export_logs_to_one_run_and_records_its_id(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        artifacts = _make_run(tmp_path)
        run = MagicMock(id="wandb-run-1")
        init = MagicMock(return_value=run)
        monkeypatch.setattr(wandb_module.wandb, "init", init)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        config = OmegaConf.create(
            {
                "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath),
                "results_dir": str(tmp_path / "results"),
                "wandb_project": "proj",
                "wandb_name": "nightly",
                "wandb_api_key": "key",  # pragma: allowlist secret
            }
        )

        result = WandbExporter.from_global_config(config).export(artifacts)

        assert result.state is ExportState.SUCCEEDED
        assert "id" not in init.call_args.kwargs
        run.config.update.assert_called_once_with({"policy_api_key": "****", "model": "m"})
        run.finish.assert_called_once()
        assert artifacts.export_done_fpath("wandb").read_text() == "wandb-run-1"


def test_skip_exporters_leaves_live_exporters_unopened(monkeypatch: MonkeyPatch) -> None:
    import nemo_gym.config.global_config as global_config

    opened = MagicMock()
    monkeypatch.setattr(global_config, "setup_exporters", opened)

    for skip in (True, False):
        opened.reset_mock()
        config = global_config.GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            skip_exporters=skip,
        )
        global_config.GlobalConfigDictParser().parse(config)
        assert opened.called is (not skip)


class TestWandbCredentials:
    @staticmethod
    def _export(tmp_path: Path, monkeypatch: MonkeyPatch, **settings: Any) -> tuple[Any, RunArtifacts]:
        artifacts = _make_run(tmp_path, saved_config={"wandb_api_key": "****", "model": "m"})
        run = MagicMock(id="wandb-run-1")
        monkeypatch.setattr(wandb_module.wandb, "init", MagicMock(return_value=run))
        config = OmegaConf.create(
            {
                "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath),
                "results_dir": str(tmp_path / "results"),
                "wandb_project": "proj",
                "wandb_name": "nightly",
                **settings,
            }
        )
        return WandbExporter.from_global_config(config).export(artifacts), artifacts

    def test_a_key_from_the_environment_is_used_when_the_command_sets_none(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "env-key")  # pragma: allowlist secret

        result, artifacts = self._export(tmp_path, monkeypatch)

        assert result.state is ExportState.SUCCEEDED, result.error
        assert os.environ["WANDB_API_KEY"] == "env-key"  # pragma: allowlist secret
        assert artifacts.export_done_fpath("wandb").read_text() == "wandb-run-1"

    def test_a_key_on_the_command_overrides_the_environment(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "env-key")  # pragma: allowlist secret

        result, _ = self._export(tmp_path, monkeypatch, wandb_api_key="command-key")  # pragma: allowlist secret

        assert result.state is ExportState.SUCCEEDED, result.error
        assert os.environ["WANDB_API_KEY"] == "command-key"  # pragma: allowlist secret
