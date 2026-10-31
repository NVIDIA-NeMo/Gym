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
"""Same-run live delivery certification through the actual SDK exporter and managed Store."""

import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from mlflow.utils.async_logging.run_operations import RunOperations
from omegaconf import OmegaConf

import nemo_gym.artifacts.exporters as registry
import nemo_gym.artifacts.exporters.mlflow as module
from nemo_gym.artifacts.exporters.base import _artifact_version
from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics
from nemo_gym.artifacts.resolved_config import write_resolved_config
from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.cli.eval import _finalize_live_exporters
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_export_receipt import _collected_mlflow


class Client:
    def __init__(self):
        self.artifacts = {}
        self.metrics = {}
        self.tags = {}
        self.status = "RUNNING"
        self.futures = []
        self.read_hook = None

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="experiment")

    def create_run(self, *args, **kwargs):
        return SimpleNamespace(info=SimpleNamespace(run_id="live-run"))

    def log_dict(self, run_id, value, path):
        self.artifacts[path] = value

    def log_batch(self, run_id, params=(), metrics=(), tags=(), **kwargs):
        self.metrics.update({m.key: m.value for m in metrics})
        self.tags.update({t.key: t.value for t in tags})
        future = Future()
        future.set_result(None)
        self.futures.append(future)
        return RunOperations([future])

    def log_table(self, run_id, data, artifact_file):
        self.artifacts[artifact_file] = {"columns": ["Rollout"], "data": [[r] for r in data["Rollout"]]}

    def download_artifacts(self, run_id, path, directory):
        if self.read_hook:
            self.read_hook()
        target = Path(directory) / path
        target.write_text(json.dumps(self.artifacts[path]))
        return str(target)

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, status=self.status),
            data=SimpleNamespace(metrics=self.metrics, tags=self.tags),
        )

    def set_terminated(self, run_id, status):
        self.status = status


@pytest.fixture
def live(tmp_path, monkeypatch):
    output = tmp_path / "artifacts" / "rollouts.jsonl"
    output.parent.mkdir()
    source = tmp_path / "inputs.jsonl"
    inputs = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(2)]
    source.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in inputs))
    prepare = lambda: (
        inputs,
        RunManifest.create(source, inputs, {}, {"agent": {"responses_api_agents": {"impl": {}}}}),
    )
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for i, row in enumerate(store.pending(3)):
            store.record_dispatch(row)
            store.record_outcome(
                row
                | {
                    "reward": float(i),
                    "response": {"output": []},
                    "evaluation_completed": True,
                    "ng_trajectory": {
                        "schema_version": "1.0",
                        "task_id": str(i),
                        "rollout_id": f"{i}-0",
                        "invocations": [{"invocation_id": "root", "status": "completed", "conversation": []}],
                        "verifier_invocations": [{"invocation_id": f"verify-{i}", "status": "completed"}],
                        "model_calls": [
                            {
                                "model_call_id": f"call-{i}",
                                "request": {"input": "private fixture"},
                                "response": {"output_text": "private fixture"},
                                "response_metadata": {"status_code": 200},
                            }
                        ],
                    },
                }
            )
    artifacts = RunArtifacts(output)
    config = OmegaConf.create(
        {
            "output_jsonl_fpath": str(output),
            "upload_rollouts": True,
            "mlflow_tracking_uri": "https://tracking.example",
            "mlflow_experiment_name": "test",
            "mlflow_run_name": "run",
        }
    )
    write_resolved_config(config, output)
    aggregate = [{"agent_ref": {"name": "agent"}, "agent_metrics": {"mean/reward": 0.5}, "key_metrics": {}}]
    artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(aggregate))
    run_health_checks(output, workers=1, run_coverage=[RolloutStore.read(output).coverage()])
    client = Client()
    monkeypatch.setattr(module, "MlflowClient", lambda **kwargs: client)
    exporter = module.MLflowExporter(config)
    exporter.setup()
    exporter.export_config()
    exporter.export_metrics(flatten_aggregate_metrics(aggregate))
    exporter.export_rollouts(exporter._read_rollouts(artifacts))
    monkeypatch.setattr(registry, "_EXPORTERS", [exporter])
    monkeypatch.setattr(registry, "_RUN_FAILED", False)
    return artifacts, exporter, client


def test_final_live_delivery_uses_same_run_and_existing_consumer(live):
    artifacts, exporter, client = live
    before = _artifact_version(artifacts)
    assert not artifacts.export_done_fpath("mlflow").exists()
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt == {
        "version": 1,
        "run_id": "live-run",
        "artifacts": before,
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    assert client.status == "FINISHED"
    assert _artifact_version(artifacts) == before
    collected = _collected_mlflow(artifacts)
    assert collected["receipt"] == {
        "run_id": "live-run",
        "current": True,
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    from nemo_gym.orchestration.recovery import mlflow_errors

    assert mlflow_errors(collected, "FINISHED", observed_run_id="live-run") == []


@pytest.mark.parametrize(
    "fault",
    [
        "async",
        "partial",
        "config",
        "metric",
        "missing_health",
        "stale_health",
        "changed_source",
        "wrong_run",
        "failed_status",
        "swallowed",
        "reopened",
        "upload_false",
        "old_receipt",
        "no_activation",
    ],
)
def test_live_finalization_never_certifies_incomplete_delivery(live, monkeypatch, fault):
    artifacts, exporter, client = live
    marker = artifacts.export_done_fpath("mlflow")
    if fault == "async":
        failed = Future()
        failed.set_exception(RuntimeError("delivery failed"))
        exporter._pending_operations.append(RunOperations([failed]))
    elif fault == "partial":

        def partial_table(*args, **kwargs):
            client.artifacts["rollouts.json"]["data"].pop()
            raise RuntimeError("table delivery interrupted")

        client.log_table = partial_table
        exporter.export_rollouts(exporter._read_rollouts(artifacts))
    elif fault == "config":
        exporter.global_config_dict["seed"] = 42
        exporter.export_config()
        del exporter.global_config_dict["seed"]
    elif fault == "metric":
        exporter.export_metrics({key: -1 for key in client.metrics})
    elif fault == "missing_health":
        (artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json").unlink()
    elif fault == "stale_health":
        artifacts.rollouts_jsonl_fpath.write_bytes(artifacts.rollouts_jsonl_fpath.read_bytes() + b"\n")
    elif fault == "changed_source":
        original = client.get_run

        def changed_run(run_id):
            artifacts.failures_jsonl_fpath.write_bytes(b"\n")
            return original(run_id)

        client.get_run = changed_run
    elif fault == "wrong_run":
        artifacts.exporter_run_id_fpath("mlflow").write_text("other-run")
    elif fault == "failed_status":
        client.set_terminated = lambda *args, **kwargs: None
    elif fault == "swallowed":
        exporter._guard("test", lambda: (_ for _ in ()).throw(RuntimeError("upload failed")))
    elif fault == "reopened":
        exporter.reopened = True
    elif fault == "upload_false":
        exporter.global_config_dict.upload_rollouts = False
    elif fault == "old_receipt":
        marker.write_text("old-run")
    elif fault == "no_activation":
        exporter.client = None
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert marker.read_text() == "old-run" if fault == "old_receipt" else not marker.exists()


def test_teardown_alone_never_certifies_live_delivery(live):
    artifacts, exporter, client = live
    exporter.teardown()
    assert not artifacts.export_done_fpath("mlflow").exists()


def test_live_upload_uses_actual_producer_default_when_key_is_absent(live):
    artifacts, exporter, client = live
    del exporter.global_config_dict.upload_rollouts
    OmegaConf.save(exporter.global_config_dict, artifacts.resolved_config_yaml_fpath)
    exporter.export_config()
    run_health_checks(
        artifacts.rollouts_jsonl_fpath,
        workers=1,
        run_coverage=[RolloutStore.read(artifacts.rollouts_jsonl_fpath).coverage()],
    )
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["uploaded_rollouts"] is True


def test_live_finalizer_streams_rollout_files_without_backend_table_download(live, monkeypatch):
    artifacts, exporter, client = live
    original = Path.read_bytes

    def guarded(path):
        if path in (artifacts.rollouts_jsonl_fpath, artifacts.failures_jsonl_fpath):
            raise AssertionError("whole rollout file read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    client.download_artifacts = lambda *a, **k: (_ for _ in ()).throw(AssertionError("whole backend table read"))
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert artifacts.export_done_fpath("mlflow").exists()


@pytest.mark.parametrize("fault", ["wrong_source", "malformed_health", "wrong_backend_run", "new_reservation"])
def test_live_finalizer_rejects_changed_completion_identity(live, fault):
    artifacts, exporter, client = live
    output = artifacts.rollouts_jsonl_fpath
    health = output.parent / "quality_summary.json"
    if fault == "wrong_source":
        summary = json.loads(health.read_text())
        summary["run"]["rollouts"] = ["another.jsonl"]
        health.write_text(json.dumps(summary))
    elif fault == "malformed_health":
        health.write_text("[]")
    elif fault == "wrong_backend_run":
        original = client.get_run

        def wrong_run(run_id):
            result = original(run_id)
            result.info.run_id = "another-run"
            return result

        client.get_run = wrong_run
    else:
        from nemo_gym.rollout_recovery import manifest_path_for

        manifest = RunManifest.model_validate_json(manifest_path_for(output).read_bytes())
        inputs = [orjson.loads(line) for line in artifacts.materialized_inputs_jsonl_fpath.read_bytes().splitlines()]
        with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=True) as store:
            row = store.selected("success")[0]
            row["_ng_attempt_index"] = 1
            store.record_dispatch(row)
        assert RolloutStore.read(output).coverage()["complete"] is False
    _finalize_live_exporters(output)
    assert not artifacts.export_done_fpath("mlflow").exists()


def test_live_delivery_retains_canonical_observations_without_raw_capture(live):
    artifacts, exporter, client = live
    saved = [orjson.loads(line) for line in artifacts.rollouts_jsonl_fpath.read_bytes().splitlines()]
    delivered = [orjson.loads(row[0]) for row in client.artifacts["rollouts.json"]["data"]]
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert artifacts.export_done_fpath("mlflow").exists()
    for original, uploaded in zip(saved, delivered, strict=True):
        trajectory = uploaded["ng_trajectory"]
        assert trajectory["verifier_invocations"] == original["ng_trajectory"]["verifier_invocations"]
        assert trajectory["task_id"] == original["ng_trajectory"]["task_id"]
        assert trajectory["rollout_id"] == original["ng_trajectory"]["rollout_id"]
        assert (
            trajectory["model_calls"][0]["model_call_id"]
            == original["ng_trajectory"]["model_calls"][0]["model_call_id"]
        )
        assert trajectory["model_calls"][0]["response_metadata"] == {"status_code": 200}
        assert "request" not in trajectory["model_calls"][0]
        assert "response" not in trajectory["model_calls"][0]
        assert "conversation" not in trajectory["invocations"][0]
        assert uploaded["reward"] == original["reward"]


def test_live_receipt_rejects_verifier_observations_not_in_acknowledged_rows(live):
    artifacts, exporter, client = live
    output = artifacts.rollouts_jsonl_fpath
    rows = [orjson.loads(line) for line in output.read_bytes().splitlines()]
    rows[0]["ng_trajectory"]["verifier_invocations"].append({"invocation_id": "late-verifier", "status": "completed"})
    output.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    run_health_checks(output, workers=1, run_coverage=[RolloutStore.read(output).coverage()])
    _finalize_live_exporters(output)
    assert not artifacts.export_done_fpath("mlflow").exists()


@pytest.mark.parametrize("projection", [None, 0, 2, True, "1"])
def test_old_or_unknown_projection_receipt_cannot_certify_canonical_delivery(live, projection):
    from nemo_gym.orchestration.recovery import mlflow_errors

    artifacts, exporter, client = live
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    marker = artifacts.export_done_fpath("mlflow")
    receipt = orjson.loads(marker.read_bytes())
    if projection is None:
        del receipt["rollout_projection_version"]
    else:
        receipt["rollout_projection_version"] = projection
    marker.write_bytes(orjson.dumps(receipt))
    before = marker.read_bytes()
    collected = _collected_mlflow(artifacts)
    assert collected["receipt"]["current"] is True
    assert any(
        "canonical trajectory metadata" in error
        for error in mlflow_errors(collected, "FINISHED", observed_run_id="live-run")
    )
    result = module.MLflowExporter(exporter.global_config_dict).export(artifacts)
    assert result.state.value == "failed"
    assert marker.read_bytes() == before


def test_malformed_canonical_observations_cannot_be_silently_certified(live):
    artifacts, exporter, client = live
    output = artifacts.rollouts_jsonl_fpath
    rows = [orjson.loads(line) for line in output.read_bytes().splitlines()]
    rows[0]["ng_trajectory"]["verifier_invocations"].append({"invocation_id": "verify-0", "status": "completed"})
    output.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    run_health_checks(output, workers=1, run_coverage=[RolloutStore.read(output).coverage()])
    _finalize_live_exporters(output)
    assert not artifacts.export_done_fpath("mlflow").exists()


def test_postrun_mlflow_export_keeps_canonical_evidence_and_projection_receipt(live, monkeypatch):
    artifacts, live_exporter, live_client = live
    from nemo_gym.orchestration.recovery import mlflow_errors

    client = Client()
    monkeypatch.setattr(module, "MlflowClient", lambda **kwargs: client)
    exporter = module.MLflowExporter(live_exporter.global_config_dict)
    result = exporter.export(artifacts)
    assert result.state.value == "succeeded"
    delivered = [orjson.loads(row[0]) for row in client.artifacts["rollouts.json"]["data"]]
    assert [row["ng_trajectory"]["verifier_invocations"][0]["invocation_id"] for row in delivered] == [
        "verify-0",
        "verify-1",
    ]
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt["rollout_projection_version"] == 1
    assert mlflow_errors(_collected_mlflow(artifacts), "FINISHED", observed_run_id=receipt["run_id"]) == []


@pytest.mark.parametrize("enabled", [False, True])
def test_live_receipt_config_projection_preserves_capture_controls(live, enabled):
    artifacts, exporter, client = live
    exporter.global_config_dict.token_id_capture = {
        "enabled": enabled,
        "dir": "/tmp/token-capture-fixture",
        "sink": None,
        "rebuild_response": False,
    }
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    TokenIdCaptureConfig.model_validate(OmegaConf.to_container(exporter.global_config_dict, resolve=False))
    before = OmegaConf.to_container(exporter.global_config_dict, resolve=False)
    write_resolved_config(exporter.global_config_dict, artifacts.rollouts_jsonl_fpath)
    exporter.export_config()
    run_health_checks(
        artifacts.rollouts_jsonl_fpath,
        workers=1,
        run_coverage=[RolloutStore.read(artifacts.rollouts_jsonl_fpath).coverage()],
    )
    version = _artifact_version(artifacts)
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert _artifact_version(artifacts) == version
    assert artifacts.export_done_fpath("mlflow").exists()
    saved = OmegaConf.to_container(OmegaConf.load(artifacts.resolved_config_yaml_fpath), resolve=False)
    assert client.artifacts[exporter.CONFIG_ARTIFACT_FILE] == saved
    assert saved["token_id_capture"] == before["token_id_capture"]
    assert OmegaConf.to_container(exporter.global_config_dict, resolve=False) == before
    assert _collected_mlflow(artifacts)["receipt"]["current"] is True


@pytest.mark.parametrize("field,value", [("enabled", True), ("sink", "other:Sink"), ("rebuild_response", True)])
def test_live_receipt_rejects_changed_capture_controls(live, field, value):
    artifacts, exporter, client = live
    exporter.global_config_dict.token_id_capture = {
        "enabled": False,
        "sink": "example:Sink",
        "rebuild_response": False,
    }
    write_resolved_config(exporter.global_config_dict, artifacts.rollouts_jsonl_fpath)
    exporter.export_config()
    exporter.global_config_dict.token_id_capture[field] = value
    write_resolved_config(exporter.global_config_dict, artifacts.rollouts_jsonl_fpath)
    run_health_checks(
        artifacts.rollouts_jsonl_fpath,
        workers=1,
        run_coverage=[RolloutStore.read(artifacts.rollouts_jsonl_fpath).coverage()],
    )
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert not artifacts.export_done_fpath("mlflow").exists()


def test_live_config_projection_masks_secret_ancestry_without_mutating_config(live):
    artifacts, exporter, client = live
    exporter.global_config_dict.token_id_capture = {"enabled": True, "sink": "example:Sink"}
    exporter.global_config_dict.environment_servers = {
        "harbor": {
            "secrets": {
                "JUDGE_AUTH": "private-auth-canary",
                "nested": {"arbitrary": 123},
                "enabled": True,
                "missing": None,
                "many": ["private-list-canary"],
                "token_id_capture": {"enabled": True, "sink": "private-sink-canary"},
            }
        }
    }
    exporter.global_config_dict.openai_default_headers = {
        "X-Auth": "private-header-canary",
        "arbitrary": "private-second-header-canary",
    }
    before = OmegaConf.to_container(exporter.global_config_dict, resolve=False)
    write_resolved_config(exporter.global_config_dict, artifacts.rollouts_jsonl_fpath)
    exporter.export_config()
    run_health_checks(
        artifacts.rollouts_jsonl_fpath,
        workers=1,
        run_coverage=[RolloutStore.read(artifacts.rollouts_jsonl_fpath).coverage()],
    )
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert artifacts.export_done_fpath("mlflow").exists()
    saved = OmegaConf.to_container(OmegaConf.load(artifacts.resolved_config_yaml_fpath), resolve=False)
    delivered = client.artifacts[exporter.CONFIG_ARTIFACT_FILE]
    assert saved == delivered
    assert saved["environment_servers"]["harbor"]["secrets"] == {
        "JUDGE_AUTH": "****",
        "nested": {"arbitrary": "****"},
        "enabled": "****",
        "missing": "****",
        "many": ["****"],
        "token_id_capture": {"enabled": "****", "sink": "****"},
    }
    assert saved["openai_default_headers"] == {"X-Auth": "****", "arbitrary": "****"}
    assert saved["token_id_capture"]["enabled"] is True
    assert OmegaConf.to_container(exporter.global_config_dict, resolve=False) == before


@pytest.mark.parametrize("environment_set", [False, True])
def test_live_config_projection_never_resolves_environment_references(live, monkeypatch, environment_set):
    artifacts, exporter, client = live
    variable = "GYM_PROJECTION_TEST_UNRESOLVED"
    if environment_set:
        monkeypatch.setenv(variable, "credential-environment-canary")
    else:
        monkeypatch.delenv(variable, raising=False)
    exporter.global_config_dict.diagnostic_url = "${oc.env:GYM_PROJECTION_TEST_UNRESOLVED}"
    exporter.global_config_dict.policy_api_key = "${oc.env:GYM_PROJECTION_TEST_UNRESOLVED}"
    before = OmegaConf.to_container(exporter.global_config_dict, resolve=False)
    write_resolved_config(exporter.global_config_dict, artifacts.rollouts_jsonl_fpath)
    exporter.export_config()
    saved = OmegaConf.to_container(OmegaConf.load(artifacts.resolved_config_yaml_fpath), resolve=False)
    assert client.artifacts[exporter.CONFIG_ARTIFACT_FILE] == saved
    assert saved["diagnostic_url"] == "${oc.env:GYM_PROJECTION_TEST_UNRESOLVED}"
    assert saved["policy_api_key"] == "****"
    assert "credential-environment-canary" not in json.dumps(saved)
    assert OmegaConf.to_container(exporter.global_config_dict, resolve=False) == before
    run_health_checks(
        artifacts.rollouts_jsonl_fpath,
        workers=1,
        run_coverage=[RolloutStore.read(artifacts.rollouts_jsonl_fpath).coverage()],
    )
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert artifacts.export_done_fpath("mlflow").exists()


@pytest.mark.parametrize("key", ["sink_kwargs", "lineage_store_kwargs"])
def test_live_capture_custom_backend_options_do_not_expose_credentials(live, key):
    artifacts, exporter, client = live
    exporter.global_config_dict.token_id_capture = {
        "enabled": True,
        "sink": "example:Sink",
        key: {
            "auth": "credential-kwargs-canary",
            "opaque": {
                "access": "credential-kwargs-canary",
                "url": "https://example.test/credential-kwargs-canary",
            },
            "enabled": False,
            "arbitrary": [17, None, "credential-kwargs-canary"],
        },
    }
    before = OmegaConf.to_container(exporter.global_config_dict, resolve=False)
    write_resolved_config(exporter.global_config_dict, artifacts.rollouts_jsonl_fpath)
    exporter.export_config()
    saved = OmegaConf.to_container(OmegaConf.load(artifacts.resolved_config_yaml_fpath), resolve=False)
    assert client.artifacts[exporter.CONFIG_ARTIFACT_FILE] == saved
    assert saved["token_id_capture"] == {
        "enabled": True,
        "sink": "example:Sink",
        key: {
            "auth": "****",
            "opaque": {"access": "****", "url": "****"},
            "enabled": "****",
            "arbitrary": ["****", "****", "****"],
        },
    }
    assert OmegaConf.to_container(exporter.global_config_dict, resolve=False) == before
    run_health_checks(
        artifacts.rollouts_jsonl_fpath,
        workers=1,
        run_coverage=[RolloutStore.read(artifacts.rollouts_jsonl_fpath).coverage()],
    )
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert artifacts.export_done_fpath("mlflow").exists()
