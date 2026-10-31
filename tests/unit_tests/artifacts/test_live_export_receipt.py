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
            store.record_outcome(row | {"reward": float(i), "response": {"output": []}, "evaluation_completed": True})
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
    OmegaConf.save(config, artifacts.resolved_config_yaml_fpath)
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
    assert receipt == {"version": 1, "run_id": "live-run", "artifacts": before, "uploaded_rollouts": True}
    assert client.status == "FINISHED"
    assert _artifact_version(artifacts) == before
    collected = _collected_mlflow(artifacts)
    assert collected["receipt"] == {"run_id": "live-run", "current": True, "uploaded_rollouts": True}
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
