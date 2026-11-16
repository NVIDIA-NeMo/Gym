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

import hashlib
import json
import stat
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from mlflow.utils.async_logging.run_operations import RunOperations
from omegaconf import OmegaConf

import nemo_gym.artifacts.exporters as registry
import nemo_gym.artifacts.exporters.mlflow as module
from nemo_gym.artifacts.exporters.base import ExportState, _artifact_version
from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics
from nemo_gym.artifacts.resolved_config import write_resolved_config
from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.cli.eval import _finalize_live_exporters
from nemo_gym.rollouts.rollout_health import run_health_checks
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.artifacts.test_export_receipt import _collected_mlflow


class Client:
    def __init__(self):
        self.artifacts = {}
        self.metrics = {}
        self.params = {}
        self.tags = {}
        self.status = "RUNNING"
        self.futures = []
        self.read_hook = None
        self.run_tags = {}

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="experiment")

    def create_run(self, *args, **kwargs):
        return SimpleNamespace(info=SimpleNamespace(run_id="live-run"))

    def log_dict(self, run_id, value, path):
        self.artifacts[path] = value

    def log_batch(self, run_id, params=(), metrics=(), tags=(), **kwargs):
        self.params.update({p.key: p.value for p in params})
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
            data=SimpleNamespace(metrics=self.metrics, params=self.params, tags=self.tags),
        )

    def set_terminated(self, run_id, status):
        self.status = status

    def set_tag(self, run_id, key, value):
        self.run_tags.setdefault(run_id, {})[key] = value


@pytest.fixture
def live(tmp_path, monkeypatch, request):
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
                                "response_metadata": {"status_code": 200, "model_operations": None},
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
    config = OmegaConf.merge(config, getattr(request, "param", {}))
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
    if config.upload_rollouts:
        exporter.export_rollouts(exporter._read_rollouts(artifacts))
    monkeypatch.setattr(registry, "_EXPORTERS", [exporter])
    monkeypatch.setattr(registry, "_RUN_FAILED", False)
    return artifacts, exporter, client


HTTP_STATUS_CONFIG = {"agent_outcome_policy": {"by_status_code": {400: "fail", 413: "fail", 422: "fail"}}}


@pytest.mark.parametrize(
    "live",
    [HTTP_STATUS_CONFIG, HTTP_STATUS_CONFIG | {"upload_rollouts": False}],
    ids=["with-rollouts", "metadata-only"],
    indirect=True,
)
def test_live_delivery_with_integer_config_keys_certifies_the_delivered_json(live, monkeypatch):
    artifacts, exporter, client = live
    post = _post_run_client(monkeypatch)
    saved_config = artifacts.resolved_config_yaml_fpath.read_bytes()
    scientific = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)

    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt["run_id"] == "live-run"
    assert receipt["artifacts"] == _artifact_version(artifacts)
    assert receipt["uploaded_rollouts"] is exporter.global_config_dict.upload_rollouts
    assert client.status == "FINISHED"
    assert post.runs == []
    assert ("rollouts.json" in client.artifacts) is exporter.global_config_dict.upload_rollouts
    assert client.artifacts["global_config.json"]["agent_outcome_policy"]["by_status_code"] == {
        "400": "fail",
        "413": "fail",
        "422": "fail",
    }
    assert artifacts.resolved_config_yaml_fpath.read_bytes() == saved_config
    assert OmegaConf.load(artifacts.resolved_config_yaml_fpath).agent_outcome_policy.by_status_code == {
        400: "fail",
        413: "fail",
        422: "fail",
    }
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    changed[0].pop("_ng_run_health_metrics")
    assert changed == scientific
    assert _collected_mlflow(artifacts)["receipt"]["current"] is True


@pytest.mark.parametrize(
    "live",
    [HTTP_STATUS_CONFIG, HTTP_STATUS_CONFIG | {"upload_rollouts": False}],
    ids=["with-rollouts", "metadata-only"],
    indirect=True,
)
def test_post_run_export_with_integer_config_keys_preserves_saved_artifacts(live, monkeypatch):
    artifacts, live_exporter, _ = live
    before = _artifact_version(artifacts)
    post = _post_run_client(monkeypatch)
    settings = OmegaConf.create(
        {key: value for key, value in live_exporter.global_config_dict.items() if key.startswith("mlflow_")}
    )
    result = module.MLflowExporter(settings).export(artifacts)

    assert result.state is ExportState.SUCCEEDED
    upload = live_exporter.global_config_dict.upload_rollouts
    assert result.detail == {"run_id": "post-run", "uploaded_rollouts": upload, "live_run_id": "live-run"}
    assert post.runs == ["post-run"] and post.status == "FINISHED"
    assert ("rollouts.json" in post.artifacts) is upload
    assert post.artifacts["global_config.json"]["agent_outcome_policy"]["by_status_code"] == {
        "400": "fail",
        "413": "fail",
        "422": "fail",
    }
    assert _artifact_version(artifacts) == before
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt == {
        "version": 1,
        "run_id": "post-run",
        "artifacts": before,
        "uploaded_rollouts": upload,
        "rollout_projection_version": 1,
    }
    assert _collected_mlflow(artifacts)["receipt"]["current"] is True


@pytest.mark.parametrize("post_run", [False, True], ids=["live", "post-run"])
def test_config_key_collisions_are_rejected_before_opening_a_backend(live, monkeypatch, post_run):
    artifacts, live_exporter, _ = live
    config = OmegaConf.create(live_exporter.global_config_dict)
    config["nested"] = [{"statuses": {400: "fail", "400": "retry"}}]
    settings = OmegaConf.create({key: value for key, value in config.items() if key.startswith("mlflow_")})
    if post_run:
        OmegaConf.save(config, artifacts.resolved_config_yaml_fpath)
    else:
        settings = config
        settings.output_jsonl_fpath = str(artifacts.rollouts_jsonl_fpath.with_stem("collision"))
    before = _artifact_version(artifacts)
    connections = []

    def connect(**kwargs):
        connections.append(kwargs)
        return Client()

    monkeypatch.setattr(module, "MlflowClient", connect)
    exporter = module.MLflowExporter(settings)
    if post_run:
        result = exporter.export(artifacts)
        assert result.state is ExportState.FAILED and "collid" in result.error
    else:
        with pytest.raises(ValueError, match="collid"):
            exporter.setup()
        assert not artifacts.rollouts_jsonl_fpath.with_name("collision_mlflow_run_id").exists()
    assert connections == []
    assert not artifacts.export_done_fpath("mlflow").exists()
    assert _artifact_version(artifacts) == before


@pytest.mark.parametrize(
    "live",
    [{"sampling": {"temperature": 0.95, "seed": 7, "choices": [1, None, "café", False]}}],
    indirect=True,
)
def test_string_config_digest_and_strict_rollout_json_are_unchanged(live):
    _, exporter, client = live
    original = OmegaConf.to_container(exporter.global_config_dict, resolve=False)
    assert client.artifacts["global_config.json"] == original
    assert (
        exporter._delivered_config_digest
        == hashlib.sha256(orjson.dumps(original, option=orjson.OPT_SORT_KEYS)).digest()
    )
    table = client.artifacts["rollouts.json"]
    with pytest.raises(TypeError, match="Dict key must be str"):
        exporter._log_rollouts([{"reward": 1.0, "nested": {400: "fail"}}])
    assert client.artifacts["rollouts.json"] == table


@pytest.mark.parametrize("live", [{"secrets": {"opaque": 2**70}}], indirect=True)
def test_live_setup_validates_the_credential_safe_configuration(live, monkeypatch):
    artifacts, exporter, client = live
    post = _post_run_client(monkeypatch)
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)

    assert exporter.global_config_dict.secrets.opaque == 2**70
    assert client.artifacts["global_config.json"]["secrets"] == {"opaque": "****"}
    assert client.status == "FINISHED" and post.runs == []
    assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["run_id"] == "live-run"


def test_a_reopened_run_logs_integer_config_keys_in_its_segment(live):
    _, exporter, client = live
    exporter.reopened = True
    exporter.global_config_dict["agent_outcome_policy"] = HTTP_STATUS_CONFIG["agent_outcome_policy"]
    exporter.export_config()
    exporter.export_metrics({})

    segments = [value for path, value in client.artifacts.items() if path.startswith("config/segment_")]
    assert len(segments) == 1
    assert segments[0]["agent_outcome_policy"]["by_status_code"] == {"400": "fail", "413": "fail", "422": "fail"}
    assert not exporter._delivery_failed


def test_final_live_delivery_uses_same_run_and_existing_consumer(live):
    artifacts, exporter, client = live
    before = _artifact_version(artifacts)
    scientific = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    assert not artifacts.export_done_fpath("mlflow").exists()
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt == {
        "version": 1,
        "run_id": "live-run",
        "artifacts": _artifact_version(artifacts),
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    assert client.status == "FINISHED"
    after = _artifact_version(artifacts)
    assert {k: v for k, v in after.items() if k != "aggregate_metrics_json_fpath"} == {
        k: v for k, v in before.items() if k != "aggregate_metrics_json_fpath"
    }
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    health = changed[0].pop("_ng_run_health_metrics")
    assert changed == scientific
    assert health["metrics"]["health/trial/known"] == 0
    assert client.metrics["health/trial/known"] == 0
    assert client.metrics["health/trial/passed"] == 0
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
    post = _post_run_client(monkeypatch)
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
    if fault == "old_receipt":
        assert marker.read_text() == "old-run" and not post.runs
    elif fault in ("missing_health", "stale_health"):
        # Final coverage or health is not verified: nothing is certified or exported.
        assert not marker.exists() and not post.runs
    elif fault == "changed_source":
        # The saved artifacts changed after their health was judged: the post-run export refuses them too.
        assert not marker.exists()
    else:
        # The live run is never certified; the completed run gets a separate post-run export instead.
        assert orjson.loads(marker.read_bytes())["run_id"] == "post-run" and post.runs == ["post-run"]


def _post_run_client(monkeypatch) -> Client:
    """The backend a post-run export opens: its runs are recorded, apart from the live run's client."""
    post = Client()
    post.runs = []

    def create_run(*args, **kwargs):
        post.runs.append("post-run")
        return SimpleNamespace(info=SimpleNamespace(run_id="post-run"))

    post.create_run = create_run
    monkeypatch.setattr(module, "MlflowClient", lambda **kwargs: post)
    return post


def test_a_reopened_live_run_is_exported_after_the_run_with_a_bound_receipt(live, monkeypatch):
    artifacts, exporter, client = live
    from nemo_gym.orchestration.recovery import mlflow_errors

    # A later segment reopened the live run an earlier segment opened, and completes the run.
    exporter.reopened = True
    post = _post_run_client(monkeypatch)
    scientific = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt == {
        "version": 1,
        "run_id": "post-run",
        "artifacts": _artifact_version(artifacts),
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    assert post.runs == ["post-run"] and post.status == "FINISHED" and "rollouts.json" in post.artifacts
    assert post.metrics["health/trial/known"] == 0
    # The live run is left to its own teardown, and the scores are unchanged.
    assert client.status == "RUNNING"
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    changed[0].pop("_ng_run_health_metrics")
    assert changed == scientific
    assert not registry.run_failed()
    collected = _collected_mlflow(artifacts)
    assert collected["receipt"]["run_id"] == "post-run" and collected["receipt"]["current"] is True
    assert mlflow_errors(collected, "FINISHED", observed_run_id="post-run") == []


def test_the_post_run_export_and_the_live_run_it_supersedes_name_each_other(live, monkeypatch):
    artifacts, exporter, client = live
    exporter.reopened = True
    post = _post_run_client(monkeypatch)
    metrics, params = dict(client.metrics), dict(client.artifacts)
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["run_id"] == "post-run"
    # Both tags go through the live exporter's client; they are tags only, on the two runs.
    assert client.run_tags == {"live-run": {"superseded_by": "post-run"}, "post-run": {"supersedes": "live-run"}}
    assert not post.run_tags
    # The live run keeps its metrics and logged artifacts; finalization adds only the health metrics it always did.
    assert {k: v for k, v in client.metrics.items() if not k.startswith("health/")} == {
        k: v for k, v in metrics.items() if not k.startswith("health/")
    }
    assert client.artifacts == params


def test_a_failed_supersession_tag_is_logged_and_keeps_the_receipt(live, monkeypatch, caplog):
    artifacts, exporter, client = live
    exporter.reopened = True
    _post_run_client(monkeypatch)

    def refused(run_id, key, value):
        raise PermissionError("tagging refused")

    client.set_tag = refused
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["run_id"] == "post-run"
    assert "could not tag" in caplog.text and "tagging refused" in caplog.text
    assert not registry.run_failed()


def test_a_certified_live_run_is_not_tagged(live):
    artifacts, exporter, client = live
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["run_id"] == "live-run"
    assert client.run_tags == {}


def test_an_existing_receipt_is_never_replaced_at_completion(live, monkeypatch):
    artifacts, exporter, client = live
    exporter.reopened = True
    post = _post_run_client(monkeypatch)
    marker = artifacts.export_done_fpath("mlflow")
    marker.write_bytes(b'{"version": 1, "run_id": "earlier-export"}')
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert marker.read_bytes() == b'{"version": 1, "run_id": "earlier-export"}' and not post.runs


def test_a_failed_post_run_export_warns_and_changes_no_score(live, monkeypatch, caplog):
    artifacts, exporter, client = live
    exporter.reopened = True
    post = _post_run_client(monkeypatch)

    def refused(*args, **kwargs):
        raise RuntimeError("tracking server refused the table")

    post.log_table = refused
    scientific = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert not artifacts.export_done_fpath("mlflow").exists()
    assert "Post-run mlflow export after live finalization failed" in caplog.text
    assert "tracking server refused the table" in caplog.text
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    changed[0].pop("_ng_run_health_metrics")
    assert changed == scientific and not registry.run_failed()


def test_a_run_without_a_configured_exporter_is_not_exported(live, monkeypatch):
    artifacts, exporter, client = live
    post = _post_run_client(monkeypatch)
    monkeypatch.setattr(registry, "_EXPORTERS", [])
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert not artifacts.export_done_fpath("mlflow").exists() and not post.runs


def test_an_exporter_that_never_certifies_live_runs_is_not_exported_again(live, monkeypatch):
    artifacts, exporter, client = live
    post = _post_run_client(monkeypatch)
    # A backend without live certification (the base `finalize_live`) keeps no receipt to complete.
    monkeypatch.setattr(module.MLflowExporter, "finalize_live", registry.BaseExporter.finalize_live)
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert not artifacts.export_done_fpath("mlflow").exists() and not post.runs


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


def test_a_live_run_that_uploads_no_rollouts_is_certified_and_not_exported_again(live, monkeypatch):
    artifacts, exporter, client = live
    post = _post_run_client(monkeypatch)
    exporter.global_config_dict.upload_rollouts = False
    exporter._delivered_rollouts.clear()
    OmegaConf.save(exporter.global_config_dict, artifacts.resolved_config_yaml_fpath)
    exporter.export_config()
    run_health_checks(
        artifacts.rollouts_jsonl_fpath,
        workers=1,
        run_coverage=[RolloutStore.read(artifacts.rollouts_jsonl_fpath).coverage()],
    )
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt["run_id"] == "live-run" and receipt["uploaded_rollouts"] is False
    assert client.status == "FINISHED" and post.runs == []  # one MLflow run, the live one


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
def test_live_finalizer_rejects_changed_completion_identity(live, monkeypatch, fault):
    artifacts, exporter, client = live
    post = _post_run_client(monkeypatch)
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
        from nemo_gym.rollouts.rollout_recovery import manifest_path_for

        manifest = RunManifest.model_validate_json(manifest_path_for(output).read_bytes())
        inputs = [orjson.loads(line) for line in artifacts.materialized_inputs_jsonl_fpath.read_bytes().splitlines()]
        with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=True) as store:
            row = store.selected("success")[0]
            row["_ng_attempt_index"] = 1
            store.record_dispatch(row)
        assert RolloutStore.read(output).coverage()["complete"] is False
    _finalize_live_exporters(output)
    if fault == "wrong_backend_run":
        # Verified saved artifacts whose live run cannot be certified get a separate post-run export.
        assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["run_id"] == "post-run"
    else:
        assert not artifacts.export_done_fpath("mlflow").exists() and not post.runs


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
        assert trajectory["model_calls"][0]["response_metadata"] == {"status_code": 200, "model_operations": None}
        assert "request" not in trajectory["model_calls"][0]
        assert "response" not in trajectory["model_calls"][0]
        assert "conversation" not in trajectory["invocations"][0]
        assert uploaded["reward"] == original["reward"]


def test_live_receipt_rejects_verifier_observations_not_in_acknowledged_rows(live, monkeypatch):
    artifacts, exporter, client = live
    _post_run_client(monkeypatch)
    output = artifacts.rollouts_jsonl_fpath
    rows = [orjson.loads(line) for line in output.read_bytes().splitlines()]
    rows[0]["ng_trajectory"]["verifier_invocations"].append({"invocation_id": "late-verifier", "status": "completed"})
    output.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    run_health_checks(output, workers=1, run_coverage=[RolloutStore.read(output).coverage()])
    _finalize_live_exporters(output)
    # The live run is not certified; the post-run export sends the saved rows, late observation included.
    assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["run_id"] == "post-run"


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
    from nemo_gym.models.token_capture.config import TokenCaptureConfig as TokenIdCaptureConfig

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
    after = _artifact_version(artifacts)
    assert {k: v for k, v in after.items() if k != "aggregate_metrics_json_fpath"} == {
        k: v for k, v in version.items() if k != "aggregate_metrics_json_fpath"
    }
    assert artifacts.export_done_fpath("mlflow").exists()
    saved = OmegaConf.to_container(OmegaConf.load(artifacts.resolved_config_yaml_fpath), resolve=False)
    assert client.artifacts[exporter.CONFIG_ARTIFACT_FILE] == saved
    assert saved["token_id_capture"] == before["token_id_capture"]
    assert OmegaConf.to_container(exporter.global_config_dict, resolve=False) == before
    assert _collected_mlflow(artifacts)["receipt"]["current"] is True


@pytest.mark.parametrize("field,value", [("enabled", True), ("sink", "other:Sink"), ("rebuild_response", True)])
def test_live_receipt_rejects_changed_capture_controls(live, monkeypatch, field, value):
    artifacts, exporter, client = live
    _post_run_client(monkeypatch)
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
    # The live run is not certified; the post-run export sends the final saved config.
    assert orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())["run_id"] == "post-run"


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


def test_postrun_exports_the_same_bound_health_metric_snapshot(live, monkeypatch):
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, live_exporter, _ = live
    scientific = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    values = _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    client = Client()
    monkeypatch.setattr(module, "MlflowClient", lambda **kwargs: client)
    exporter = module.MLflowExporter(live_exporter.global_config_dict)
    result = exporter.export(artifacts)
    assert result.state.value == "succeeded"
    assert all(client.metrics[key] == value for key, value in values.items())
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    payload = changed[0].pop("_ng_run_health_metrics")
    assert changed == scientific
    summary = orjson.loads((artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json").read_bytes())
    assert payload == summary["run"]["stats"]["acceptance_health"]
    assert payload["metrics"] == values
    assert _collected_mlflow(artifacts)["receipt"]["current"] is True
    # The metric coverage fact is exported with the other health fields; its source is a key, as metrics are numbers.
    assert values["health/metric_coverage/expected"] == 2
    assert client.metrics["health/metric_coverage/trials"] == values["health/metric_coverage/trials"]
    assert [key for key in values if key.startswith("health/metric_coverage/source/")] == [
        "health/metric_coverage/source/saved_rows"
    ]


@pytest.mark.parametrize("mode", [0o664, 0o640])
def test_postrun_health_rewrite_keeps_each_files_mode(live, mode):
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, _, _ = live
    summary = artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json"
    for path in (artifacts.aggregate_metrics_json_fpath, summary):
        path.chmod(mode)
    _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    assert [stat.S_IMODE(p.stat().st_mode) for p in (artifacts.aggregate_metrics_json_fpath, summary)] == [mode, mode]


def test_postrun_exports_a_benchmark_declared_metric_coverage(live):
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, _, _ = live
    entries = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    entries[0]["agent_metrics"] |= {"metric_coverage/trials": 1, "metric_coverage/expected": 2}
    artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(entries))
    values = _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    assert (values["health/metric_coverage/trials"], values["health/metric_coverage/expected"]) == (1, 2)
    assert values["health/metric_coverage/source/benchmark"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "stale", "nan", "non_health_key"])
def test_postrun_rejects_unbound_or_malformed_health_metrics(live, fault):
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, exporter, _ = live
    _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    entries = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    payload = entries[0]["_ng_run_health_metrics"]
    if fault == "duplicate":
        entries.append(dict(entries[0]))
    elif fault == "stale":
        payload["source_artifacts"]["rollouts_jsonl_fpath"] = "changed"
    elif fault == "nan":
        payload["metrics"]["health/trial/known"] = float("nan")
    elif fault == "non_health_key":
        payload["metrics"]["agent/mean/reward"] = 1
    artifacts.aggregate_metrics_json_fpath.write_text(json.dumps(entries))
    result = module.MLflowExporter(exporter.global_config_dict).export(artifacts)
    assert result.state.value == "failed"
    assert not artifacts.export_done_fpath("mlflow").exists()


def test_current_unknown_health_replaces_stale_healthy_metadata_without_changing_scores(live):
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, _, _ = live
    entries = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    original = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    entries[0]["_ng_run_health_metrics"] = {
        "metrics": {"health/trial/known": 1, "health/trial/passed": 1, "health/trial/failure_rate": 0}
    }
    artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(entries))
    values = _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    assert values["health/trial/known"] == 0
    assert values["health/trial/passed"] == 0 and "health/trial/failure_rate" not in values
    assert values["health/trial/expected"] == 2
    assert values["health/trial/failed_lower_bound"] == 0
    assert values["health/trial/failure_rate_lower_bound"] == 0
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    changed[0].pop("_ng_run_health_metrics")
    assert changed == original


def test_empty_aggregate_does_not_invent_an_agent_or_health_metric(live):
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, _, _ = live
    artifacts.aggregate_metrics_json_fpath.write_text("[]")
    before = artifacts.aggregate_metrics_json_fpath.read_bytes()
    assert _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath) == {}
    assert artifacts.aggregate_metrics_json_fpath.read_bytes() == before


def test_artifact_change_during_health_derivation_leaves_aggregate_and_summary_untouched(live, monkeypatch):
    import subprocess

    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, _, _ = live
    health = artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json"
    before = (artifacts.aggregate_metrics_json_fpath.read_bytes(), health.read_bytes())
    original = subprocess.run

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        artifacts.rollouts_jsonl_fpath.write_bytes(artifacts.rollouts_jsonl_fpath.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(subprocess, "run", changed)
    with pytest.raises(RuntimeError, match="changed"):
        _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    assert (artifacts.aggregate_metrics_json_fpath.read_bytes(), health.read_bytes()) == before
    assert not artifacts.export_done_fpath("mlflow").exists()


@pytest.mark.parametrize("source", ["quality", "manifest", "journal"])
def test_existing_receipt_does_not_certify_changed_health_inputs_or_resend(live, source):
    artifacts, exporter, client = live
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    marker = artifacts.export_done_fpath("mlflow")
    receipt = marker.read_bytes()
    before_backend = (dict(client.metrics), dict(client.tags), dict(client.artifacts))
    if source == "quality":
        path = artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json"
        summary = orjson.loads(path.read_bytes())
        summary["run"]["stats"]["responses"]["faulted"] += 1
        path.write_bytes(orjson.dumps(summary))
    else:
        suffix = "manifest.json" if source == "manifest" else "attempts.jsonl"
        path = artifacts.rollouts_jsonl_fpath.with_name(f"rollouts_{suffix}")
        path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n")
    result = module.MLflowExporter(exporter.global_config_dict).export(artifacts)
    assert result.state.value == "failed"
    assert marker.read_bytes() == receipt
    assert _collected_mlflow(artifacts)["receipt"]["current"] is False
    assert (client.metrics, client.tags, client.artifacts) == before_backend


def test_unknown_snapshot_invalidates_interpretation_of_prior_backend_pass_metrics(live):
    artifacts, exporter, client = live
    exporter.export_metrics({"health/trial/known": 1, "health/trial/passed": 1, "health/trial/failure_rate": 0})
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    assert client.metrics["health/trial/known"] == 0
    # Unknown resets the current pass flag without fabricating a failure count or rate.
    assert client.metrics["health/trial/passed"] == 0
    payload = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())[0]["_ng_run_health_metrics"]["metrics"]
    assert payload["health/trial/known"] == 0
    assert payload["health/trial/passed"] == 0 and "health/trial/failure_rate" not in payload


@pytest.mark.parametrize("health_first", [True, False])
def test_reserved_health_metric_names_cannot_collide_with_scientific_metrics(health_first):
    from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics

    health = {
        "agent_ref": {"name": "policy"},
        "_ng_run_health_metrics": {
            "source_artifacts": {},
            "metrics": {"health/response/known": 0, "health/response/passed": 0},
        },
    }
    scientific = {"agent_ref": {"name": "health"}, "agent_metrics": {"response/passed": 1}}
    entries = [health, scientific] if health_first else [scientific, health]
    with pytest.raises(ValueError, match="collide"):
        flatten_aggregate_metrics(entries, source_artifacts={})


@pytest.mark.parametrize("raw", ['{"run":{"stats":{}},"other":NaN}', "{invalid", "[]"])
def test_existing_unhashable_health_summary_is_not_treated_as_absent(live, raw):
    from nemo_gym.artifacts.exporters.base import _artifact_version
    from nemo_gym.artifacts.exporters.metrics import health_source_artifacts

    artifacts, _, _ = live
    path = artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="cannot be bound"):
        health_source_artifacts(artifacts, _artifact_version(artifacts))


@pytest.mark.parametrize("case", ["healthy", "wrong", "recovered", "invalid"])
def test_health_export_projects_validated_collector_facts_without_changing_scores(live, monkeypatch, case):
    """Explicit collector projection boundary; real classifier, persisted metadata and SDK delivery."""
    import nemo_gym.orchestration.recovery as recovery
    from nemo_gym.cli.eval import _prepare_run_health_metrics
    from tests.unit_tests.orchestration.test_llm_judge_trial_coverage import _recovered_fixture
    from tests.unit_tests.orchestration.test_tool_trial_coverage import fixture

    artifacts, exporter, client = live
    if case in {"recovered", "invalid"}:
        collected, _, grade = _recovered_fixture()
        if case == "invalid":
            grade["outcome"] = "unknown"
    else:
        collected = fixture(case)
    # Preserve the two upstream counters and all their actual coverage metadata.
    collected.quality["observed_upstream_recovery"] = {
        "observed_attempt_errors": 0,
        "observed_failed_operations": 0,
        "recovered_attempt_errors": 0,
        "recovered_operations": 0,
        "operations_without_terminal_observation": 0,
    }
    before = collected.model_dump()
    scientific = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    monkeypatch.setattr(recovery, "select_managed_evidence", lambda *_: collected)
    values = _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    assert collected.model_dump() == before
    assert values["health/response/known"] == 1
    assert values["health/trial/known"] == 1
    # An unknown grading outcome is a counted judge_unparseable fault; one over the gate limit is a known lower bound.
    exact = "" if case != "invalid" else "_lower_bound"
    assert values["health/trial/counts_known"] == int(case != "invalid")
    assert values["health/trial/expected"] == 1
    assert values[f"health/trial/failed{exact}"] == int(case == "invalid")
    assert values["health/trial/passed"] == int(case != "invalid")
    assert values[f"health/trial/failure_rate{exact}"] == int(case == "invalid")
    assert values["health/upstream_recovery/known"] == 1
    assert values["health/grading_recovery/judge_recovered_attempts"] == int(case == "recovered")
    assert values["health/grading_recovery/judge_recovered_operations"] == int(case == "recovered")
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    changed[0].pop("_ng_run_health_metrics")
    assert changed == scientific
    exporter.export_metrics(values)
    assert all(client.metrics[key] == value for key, value in values.items())


@pytest.mark.parametrize("path", ["live", "postrun"])
def test_health_collision_refuses_before_artifact_or_backend_mutation(live, monkeypatch, path):
    from nemo_gym.cli import eval as cli

    artifacts, exporter, client = live
    entries = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    entries[0]["agent_ref"]["name"] = "health"
    entries[0]["agent_metrics"] = {"trial/passed": 123}
    artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(entries))
    exporter.export_metrics({"health/trial/passed": 123})
    protected = [
        artifacts.rollouts_jsonl_fpath,
        artifacts.aggregate_metrics_json_fpath,
        artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json",
    ]
    before = {str(p): p.read_bytes() for p in protected}
    backend = (dict(client.metrics), dict(client.tags), dict(client.artifacts))
    if path == "live":
        _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    else:
        config = OmegaConf.create(OmegaConf.to_container(exporter.global_config_dict))
        config.update({"exporter": "mlflow", "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath)})
        monkeypatch.setattr(cli, "get_global_config_dict", lambda **kwargs: config)
        with pytest.raises(ValueError, match="collide"):
            cli.export_run()
    assert not artifacts.export_done_fpath("mlflow").exists()
    assert client.metrics["health/trial/passed"] == 123
    assert backend == (client.metrics, client.tags, client.artifacts)
    assert before == {str(p): p.read_bytes() for p in protected}


@pytest.mark.parametrize("aggregate", [None, b"{invalid", b"null", b"{}"])
def test_unreadable_aggregate_does_not_write_health_or_deliver_metrics(live, aggregate):
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    artifacts, _, client = live
    path = artifacts.aggregate_metrics_json_fpath
    if aggregate is None:
        path.unlink()
    else:
        path.write_bytes(aggregate)
    health = artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json"
    before = health.read_bytes()
    backend = (dict(client.metrics), dict(client.tags), dict(client.artifacts))

    assert _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath) == {}

    assert (path.read_bytes() if path.exists() else None) == aggregate
    assert health.read_bytes() == before
    assert backend == (client.metrics, client.tags, client.artifacts)
    assert not artifacts.export_done_fpath("mlflow").exists()


def test_live_health_delivery_cannot_certify_a_changed_summary(live, monkeypatch, caplog):
    artifacts, exporter, client = live
    health = artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json"
    rows = artifacts.rollouts_jsonl_fpath.read_bytes()
    scientific = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    original = type(exporter).export_metrics

    def change_summary(self, metrics):
        original(self, metrics)
        health.write_bytes(health.read_bytes() + b"\n")

    monkeypatch.setattr(type(exporter), "export_metrics", change_summary)
    _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)

    assert "Final run artifacts changed before live finalization" in caplog.text
    assert not artifacts.export_done_fpath("mlflow").exists()
    assert client.status == "RUNNING"
    assert artifacts.rollouts_jsonl_fpath.read_bytes() == rows
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    changed[0].pop("_ng_run_health_metrics")
    assert changed == scientific
    assert client.metrics["agent/mean/reward"] == 0.5


@pytest.mark.parametrize("path", ["live", "postrun"])
@pytest.mark.parametrize(
    "scientific_key,canonical_key",
    [
        ("trial//passed", "health/trial/passed"),
        ("/trial/passed", "health/trial/passed"),
        ("trial/passed/", "health/trial/passed"),
        ("grading?recovery/known", "health/grading_recovery/known"),
        ("upstream?recovery/known", "health/upstream_recovery/known"),
    ],
)
def test_canonical_health_collision_refuses_before_any_mutation(
    live, monkeypatch, path, scientific_key, canonical_key
):
    from nemo_gym.cli import eval as cli

    artifacts, exporter, client = live
    entries = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    entries[0]["agent_ref"]["name"] = "health"
    entries[0]["agent_metrics"] = {scientific_key: 123}
    artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(entries))
    exporter.export_metrics({f"health/{scientific_key}": 123})
    assert client.metrics[canonical_key] == 123
    protected = [
        artifacts.rollouts_jsonl_fpath,
        artifacts.aggregate_metrics_json_fpath,
        artifacts.rollouts_jsonl_fpath.parent / "quality_summary.json",
    ]
    before = {str(p): p.read_bytes() for p in protected}
    backend = (dict(client.metrics), dict(client.tags), dict(client.artifacts), client.status)

    if path == "live":
        _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    else:
        config = OmegaConf.create(OmegaConf.to_container(exporter.global_config_dict))
        config.update({"exporter": "mlflow", "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath)})
        monkeypatch.setattr(cli, "get_global_config_dict", lambda **kwargs: config)
        with pytest.raises(ValueError, match="collide"):
            cli.export_run()

    assert client.metrics[canonical_key] == 123
    assert backend == (client.metrics, client.tags, client.artifacts, client.status)
    assert before == {str(p): p.read_bytes() for p in protected}
    assert not artifacts.export_done_fpath("mlflow").exists()


@pytest.mark.parametrize("path", ["live", "postrun"])
@pytest.mark.parametrize("scientific_key", ["score//value", "trial/./passed"])
def test_noncolliding_sanitized_scientific_metric_still_exports(live, monkeypatch, path, scientific_key):
    from nemo_gym.cli import eval as cli

    artifacts, exporter, client = live
    entries = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    entries[0]["agent_ref"]["name"] = "health"
    entries[0]["agent_metrics"] = {scientific_key: 123}
    artifacts.aggregate_metrics_json_fpath.write_bytes(orjson.dumps(entries))
    rows = artifacts.rollouts_jsonl_fpath.read_bytes()
    exporter.export_metrics({f"health/{scientific_key}": 123})
    canonical_key = module._sanitize_key(f"health/{scientific_key}")

    if path == "live":
        _finalize_live_exporters(artifacts.rollouts_jsonl_fpath)
    else:
        config = OmegaConf.create(OmegaConf.to_container(exporter.global_config_dict))
        config.update({"exporter": "mlflow", "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath)})
        monkeypatch.setattr(cli, "get_global_config_dict", lambda **kwargs: config)
        cli.export_run()

    assert client.metrics[canonical_key] == 123
    assert client.metrics["health/trial/passed"] == 0
    assert client.status == "FINISHED"
    assert artifacts.export_done_fpath("mlflow").exists()
    assert artifacts.rollouts_jsonl_fpath.read_bytes() == rows
    changed = orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes())
    changed[0].pop("_ng_run_health_metrics")
    assert changed == entries


@pytest.mark.parametrize("case", ["settled_zero", "missing"])
def test_the_exported_trial_rate_uses_the_gates_denominator_and_reports_omissions(live, monkeypatch, case):
    """49 scored trials, then one settled zero (or one missing row), and 50 rows an accepted stage omitted."""
    import nemo_gym.orchestration.recovery as recovery
    from nemo_gym.artifacts.run_artifacts import ACCEPTED_STAGE_OMISSION
    from nemo_gym.cli.eval import _prepare_run_health_metrics
    from tests.unit_tests.orchestration.test_evaluation_health import _run

    artifacts, _, _ = live
    collected = _run(49, settled={"model_transport_error": 1} if case == "settled_zero" else None)
    if case == "missing":
        collected.materialized.append({"_ng_task_index": 49, "_ng_rollout_index": 0, "_ng_collected_task_id": "49"})
    for task in range(50, 100):
        collected.materialized.append(
            {
                "_ng_task_index": task,
                "_ng_rollout_index": 0,
                "_ng_collected_task_id": str(task),
                ACCEPTED_STAGE_OMISSION: {"stage_index": 1, "tolerate_unresolved": True},
            }
        )
    monkeypatch.setattr(recovery, "select_managed_evidence", lambda *_: collected)
    decision = recovery.decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert decision.counts.get("omitted") == 50
    passed, reason = recovery.trial_health(decision, collected)
    values = _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
    assert values["health/trial/expected"] == 50 and values["health/trial/omitted"] == 50
    assert values["health/trial/counts_known"] == 0
    if case == "settled_zero":
        # The gate rejects 1 of 50 (2%); the export reports the same population, not 1 of 100.
        assert passed is False and "evaluation-side faults: 1 of 50 (2.00%) is not below 2%" in reason
        assert (values["health/trial/known"], values["health/trial/passed"]) == (1, 0)
        assert values["health/trial/failed_lower_bound"] == 1
        assert values["health/trial/failure_rate_lower_bound"] == 0.02
        assert values["health/trial_failures/request"] == 1 and values["health/trial_failures/rate"] == 0.02
    else:
        # Not evaluated: the known count is a lower bound over the same denominator.
        assert passed is None and reason.startswith("trial: not evaluated")
        assert (values["health/trial/known"], values["health/trial/passed"]) == (0, 0)
        assert values["health/trial/failed_lower_bound"] == 0
        assert values["health/trial/failure_rate_lower_bound"] == 0
        assert values["health/trial_failures/rate"] == 0 and values["health/response_failures/rate"] == 0
