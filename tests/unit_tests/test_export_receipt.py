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
"""The post-run export's version-1 completion receipt, as BaseExporter writes it and recovery reads it."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, Optional

import orjson
import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.artifacts.exporters.base import BaseExporter, ExportState
from nemo_gym.artifacts.run_artifacts import RunArtifacts
from nemo_gym.orchestration.recovery import COLLECT_SCRIPT, CollectedEvidence, mlflow_errors


ARTIFACT_NAMES = (
    "rollouts_jsonl_fpath",
    "resolved_config_yaml_fpath",
    "aggregate_metrics_json_fpath",
    "failures_jsonl_fpath",
    "materialized_inputs_jsonl_fpath",
)


class ReceiptBackend(BaseExporter):
    """A backend whose run is named like the MLflow exporter's, so recovery's collector reads its receipt."""

    name: ClassVar[str] = "mlflow"
    fail_on: ClassVar[Optional[str]] = None
    on_metrics: ClassVar[Any] = None

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.calls: list[str] = []

    def setup(self) -> None:
        self.calls.append("setup")
        self._record_run_id("run-1")

    def teardown(self) -> None:
        self.calls.append("teardown")

    def _log(self, what: str) -> None:
        if self.fail_on == what:
            raise RuntimeError(f"backend rejected the {what}")
        self.calls.append(what)

    def _log_config(self, config_dict: DictConfig) -> None:
        self._log("config")

    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self._log("metrics")
        hook = type(self).on_metrics  # a plain function, not bound to the backend
        if hook is not None:
            hook()

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        self._log("rollouts")


@pytest.fixture(autouse=True)
def _reset() -> None:
    ReceiptBackend.fail_on = None
    ReceiptBackend.on_metrics = None


def _run(tmp_path: Path) -> RunArtifacts:
    art = tmp_path / "artifacts"
    art.mkdir()
    artifacts = RunArtifacts(art / "rollouts.jsonl")
    rows = [
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0},
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "reward": 0.0},
    ]
    artifacts.rollouts_jsonl_fpath.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
    artifacts.materialized_inputs_jsonl_fpath.write_bytes(
        b"".join(orjson.dumps({"_ng_task_index": t, "_ng_rollout_index": 0}) + b"\n" for t in range(2))
    )
    artifacts.failures_jsonl_fpath.write_bytes(b"")
    artifacts.aggregate_metrics_json_fpath.write_bytes(
        orjson.dumps([{"agent_ref": {"name": "a"}, "agent_metrics": {"mean/reward": 0.5}, "key_metrics": {}}])
    )
    artifacts.resolved_config_yaml_fpath.write_text("model: m\n")
    stamp = artifacts.rollouts_jsonl_fpath.stat().st_mtime + 10
    os.utime(artifacts.aggregate_metrics_json_fpath, (stamp, stamp))
    return artifacts


def _backend(artifacts: RunArtifacts, **overrides: Any) -> ReceiptBackend:
    return ReceiptBackend(OmegaConf.create({"rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath), **overrides}))


def _digests(artifacts: RunArtifacts) -> dict[str, Optional[str]]:
    paths = {name: getattr(artifacts, name) for name in ARTIFACT_NAMES}
    return {name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for name, p in paths.items()}


def _collected_mlflow(artifacts: RunArtifacts) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, "-c", COLLECT_SCRIPT, str(artifacts.rollouts_jsonl_fpath.parent.parent)],
        capture_output=True,
        text=True,
        check=True,
    )
    return CollectedEvidence.model_validate_json(out.stdout).mlflow


def test_a_complete_export_writes_a_version_1_receipt_of_what_it_sent(tmp_path: Path) -> None:
    artifacts = _run(tmp_path)
    result = _backend(artifacts).export(artifacts)
    assert result.state is ExportState.SUCCEEDED
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt == {
        "version": 1,
        "run_id": "run-1",
        "artifacts": _digests(artifacts),
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    # Recovery's collector binds the receipt to the current files; only FINISHED for the receipt's own run passes.
    local = _collected_mlflow(artifacts)
    assert local["receipt"] == {
        "run_id": "run-1",
        "current": True,
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    assert mlflow_errors(local, "FINISHED", observed_run_id="run-1") == []
    assert mlflow_errors(local, "FINISHED", observed_run_id="another-run")
    assert mlflow_errors(local, "FAILED", observed_run_id="run-1")
    assert mlflow_errors(local, None, observed_run_id=None)


def test_a_repeat_of_an_unchanged_run_is_skipped_by_its_receipt(tmp_path: Path) -> None:
    artifacts = _run(tmp_path)
    assert _backend(artifacts).export(artifacts).state is ExportState.SUCCEEDED
    backend = _backend(artifacts)
    result = backend.export(artifacts)
    assert result.state is ExportState.SKIPPED and result.detail == {"reason": "already exported", "run_id": "run-1"}
    assert backend.calls == []


def test_a_legacy_run_id_marker_is_not_trusted_or_upgraded(tmp_path: Path) -> None:
    artifacts = _run(tmp_path)
    marker = artifacts.export_done_fpath("mlflow")
    marker.write_text("run-0")
    backend = _backend(artifacts)
    result = backend.export(artifacts)
    assert result.state is ExportState.FAILED and "legacy or malformed export receipt" in result.error
    assert marker.read_text() == "run-0" and backend.calls == []
    local = _collected_mlflow(artifacts)
    assert local["receipt"] == {"run_id": "run-0", "current": None}
    assert mlflow_errors(local, "FINISHED", observed_run_id="run-0")


@pytest.mark.parametrize("fail_on", ["config", "metrics", "rollouts"])
def test_a_partial_export_writes_no_receipt(tmp_path: Path, fail_on: str) -> None:
    artifacts = _run(tmp_path)
    ReceiptBackend.fail_on = fail_on
    result = _backend(artifacts).export(artifacts)
    assert result.state is ExportState.FAILED
    assert not artifacts.export_done_fpath("mlflow").exists()


def test_an_export_without_rollouts_cannot_claim_row_delivery(tmp_path: Path) -> None:
    artifacts = _run(tmp_path)
    backend = _backend(artifacts, upload_rollouts=False)
    assert backend.export(artifacts).state is ExportState.SUCCEEDED
    assert "rollouts" not in backend.calls
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    assert receipt["uploaded_rollouts"] is False
    errors = mlflow_errors(_collected_mlflow(artifacts), "FINISHED", observed_run_id="run-1")
    assert errors and all("upload" in e for e in errors)


def test_changed_content_after_export_fails_closed_and_keeps_the_receipt(tmp_path: Path) -> None:
    artifacts = _run(tmp_path)
    assert _backend(artifacts).export(artifacts).state is ExportState.SUCCEEDED
    marker = artifacts.export_done_fpath("mlflow")
    receipt_bytes = marker.read_bytes()
    path = artifacts.rollouts_jsonl_fpath
    before = path.stat()
    content = path.read_bytes()
    changed = content.replace(b'"reward":0.0', b'"reward":1.0')
    assert changed != content and len(changed) == len(content)
    path.write_bytes(changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.utime(artifacts.aggregate_metrics_json_fpath, (before.st_mtime + 10, before.st_mtime + 10))
    # Same size and timestamp, different content: the receipt no longer describes the run.
    stale = _collected_mlflow(artifacts)
    assert stale["receipt"]["current"] is False
    assert mlflow_errors(stale, "FINISHED", observed_run_id="run-1")
    backend = _backend(artifacts)
    result = backend.export(artifacts)
    assert result.state is ExportState.FAILED and "does not match the current run artifacts" in result.error
    assert backend.calls == []
    assert marker.read_bytes() == receipt_bytes


def test_content_changed_during_export_writes_no_receipt(tmp_path: Path) -> None:
    artifacts = _run(tmp_path)
    ReceiptBackend.on_metrics = lambda: artifacts.failures_jsonl_fpath.write_bytes(b'{"_ng_task_index": 1}\n')
    result = _backend(artifacts).export(artifacts)
    assert result.state is ExportState.FAILED and "changed during export" in result.error
    assert not artifacts.export_done_fpath("mlflow").exists()


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        json.dumps({"version": 2, "run_id": "r", "artifacts": {}}),
        json.dumps({"version": 1, "run_id": "", "artifacts": {}}),
        json.dumps({"version": 1, "run_id": "r", "artifacts": {"rollouts_jsonl_fpath": "x"}}),
    ],
)
def test_a_malformed_receipt_fails_closed(tmp_path: Path, text: str) -> None:
    artifacts = _run(tmp_path)
    artifacts.export_done_fpath("mlflow").write_text(text)
    backend = _backend(artifacts)
    result = backend.export(artifacts)
    assert result.state is ExportState.FAILED and backend.calls == []
    assert artifacts.export_done_fpath("mlflow").read_text() == text


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"version": 1.0},
        {"run_id": "   "},
        {"uploaded_rollouts": None},
        {"uploaded_rollouts": 1},
        {"uploaded_rollouts": "true"},
        "drop uploaded_rollouts",
    ],
)
def test_a_receipt_with_a_coerced_or_missing_field_fails_closed(tmp_path: Path, change: Any) -> None:
    artifacts = _run(tmp_path)
    assert _backend(artifacts).export(artifacts).state is ExportState.SUCCEEDED
    marker = artifacts.export_done_fpath("mlflow")
    receipt = orjson.loads(marker.read_bytes())
    if change == "drop uploaded_rollouts":
        del receipt["uploaded_rollouts"]
    else:
        receipt.update(change)
    marker.write_bytes(orjson.dumps(receipt))
    written = marker.read_bytes()
    backend = _backend(artifacts)
    result = backend.export(artifacts)
    assert result.state is ExportState.FAILED and backend.calls == []
    assert marker.read_bytes() == written
