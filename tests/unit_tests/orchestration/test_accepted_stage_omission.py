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
"""A planned row that an accepted partial stage omitted: complete by the stage's policy, stated, never planned."""

import json
import os
import subprocess
import sys
from typing import Any, ClassVar, Optional

import orjson
import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.artifacts.exporters.base import BaseExporter, ExportState
from nemo_gym.artifacts.run_artifacts import ACCEPTED_STAGE_OMISSION, RunArtifacts
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    RecoveryBudget,
    decide_recovery,
    gate_task_summary,
)


_BUDGET = RecoveryBudget(max_infra_retries=2, infra_retries_used=0, walltime_used_s=0.0, live_job=False)
_MARK = {"stage_index": 1, "tolerate_unresolved": True}
_STATED = "1 omitted by an accepted partial stage (stage_index 1, tolerate_unresolved)"
# The omitted row's last attempt timed out, a failure a resume would otherwise run again.
_TIMEOUT = {
    "_ng_task_index": 2,
    "_ng_rollout_index": 0,
    "_line": 0,
    "_ng_failure_class": "agent_timeout",
    "failure_kind": "agent_timeout",
}


def _planned(marked: bool) -> list[dict]:
    materialized = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(3)]
    if marked:
        materialized[2][ACCEPTED_STAGE_OMISSION] = _MARK
    return materialized


_RESULTS = [{"_ng_task_index": i, "_ng_rollout_index": 0, "reward": 1.0} for i in range(2)]


def _decide(marked: bool):
    return decide_recovery(materialized=_planned(marked), results=_RESULTS, failures=[_TIMEOUT], budget=_BUDGET)


def _gate(decision) -> dict:
    return gate_task_summary(
        decision, score=1.0, summary_valid=True, summary_reasons=[], health_passed=True, health_reasons=[]
    )


def test_an_accepted_omission_is_complete_typed_and_never_planned_again():
    decision = _decide(marked=True)
    # Never planned again, although its recorded timeout is retryable, and neither scored nor a failure.
    assert (decision.action.value, decision.complete) == ("noop", True), decision.reason
    assert decision.counts == {"scored": 2, "omitted": 1}
    [item] = [item for item in decision.evidence if item.identity == (2, 0)]
    assert (item.state.value, item.failure_kind, item.reason) == (
        "omitted",
        ACCEPTED_STAGE_OMISSION,
        "(stage_index 1, tolerate_unresolved)",
    )
    assert decision.reason.endswith(f"or omitted by an accepted partial stage: {_STATED}")


def test_without_the_mark_the_same_row_is_planned_again():
    decision = _decide(marked=False)
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 2, "retryable": 1}


def test_the_gate_passes_states_the_omission_and_counts_it_on_its_own():
    task = _gate(_decide(marked=True))
    # Stated in the gate text, and counted on its own beside scored and expected.
    assert task["status"] == "SUCCESS", task["errors"]
    assert (task["expected_samples"], task["scored_samples"], task["omitted_samples"]) == (3, 2, 1)
    assert _STATED in task["warnings"]


@pytest.mark.parametrize("marked", [True, False])
def test_the_collector_keeps_the_mark_from_the_planned_inputs(tmp_path, marked):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "prompt": "not collected"}
    if marked:
        row[ACCEPTED_STAGE_OMISSION] = _MARK
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text(json.dumps(row) + "\n")
    (artifacts / "rollouts.jsonl").write_text("")
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    [collected] = CollectedEvidence.model_validate_json(raw.stdout).materialized
    assert collected.get(ACCEPTED_STAGE_OMISSION) == (_MARK if marked else None)
    assert "prompt" not in collected


class _Backend(BaseExporter):
    """A backend that accepts every call and names its run, so the export writes its receipt."""

    name: ClassVar[str] = "mlflow"

    def setup(self) -> None:
        self._record_run_id("run-1")

    def teardown(self) -> None:
        pass

    def _log_config(self, config_dict: DictConfig) -> None:
        pass

    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        pass


def test_the_export_receipt_carries_the_gate_count(tmp_path):
    (tmp_path / "artifacts").mkdir()
    artifacts = RunArtifacts(tmp_path / "artifacts" / "rollouts.jsonl")
    planned = _planned(marked=True)
    artifacts.rollouts_jsonl_fpath.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in _RESULTS))
    artifacts.materialized_inputs_jsonl_fpath.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in planned))
    artifacts.failures_jsonl_fpath.write_bytes(b"")
    artifacts.aggregate_metrics_json_fpath.write_bytes(
        orjson.dumps([{"agent_ref": {"name": "a"}, "agent_metrics": {"mean/reward": 1.0}, "key_metrics": {}}])
    )
    artifacts.resolved_config_yaml_fpath.write_text("model: m\n")
    stamp = artifacts.rollouts_jsonl_fpath.stat().st_mtime + 10
    os.utime(artifacts.aggregate_metrics_json_fpath, (stamp, stamp))
    exporter = _Backend(OmegaConf.create({"rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath)}))
    assert exporter.export(artifacts).state is ExportState.SUCCEEDED
    receipt = orjson.loads(artifacts.export_done_fpath("mlflow").read_bytes())
    gate = _gate(decide_recovery(materialized=planned, results=_RESULTS, failures=[], budget=_BUDGET))
    # The receipt carries the gate's own number; a run without omissions keeps the receipt as it was.
    assert receipt["accepted_stage_omissions"] == gate["omitted_samples"] == 1
