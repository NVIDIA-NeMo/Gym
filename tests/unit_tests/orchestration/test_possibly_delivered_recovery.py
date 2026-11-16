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
"""Recovery plans a retry for a possibly delivered /run that saved no response, and never for one that did.

Real Store, real stdlib collector and the real auto-resume prologue; only the scheduler is simulated.
"""

import hashlib
import json
import subprocess
import sys

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.orchestration.test_resume_script import _run_prologue
from tests.unit_tests.rollouts.test_possibly_delivered_replay import _lost_connection_row


_PARTIAL = {"object": "response", "id": "resp-kept", "status": "incomplete", "output": []}


def _lost_connection_run(job, *, scored=3, lost=2, lost_with_partial=0):
    """A run whose driver finished with lost /run connections: sidecar rows with or without a partial response."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    total = scored + lost + lost_with_partial
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(total)]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SLURM_JOB_ID", "100")
        with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
            for index, row in enumerate(rows):
                row = dict(row)
                store.allocate_attempt(row)
                if index < scored:
                    store.record_outcome(row | {"reward": 1.0})
                    continue
                failure = _lost_connection_row(
                    **({"_ng_failure_partial_response": _PARTIAL} if index >= scored + lost else {})
                )
                failure["_ng_failure_record"] = failure["_ng_failure_record"] | {
                    "run_id": store.manifest.run_id,
                    "episode_id": {"rollout_id": f"{index}-0", "attempt": 0},
                }
                store.record_outcome(row | failure)
    (job / ".gym_job_chain").write_text("100\n")
    (job / ".gym_accumulated_walltime").write_text("600\n")


def _decide(job):
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
    assert collected.managed_reader_error is None
    return decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)


def _digests(job):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").glob("*") if p.is_file()}


def test_lost_connections_with_no_saved_response_are_planned_as_retries(tmp_path):
    _lost_connection_run(tmp_path)
    decision = _decide(tmp_path)
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 3, "retryable": 2}
    assert [item.attempts for item in decision.evidence if item.state == "retryable"] == [1, 1]


def test_a_lost_connection_that_kept_a_partial_response_is_reported_not_retried(tmp_path):
    _lost_connection_run(tmp_path, lost_with_partial=1)
    decision = _decide(tmp_path)
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 3, "retryable": 2, "unknown": 1}
    assert "retryable 2 (resumed), unknown 1" in decision.reason


def test_the_guard_continues_a_failed_allocation_with_lost_connections_as_an_infra_retry(tmp_path):
    _lost_connection_run(tmp_path)
    before = _digests(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "FAILED", exit_code="75:0", prev_job_id="100")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Saved recovery evidence: 2 identity(ies) need the policy again" in result.stdout
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"
    assert _digests(tmp_path) == before


def test_the_guard_names_the_saved_evidence_when_nothing_is_retryable(tmp_path):
    _lost_connection_run(tmp_path, lost=0, lost_with_partial=1)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "FAILED", exit_code="75:0", prev_job_id="100")
    assert "REACHED_WORK" not in result.stdout
    marker = (tmp_path / ".gym_chain_incomplete").read_text()
    assert "without an infrastructure cause" not in marker
    assert marker.startswith("Saved recovery evidence has no eligible retries; 1 failure(s)")
    assert "retryable 0 (resumed), unknown 1" in marker
    assert marker.rstrip().endswith("(previous job 100 ended FAILED)")
