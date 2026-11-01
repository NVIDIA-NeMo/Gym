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
"""A scheduler cut (walltime or preemption) leaves reservations without rows; recovery resumes them.

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
    allocation_cut_from_accounting,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_resume_script import _run_prologue


def _cut_run(job, *, scored=2, in_flight=2, not_started=2, masked=0, unowned=0, allocation="100"):
    """A run cut while ``in_flight`` reservations had no row; ``masked`` results and ``unowned`` failures."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    total = scored + in_flight + not_started + masked + unowned
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(total)]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SLURM_JOB_ID", allocation)
        with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
            index = 0
            for _ in range(scored):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"reward": 1.0})
                index += 1
            for _ in range(masked):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"reward": 0.0, "mask_sample": True, "evaluation_completed": False})
                index += 1
            for _ in range(unowned):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"_ng_failure_class": "unowned_class", "failure_kind": "not_registered"})
                index += 1
            for _ in range(in_flight):
                # Reserved and dispatched; the allocation ended before any outcome was saved.
                store.record_dispatch(dict(rows[index]))
                index += 1
    (job / ".gym_job_chain").write_text(f"{allocation}\n")
    (job / ".gym_accumulated_walltime").write_text("14400\n")
    return artifacts / "rollouts.jsonl"


def _decide(job, *, allocation_cut, live_job=False):
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
    assert collected.managed_reader_error is None
    return collected, decide_from_collected(
        collected, max_infra_retries=2, max_walltime_s=None, live_job=live_job, allocation_cut=allocation_cut
    )


def _digests(job):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").glob("*") if p.is_file()}


def test_in_flight_reservations_at_a_scheduler_cut_resume_with_never_started_rows(tmp_path):
    _cut_run(tmp_path)
    before = _digests(tmp_path)
    collected, decision = _decide(tmp_path, allocation_cut=True)
    assert collected.coverage_allocation_id == "100"
    assert decision.action.value == "resubmit"
    assert decision.counts == {"scored": 2, "missing": 4}
    assert decision.reason.startswith("4 identity(ies) need the policy again")
    assert _digests(tmp_path) == before


def test_without_a_scheduler_cut_reservations_stay_unknown_and_the_stop_names_three_populations(tmp_path):
    _cut_run(tmp_path)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "stop"
    assert decision.counts == {"scored": 2, "unknown": 2, "missing": 2}
    assert "a resume would run 2 of them again" in decision.reason
    assert (
        "retryable 2 (resumed), unknown 2 (2 reservation_without_outcome) (reported, not retried), "
        "retained 2 (saved rows kept)"
    ) in decision.reason


def test_a_scheduler_cut_is_not_applied_while_a_job_of_the_run_is_live(tmp_path):
    _cut_run(tmp_path)
    _, decision = _decide(tmp_path, allocation_cut=True, live_job=True)
    assert decision.action.value == "stop"
    assert decision.counts["unknown"] == 2


def test_unknown_rows_a_resume_never_reruns_do_not_block_never_started_rows(tmp_path):
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=3, masked=2)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "resubmit"
    assert decision.counts == {"scored": 1, "unknown": 2, "missing": 3}
    assert decision.reason == (
        "3 identity(ies) need the policy again: a Gym resume runs only missing and retryable ones; "
        "retryable 3 (resumed), unknown 2 (2 result_without_usable_grade) (reported, not retried), "
        "retained 1 (saved rows kept)"
    )


def test_unknown_failures_a_resume_would_rerun_still_stop(tmp_path):
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=1, unowned=1)
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert decision.action.value == "stop"
    assert "nobody owns them and a resume would run 1 of them again" in decision.reason
    assert "unknown 1 (1 not_registered)" in decision.reason


def test_only_unknown_rows_left_stop_with_the_populations(tmp_path):
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=0, masked=2)
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert decision.action.value == "stop"
    assert decision.reason.endswith(
        "retryable 0 (resumed), unknown 2 (2 result_without_usable_grade) (reported, not retried), "
        "retained 1 (saved rows kept)"
    )


@pytest.mark.parametrize(
    "accounting,allocation,cut",
    [
        ("100|TIMEOUT|14400\n101|FAILED|90", "100", True),
        ("100|PREEMPTED|300", "100", True),
        ("100|CANCELLED by 7|300", "100", False),
        ("100|FAILED|300", "100", False),
        ("100|TIMEOUT|14400", "101", False),
        ("100|TIMEOUT|14400", None, False),
        ("malformed", "100", False),
    ],
)
def test_the_cut_is_read_from_the_allocation_that_last_wrote_the_store(accounting, allocation, cut):
    assert allocation_cut_from_accounting(accounting, allocation) is cut


@pytest.mark.parametrize("state,debit", [("TIMEOUT", False), ("PREEMPTED", True)])
def test_the_guard_continues_a_cut_allocation_from_its_saved_rows(tmp_path, state, debit):
    _cut_run(tmp_path)
    before = _digests(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), state, exit_code="0:15", prev_job_id="100")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Saved recovery evidence: 4 identity(ies) need the policy again" in result.stdout
    assert (tmp_path / ".gym_infra_retries").exists() is debit
    assert not (tmp_path / ".gym_chain_incomplete").exists()
    assert _digests(tmp_path) == before


def test_the_guard_stop_names_the_three_populations_when_the_cut_is_not_the_store_writer(tmp_path):
    _cut_run(tmp_path, allocation="99")
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "TIMEOUT", exit_code="0:15", prev_job_id="100")
    assert "REACHED_WORK" not in result.stdout
    marker = (tmp_path / ".gym_chain_incomplete").read_text()
    assert marker.startswith("Saved recovery evidence has no eligible retries; 2 failure(s) have no registered kind")
    assert "retryable 2 (resumed), unknown 2 (2 reservation_without_outcome)" in marker
    assert marker.rstrip().endswith("(previous job 100 ended TIMEOUT)")
