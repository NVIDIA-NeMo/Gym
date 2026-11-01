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
    chain_accounting,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_eval_resubmit import _handle, _resumable_run, _sbatch_env
from tests.unit_tests.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.test_resume_script import _run_prologue


def _cut_run(job, *, scored=2, in_flight=2, not_started=2, masked=0, unowned=0, allocation="100", chain=True):
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
    if chain:
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


def test_a_live_job_is_reported_before_rows_nobody_owns(tmp_path):
    from nemo_gym.orchestration.recovery import gate_task_summary

    _cut_run(tmp_path, masked=1)
    _, decision = _decide(tmp_path, allocation_cut=False, live_job=True)
    assert (decision.action.value, decision.scheduler_live) == ("stop", True)
    assert decision.reason == "a job of this run is still pending or running; it is not queued twice"
    gate = gate_task_summary(
        decision, score=None, summary_valid=None, summary_reasons=[], health_passed=None, health_reasons=[]
    )
    assert gate["status"] == "RUNNING"


def test_a_pending_job_that_has_not_written_its_inventory_is_reported_as_live():
    collected = CollectedEvidence(materialized=None, results=None, failures=[])
    pending = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=True)
    assert (pending.action.value, pending.scheduler_live) == ("stop", True)
    assert pending.reason == "a job of this run is still pending or running; it is not queued twice"
    ended = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert (ended.reason, ended.scheduler_live) == ("the run has no materialized inventory to recover against", False)


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
        ("100|TIMEOUT|14400|0\n101|FAILED|90|0", "100", True),
        ("100|PREEMPTED|300|0", "100", True),
        ("100|CANCELLED by 7|300|0", "100", True),
        ("100|NODE_FAIL|300|0", "100", True),
        # A requeued id: only its first incarnation ran work; the requeued one stopped in the prologue.
        ("100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0", "100", True),
        ("100|FAILED|300|0", "100", False),
        ("100|TIMEOUT|14400|0", "101", False),
        ("100|TIMEOUT|14400|0", None, False),
    ],
)
def test_the_cut_is_read_from_the_allocation_that_last_wrote_the_store(accounting, allocation, cut):
    assert allocation_cut_from_accounting(chain_accounting(accounting), allocation) is cut


@pytest.mark.parametrize("accounting", ["malformed", "100|TIMEOUT|14400", "100|TIMEOUT|x|0"])
def test_malformed_chain_accounting_is_unknown(accounting):
    assert chain_accounting(accounting) is None


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


@pytest.mark.parametrize("state", ["CANCELLED by 1000", "NODE_FAIL"])
def test_recover_resumes_reservations_after_an_operator_cancel_or_a_node_failure(tmp_path, slurm_bin, state):
    from nemo_gym.orchestration.submit import recover

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=2, in_flight=2, not_started=1, chain=False)
    (slurm_bin / "accounting").write_text(f"100|{state}|7200|0\n101|FAILED|60|0\n")
    before = _digests(job)
    result = recover(*_handle(job))
    assert result.decision.action.value == "resubmit", result.decision.reason
    assert result.decision.counts == {"scored": 2, "missing": 3}
    assert result.dispatched == "resubmit"
    assert _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
    assert _digests(job) == before


def test_the_guard_never_continues_after_an_operator_cancel(tmp_path):
    _cut_run(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "CANCELLED", exit_code="0:15", prev_job_id="100")
    assert "REACHED_WORK" not in result.stdout
    assert (tmp_path / ".gym_chain_incomplete").read_text() == (
        "Previous job 100 was cancelled. Chain stopped with rollouts missing; resubmit to continue.\n"
    )


_REQUEUE_MARKER = (
    "GYM_ACCOUNTING_UNKNOWN: Requeued allocation elapsed time is unknown; "
    "recover only after complete terminal accounting.\n"
)


def _requeued_run(tmp_path, slurm_bin, accounting, *, marker=_REQUEUE_MARKER, stored_walltime=None):
    """Allocation 100 ran the Store, was preempted and requeued; the requeue and its follow-up 101 stopped early."""
    job, script, resume = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=3, in_flight=2, not_started=0, chain=False)
    (job / ".gym_infra_retries").unlink()
    (job / ".gym_accumulated_walltime").unlink()
    if stored_walltime is not None:
        (job / ".gym_accumulated_walltime").write_text(f"{stored_walltime}\n")
    (job / ".gym_chain_incomplete").write_text(marker)
    (job / ".gym_followup_queued.100").write_text("101\n")
    (slurm_bin / "accounting").write_text(accounting)
    return job, script, resume


_TERMINAL_REQUEUE = "100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0\n"


def test_a_terminal_requeued_chain_is_reconciled_and_resumes_its_in_flight_rows(tmp_path, slurm_bin):
    from nemo_gym.orchestration.executors.resume_script import REQUEUE_RECONCILED_NOTE
    from nemo_gym.orchestration.submit import inspect_recovery, recover
    from tests.unit_tests.test_eval_resubmit import _start_queued_job

    job, script, resume = _requeued_run(tmp_path, slurm_bin, _TERMINAL_REQUEUE)
    before = _digests(job)
    inspected = inspect_recovery(*_handle(job))
    assert inspected.decision.action.value == "resubmit", inspected.decision.reason
    assert inspected.decision.counts == {"scored": 3, "missing": 2}
    assert inspected.dispatched is None and (job / ".gym_chain_incomplete").read_text() == _REQUEUE_MARKER

    result = recover(*_handle(job))
    assert result.dispatched == "resubmit"
    assert _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
    # Every incarnation is charged once: 15000 + 14 + 16 seconds.
    assert (job / ".gym_accumulated_walltime").read_text() == "15030\n"
    assert (job / ".gym_chain_incomplete").read_text() == REQUEUE_RECONCILED_NOTE + "\n"
    assert _digests(job) == before
    # The queued job passes the prologue's accounting checks, spends one infra retry and resumes saved rows.
    args = _start_queued_job(script, resume, job, slurm_bin)
    assert args[-1] == "++resume_from_cache=True"
    assert (job / ".gym_infra_retries").read_text() == "1\n"


@pytest.mark.parametrize(
    "accounting,marker,stored",
    [
        # The first incarnation's record is absent: the requeued chain is not completely accounted.
        ("100|FAILED|14|1\n101|FAILED|16|0\n", _REQUEUE_MARKER, None),
        # The requeued incarnation is still running.
        ("100|PREEMPTED|15000|0\n100|RUNNING|14|1\n101|FAILED|16|0\n", _REQUEUE_MARKER, None),
        # The stored counter does not match the jobs before the requeue: the budget epoch is unknown.
        (_TERMINAL_REQUEUE, _REQUEUE_MARKER, 7200),
        # Another unknown reason is never reconciled from scheduler records.
        (
            _TERMINAL_REQUEUE,
            "GYM_ACCOUNTING_UNKNOWN: Follow-up submission may have succeeded without an observed reply.\n",
            None,
        ),
    ],
)
def test_an_incomplete_or_other_unknown_accounting_is_not_reconciled(tmp_path, slurm_bin, accounting, marker, stored):
    from nemo_gym.orchestration.submit import recover

    job, _, _ = _requeued_run(tmp_path, slurm_bin, accounting, marker=marker, stored_walltime=stored)
    result = recover(*_handle(job))
    assert result.decision.action.value == "stop" and result.dispatched is None
    assert not (slurm_bin / "env").exists()
    assert (job / ".gym_chain_incomplete").read_text() == marker
