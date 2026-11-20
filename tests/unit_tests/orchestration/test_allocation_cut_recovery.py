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

import json

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from tests.helpers.recovery_cut_runs import _cut_run, _digests
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run, _sbatch_env
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resume_script import _run_prologue


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

    # After the inventory a timeout still continues: test_the_guard_continues_a_cut_allocation_from_its_saved_rows.


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
    from nemo_gym.orchestration.executors.slurm.resume_script import REQUEUE_RECONCILED_NOTE
    from nemo_gym.orchestration.submit import inspect_recovery, recover
    from tests.unit_tests.orchestration.test_eval_resubmit import _start_queued_job

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


def test_a_requeued_then_failed_incarnation_is_accounted_and_recoverable(tmp_path, slurm_bin):
    from nemo_gym.orchestration.executors.slurm.resume_script import REQUEUE_RECONCILED_NOTE
    from nemo_gym.orchestration.submit import recover

    # A node prolog failure: the first incarnation is REQUEUED with its Start and End recorded, the second FAILED.
    job, _, _ = _requeued_run(tmp_path, slurm_bin, "100|REQUEUED|36|0\n100|FAILED|15|1\n101|FAILED|16|0\n")
    result = recover(*_handle(job))
    assert result.dispatched == "resubmit", result.decision.reason
    assert (job / ".gym_accumulated_walltime").read_text() == "67\n"
    assert (job / ".gym_chain_incomplete").read_text() == REQUEUE_RECONCILED_NOTE + "\n"


def test_a_requeued_incarnation_without_an_end_still_blocks_recovery(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    # The last incarnation is still REQUEUED (no End yet): its elapsed time is unknown.
    job, _, _ = _requeued_run(tmp_path, slurm_bin, "100|REQUEUED|36|0\n101|FAILED|16|0\n")
    result = recover(*_handle(job))
    # A requeued incarnation that has not ended is a live continuation: nothing is decided or queued.
    assert (result.decision.action.value, result.dispatched) == ("noop", None)
    assert (job / ".gym_chain_incomplete").read_text() == _REQUEUE_MARKER


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
    # A requeued incarnation that still runs is a live continuation: nothing to decide, nothing queued.
    live = "RUNNING" in accounting
    assert result.decision.action.value == ("noop" if live else "stop") and result.dispatched is None
    assert not (slurm_bin / "env").exists()
    assert (job / ".gym_chain_incomplete").read_text() == marker


_TWICE_REQUEUED = "100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0\n102|PREEMPTED|16455|0\n102|FAILED|13|1\n103|FAILED|12|0\n"


def _twice_requeued_run(tmp_path, slurm_bin, *, stored_walltime):
    """100's requeue was reconciled and the chain went on; 102 ran the Store, was preempted and requeued."""
    job, script, resume = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=3, in_flight=2, not_started=0, allocation="102", chain=False)
    (job / ".gym_job_chain").write_text("100\n101\n102\n103\n")
    (job / ".gym_infra_retries").write_text("1\n")
    (job / ".gym_accumulated_walltime").write_text(f"{stored_walltime}\n")
    (job / ".gym_chain_incomplete").write_text(_REQUEUE_MARKER)
    (job / ".gym_followup_queued.102").write_text("103\n")
    (slurm_bin / "accounting").write_text(_TWICE_REQUEUED)
    return job


def test_a_second_requeue_is_reconciled_from_every_allocation_and_named(tmp_path, slurm_bin):
    from nemo_gym.orchestration.executors.slurm.resume_script import REQUEUE_RECONCILED_NOTE
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    # Stored before 102 ran: both incarnations of 100 and the 16 s of 101.
    job = _twice_requeued_run(tmp_path, slurm_bin, stored_walltime=15030)
    decision = inspect_recovery(*_handle(job)).decision
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 3, "missing": 2}
    assert decision.reason.endswith("; job 102 was preempted and requeued; its 2 allocations account for 16468 s")

    result = recover(*_handle(job))
    assert result.dispatched == "resubmit"
    # Every allocation of the chain is charged once: 15030 + 16455 + 13 + 12 seconds.
    assert (job / ".gym_accumulated_walltime").read_text() == "31510\n"
    assert (job / ".gym_chain_incomplete").read_text() == REQUEUE_RECONCILED_NOTE + "\n"


def test_a_second_requeue_whose_stored_walltime_does_not_match_is_not_reconciled(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    # 15000 leaves out the 14 s and 16 s before 102: which allocations it covers is unknown.
    job = _twice_requeued_run(tmp_path, slurm_bin, stored_walltime=15000)
    result = recover(*_handle(job))
    assert (result.decision.action.value, result.dispatched) == ("stop", None)
    assert "requeued" not in result.decision.reason
    assert (job / ".gym_chain_incomplete").read_text() == _REQUEUE_MARKER


def _inspected_cut(tmp_path, slurm_bin, accounting):
    """A cut allocation 100 of a chain 100, 101, inspected the way the operator's login host does it."""
    from nemo_gym.orchestration.submit import inspect_recovery

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=2, in_flight=2, not_started=2, chain=False)
    (job / ".gym_job_chain").write_text("100\n101\n")
    (slurm_bin / "accounting").write_text(accounting)
    return job, inspect_recovery(*_handle(job))


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_the_inspect_reports_a_live_continuation_of_the_chain_and_decides_nothing(tmp_path, slurm_bin, state):
    job, inspected = _inspected_cut(tmp_path, slurm_bin, f"100|TIMEOUT|7200|0\n101|{state}|0|0\n")
    decision = inspected.decision
    assert (decision.action.value, decision.scheduler_live) == ("noop", True), decision.reason
    assert decision.reason == f"continuation 101 ({state}) live; nothing to decide"
    assert inspected.gate_task["status"] == "RUNNING" and inspected.dispatched is None
    assert not (slurm_bin / "env").exists()


def test_the_inspect_and_the_guard_classify_a_cut_allocation_the_same_way(tmp_path, slurm_bin):
    job, inspected = _inspected_cut(tmp_path, slurm_bin, "100|TIMEOUT|7200|0\n101|FAILED|60|0\n")
    assert inspected.decision.action.value == "resubmit", inspected.decision.reason
    assert inspected.decision.counts == {"scored": 2, "missing": 4}
    assert inspected.decision.reason.startswith("4 identity(ies) need the policy again")
    # The in-allocation guard of the next job, on the same saved files, sees the same four.
    result = _run_prologue(job, ResumeConfig(max_retries=2), "TIMEOUT", exit_code="0:15", prev_job_id="100")
    assert "Saved recovery evidence: 4 identity(ies) need the policy again" in result.stdout, result.stdout


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED"])
def test_the_inspect_resumes_reservations_of_an_allocation_that_ended_by_any_path(tmp_path, slurm_bin, state):
    _, inspected = _inspected_cut(tmp_path, slurm_bin, f"100|{state}|7200|0\n101|FAILED|60|0\n")
    assert inspected.decision.action.value == "resubmit", inspected.decision.reason
    assert inspected.decision.counts == {"scored": 2, "missing": 4}


def test_reservations_of_an_allocation_still_running_are_not_decided(tmp_path, slurm_bin):
    _, inspected = _inspected_cut(tmp_path, slurm_bin, "100|RUNNING|7200|0\n")
    assert (inspected.decision.action.value, inspected.decision.scheduler_live) == ("noop", True)
    assert inspected.decision.counts["unknown"] == 2


def test_reservations_without_a_known_ending_never_read_as_failures_nobody_owns(tmp_path, slurm_bin):
    # The accounting has no record of the allocation that made them.
    _, inspected = _inspected_cut(tmp_path, slurm_bin, "101|FAILED|60|0\n")
    reason = inspected.decision.reason
    assert inspected.decision.action.value == "stop"
    assert reason.startswith(
        "2 reservation(s) have no saved outcome and the allocation that made them is not known to have ended"
    )
    assert "nobody owns them" not in reason


def _without_named_writer(job):
    """The coverage report as a worker older than allocation_id writes it: no allocation_id key."""
    report_path = job / "artifacts" / "rollouts_coverage.json"
    report = json.loads(report_path.read_text())
    del report["allocation_id"]
    report_path.write_text(json.dumps(report))


def test_the_inspect_resumes_reservations_when_the_coverage_names_no_writer_and_the_chain_ended(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery

    job, _ = _inspected_cut(tmp_path, slurm_bin, "100|TIMEOUT|7200|0\n101|FAILED|60|0\n")
    _without_named_writer(job)
    inspected = inspect_recovery(*_handle(job))
    assert inspected.decision.action.value == "resubmit", inspected.decision.reason
    assert inspected.decision.counts == {"scored": 2, "missing": 4}


def test_reservations_stay_unknown_when_the_coverage_names_no_writer_and_a_chain_record_is_missing(
    tmp_path, slurm_bin
):
    from nemo_gym.orchestration.submit import inspect_recovery

    job, _ = _inspected_cut(tmp_path, slurm_bin, "100|TIMEOUT|7200|0\n")
    _without_named_writer(job)
    inspected = inspect_recovery(*_handle(job))
    assert inspected.decision.action.value == "stop"
    assert inspected.decision.counts["unknown"] == 2
