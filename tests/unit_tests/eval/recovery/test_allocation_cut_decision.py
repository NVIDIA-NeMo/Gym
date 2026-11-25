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
"""A run cut by its scheduler leaves reservations without rows; what recovery decides for them.

Real Store and real stdlib collector; only the scheduler is simulated.
"""

import subprocess
import sys

import pytest

from nemo_gym.eval.recovery.continuation import (
    allocation_cut_from_accounting,
    chain_accounting,
    store_writer_cut_from_accounting,
)
from nemo_gym.eval.recovery.evidence import (
    COLLECT_SCRIPT,
    CollectedEvidence,
)
from nemo_gym.eval.recovery.planner import decide_from_collected
from tests.helpers.recovery_cut_runs import _cut_run, _decide, _digests, _hang_until_cut


def test_the_in_job_admission_verifies_and_reads_the_store_once(tmp_path, monkeypatch, capsys):
    from nemo_gym.eval.recovery import continuation as recovery

    job = tmp_path / "job"
    _cut_run(job, scored=2, in_flight=2, not_started=2)
    calls = []
    select = recovery.select_managed_evidence
    monkeypatch.setattr(
        recovery,
        "select_managed_evidence",
        lambda collected, job_dir: calls.append(job_dir) or select(collected, job_dir),
    )
    assert recovery.continuation_admission(str(job), 2, None, "0:15", "100", "TIMEOUT") == "segment"
    # Each selection hashes every source file and reads the whole Store: one per admission is enough.
    assert calls == [job]
    assert "4 identity(ies) need the policy again" in capsys.readouterr().out


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
    assert (decision.action.value, decision.scheduler_live) == ("noop", True)
    assert decision.counts["unknown"] == 2


def test_a_live_job_is_reported_before_rows_nobody_owns(tmp_path):
    from nemo_gym.eval.recovery.gate import gate_task_summary

    _cut_run(tmp_path, masked=1)
    _, decision = _decide(tmp_path, allocation_cut=False, live_job=True)
    assert (decision.action.value, decision.scheduler_live) == ("noop", True)
    assert (
        decision.reason == "a job of this run is still pending or running; nothing to decide, it is not queued twice"
    )
    gate = gate_task_summary(
        decision, score=None, summary_valid=None, summary_reasons=[], health_passed=None, health_reasons=[]
    )
    assert gate["status"] == "RUNNING"


def test_a_pending_job_that_has_not_written_its_inventory_is_reported_as_live():
    collected = CollectedEvidence(materialized=None, results=None, failures=[])
    pending = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=True)
    assert (pending.action.value, pending.scheduler_live) == ("noop", True)
    assert pending.reason == "a job of this run is still pending or running; nothing to decide, it is not queued twice"
    named = decide_from_collected(
        collected, max_infra_retries=2, max_walltime_s=None, live_job=True, live_job_name="101 (RUNNING)"
    )
    assert (named.action.value, named.reason) == ("noop", "continuation 101 (RUNNING) live; nothing to decide")


def test_a_run_that_ended_before_its_inventory_is_a_fresh_restart():
    collected = CollectedEvidence(materialized=None, results=None, failures=[])
    ended = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert (ended.action.value, ended.scheduler_live, ended.expected) == ("fresh-restart", False, 0)
    assert ended.reason == (
        "nothing to recover: the run ended before the task inventory was written; "
        "resubmit it as a fresh run with a longer walltime"
    )
    # Rows saved without an inventory did not end before it, and a fresh run does not start over them.
    saved = collected.model_copy(update={"results": [{"_ng_task_index": 0, "_ng_rollout_index": 0}]})
    stopped = decide_from_collected(saved, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert stopped.action.value == "stop"
    assert stopped.reason == "the run has no materialized inventory to recover against"


def test_masked_results_are_retained_and_do_not_block_never_started_rows(tmp_path):
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=3, masked=2)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "resubmit"
    assert decision.counts == {"scored": 1, "masked": 2, "missing": 3}
    assert decision.reason == (
        "3 identity(ies) need the policy again: a Gym resume runs only missing and retryable ones; "
        "retryable 3 (resumed), unknown 0 (reported, not retried), retained 1 (saved rows kept), "
        "masked 2 (completed, not measured, kept)"
    )


def test_unknown_failures_a_resume_would_rerun_still_stop(tmp_path):
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=1, unowned=1)
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert decision.action.value == "stop"
    assert "nobody owns them and a resume would run 1 of them again" in decision.reason
    assert "unknown 1 (1 unregistered kind not_registered)" in decision.reason


def test_only_masked_results_left_stop_and_say_they_are_kept(tmp_path):
    from nemo_gym.eval.recovery.gate import gate_task_summary

    _cut_run(tmp_path, scored=1, in_flight=0, not_started=0, masked=2)
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert decision.action.value == "stop"
    assert decision.reason == (
        "2 completed result(s) are masked and not measured: they are kept and a resume would run nothing; the run "
        "is not complete; retryable 0 (resumed), unknown 0 (reported, not retried), retained 1 (saved rows kept), "
        "masked 2 (completed, not measured, kept)"
    )
    assert not decision.complete
    gate = gate_task_summary(
        decision, score=1.0, summary_valid=True, summary_reasons=[], health_passed=True, health_reasons=[]
    )
    assert gate["status"] == "FAILED"
    assert gate["errors"][0] == "2 masked results, completed but not measured: [(1, 0), (2, 0)]"
    assert all("nobody owns" not in error for error in gate["errors"])


@pytest.mark.parametrize(
    "accounting,allocation,cut",
    [
        ("100|TIMEOUT|14400|0\n101|FAILED|90|0", "100", True),
        ("100|PREEMPTED|300|0", "100", True),
        ("100|CANCELLED by 7|300|0", "100", True),
        ("100|NODE_FAIL|300|0", "100", True),
        # A requeued id: only its first incarnation ran work; the requeued one stopped in the prologue.
        ("100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0", "100", True),
        # Any ending counts: a failed driver, or one that completed through its no-progress exit.
        ("100|FAILED|300|0", "100", True),
        ("100|COMPLETED|16|0", "100", True),
        ("100|OUT_OF_MEMORY|300|0", "100", True),
        # Still alive, or its state is unknown: not an ending.
        ("100|RUNNING|300|0", "100", False),
        ("100|PENDING|0|0", "100", False),
        ("101|FAILED|60|0", "100", False),
        ("100|TIMEOUT|14400|0", "101", False),
        ("100|TIMEOUT|14400|0", None, False),
    ],
)
def test_the_cut_is_read_from_the_allocation_that_last_wrote_the_store(accounting, allocation, cut):
    assert allocation_cut_from_accounting(chain_accounting(accounting), allocation) is cut


@pytest.mark.parametrize(
    "accounting,allocation,unrecorded,chain,cut",
    [
        # The named writer decides, whatever else the chain holds.
        ("100|TIMEOUT|14400|0\n101|RUNNING|60|0", "100", False, ["100", "101"], True),
        # No allocation_id key (a worker older than it): every allocation of the chain must have ended.
        ("100|TIMEOUT|14400|0\n101|FAILED|60|0", None, True, ["100", "101"], True),
        ("100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0", None, True, ["100", "101"], True),
        ("100|TIMEOUT|14400|0\n101|RUNNING|60|0", None, True, ["100", "101"], False),
        ("100|TIMEOUT|14400|0", None, True, ["100", "101"], False),
        ("100|TIMEOUT|14400|0", None, True, [], False),
        # A recorded null (a writer outside Slurm) is not in the chain: unknown, however the chain ended.
        ("100|TIMEOUT|14400|0\n101|FAILED|60|0", None, False, ["100", "101"], False),
    ],
)
def test_a_store_without_a_named_writer_is_cut_once_every_allocation_of_the_chain_ended(
    accounting, allocation, unrecorded, chain, cut
):
    collected = CollectedEvidence(
        materialized=None,
        results=None,
        failures=[],
        job_chain=chain,
        coverage_allocation_id=allocation,
        coverage_writer_unrecorded=unrecorded,
    )
    assert store_writer_cut_from_accounting(chain_accounting(accounting), collected) is cut


@pytest.mark.parametrize("accounting", ["malformed", "100|TIMEOUT|14400", "100|TIMEOUT|x|0"])
def test_malformed_chain_accounting_is_unknown(accounting):
    assert chain_accounting(accounting) is None


def test_an_installer_error_with_a_credential_never_leaves_the_run_host(tmp_path):
    secret = "synthetic-install-secret"
    job = tmp_path / "gym-job-x" / "gsm8k"
    job.mkdir(parents=True)
    (job / ".gym_chain_incomplete").write_text(
        "Previous job 100 ended FAILED: driver_install_failed: fatal: unable to access "
        f"'https://user:{secret}@example.invalid/gym.git/': Could not resolve host; the chain does not retry it; "
        "see logs/driver-100.log\n"
    )
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    assert secret not in raw.stdout
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    # The typed failure and the driver-log reference stay; the raw log on the run host holds the detail.
    assert "driver_install_failed" in collected.chain_continuation_note
    assert "see logs/driver-100.log" in collected.chain_continuation_note
    decision = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert secret not in decision.reason
    assert "driver_install_failed" in decision.reason


def test_a_scheduler_cut_with_never_started_rows_stays_a_free_segment_beside_retryable_failures(tmp_path):
    from nemo_gym.eval.recovery import continuation as recovery

    # Two rows failed on provider errors earlier in the run and the allocation then ran out of walltime with work
    # never started: the continuation is planned, so it must not spend the run's infra retries even if they are spent.
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=2, retryable=2)
    (tmp_path / ".gym_infra_retries").write_text("2\n")
    assert recovery.continuation_admission(str(tmp_path), 2, None, "0:15", "100", "TIMEOUT") == "segment"


def test_a_scheduler_cut_with_only_retryable_failures_left_spends_an_infra_retry(tmp_path):
    from nemo_gym.eval.recovery import continuation as recovery

    _cut_run(tmp_path, scored=1, in_flight=0, not_started=0, retryable=2)
    assert recovery.continuation_admission(str(tmp_path), 2, None, "0:15", "100", "TIMEOUT") == "retry"
    (tmp_path / ".gym_infra_retries").write_text("2\n")
    assert recovery.continuation_admission(str(tmp_path), 2, None, "0:15", "100", "TIMEOUT") == "stop"


@pytest.mark.parametrize("retryable", [0, 2])
def test_rows_that_hang_until_every_cut_stop_the_chain_without_a_walltime_cap(tmp_path, retryable):
    from nemo_gym.eval.recovery import continuation as recovery

    # Rows dispatched in every allocation and cut with no outcome: the first end of an allocation is free for a row,
    # each later one spends an attempt of its cap, so the chain stops after the attempt cap even with no max_walltime
    # and with the run's infra retries already spent.
    _cut_run(tmp_path, scored=1, in_flight=2, not_started=0, retryable=retryable)
    (tmp_path / ".gym_infra_retries").write_text("2\n")
    admissions = []
    for allocation in range(100, 110):
        admissions.append(recovery.continuation_admission(str(tmp_path), 2, None, "0:15", str(allocation), "TIMEOUT"))
        if admissions[-1] != "segment":
            break
        _hang_until_cut(tmp_path, allocation=str(allocation + 1))
    assert admissions == ["segment"] * 4 + ["stop"]
