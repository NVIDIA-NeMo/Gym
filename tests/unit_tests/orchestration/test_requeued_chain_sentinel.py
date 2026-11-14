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
"""A sentinel continues a chain that a Slurm requeue stopped, once the scheduler's terminal records account for it.

Real Store, real collector, the real rendered prologue and the real installed-runtime reader; only sacct, squeue and
sbatch are simulated.
"""

import shutil
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.resume_script import (
    ACCOUNTING_UNKNOWN_PREFIX,
    INCOMPLETE_MARKER_NAME,
    REQUEUE_RECONCILED_NOTE,
    REQUEUE_UNKNOWN_REASON,
    RESUBMIT_LOCK_NAME,
    render_resume_prologue,
)
from tests.unit_tests.orchestration.test_allocation_cut_recovery import _cut_run, _digests
from tests.unit_tests.orchestration.test_resume_script import _run_prologue, _sbatch_calls


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")

_REQUEUE_MARKER = f"{ACCOUNTING_UNKNOWN_PREFIX} {REQUEUE_UNKNOWN_REASON}\n"
_RESUME = ResumeConfig(max_retries=3, max_walltime="16:00:00")
# 100 ran the Store and was preempted; its requeued incarnation and its follow-up 777 stopped on the requeue marker.
_TERMINAL = "100|PREEMPTED|15000|0\n100|FAILED|14|1\n777|FAILED|16|0\n"
_TWICE_REQUEUED = "100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0\n102|PREEMPTED|16455|0\n102|FAILED|13|1\n103|FAILED|12|0\n"


def _requeued_chain(
    job: Path,
    accounting: str | list[str],
    *,
    chain: str = "100\n777\n",
    allocation: str = "100",
    followup: str = "777",
    stored_walltime: int | None = None,
    retries: int | None = None,
    inventory: bool = True,
) -> None:
    """The job dir after the requeue; ``accounting`` is what sacct answers, poll by poll when it is a list."""
    if inventory:
        _cut_run(job, scored=3, in_flight=2, not_started=0, allocation=allocation, chain=False)
    (job / ".gym_job_chain").write_text(chain)
    if stored_walltime is not None:
        (job / ".gym_accumulated_walltime").write_text(f"{stored_walltime}\n")
    if retries is not None:
        (job / ".gym_infra_retries").write_text(f"{retries}\n")
    (job / INCOMPLETE_MARKER_NAME).write_text(_REQUEUE_MARKER)
    (job / f".gym_followup_queued.{allocation}").write_text(f"{followup}\n")
    polls = accounting if isinstance(accounting, list) else [accounting]
    for number, answer in enumerate(polls[:-1], start=1):
        (job / f"chain_accounting.{number}").write_text(answer)
    (job / "chain_accounting").write_text(polls[-1])


def _sentinel(job: Path, resume: ResumeConfig = _RESUME, *, parent: str = "100", followup: str = "777"):
    """Sentinel 200 of follow-up ``followup``, queued by ``parent``: it starts once the follow-up has ended."""
    return _run_prologue(
        job,
        resume,
        "FAILED",
        prev_job_id=parent,
        extra_env={"GYM_FOLLOWUP_SENTINEL": followup, "FAKE_FOLLOWUP": followup, "FAKE_FOLLOWUP_ELAPSED": "16"},
    )


def _chain_polls(job: Path) -> int:
    polls = job / "chain_polls"
    return int(polls.read_text()) if polls.exists() else 0


def _queued_env(job: Path) -> list[str]:
    calls = job / "sbatch_env"
    return calls.read_text().splitlines() if calls.exists() else []


def _assert_stopped_as_before(job: Path, before: dict[str, bytes]) -> None:
    assert _sbatch_calls(job) == []
    assert (job / INCOMPLETE_MARKER_NAME).read_text() == _REQUEUE_MARKER
    after = {name: (job / name).read_bytes() for name in before if (job / name).exists()}
    assert after == before


def _budget_files(job: Path) -> dict[str, bytes]:
    names = [".gym_job_chain", ".gym_accumulated_walltime", ".gym_infra_retries", INCOMPLETE_MARKER_NAME]
    return {name: (job / name).read_bytes() for name in names if (job / name).exists()}


def test_a_single_requeue_continues_once_its_records_settle(tmp_path):
    # The requeued incarnation's record lands only on the third poll.
    _requeued_chain(
        tmp_path,
        [
            "100|PREEMPTED|15000|0\n777|FAILED|16|0\n",
            "100|PREEMPTED|15000|0\n100|RUNNING|14|1\n777|FAILED|16|0\n",
            _TERMINAL,
        ],
    )
    rows = _digests(tmp_path)

    result = _sentinel(tmp_path)

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    assert _chain_polls(tmp_path) == 3
    # One recovery, queued without a dependency, keeping the budget, as gym eval resubmit --recover queues it.
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [str(tmp_path / "job.sh")]
    assert _queued_env(tmp_path) == ["1|1"]
    # Every incarnation is charged once: 15000 + 14 + 16 seconds.
    assert (tmp_path / ".gym_accumulated_walltime").read_text() == "15030\n"
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == REQUEUE_RECONCILED_NOTE + "\n"
    assert not (tmp_path / RESUBMIT_LOCK_NAME).exists()
    # The sentinel runs no work and stays out of the chain; the saved rows are untouched.
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n"
    assert _digests(tmp_path) == rows
    assert "job 100 was preempted and requeued; its 2 allocations account for 15014 s" in result.stdout

    # The queued recovery passes the prologue's accounting checks, spends one infra retry and reaches the work.
    queued = _run_prologue(tmp_path, _RESUME, None, job_id="999", keep_budget=True)
    assert queued.returncode == 0 and "REACHED_WORK" in queued.stdout, queued.stdout + queued.stderr
    assert (tmp_path / ".gym_infra_retries").read_text() == "1\n"
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n999\n"


def test_a_requeued_follow_up_settles_on_its_requeued_record(tmp_path):
    # Slurm keeps the earlier incarnation of a requeued job as a REQUEUED row for good, not a PREEMPTED one: the
    # follow-up 777 was requeued once and stopped on the marker in its second incarnation.
    _requeued_chain(
        tmp_path,
        "100|FAILED|15000|0\n777|REQUEUED|36|0\n777|FAILED|14|1\n",
        stored_walltime=15000,
    )

    result = _sentinel(tmp_path)

    assert result.returncode == 0 and "did not settle" not in result.stdout, result.stdout + result.stderr
    assert _chain_polls(tmp_path) == 1
    assert len(_sbatch_calls(tmp_path)) == 1 and _queued_env(tmp_path) == ["1|1"]
    # 15000 + 36 + 14 seconds.
    assert (tmp_path / ".gym_accumulated_walltime").read_text() == "15050\n"
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == REQUEUE_RECONCILED_NOTE + "\n"


def test_a_last_incarnation_still_requeued_does_not_settle(tmp_path):
    # No later incarnation has run yet, so the REQUEUED row is not an ended allocation.
    _requeued_chain(tmp_path, "100|FAILED|15000|0\n777|REQUEUED|36|0\n777|REQUEUED|0|1\n")
    before = _budget_files(tmp_path)

    result = _sentinel(tmp_path)

    assert result.returncode == 0 and "did not settle" in result.stdout
    _assert_stopped_as_before(tmp_path, before)


def test_a_second_requeue_is_reconciled_from_every_allocation(tmp_path):
    # 100's requeue was reconciled before and the chain went on; 102 ran the Store and was requeued.
    _requeued_chain(
        tmp_path,
        _TWICE_REQUEUED,
        chain="100\n101\n102\n103\n",
        allocation="102",
        followup="103",
        stored_walltime=15030,
        retries=1,
    )

    result = _sentinel(tmp_path, parent="102", followup="103")

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_sbatch_calls(tmp_path)) == 1 and _queued_env(tmp_path) == ["1|1"]
    # 15030 + 16455 + 13 + 12 seconds.
    assert (tmp_path / ".gym_accumulated_walltime").read_text() == "31510\n"
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == REQUEUE_RECONCILED_NOTE + "\n"


def test_records_that_never_settle_leave_the_chain_stopped_as_before(tmp_path):
    _requeued_chain(tmp_path, "100|PREEMPTED|15000|0\n100|RUNNING|14|1\n777|FAILED|16|0\n")
    before = _budget_files(tmp_path)

    result = _sentinel(tmp_path)

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout
    # Every 30 s for 5 minutes, then the same stop with the same typed record.
    assert _chain_polls(tmp_path) == 11
    _assert_stopped_as_before(tmp_path, before)
    assert "did not settle" in result.stdout


def test_a_budget_exhausted_by_the_reconcile_stops_without_a_submission(tmp_path):
    _requeued_chain(tmp_path, _TERMINAL)
    before = _budget_files(tmp_path)

    # The four-hour cap is below the 15030 s the requeued chain spent.
    result = _sentinel(tmp_path, ResumeConfig(max_retries=3, max_walltime="04:00:00"))

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout
    _assert_stopped_as_before(tmp_path, before)
    assert "walltime" in result.stdout


def test_an_operator_resubmit_holding_the_lock_is_not_queued_twice(tmp_path):
    _requeued_chain(tmp_path, _TERMINAL)
    (tmp_path / RESUBMIT_LOCK_NAME).write_text("")
    before = _budget_files(tmp_path)

    result = _sentinel(tmp_path)

    assert result.returncode == 0
    _assert_stopped_as_before(tmp_path, before)
    # The lock is the operator's: the sentinel leaves it.
    assert (tmp_path / RESUBMIT_LOCK_NAME).exists()


def test_a_job_of_the_run_already_queued_is_not_queued_twice(tmp_path):
    _requeued_chain(tmp_path, _TERMINAL)
    # An operator recovery queued 4242 and released the lock before it wrote anything else this test checks.
    (tmp_path / "pending").write_text(f"4242|(null)|{tmp_path}\n")
    before = _budget_files(tmp_path)

    result = _sentinel(tmp_path)

    assert result.returncode == 0
    _assert_stopped_as_before(tmp_path, before)


def test_a_second_sentinel_after_the_reconcile_queues_nothing(tmp_path):
    _requeued_chain(tmp_path, _TERMINAL)
    assert _sentinel(tmp_path).returncode == 0
    assert len(_sbatch_calls(tmp_path)) == 1
    (tmp_path / "pending").unlink(missing_ok=True)

    again = _sentinel(tmp_path)

    assert again.returncode == 0 and "REACHED_WORK" not in again.stdout
    assert len(_sbatch_calls(tmp_path)) == 1
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == REQUEUE_RECONCILED_NOTE + "\n"
    assert (tmp_path / ".gym_accumulated_walltime").read_text() == "15030\n"


def test_an_unconfirmed_submission_is_recorded_as_unknown_accounting(tmp_path):
    _requeued_chain(tmp_path, _TERMINAL)

    result = _run_prologue(
        tmp_path,
        _RESUME,
        "FAILED",
        sbatch_loses_reply=True,
        extra_env={"GYM_FOLLOWUP_SENTINEL": "777", "FAKE_FOLLOWUP": "777", "FAKE_FOLLOWUP_ELAPSED": "16"},
    )

    assert result.returncode == 0
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text().startswith(f"{ACCOUNTING_UNKNOWN_PREFIX} Resubmit")
    assert not (tmp_path / RESUBMIT_LOCK_NAME).exists()


def test_a_requeue_before_the_task_inventory_stays_a_stop(tmp_path):
    _requeued_chain(tmp_path, _TERMINAL, inventory=False)
    before = _budget_files(tmp_path)

    result = _sentinel(tmp_path)

    assert result.returncode == 0
    _assert_stopped_as_before(tmp_path, before)


def test_the_rendered_job_script_carries_the_sentinel_reconcile():
    from tests.unit_tests.orchestration.test_eval_resubmit import _resumable_run

    def rendered(tmp: Path) -> str:
        job, script, resume = _resumable_run(tmp)
        start = script.index("_gym_admission_output=$(") + len("_gym_admission_output=$(")
        admission = script[start : script.index(") || _gym_stop_incomplete", start)]
        prologue = render_resume_prologue(resume, admission_command=admission)
        assert prologue in script
        return script

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        script = rendered(Path(tmp))
    # The sentinel hands the chain's records to the same installed reader the guard runs.
    reconcile = script[script.index("_gym_reconcile_requeue() {") :]
    assert 'export GYM_CHAIN_ACCOUNTING="$_gym_requeue_rows"' in reconcile
    assert "srun --overlap" in reconcile.split("unset GYM_CHAIN_ACCOUNTING")[0]
    assert "sacct -D -n -P -X -j" in reconcile
