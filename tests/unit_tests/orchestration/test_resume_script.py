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

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.resume_script import (
    ACCOUNTING_UNKNOWN_PREFIX,
    CHAIN_END_JOURNAL_NAME,
    DRIVER_FAILURE_NAME,
    EXPORT_RECEIPT_MISSING,
    EXPORT_RECEIPT_MISSING_REASON,
    FOLLOW_UP_SUBMISSION_REFUSED,
    FRESH_RESTART_MARKER_NAME,
    GUARDED_DRIVER_FAILURE_KINDS,
    INCOMPLETE_MARKER_NAME,
    NO_ELIGIBLE_RETRY,
    NO_ELIGIBLE_RETRY_EXIT,
    NO_ELIGIBLE_RETRY_REASON,
    PROVISIONAL_MARKER_PREFIX,
    ROWS_AT_START_NAME,
    _walltime_to_seconds,
    render_resume_epilogue,
    render_resume_prologue,
)


# ---------------------------------------------------------------------------
# _walltime_to_seconds
# ---------------------------------------------------------------------------


def test_minutes_only():
    assert _walltime_to_seconds("90") == 90 * 60


def test_minutes_seconds():
    assert _walltime_to_seconds("05:30") == 5 * 60 + 30


def test_hours_minutes_seconds():
    assert _walltime_to_seconds("02:00:00") == 2 * 3600


def test_days_hours():
    assert _walltime_to_seconds("1-06") == 86400 + 6 * 3600


def test_days_hours_minutes_seconds():
    assert _walltime_to_seconds("2-00:00:00") == 2 * 86400


def test_invalid_duration_raises():
    with pytest.raises(ValueError, match="Invalid max_walltime"):
        _walltime_to_seconds("not-a-duration")


def test_invalid_day_component_raises():
    with pytest.raises(ValueError, match="day component must be an integer"):
        _walltime_to_seconds("x-06:00:00")


# ---------------------------------------------------------------------------
# render_resume_prologue
# ---------------------------------------------------------------------------


def test_default_prologue_has_no_walltime_check():
    out = render_resume_prologue(ResumeConfig())
    assert "_gym_accumulated >=" not in out
    assert "_gym_max_retries=3" in out  # default max_retries
    assert "-ge $_gym_max_retries" in out


def test_prologue_with_max_walltime_adds_check():
    out = render_resume_prologue(ResumeConfig(max_walltime="48:00:00"))
    assert "_gym_accumulated >= 172800" in out


def test_prologue_uses_custom_max_retries():
    out = render_resume_prologue(ResumeConfig(max_retries=9))
    assert "_gym_max_retries=9" in out
    assert "Infra retry limit ($_gym_max_retries)" in out


def test_prologue_queues_own_successor():
    out = render_resume_prologue(ResumeConfig())
    assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("resume", [ResumeConfig(), ResumeConfig(max_walltime="12:00:00")])
def test_rendered_prologue_is_valid_bash(resume, tmp_path):
    script = tmp_path / "prologue.sh"
    # The epilogue follows the driver step in job.sh and calls the prologue's functions.
    script.write_text(render_resume_prologue(resume) + "\n" + render_resume_epilogue())
    subprocess.run(["bash", "-n", str(script)], check=True)


# ---------------------------------------------------------------------------
# Running the prologue against fake sacct/sbatch
# ---------------------------------------------------------------------------

_FAKE_SACCT = """#!/bin/bash
echo call >> "$FAKE_LOG_DIR/sacct_calls"
# The node dies under the job: kill the job's shell, found by its script path (never the test runner).
if [[ -n "${FAKE_SACCT_KILLS_JOB:-}" ]]; then pkill -KILL -f "$FAKE_LOG_DIR/job.sh"; exit 1; fi
# A chain query (a sentinel waiting on a requeued chain) reads chain_accounting.<poll>, else chain_accounting.
if [[ "$*" == *Restarts* ]]; then
    _poll=$(( $(cat "$FAKE_LOG_DIR/chain_polls" 2>/dev/null || echo 0) + 1 )); echo "$_poll" > "$FAKE_LOG_DIR/chain_polls"
    if [[ -e "$FAKE_LOG_DIR/chain_accounting.$_poll" ]]; then cat "$FAKE_LOG_DIR/chain_accounting.$_poll"
    else cat "$FAKE_LOG_DIR/chain_accounting" 2>/dev/null; fi
    exit 0
fi
if [[ -n "${FAKE_FOLLOWUP:-}" && " $* " == *" -j ${FAKE_FOLLOWUP} "* ]]; then echo "${FAKE_FOLLOWUP_ELAPSED:-}"; exit 0; fi
if [[ "$*" == *State,ElapsedRaw* ]]; then echo "${FAKE_STATE:-}|${FAKE_ELAPSED:-}";
elif [[ "$*" == *ExitCode* ]]; then echo "${FAKE_EXIT:-1:0}";
elif [[ "$*" == *ElapsedRaw* ]]; then echo "${FAKE_ELAPSED:-}"; else echo "${FAKE_STATE:-}"; fi
"""
_FAKE_SBATCH = """#!/bin/bash
echo "$* sentinel=${GYM_FOLLOWUP_SENTINEL:-}" >> "$FAKE_LOG_DIR/sbatch_calls"
echo "${GYM_RESUBMIT:-}|${GYM_RESUBMIT_KEEP_BUDGET:-}" >> "$FAKE_LOG_DIR/sbatch_env"
if [[ -n "${FAKE_SBATCH_FAILS:-}" ]]; then echo "sbatch: error: Socket timed out"; exit 1; fi
# The scheduler's refusal, as at a QOS submit limit: nothing is queued, and sbatch says why on stderr.
if [[ -n "${FAKE_SBATCH_REFUSES:-}" ]]; then printf '%s\n' "$FAKE_SBATCH_REFUSES" >&2; exit 1; fi
# Every queued job is 999 unless a test gives the sentinel its own id, to tell the two apart.
_id=999
[[ -z "${GYM_FOLLOWUP_SENTINEL:-}" || -z "${FAKE_SENTINEL_ID:-}" ]] || _id=$FAKE_SENTINEL_ID
# The queued follow-up waits on its dependency in this job dir, as squeue would list it. In a hidden partition
# squeue lists it only with --all.
_pending="$FAKE_LOG_DIR/pending${FAKE_HIDDEN_PARTITION:+_hidden}"
echo "$_id|${1#--dependency=}(unfulfilled)|$PWD" >> "$_pending"
# A lost reply: the job exists, but sbatch reports a failure.
if [[ -n "${FAKE_SBATCH_LOSES_REPLY:-}" ]]; then echo "sbatch: error: Socket timed out on send/recv operation"; exit 1; fi
echo "Submitted batch job $_id"
"""
_FAKE_SCANCEL = """#!/bin/bash
echo "$*" >> "$FAKE_LOG_DIR/scancel_calls"
if [[ " ${FAKE_SCANCEL_FAILS:-} " == *" $1 "* ]]; then
    echo "scancel: error: Kill job error on job id $1: Invalid job id specified" >&2
    exit 1
fi
"""
_FAKE_SCONTROL = """#!/bin/bash
echo "$*" >> "$FAKE_LOG_DIR/scontrol_calls"
if [[ -n "${FAKE_SCONTROL_FAILS:-}" ]]; then exit 1; fi
if [[ "$1" == show ]]; then
    if [[ -e "$FAKE_LOG_DIR/time_limit" ]]; then
        echo "JobId=$SLURM_JOB_ID TimeLimit=$(cat "$FAKE_LOG_DIR/time_limit")"
    else
        echo "JobId=$SLURM_JOB_ID TimeLimit=${FAKE_TIME_LIMIT:-04:00:00}"
    fi
elif [[ "$1" == update ]]; then
    [[ "${FAKE_UPDATE_RESULT:-}" != failed ]] || exit 1
    if [[ "${FAKE_UPDATE_RESULT:-}" == unknown ]]; then echo UNKNOWN > "$FAKE_LOG_DIR/time_limit"
    elif [[ "${FAKE_UPDATE_RESULT:-}" != unchanged ]]; then
        minutes=${3#TimeLimit=}
        printf '%02d:%02d:00\\n' "$((minutes / 60))" "$((minutes % 60))" > "$FAKE_LOG_DIR/time_limit"
    fi
fi
"""
_FAKE_SQUEUE = """#!/bin/bash
if [[ -n "${FAKE_SQUEUE_FAILS:-}" ]]; then echo "squeue: error: Unable to contact slurm controller"; exit 1; fi
cat "$FAKE_LOG_DIR/pending" 2>/dev/null
[[ " $* " == *" --all "* ]] && cat "$FAKE_LOG_DIR/pending_hidden" 2>/dev/null
exit 0
"""


def _run_prologue(
    tmp_path,
    resume: ResumeConfig,
    prev_state: str | None,
    *,
    exit_code: str = "1:0",
    job_id: str = "200",
    prev_job_id: str = "100",
    sbatch_fails: bool = False,
    squeue_fails: bool = False,
    hidden_partition: bool = False,
    sbatch_loses_reply: bool = False,
    elapsed: str = "60",
    time_limit: str = "04:00:00",
    scontrol_fails: bool = False,
    keep_budget: bool = False,
    update_result: str = "",
    restart_count: str = "0",
    manual_resubmit: bool = False,
    admission_command: str | None = None,
    extra_env: dict[str, str] | None = None,
    driver: str | None = None,
    sentinel_id: str = "",
    scancel_fails: str = "",
) -> subprocess.CompletedProcess:
    """Run the prologue as job ``job_id`` after ``prev_job_id`` ended ``prev_state`` (None: the first segment).

    With ``driver``, that shell is the driver step and the epilogue runs after it, as in job.sh.
    ``sentinel_id`` gives the queued sentinel its own id (the follow-up is 999); ``scancel_fails`` names the ids
    scancel refuses.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (
        ("sacct", _FAKE_SACCT),
        ("sbatch", _FAKE_SBATCH),
        ("squeue", _FAKE_SQUEUE),
        ("scontrol", _FAKE_SCONTROL),
        ("scancel", _FAKE_SCANCEL),
        ("sleep", "#!/bin/bash\n"),
    ):
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    script = tmp_path / "job.sh"
    if admission_command is None:
        # Explicit installed-runtime boundary: host PATH below intentionally has no Gym installation.
        code = (
            f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r}); "
            "from nemo_gym.orchestration.recovery import continuation_admission; "
            f"print('__GYM_CONTINUATION:' + continuation_admission({str(tmp_path)!r}, "
            f"{resume.max_retries}, {_walltime_to_seconds(resume.max_walltime) if resume.max_walltime else None!r}, sys.argv[1], sys.argv[2], sys.argv[3]))"
        )
        admission_command = f'{shlex.quote(sys.executable)} -c {shlex.quote(code)} "$_gym_exit_code" "$_prev_slurm_job_id" "$_prev_state"'
    body = render_resume_prologue(resume, admission_command=admission_command) + "\necho REACHED_WORK\n"
    if driver is not None:
        # The driver step, then the epilogue job.sh runs after it.
        body += f"{driver}\n" + render_resume_epilogue()
    script.write_text(body)
    args = ["bash", str(script)] + ([] if prev_state is None else [prev_job_id])
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SLURM_JOB_ID": job_id,
        "FAKE_LOG_DIR": str(tmp_path),
        "FAKE_STATE": prev_state or "",
        "FAKE_EXIT": exit_code,
        "FAKE_ELAPSED": elapsed,
        "FAKE_TIME_LIMIT": time_limit,
        "FAKE_UPDATE_RESULT": update_result,
        "SLURM_RESTART_COUNT": restart_count,
        **({"GYM_RESUBMIT": "1"} if manual_resubmit else {}),
        **({"FAKE_SCONTROL_FAILS": "1"} if scontrol_fails else {}),
        **({"GYM_RESUBMIT": "1", "GYM_RESUBMIT_KEEP_BUDGET": "1"} if keep_budget else {}),
        **({"FAKE_SBATCH_FAILS": "1"} if sbatch_fails else {}),
        **({"FAKE_SQUEUE_FAILS": "1"} if squeue_fails else {}),
        **({"FAKE_HIDDEN_PARTITION": "1"} if hidden_partition else {}),
        **({"FAKE_SBATCH_LOSES_REPLY": "1"} if sbatch_loses_reply else {}),
        **({"FAKE_SENTINEL_ID": sentinel_id} if sentinel_id else {}),
        **({"FAKE_SCANCEL_FAILS": scancel_fails} if scancel_fails else {}),
        **(extra_env or {}),
    }
    return subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_max_retries_allows_exactly_that_many_retries(tmp_path):
    resume = ResumeConfig(max_retries=3)
    for _ in range(3):
        result = _run_prologue(tmp_path, resume, "PREEMPTED")
        assert result.returncode == 0, result.stdout
        assert "REACHED_WORK" in result.stdout

    result = _run_prologue(tmp_path, resume, "PREEMPTED")
    assert result.returncode == 1
    assert "REACHED_WORK" not in result.stdout
    assert "Infra retry limit" in (tmp_path / INCOMPLETE_MARKER_NAME).read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_empty_sacct_stops_without_spending_a_retry(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=1), "")

    assert result.returncode == 1, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert "unknown" in result.stdout
    assert not (tmp_path / ".gym_infra_retries").exists()
    assert len((tmp_path / "sacct_calls").read_text().splitlines()) > 2


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_successor_is_killed_if_its_dependency_can_never_be_met(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(), None)

    assert result.returncode == 0, result.stdout
    sbatch_args = (tmp_path / "sbatch_calls").read_text()
    assert "--dependency=afternotok:200" in sbatch_args
    assert "--kill-on-invalid-dep=yes" in sbatch_args


@pytest.mark.parametrize("exit_code,continues", [("75:0", True), ("76:0", False)])
@pytest.mark.parametrize("max_retries", [0, 3])
def test_incomplete_evaluation_chain_obeys_retryability(tmp_path, monkeypatch, exit_code, continues, max_retries):
    from tests.unit_tests.orchestration.test_managed_recovery_selection import prepare_managed

    monkeypatch.setenv("SLURM_JOB_ID", "100")
    # Real saved outcomes and untouched pending inputs, not an exit-code-only permission token.
    prepare_managed(tmp_path, "missing" if continues else "healthy", n=2)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=max_retries), "FAILED", exit_code=exit_code)
    assert result.returncode == (0 if continues else 1), result.stdout
    assert ("REACHED_WORK" in result.stdout) is continues
    assert (tmp_path / "sbatch_calls").exists() is continues
    assert not (tmp_path / ".gym_infra_retries").exists()
    incomplete = tmp_path / INCOMPLETE_MARKER_NAME
    assert incomplete.exists() is not continues
    if not continues:
        # Both rollouts are saved: the run is complete, so the stop names its missing export, not the retry budget.
        assert incomplete.read_text() == EXPORT_RECEIPT_MISSING_REASON + "\n"


def test_retryable_evaluation_still_obeys_walltime_cap(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="00:01:00"), "FAILED", exit_code="75:0")
    assert result.returncode == 1
    assert "REACHED_WORK" not in result.stdout
    assert not (tmp_path / "sbatch_calls").exists()
    assert "walltime cap" in (tmp_path / INCOMPLETE_MARKER_NAME).read_text()


def _job_dir(tmp_path) -> str:
    # The prologue's OUTPUT_DIR is the shell's $PWD, the physical path of the job dir.
    return os.path.realpath(tmp_path)


def _write_inventory(tmp_path) -> None:
    """The task inventory a segment writes before any rollout: after it, a timed-out segment is continued."""
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    (tmp_path / "artifacts" / "rollouts_materialized_inputs.jsonl").write_text(
        '{"_ng_task_index": 0, "_ng_rollout_index": 0}\n'
    )


def _sbatch_calls(tmp_path) -> list[str]:
    calls = tmp_path / "sbatch_calls"
    return calls.read_text().splitlines() if calls.exists() else []


def _scancel_calls(tmp_path) -> list[str]:
    calls = tmp_path / "scancel_calls"
    return calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("prev_state", [None, "TIMEOUT"])
def test_a_requeued_job_does_not_queue_a_second_follow_up(tmp_path, prev_state):
    """Slurm reruns job.sh under the same job id on a requeue (node failure, preemption with --requeue)."""
    _write_inventory(tmp_path)
    first = _run_prologue(tmp_path, ResumeConfig(), prev_state)
    requeued = _run_prologue(tmp_path, ResumeConfig(), prev_state)

    assert "REACHED_WORK" in first.stdout and "REACHED_WORK" in requeued.stdout
    # The first run queued the follow-up and its sentinel; the requeue queued nothing.
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afterany:999",
    ]
    assert "already queued" in requeued.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("claim", ["file", "directory"])
def test_a_claim_left_by_an_interrupted_submission_stops_without_an_observed_job(tmp_path, claim):
    """No queue entry cannot distinguish a failed submit from a successor that already ran."""
    marker = tmp_path / ".gym_followup_queued.200"
    marker.mkdir() if claim == "directory" else marker.touch()  # a directory is #4840's marker format

    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert requeued.returncode == 1 and "REACHED_WORK" not in requeued.stdout
    assert _sbatch_calls(tmp_path) == []
    assert marker.exists()
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text().startswith("GYM_ACCOUNTING_UNKNOWN:")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_the_claim_records_the_follow_up_and_a_requeue_finds_it_pending(tmp_path):
    _run_prologue(tmp_path, ResumeConfig(), None)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert (tmp_path / ".gym_followup_queued.200").read_text() == "999\n"
    assert (tmp_path / ".gym_followup_sentinel.200").read_text() == "999\n"
    assert "already queued: 999" in requeued.stdout
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afterany:999",
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_an_empty_claim_with_its_follow_up_pending_queues_none(tmp_path):
    """The shell died after sbatch queued the follow-up but before the claim recorded its id."""
    (tmp_path / ".gym_followup_queued.200").touch()
    (tmp_path / "pending").write_text(f"999|afternotok:200(unfulfilled)|{_job_dir(tmp_path)}\n")

    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert "already queued: 999" in requeued.stdout
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_pending_on_another_job_does_not_count(tmp_path):
    (tmp_path / ".gym_followup_queued.200").touch()
    job_dir = _job_dir(tmp_path)
    (tmp_path / "pending").write_text(
        f"998|afternotok:2000(unfulfilled)|{job_dir}\n997|afterok:200(unfulfilled)|{job_dir}\n"
    )

    _run_prologue(tmp_path, ResumeConfig(), None)

    assert _sbatch_calls(tmp_path) == []
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text().startswith("GYM_ACCOUNTING_UNKNOWN:")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_claim_is_kept_when_squeue_fails(tmp_path):
    (tmp_path / ".gym_followup_queued.200").touch()

    requeued = _run_prologue(tmp_path, ResumeConfig(), None, squeue_fails=True)

    assert requeued.returncode == 1 and "REACHED_WORK" not in requeued.stdout
    assert "squeue failed" in requeued.stdout
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_each_resumed_segment_still_queues_its_own_follow_up(tmp_path):
    _write_inventory(tmp_path)
    _run_prologue(tmp_path, ResumeConfig(), None, job_id="200")
    _run_prologue(tmp_path, ResumeConfig(), None, job_id="200")  # requeue of the first segment
    resumed = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", job_id="999", prev_job_id="200")

    assert "REACHED_WORK" in resumed.stdout
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afterany:999",
        "--dependency=afternotok:999",
        "--dependency=afterany:999",
    ]


# sbatch's reply when a QOS caps the jobs a user may have submitted.
_QOS_SUBMIT_LIMIT_REPLY = (
    "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
    "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
    "(job submit limit, user's size and/or time limits)"
)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_the_scheduler_refuses_is_a_typed_stop_not_unknown_accounting(tmp_path):
    refused = _run_prologue(tmp_path, ResumeConfig(), None, extra_env={"FAKE_SBATCH_REFUSES": _QOS_SUBMIT_LIMIT_REPLY})

    # This allocation still does its work; nothing was queued, so no sentinel either.
    assert refused.returncode == 0 and "REACHED_WORK" in refused.stdout, refused.stdout + refused.stderr
    assert len(_sbatch_calls(tmp_path)) == 1
    marker = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert marker == (
        f"{FOLLOW_UP_SUBMISSION_REFUSED}: sbatch exited 1: sbatch: error: QOSMaxSubmitJobPerUserLimit "
        "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
        "(job submit limit, user's size and/or time limits)\n"
    )
    assert "QOSMaxSubmitJobPerUserLimit" in refused.stdout and "gym eval resubmit --recover" in refused.stdout
    # Its claim named no job: none is left behind for a later reader to take as an unobserved follow-up.
    assert not list(tmp_path.glob(".gym_followup_queued.*"))


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "reply,refused",
    [
        (_QOS_SUBMIT_LIMIT_REPLY, True),
        (
            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified",
            True,
        ),
        ("sbatch: error: Batch job submission failed: Requested node configuration is not available", True),
        ("sbatch: error: Batch job submission failed: Job dependency problem", True),
        # The request or its reply may have been lost: the job may exist, so the accounting stays unknown.
        ("sbatch: error: Batch job submission failed: Socket timed out on send/recv operation", False),
        ("sbatch: error: Batch job submission failed: Transport endpoint is not connected", False),
        ("sbatch: error: Batch job submission failed: Broken pipe", False),
        ("sbatch: error: Batch job submission failed: Unspecified error", False),
        ("sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)", False),
    ],
)
def test_only_a_definite_refusal_of_the_follow_up_is_a_typed_stop(tmp_path, reply, refused):
    _run_prologue(tmp_path, ResumeConfig(), None, extra_env={"FAKE_SBATCH_REFUSES": reply})

    marker = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert marker.startswith(FOLLOW_UP_SUBMISSION_REFUSED if refused else ACCOUNTING_UNKNOWN_PREFIX), marker


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_recovery_resubmits_a_run_whose_follow_up_the_scheduler_refused(tmp_path):
    from nemo_gym.orchestration.recovery import RecoveryAction, decide_from_collected
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect, ident, write_job_dir

    # The segment's prologue took its follow-up claim, and the scheduler refused the submission.
    job = write_job_dir(tmp_path, n=2, results=[ident(0, reward=1.0)], failures=[])
    _run_prologue(job, ResumeConfig(), None, extra_env={"FAKE_SBATCH_REFUSES": _QOS_SUBMIT_LIMIT_REPLY})
    collected = collect(job)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    # The accounting is known: the chain stopped on a refusal, not on an unobserved submission.
    assert collected.infra_retries_state != "unknown" and collected.walltime_state != "unknown"
    assert decision.action == RecoveryAction.RESUBMIT, decision.reason
    assert f"the chain recorded: {FOLLOW_UP_SUBMISSION_REFUSED}" in decision.reason


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_submission_without_a_definite_reply_stays_unknown_and_logs_the_reply(tmp_path):
    failed = _run_prologue(tmp_path, ResumeConfig(), None, sbatch_loses_reply=True)

    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == (
        f"{ACCOUNTING_UNKNOWN_PREFIX} Follow-up submission may have succeeded without an observed reply.\n"
    )
    assert "sbatch exited 1: sbatch: error: Socket timed out on send/recv operation" in failed.stdout


def test_inspection_names_a_chain_stopped_on_a_refused_follow_up(tmp_path):
    from nemo_gym.orchestration.recovery import decide_from_collected
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect, ident, write_job_dir

    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    line = f"{FOLLOW_UP_SUBMISSION_REFUSED}: sbatch exited 1: sbatch: error: QOSMaxSubmitJobPerUserLimit"
    (job / INCOMPLETE_MARKER_NAME).write_text(line + "\n")
    collected = collect(job)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert collected.chain_continuation_note == line
    assert f"the chain recorded: {line}" in decision.reason


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_failed_submission_uncertainty_survives_a_later_invocation(tmp_path):
    failed = _run_prologue(tmp_path, ResumeConfig(), None, sbatch_fails=True)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert "Failed to submit auto-resume follow-up" in failed.stdout
    assert requeued.returncode == 1 and "REACHED_WORK" not in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_the_walltime_cap_stop_names_a_continuation_the_cli_accepts(tmp_path, monkeypatch):
    import nemo_gym.orchestration.submit as submit_module
    from nemo_gym.cli.main import main

    _write_inventory(tmp_path)
    (tmp_path / ".gym_accumulated_walltime").write_text("3590")
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), "TIMEOUT")
    hint = re.search(r"continue with gym eval resubmit (--recover [^,]*),", result.stdout)
    assert hint is not None, result.stdout
    options = shlex.split(hint[1].replace("HH:MM:SS", "02:00:00").replace("TEXT", "longer"))

    calls = []

    def dispatched(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop after the arguments were accepted")

    monkeypatch.setattr(submit_module, "recover", dispatched)
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "resubmit", "gym-job-1", "bench", *options])
    with pytest.raises((RuntimeError, SystemExit)):
        main()

    # The hinted options parse and reach the recovery as given.
    assert len(calls) == 1
    assert (calls[0]["max_walltime"], calls[0]["walltime_reason"]) == ("02:00:00", "longer")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_with_less_than_a_minute_left_names_a_continuation_the_cli_accepts(tmp_path, monkeypatch):
    import nemo_gym.orchestration.submit as submit_module
    from nemo_gym.cli.main import main

    _write_inventory(tmp_path)
    # 3500 s stored plus the predecessor's 60 s leaves 40 s of the 3600 s cap: under a minute.
    (tmp_path / ".gym_accumulated_walltime").write_text("3500")
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), "TIMEOUT")
    assert result.returncode == 1 and "Less than one minute remains in the walltime budget (3560s of 3600s)" in (
        result.stdout
    )
    hint = re.search(r"continue with gym eval resubmit (--recover [^,]*),", result.stdout)
    assert hint is not None, result.stdout
    options = shlex.split(hint[1].replace("HH:MM:SS", "02:00:00").replace("TEXT", "longer"))

    calls = []

    def dispatched(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop after the arguments were accepted")

    monkeypatch.setattr(submit_module, "recover", dispatched)
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "resubmit", "gym-job-1", "bench", *options])
    with pytest.raises((RuntimeError, SystemExit)):
        main()
    assert len(calls) == 1
    assert (calls[0]["max_walltime"], calls[0]["walltime_reason"]) == ("02:00:00", "longer")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_stopped_at_the_walltime_cap_exits_non_zero_and_marks_the_run_incomplete(tmp_path):
    _write_inventory(tmp_path)
    (tmp_path / ".gym_accumulated_walltime").write_text("3590")

    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), "TIMEOUT")

    assert result.returncode == 1, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert (
        "walltime cap with rollouts missing; continue with gym eval resubmit --recover --max-walltime HH:MM:SS "
        "--walltime-reason TEXT, with HH:MM:SS above the cap: the retry and walltime budgets carry over."
        in result.stdout
    )
    assert "resubmit to continue" not in result.stdout
    assert "walltime cap" in (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert not (tmp_path / "sbatch_calls").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("state", ["CANCELLED", "CANCELLED by 1234"])
def test_a_chain_stopped_after_a_cancel_exits_non_zero_and_marks_the_run_incomplete(tmp_path, state):
    result = _run_prologue(tmp_path, ResumeConfig(), state)

    assert result.returncode == 1, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert (
        "was cancelled. Chain stopped with rollouts missing; continue with gym eval resubmit --recover, which "
        "carries the retry and walltime budgets over." in result.stdout
    )
    assert (tmp_path / INCOMPLETE_MARKER_NAME).exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_whose_previous_job_completed_still_exits_zero(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), "COMPLETED")

    assert result.returncode == 0, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_that_continues_clears_an_earlier_incomplete_marker(tmp_path):
    _write_inventory(tmp_path)
    (tmp_path / INCOMPLETE_MARKER_NAME).write_text("stopped earlier")

    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="10:00:00"), "TIMEOUT")

    assert result.returncode == 0, result.stdout
    assert "REACHED_WORK" in result.stdout
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("claim", ["file", "directory"])
def test_a_dependent_job_of_another_run_is_not_this_runs_follow_up(tmp_path, claim):
    """Another run's job waits on this job id too; only a job in this job dir is this run's follow-up."""
    marker = tmp_path / ".gym_followup_queued.200"
    marker.mkdir() if claim == "directory" else marker.touch()
    (tmp_path / "pending").write_text("998|afternotok:200(unfulfilled)|/another/run/dir\n")

    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert requeued.returncode == 1 and "REACHED_WORK" not in requeued.stdout
    assert _sbatch_calls(tmp_path) == []
    assert marker.exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("claim", ["empty", "recorded", "directory"])
def test_a_follow_up_pending_in_this_job_dir_queues_none(tmp_path, claim):
    marker = tmp_path / ".gym_followup_queued.200"
    if claim == "directory":
        marker.mkdir()
    else:
        marker.write_text("" if claim == "empty" else "999\n")
    (tmp_path / "pending").write_text(f"999|afternotok:200(unfulfilled)|{_job_dir(tmp_path)}\n")

    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    if claim == "directory":
        assert requeued.returncode == 1 and "REACHED_WORK" not in requeued.stdout
    else:
        assert "REACHED_WORK" in requeued.stdout
        assert "already queued: 999" in requeued.stdout
        assert marker.read_text() == "999\n"
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_listed_through_a_symlinked_job_dir_still_counts(tmp_path):
    """squeue reports the --chdir path as given, which can be a symlink, while $PWD in the job is the resolved
    path: the same directory must match, or every requeue would queue another follow-up."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(job_dir, target_is_directory=True)
    (job_dir / ".gym_followup_queued.200").touch()
    (job_dir / "pending").write_text(f"999|afternotok:200(unfulfilled)|{link}\n")

    requeued = _run_prologue(job_dir, ResumeConfig(), None)

    assert "REACHED_WORK" in requeued.stdout
    assert "already queued: 999" in requeued.stdout
    assert _sbatch_calls(job_dir) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_symlink_to_another_dir_does_not_count(tmp_path):
    job_dir = tmp_path / "job"
    other = tmp_path / "other"
    job_dir.mkdir()
    other.mkdir()
    link = tmp_path / "link"
    link.symlink_to(other, target_is_directory=True)
    (job_dir / ".gym_followup_queued.200").touch()
    (job_dir / "pending").write_text(f"998|afternotok:200(unfulfilled)|{link}\n")

    _run_prologue(job_dir, ResumeConfig(), None)

    assert _sbatch_calls(job_dir) == []
    assert (job_dir / INCOMPLETE_MARKER_NAME).read_text().startswith("GYM_ACCOUNTING_UNKNOWN:")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_pending_in_a_hidden_partition_is_found(tmp_path):
    """squeue lists jobs in a hidden partition, ours included, only with --all."""
    _run_prologue(tmp_path, ResumeConfig(), None, hidden_partition=True)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None, hidden_partition=True)

    assert "already queued: 999" in requeued.stdout
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afterany:999",
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_whose_sbatch_reply_was_lost_is_not_queued_again(tmp_path):
    """A lost reply remains unknown even when a later invocation sees some pending job."""
    failed = _run_prologue(tmp_path, ResumeConfig(), None, sbatch_loses_reply=True)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert "Failed to submit auto-resume follow-up" in failed.stdout
    assert requeued.returncode == 1 and "REACHED_WORK" not in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("state", ["FAILED", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"])
def test_zero_retry_budget_never_reaches_work_after_failure(tmp_path, state):
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=0), state)
    assert result.returncode == 1
    assert "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []
    assert not (tmp_path / ".gym_infra_retries").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("state", ["PREEMPTED", "NODE_FAIL", "BOOT_FAIL"])
def test_infrastructure_retry_is_spent_once_before_work(tmp_path, state):
    resume = ResumeConfig(max_retries=1)
    first = _run_prologue(tmp_path, resume, state)
    assert first.returncode == 0 and "REACHED_WORK" in first.stdout
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"
    second = _run_prologue(tmp_path, resume, state, job_id="300")
    assert second.returncode == 1 and "REACHED_WORK" not in second.stdout
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "state,elapsed",
    [
        ("", "60"),
        ("RUNNING", "60"),
        ("UNRECOGNIZED", "60"),
        ("FAILED", ""),
        ("NODE_FAIL", "-1"),
        ("TIMEOUT", "nan"),
        ("FAILED", "1+2"),
        ("FAILED", "60\nFAILED|60"),
    ],
)
def test_unknown_accounting_cannot_run_or_change_counters(tmp_path, state, elapsed):
    (tmp_path / ".gym_accumulated_walltime").write_text("600\n")
    (tmp_path / ".gym_infra_retries").write_text("1\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="04:00:00"), state, elapsed=elapsed)
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []
    assert (tmp_path / ".gym_accumulated_walltime").read_text() == "600\n"
    assert (tmp_path / ".gym_infra_retries").read_text() == "1\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("state", ["TIMEOUT", "BOOT_FAIL", "PREEMPTED", "NODE_FAIL"])
def test_remaining_walltime_shortens_four_hour_allocation(tmp_path, state):
    _write_inventory(tmp_path)
    (tmp_path / ".gym_accumulated_walltime").write_text("3600\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="04:00:00"), state, elapsed="3630")
    assert result.returncode == 0 and "REACHED_WORK" in result.stdout
    assert "update JobId=200 TimeLimit=119" in (tmp_path / "scontrol_calls").read_text()
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "7230"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("state", ["TIMEOUT", "FAILED", "PREEMPTED", "NODE_FAIL"])
def test_all_failure_states_obey_exhausted_walltime(tmp_path, state):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="00:01:00"), state)
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_remaining_walltime_never_increases_existing_allocation(tmp_path):
    _write_inventory(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="04:00:00"), "TIMEOUT", time_limit="00:30:00")
    assert result.returncode == 0
    assert "update" not in (tmp_path / "scontrol_calls").read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("limit,failed", [("UNKNOWN", False), ("", True)])
def test_unavailable_allocation_limit_stops_before_work(tmp_path, limit, failed):
    result = _run_prologue(
        tmp_path, ResumeConfig(max_walltime="04:00:00"), None, time_limit=limit, scontrol_fails=failed
    )
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("counter", [".gym_accumulated_walltime", ".gym_infra_retries"])
@pytest.mark.parametrize("value", ["", "-1", "nan", "1+2", "99999999999999999999999"])
def test_malformed_stored_counter_stops_before_work(tmp_path, counter, value):
    (tmp_path / counter).write_text(value)
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="04:00:00"), "FAILED")
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []
    assert (tmp_path / counter).read_text() == value


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_subminute_allowance_stops_instead_of_rounding_up(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="00:01:30"), "TIMEOUT")
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_recover_start_spends_retry_and_preserves_remaining_walltime(tmp_path):
    (tmp_path / ".gym_infra_retries").write_text("2\n")
    (tmp_path / ".gym_accumulated_walltime").write_text("7200\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=3, max_walltime="04:00:00"), None, keep_budget=True)
    assert result.returncode == 0 and "REACHED_WORK" in result.stdout
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "3"
    assert (tmp_path / ".gym_accumulated_walltime").read_text() == "7200\n"
    assert "update JobId=200 TimeLimit=120" in (tmp_path / "scontrol_calls").read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("update_result", ["failed", "unknown", "unchanged"])
def test_failed_or_unconfirmed_time_limit_shrink_stops_before_work(tmp_path, update_result):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), "NODE_FAIL", update_result=update_result)
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []
    assert not (tmp_path / ".gym_infra_retries").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("limit", ["UNLIMITED", "1-04:00:00"])
def test_unlimited_or_multiday_allocation_is_capped(tmp_path, limit):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), None, time_limit=limit)
    assert result.returncode == 0 and "REACHED_WORK" in result.stdout
    assert (tmp_path / "time_limit").read_text().strip() == "01:00:00"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("restart_count", ["1", "2", "-1", "unknown"])
@pytest.mark.parametrize("keep_budget", [False, True])
def test_requeue_with_unaccounted_previous_incarnation_stops_without_recounting(tmp_path, restart_count, keep_budget):
    # Same job id, predecessor and queued follow-up as before Slurm requeued the allocation.
    (tmp_path / ".gym_job_chain").write_text("100\n200\n")
    (tmp_path / ".gym_followup_queued.200").write_text("999\n")
    (tmp_path / "pending").write_text(f"999|afternotok:200(unfulfilled)|{tmp_path}\n")
    (tmp_path / ".gym_accumulated_walltime").write_text("600\n")
    (tmp_path / ".gym_infra_retries").write_text("1\n")
    answer = tmp_path / "saved-answer.jsonl"
    answer.write_text('{"reward":0,"answer":"saved"}\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result = _run_prologue(
        tmp_path,
        ResumeConfig(max_retries=3, max_walltime="04:00:00"),
        "NODE_FAIL",
        restart_count=restart_count,
        keep_budget=keep_budget,
    )
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []
    assert not (tmp_path / "sacct_calls").exists()
    assert not (tmp_path / "scontrol_calls").exists()
    assert all((tmp_path / name).read_bytes() == contents for name, contents in before.items())


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_requeued_manual_resubmit_does_not_reset_spent_history_again(tmp_path):
    (tmp_path / ".gym_accumulated_walltime").write_text("600\n")
    (tmp_path / ".gym_infra_retries").write_text("1\n")
    result = _run_prologue(
        tmp_path, ResumeConfig(max_walltime="04:00:00"), None, restart_count="1", manual_resubmit=True
    )
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert (tmp_path / ".gym_accumulated_walltime").read_text() == "600\n"
    assert (tmp_path / ".gym_infra_retries").read_text() == "1\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("state", ["FAILED", "OUT_OF_MEMORY", "DEADLINE"])
@pytest.mark.parametrize("saved_output", [False, True])
def test_a_failure_without_scheduler_infrastructure_cause_stops_and_keeps_its_elapsed_time(
    tmp_path, state, saved_output
):
    # A FAILED job's exit code does not tell a broken config from a broken node, and the same allocation runs the
    # same config again. So, with retries and walltime left, the chain stops instead of spending another job.
    (tmp_path / ".gym_accumulated_walltime").write_text("600\n")
    (tmp_path / ".gym_infra_retries").write_text("1\n")
    if saved_output:
        (tmp_path / "rollouts.jsonl").write_text('{"reward": 0.0}\n')
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=3, max_walltime="04:00:00"), state, elapsed="42")
    assert result.returncode == 1, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "642"
    assert (tmp_path / ".gym_infra_retries").read_text() == "1\n"
    assert state in (tmp_path / ".gym_chain_incomplete").read_text()
    assert (tmp_path / ".gym_job_chain").read_text() == "200\n"
    if saved_output:
        assert (tmp_path / "rollouts.jsonl").read_text() == '{"reward": 0.0}\n'


def test_the_prologue_carries_the_scheduler_cause_capability():
    from nemo_gym.orchestration.executors.resume_script import SCHEDULER_CAUSE_MARKER

    assert SCHEDULER_CAUSE_MARKER in render_resume_prologue(ResumeConfig(max_retries=1)).splitlines()


@pytest.mark.parametrize("manual_reset", [False, True])
def test_lost_reply_and_guard_stopped_follower_cannot_reopen_unknown_budget(tmp_path, manual_reset):
    resume = ResumeConfig(max_retries=3, max_walltime="01:00:00")
    first = _run_prologue(tmp_path, resume, None, job_id="100", sbatch_loses_reply=True)
    follower = _run_prologue(tmp_path, resume, "", job_id="999", prev_job_id="100")
    assert first.returncode == 0 and "REACHED_WORK" in first.stdout
    assert follower.returncode == 1 and "REACHED_WORK" not in follower.stdout
    (tmp_path / ".gym_accumulated_walltime").write_text("123\n")
    (tmp_path / ".gym_infra_retries").write_text("1\n")
    before = {name: (tmp_path / name).read_bytes() for name in (".gym_accumulated_walltime", ".gym_infra_retries")}
    later = _run_prologue(tmp_path, resume, None, job_id="200", manual_resubmit=manual_reset)
    assert later.returncode == 1 and "REACHED_WORK" not in later.stdout
    assert (tmp_path / ".gym_job_chain").read_text().split() == ["100", "999", "200"]
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text().startswith("GYM_ACCOUNTING_UNKNOWN:")
    assert len(_sbatch_calls(tmp_path)) == 1


def test_unavailable_accounting_records_the_guard_stopped_allocation(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(), "", job_id="200")
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert (tmp_path / ".gym_job_chain").read_text() == "200\n"
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text().startswith("GYM_ACCOUNTING_UNKNOWN:")


def test_multiple_observed_successors_are_not_reduced_to_the_first(tmp_path):
    (tmp_path / ".gym_followup_queued.200").touch()
    (tmp_path / "pending").write_text(
        f"998|afternotok:200(unfulfilled)|{tmp_path}\n999|afternotok:200(unfulfilled)|{tmp_path}\n"
    )
    result = _run_prologue(tmp_path, ResumeConfig(), None)
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert _sbatch_calls(tmp_path) == []


def test_manual_follower_without_predecessor_cannot_claim_a_fresh_chain(tmp_path):
    _run_prologue(tmp_path, ResumeConfig(), None, job_id="100")
    result = _run_prologue(tmp_path, ResumeConfig(), None, job_id="200")
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert (tmp_path / ".gym_job_chain").read_text().split() == ["100", "200"]


def test_collector_preserves_unknown_budget_even_with_apparently_complete_counters(tmp_path):
    import json

    from nemo_gym.orchestration.recovery import COLLECT_SCRIPT, CollectedEvidence, decide_from_collected

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text(
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0}) + "\n"
    )
    (artifacts / "rollouts.jsonl").write_text("")
    (tmp_path / ".gym_accumulated_walltime").write_text("60\n")
    (tmp_path / ".gym_infra_retries").write_text("1\n")
    (tmp_path / INCOMPLETE_MARKER_NAME).write_text("GYM_ACCOUNTING_UNKNOWN: prior follower unobserved\n")
    raw = subprocess.check_output([sys.executable, "-c", COLLECT_SCRIPT, str(tmp_path)], text=True)
    collected = CollectedEvidence.model_validate_json(raw)
    assert collected.walltime_used_s == 60 and collected.infra_retries_used == 1
    assert collected.walltime_state == collected.infra_retries_state == "unknown"
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=3600, live_job=False)
    assert decision.action.value == "stop"


@pytest.mark.parametrize("name", [".gym_accumulated_walltime", ".gym_infra_retries"])
def test_manual_reset_cannot_erase_malformed_spent_counters(tmp_path, name):
    (tmp_path / name).write_text("unknown\n")
    result = _run_prologue(tmp_path, ResumeConfig(), None, manual_resubmit=True)
    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert (tmp_path / name).read_text() == "unknown\n"
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text().startswith("GYM_ACCOUNTING_UNKNOWN:")


@pytest.mark.parametrize("mutation", [None, "create", "change", "remove"])
def test_collector_binds_dynamic_followup_claim_set_and_contents(tmp_path, mutation):
    import json

    from nemo_gym.orchestration.recovery import COLLECT_SCRIPT, CollectedEvidence

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text(
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0}) + "\n"
    )
    (artifacts / "rollouts.jsonl").write_text("")
    (tmp_path / ".gym_accumulated_walltime").write_text("60\n")
    (tmp_path / ".gym_infra_retries").write_text("1\n")
    claim = tmp_path / ".gym_followup_queued.100"
    if mutation != "create":
        claim.write_text("999\n")
    prefix = ""
    if mutation is not None:
        operation = (
            f"os.unlink({str(claim)!r})"
            if mutation == "remove"
            else f"_original_open({str(claim)!r}, 'w').write('998\\n')"
        )
        prefix = (
            "import builtins,os\n_original_open=builtins.open\n_changed=False\n"
            "def open(path, *args, **kwargs):\n"
            "    global _changed\n"
            "    if not _changed and str(path).endswith('/rollouts.jsonl') and 'source_snapshot' in globals():\n"
            "        _changed=True\n"
            f"        {operation}\n"
            "    return _original_open(path,*args,**kwargs)\n"
        )
    raw = subprocess.check_output([sys.executable, "-c", prefix + COLLECT_SCRIPT, str(tmp_path)], text=True)
    collected = CollectedEvidence.model_validate_json(raw)
    assert collected.infra_retries_used == 1 and collected.walltime_used_s == 60
    if mutation is None:
        assert collected.infra_retries_state == collected.walltime_state == "ok"
        assert collected.source_digests is not None
    else:
        assert collected.infra_retries_state == collected.walltime_state == "unknown"
        assert collected.source_digests is None


def _sentinel_of(followup_elapsed: str, **extra: str) -> dict[str, str]:
    """The environment of sentinel 200 for follow-up 777 of job 100, with 777's elapsed time in accounting."""
    return {"GYM_FOLLOWUP_SENTINEL": "777", "FAKE_FOLLOWUP": "777", "FAKE_FOLLOWUP_ELAPSED": followup_elapsed, **extra}


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_takes_the_place_of_a_follow_up_that_died_before_it_recorded_itself(tmp_path):
    _write_inventory(tmp_path)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of("14"))

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Follow-up 777 ended after 14s before it recorded itself" in result.stdout
    # The dead follow-up is persisted before the sentinel, and its 14 s are debited with the parent's 60 s.
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n200\n"
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "74"
    # The infrastructure ended the follow-up, so the takeover spends one infra retry.
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"
    # It continues as the follow-up would: its own follow-up and sentinel, which are ordinary hops. Only the new
    # sentinel is told which follow-up to watch; the follow-up does not inherit this sentinel's role.
    assert [(call.split()[0], call.split()[-1]) for call in _sbatch_calls(tmp_path)] == [
        ("--dependency=afternotok:200", "sentinel="),
        ("--dependency=afterany:999", "sentinel=999"),
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_dead_follow_up_is_debited_once_and_stays_in_the_chain_for_the_next_hop(tmp_path):
    _write_inventory(tmp_path)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    (tmp_path / ".gym_accumulated_walltime").write_text("1000\n")
    taken_over = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of("14"))
    assert taken_over.returncode == 0 and "REACHED_WORK" in taken_over.stdout, taken_over.stdout + taken_over.stderr
    # Parent 100 (60 s) and dead follow-up 777 (14 s) are both spent: 1000 + 60 + 14.
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "1074"

    # The sentinel's own follow-up 999 debits only its predecessor 200 (30 s); 777 is not counted again.
    # 200 saved a rollout, so its timeout is not a segment that would repeat itself.
    _store(tmp_path, [_RESULT])
    next_hop = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", job_id="999", prev_job_id="200", elapsed="30")
    assert next_hop.returncode == 0 and "REACHED_WORK" in next_hop.stdout, next_hop.stdout + next_hop.stderr
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "1104"
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n200\n999\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "chain,elapsed,said",
    [
        ("100\n777\n", "14", "Follow-up 777 recorded itself in the chain"),
        ("100\n", "0", "Follow-up 777 never started"),
    ],
    ids=["follow-up-ran", "follow-up-cancelled"],
)
def test_a_sentinel_leaves_the_chain_as_it_is_when_its_follow_up_ran_or_never_started(tmp_path, chain, elapsed, said):
    (tmp_path / ".gym_job_chain").write_text(chain)
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of(elapsed))

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout
    assert said in result.stdout
    assert (tmp_path / ".gym_job_chain").read_text() == chain
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_that_cannot_read_its_follow_up_stops_with_unknown_accounting(tmp_path):
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of(""))

    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert (
        (tmp_path / INCOMPLETE_MARKER_NAME)
        .read_text()
        .startswith("GYM_ACCOUNTING_UNKNOWN: Follow-up 777 accounting is unknown")
    )
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n"
    assert _sbatch_calls(tmp_path) == []
    # The stop names the operator's continuation; the marker keeps only the reason its readers match.
    assert "gym eval resubmit --recover --max-walltime HH:MM:SS --walltime-reason TEXT" in result.stdout
    assert "--recover" not in (tmp_path / INCOMPLETE_MARKER_NAME).read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_whose_accounting_is_already_unknown_names_the_continuation_and_keeps_its_marker(tmp_path):
    marker = "GYM_ACCOUNTING_UNKNOWN: Cannot record the allocation.\n"
    (tmp_path / INCOMPLETE_MARKER_NAME).write_text(marker)
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT")

    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert "Allocation accounting remains unknown" in result.stdout
    assert "gym eval resubmit --recover --max-walltime HH:MM:SS --walltime-reason TEXT" in result.stdout
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == marker
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_stops_at_the_infra_retry_cap_so_dying_follow_ups_cannot_loop(tmp_path):
    _write_inventory(tmp_path)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    (tmp_path / ".gym_infra_retries").write_text("2\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "TIMEOUT", extra_env=_sentinel_of("14"))

    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    assert "Infra retry limit (2) reached" in (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert _sbatch_calls(tmp_path) == []


def _provisional_line(job_id: str) -> str:
    return f"{PROVISIONAL_MARKER_PREFIX}{job_id} started and had not queued its follow-up; if this line remains, ..."


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("pkill") is None, reason="bash and pkill needed")
def test_a_continuation_killed_before_its_guard_leaves_its_provisional_line(tmp_path):
    _write_inventory(tmp_path)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    (tmp_path / INCOMPLETE_MARKER_NAME).write_text("An earlier stop.\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env={"FAKE_SACCT_KILLS_JOB": "1"})

    assert result.returncode == -9 and "REACHED_WORK" not in result.stdout
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n200\n"
    marker = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert marker.startswith(f"{PROVISIONAL_MARKER_PREFIX}200 started") and "gym eval resubmit --recover" in marker
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_takes_over_a_follow_up_that_recorded_itself_and_died_before_its_guard(tmp_path):
    _write_inventory(tmp_path)
    (tmp_path / ".gym_job_chain").write_text("100\n777\n")
    (tmp_path / INCOMPLETE_MARKER_NAME).write_text(_provisional_line("777") + "\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of("13"))

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "after it recorded itself and before it queued its follow-up" in result.stdout
    # 777 is in the chain once; its 13 s are debited with the parent's 60 s; the takeover spends an infra retry.
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n200\n"
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "73"
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afterany:999",
    ]
    # Its own follow-up is claimed, so its provisional line is gone.
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("case", ["deliberate-stop", "claimed"])
def test_a_sentinel_leaves_a_follow_up_that_stopped_the_chain_or_claimed_its_successor(tmp_path, case):
    (tmp_path / ".gym_job_chain").write_text("100\n777\n")
    if case == "deliberate-stop":
        (tmp_path / INCOMPLETE_MARKER_NAME).write_text("Infra retry limit (3) reached. Stopping.\n")
    else:
        (tmp_path / INCOMPLETE_MARKER_NAME).write_text(_provisional_line("777") + "\n")
        (tmp_path / ".gym_followup_queued.777").write_text("888\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of("13"))

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout
    assert "Follow-up 777 recorded itself in the chain" in result.stdout
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n"
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_finished_run_clears_the_provisional_line(tmp_path):
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "COMPLETED")

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()


def test_inspection_names_a_continuation_that_died_before_its_guard(tmp_path):
    from nemo_gym.orchestration.recovery import decide_from_collected
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect, ident, write_job_dir

    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    (job / INCOMPLETE_MARKER_NAME).write_text(_provisional_line("777") + "\n")
    collected = collect(job)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert collected.chain_continuation_note.startswith(f"{PROVISIONAL_MARKER_PREFIX}777 started")
    assert f"the chain recorded: {PROVISIONAL_MARKER_PREFIX}777 started" in decision.reason

    (job / INCOMPLETE_MARKER_NAME).write_text("Infra retry limit (3) reached. Stopping.\n")
    assert (
        "the chain recorded"
        not in decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False).reason
    )


_RESULT = '{"_ng_task_index":0,"_ng_rollout_index":0,"reward":1.0}'
_KILLED = '{"_ng_task_index":1,"_ng_rollout_index":0,"_ng_failure_class":"kill_shaped"}'
_NOT_SENT = '{"_ng_task_index":2,"_ng_rollout_index":0,"_ng_failure_class":"transient","_ng_failure_record":{"delivery":"not_sent"}}'
_FAILED = '{"_ng_task_index":3,"_ng_rollout_index":0,"_ng_failure_class":"transient"}'
_KILLED_TERMINAL = (
    '{"_ng_task_index":4,"_ng_rollout_index":0,"_ng_failure_class":"kill_shaped","_ng_failure_terminal":true}'
)


def _store(tmp_path, results=(), failures=(), tail="") -> None:
    """The Store's two append-only files; ``tail`` is an unterminated last line of the results file."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "rollouts.jsonl").write_text("".join(f"{row}\n" for row in results) + tail)
    (artifacts / "rollouts_failures.jsonl").write_text("".join(f"{row}\n" for row in failures))


def _rows_at_start(tmp_path, job_id, value) -> None:
    (tmp_path / f"{ROWS_AT_START_NAME}.{job_id}").write_text(f"{value}\n")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    ("results", "failures", "tail"),
    [
        ([_RESULT, _FAILED], [], ""),
        ([_RESULT, _FAILED], [_KILLED, _NOT_SENT], ""),
        ([_RESULT, _FAILED], [], '{"_ng_task_index":5,"rew'),
    ],
    ids=["nothing-written", "only-interruptions-written", "only-a-cut-row-written"],
)
def test_a_resumed_segment_that_timed_out_without_saving_a_rollout_ends_the_chain(tmp_path, results, failures, tail):
    _write_inventory(tmp_path)
    _store(tmp_path, results, failures, tail)
    _rows_at_start(tmp_path, "100", 2)
    (tmp_path / ".gym_job_chain").write_text("50\n100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", elapsed="14400")

    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout
    marker = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert marker.startswith("Previous job 100 ended TIMEOUT after 14400s: the segment saved no rollout")
    assert "--recover --max-walltime HH:MM:SS --walltime-reason TEXT" in marker
    assert _sbatch_calls(tmp_path) == []
    # The segment's time is still accounted.
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "14400"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    ("failures", "start"),
    [([], "2"), ([_FAILED], "3"), ([_KILLED_TERMINAL], "3"), ([], None), ([], "unreadable")],
    ids=["saved-a-result", "saved-a-failure", "a-killed-row-reached-its-cap", "no-record", "unreadable-record"],
)
def test_a_resumed_segment_that_saved_a_row_or_left_no_record_continues(tmp_path, failures, start):
    _write_inventory(tmp_path)
    _store(tmp_path, [_RESULT, _FAILED, _RESULT], failures)
    if start is not None:
        _rows_at_start(tmp_path, "100", start)
    # A segment killed at its walltime never flushed coverage: the report still says it saved nothing.
    report = {"outcomes_recorded": 0, "progress_outcomes_recorded": 0, "allocation_id": "100"}
    (tmp_path / "artifacts" / "rollouts_coverage.json").write_text(json.dumps(report))
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT")

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "saved no rollout" not in result.stdout
    # This segment records its own start for the next one.
    assert (tmp_path / f"{ROWS_AT_START_NAME}.200").read_text().strip() == str(3 + len(failures))


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_the_first_segment_records_no_start_so_its_successor_is_never_stopped_for_it(tmp_path):
    _write_inventory(tmp_path)
    _store(tmp_path, [_RESULT])
    result = _run_prologue(tmp_path, ResumeConfig(), None)

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert not (tmp_path / f"{ROWS_AT_START_NAME}.200").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_the_recovery_the_no_progress_stop_names_runs_and_records_its_start(tmp_path):
    _write_inventory(tmp_path)
    _store(tmp_path, [_RESULT, _FAILED])
    _rows_at_start(tmp_path, "100", 2)
    (tmp_path / ".gym_job_chain").write_text("50\n100\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="08:00:00"), None, keep_budget=True)

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "saved no rollout" not in result.stdout
    assert (tmp_path / f"{ROWS_AT_START_NAME}.200").read_text().strip() == "2"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_cut_before_the_inventory_still_takes_the_fresh_restart_path(tmp_path):
    _rows_at_start(tmp_path, "100", 0)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT")

    assert result.returncode == 1
    marker = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert "before the task inventory was written" in marker and "saved no rollout" not in marker


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_records_why_the_chain_stopped_when_its_follow_up_was_cancelled_before_it_started(tmp_path):
    # The parent timed out, so its follow-up should have run; someone cancelled it before it started.
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of("0"))

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    record = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert record.startswith("Follow-up 777 was cancelled before it started")
    assert "job 100 ended TIMEOUT" in record and "gym eval resubmit" in record and "--recover" in record
    assert record.strip() in result.stdout
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n"
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_records_nothing_when_its_parent_completed_and_slurm_cancelled_the_follow_up(tmp_path):
    # A completed segment finished the run: Slurm cancels its afternotok follow-up, and the chain is done.
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "COMPLETED", extra_env=_sentinel_of("0"))

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()
    assert "job 100 completed" in result.stdout
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "existing",
    ["Accumulated walltime reached the cap; continue with gym eval resubmit --recover.", None],
    ids=["deliberate-stop-kept", "provisional-line-replaced"],
)
def test_a_cancelled_follow_up_keeps_a_deliberate_stop_and_replaces_a_provisional_line(tmp_path, existing):
    (tmp_path / ".gym_job_chain").write_text("100\n")
    marker = tmp_path / INCOMPLETE_MARKER_NAME
    marker.write_text((existing or f"{PROVISIONAL_MARKER_PREFIX}100 started and had not queued its follow-up") + "\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "FAILED", extra_env=_sentinel_of("0"))

    assert result.returncode == 0, result.stdout + result.stderr
    if existing:
        assert marker.read_text() == existing + "\n"
    else:
        assert marker.read_text().startswith("Follow-up 777 was cancelled before it started")


def test_inspection_names_a_follow_up_cancelled_before_it_started(tmp_path):
    from nemo_gym.orchestration.recovery import decide_from_collected
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect, ident, write_job_dir

    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    line = "Follow-up 777 was cancelled before it started (job 100 ended TIMEOUT); the chain stopped."
    (job / INCOMPLETE_MARKER_NAME).write_text(line + "\n")
    collected = collect(job)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert collected.chain_continuation_note == line
    assert f"the chain recorded: {line}" in decision.reason


def test_inspection_names_a_chain_stopped_on_rejected_sandbox_credentials(tmp_path):
    from nemo_gym.orchestration.executors.resume_script import CREDENTIALS_REJECTED_REASON
    from nemo_gym.orchestration.recovery import decide_from_collected
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect, ident, write_job_dir

    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    line = f"Previous job 100 ended FAILED: {CREDENTIALS_REJECTED_REASON}"
    (job / INCOMPLETE_MARKER_NAME).write_text(line + "\n")
    collected = collect(job)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert collected.chain_continuation_note == line
    assert "the chain recorded: Previous job 100 ended FAILED: provider_credentials_rejected" in decision.reason


# ---------------------------------------------------------------------------
# A complete, exported run ends the chain before any retry or budget decision
# ---------------------------------------------------------------------------

_MLFLOW_CONFIG = "mlflow_tracking_uri: http://tracking.example\nmlflow_experiment_name: evals\nmlflow_run_name: run\n"


def _finished_run(tmp_path, monkeypatch, case="healthy", config=_MLFLOW_CONFIG, receipt="current") -> None:
    """A managed run of two rollouts as its last segment left it: Store, saved config, aggregate and export receipt."""
    from tests.unit_tests.orchestration.test_managed_recovery_selection import prepare_managed

    monkeypatch.setenv("SLURM_JOB_ID", "100")
    prepare_managed(tmp_path, case, n=2)
    artifacts = tmp_path / "artifacts"
    (artifacts / "rollouts_resolved_config.yaml").write_text(config)
    (artifacts / "rollouts_aggregate_metrics.json").write_text("[]\n")
    if receipt is None:
        return
    files = {
        "rollouts_jsonl_fpath": "rollouts.jsonl",
        "resolved_config_yaml_fpath": "rollouts_resolved_config.yaml",
        "aggregate_metrics_json_fpath": "rollouts_aggregate_metrics.json",
        "failures_jsonl_fpath": "rollouts_failures.jsonl",
        "materialized_inputs_jsonl_fpath": "rollouts_materialized_inputs.jsonl",
    }
    digests = {key: hashlib.sha256((artifacts / name).read_bytes()).hexdigest() for key, name in files.items()}
    receipt_value = {"version": 1, "run_id": "tracked", "artifacts": digests, "uploaded_rollouts": True}
    (artifacts / "rollouts_mlflow_export_done").write_text(json.dumps(receipt_value, separators=(",", ":")))
    if receipt == "stale":
        # The rollouts changed after the export: the receipt no longer describes them.
        with (artifacts / "rollouts.jsonl").open("a") as rollouts:
            rollouts.write("\n")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("prev_state", ["TIMEOUT", "FAILED", "PREEMPTED"])
def test_a_follow_up_ends_the_chain_of_a_complete_exported_run_whatever_the_previous_state(
    tmp_path, monkeypatch, prev_state
):
    # The segment saved the last rollout and the export receipt, then hit its walltime.
    _finished_run(tmp_path, monkeypatch)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), prev_state)

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    assert (
        "Run complete: the Store covers all 2 expected rollouts and its MLflow export receipt is current; "
        "job 200 ends the chain with nothing to recover." in result.stdout
    )
    assert "no eligible retries" not in result.stdout and "--recover" not in result.stdout
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()
    assert _sbatch_calls(tmp_path) == []
    assert not (tmp_path / ".gym_infra_retries").exists()
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n200\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "chain,elapsed,marker",
    [("100\n", "14", None), ("100\n", "0", None), ("100\n777\n", "13", _provisional_line("777"))],
    ids=["follow-up-died-unrecorded", "follow-up-cancelled", "follow-up-died-before-its-guard"],
)
def test_a_sentinel_ends_the_chain_of_a_complete_exported_run_instead_of_taking_over(
    tmp_path, monkeypatch, chain, elapsed, marker
):
    _finished_run(tmp_path, monkeypatch)
    (tmp_path / ".gym_job_chain").write_text(chain)
    if marker is not None:
        (tmp_path / INCOMPLETE_MARKER_NAME).write_text(marker + "\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of(elapsed))

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    assert "Run complete: the Store covers all 2 expected rollouts and its MLflow export receipt is current" in (
        result.stdout
    )
    assert "--recover" not in result.stdout
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()
    assert _sbatch_calls(tmp_path) == []
    assert not (tmp_path / ".gym_infra_retries").exists()
    assert (tmp_path / ".gym_job_chain").read_text() == chain


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("receipt", [None, "stale"], ids=["no-receipt", "stale-receipt"])
@pytest.mark.parametrize("sentinel", [False, True], ids=["follow-up", "sentinel"])
def test_a_complete_run_without_a_current_export_receipt_names_the_missing_export(
    tmp_path, monkeypatch, receipt, sentinel
):
    _finished_run(tmp_path, monkeypatch, receipt=receipt)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of("14") if sentinel else None)

    assert result.returncode == 1 and "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    marker = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert marker == EXPORT_RECEIPT_MISSING_REASON + "\n"
    assert marker.startswith(f"{EXPORT_RECEIPT_MISSING}: the run is complete")
    assert "gym eval resubmit --recover" in marker
    assert "no eligible retries" not in result.stdout + marker
    assert "Run complete" not in result.stdout
    assert _sbatch_calls(tmp_path) == []
    assert not (tmp_path / ".gym_infra_retries").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "config,said",
    [
        (_MLFLOW_CONFIG + "mlflow_tracking_required: false\n", "it exports nothing by design"),
        ("mlflow_tracking_uri: null\nmlflow_experiment_name: evals\nmlflow_run_name: run\n", "configures no MLflow"),
        ("num_repeats: 1\n", "configures no MLflow export (mlflow_tracking_uri is unset)"),
    ],
    ids=["tracking-not-required", "tracking-uri-null", "no-mlflow-names"],
)
def test_a_complete_run_that_exports_nothing_needs_no_receipt(tmp_path, monkeypatch, config, said):
    _finished_run(tmp_path, monkeypatch, config=config, receipt=None)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT")

    assert result.returncode == 0 and "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    assert "Run complete: the Store covers all 2 expected rollouts and " in result.stdout and said in result.stdout
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_an_incomplete_run_with_an_export_receipt_continues_as_before(tmp_path, monkeypatch):
    # One rollout is still missing: a receipt (here, of the partial run) does not make the run complete.
    _finished_run(tmp_path, monkeypatch, case="missing")
    coverage = json.loads((tmp_path / "artifacts" / "rollouts_coverage.json").read_text())
    assert coverage["complete"] is False and coverage["expected"] == 2
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT")

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Run complete" not in result.stdout
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afterany:999",
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("mismatch", ["inventory", "run_id"])
def test_a_coverage_report_of_another_inventory_or_run_does_not_end_the_chain(tmp_path, monkeypatch, mismatch):
    _finished_run(tmp_path, monkeypatch)
    artifacts = tmp_path / "artifacts"
    if mismatch == "inventory":
        with (artifacts / "rollouts_materialized_inputs.jsonl").open("a") as inputs:
            inputs.write('{"_ng_task_index": 2, "_ng_rollout_index": 0}\n')
    else:
        report = json.loads((artifacts / "rollouts_coverage.json").read_text())
        (artifacts / "rollouts_coverage.json").write_text(json.dumps(report | {"run_id": "0" * 32}, indent=2) + "\n")
    (tmp_path / ".gym_job_chain").write_text("100\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT")

    assert "Run complete" not in result.stdout, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_keeps_a_deliberate_stop_on_a_complete_run(tmp_path, monkeypatch):
    _finished_run(tmp_path, monkeypatch)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    stop = "Accumulated walltime reached the cap; continue with gym eval resubmit --recover."
    (tmp_path / INCOMPLETE_MARKER_NAME).write_text(stop + "\n")
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", extra_env=_sentinel_of("0"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Run complete" not in result.stdout
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == stop + "\n"
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("pkill") is None, reason="bash and pkill needed")
@pytest.mark.parametrize(
    "death",
    [{"admission_command": "kill -9 $$"}, {"extra_env": {"FAKE_SACCT_KILLS_JOB": "1"}}],
    ids=["after-the-counter-write", "before-the-counter-write"],
)
def test_a_sentinel_debits_the_parent_once_when_its_follow_up_died_around_the_counter_write(tmp_path, death):
    _write_inventory(tmp_path)
    (tmp_path / "artifacts" / "rollouts_manifest.json").write_text("{}\n")  # the follow-up runs admission
    (tmp_path / ".gym_job_chain").write_text("100\n")
    # Follow-up 777 of parent 100 (60 s) records itself, then dies (in admission, after it wrote the counter, or in
    # its accounting query, before).
    followup = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", job_id="777", **death)
    assert followup.returncode == -9 and "REACHED_WORK" not in followup.stdout, followup.stdout + followup.stderr
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n"
    after_followup = (
        (tmp_path / ".gym_accumulated_walltime").read_text().strip() if "admission_command" in death else ""
    )
    assert after_followup in ("60", "")

    sentinel = _run_prologue(
        tmp_path,
        ResumeConfig(),
        "TIMEOUT",
        admission_command="echo __GYM_CONTINUATION:segment",
        extra_env=_sentinel_of("13"),
    )

    assert sentinel.returncode == 0 and "REACHED_WORK" in sentinel.stdout, sentinel.stdout + sentinel.stderr
    # Parent 100 (60 s) and dead follow-up 777 (13 s), each once: 73, not 60 + 60 + 13.
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "73"
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n777\n200\n"


# ---------------------------------------------------------------------------
# A segment that completes the run cancels the sentinel and follow-up it queued
# ---------------------------------------------------------------------------

# The driver step of a segment that saves the last rollout and exports the run: it leaves a finished run's artifacts.
_FINISHES_THE_RUN = "cp -R done/artifacts/. artifacts/"


def _segment_finishing_the_run(tmp_path, monkeypatch, **finished) -> None:
    """Job 200 continues job 100 of a run with a rollout missing; `_FINISHES_THE_RUN` leaves it as `finished` says."""
    _finished_run(tmp_path / "done", monkeypatch, **finished)
    _finished_run(tmp_path, monkeypatch, case="missing")
    (tmp_path / ".gym_job_chain").write_text("100\n")


def _chain_end_entries(tmp_path) -> list[dict]:
    journal = tmp_path / CHAIN_END_JOURNAL_NAME
    return [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_segment_that_completes_the_run_cancels_its_own_sentinel_and_follow_up(tmp_path, monkeypatch):
    _segment_finishing_the_run(tmp_path, monkeypatch)
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", driver=_FINISHES_THE_RUN, sentinel_id="998")

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    # Like every segment, it queued both before its work.
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afterany:999",
    ]
    # The sentinel first: once its follow-up is gone, its afterany dependency is met and it could start.
    assert _scancel_calls(tmp_path) == ["998", "999"]
    # The record the sentinel would have written, from the segment that finished the run.
    assert (
        "Run complete: the Store covers all 2 expected rollouts and its MLflow export receipt is current; "
        "job 200 ends the chain with nothing to recover." in result.stdout
    )
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()
    [entry] = _chain_end_entries(tmp_path)
    assert entry["job_id"] == "200" and entry["ended"] == "run_complete"
    assert entry["cancelled"] == {
        "sentinel": {"job_id": "998", "result": "cancelled"},
        "follow_up": {"job_id": "999", "result": "cancelled"},
    }
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", entry["at"])
    # The chain's records stay as they were: the job is in the chain and its claims name what it queued.
    assert (tmp_path / ".gym_job_chain").read_text() == "100\n200\n"
    assert (tmp_path / ".gym_followup_queued.200").read_text() == "999\n"
    assert (tmp_path / ".gym_followup_sentinel.200").read_text() == "998\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("status", [1, 143])
def test_a_segment_whose_driver_failed_keeps_its_status_and_its_queued_jobs(tmp_path, monkeypatch, status):
    # The job fails, so its follow-up starts; the follow-up's own prologue ends the chain of the complete run.
    _segment_finishing_the_run(tmp_path, monkeypatch)
    result = _run_prologue(
        tmp_path, ResumeConfig(), "TIMEOUT", driver=f"{_FINISHES_THE_RUN}; ( exit {status} )", sentinel_id="998"
    )

    assert result.returncode == status, result.stdout + result.stderr
    assert "REACHED_WORK" in result.stdout and "Run complete" not in result.stdout
    assert _scancel_calls(tmp_path) == []
    assert _chain_end_entries(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_segment_with_nothing_eligible_for_retry_ends_its_own_chain(tmp_path, monkeypatch):
    # Exit 76: the follow-up would start on a full allocation only to stop. The segment stops the chain itself.
    _segment_finishing_the_run(tmp_path, monkeypatch)
    result = _run_prologue(
        tmp_path, ResumeConfig(), "TIMEOUT", driver=f"( exit {NO_ELIGIBLE_RETRY_EXIT} )", sentinel_id="998"
    )

    assert result.returncode == NO_ELIGIBLE_RETRY_EXIT, result.stdout + result.stderr
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == NO_ELIGIBLE_RETRY_REASON + "\n"
    assert NO_ELIGIBLE_RETRY_REASON in result.stdout and "Run complete" not in result.stdout
    assert _scancel_calls(tmp_path) == ["998", "999"]
    [entry] = _chain_end_entries(tmp_path)
    assert entry["ended"] == NO_ELIGIBLE_RETRY and entry["job_id"] == "200"
    assert entry["cancelled"] == {
        "sentinel": {"job_id": "998", "result": "cancelled"},
        "follow_up": {"job_id": "999", "result": "cancelled"},
    }


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("kind", GUARDED_DRIVER_FAILURE_KINDS)
def test_a_driver_that_exits_76_with_a_typed_failure_leaves_the_stop_to_its_follow_up(tmp_path, monkeypatch, kind):
    # The follow-up's prologue turns the kind into its own stop and remedy, so the segment keeps its queued jobs.
    _segment_finishing_the_run(tmp_path, monkeypatch)
    failure = tmp_path / f"{DRIVER_FAILURE_NAME}.200"
    result = _run_prologue(
        tmp_path,
        ResumeConfig(),
        "TIMEOUT",
        driver=f"echo {kind} > {shlex.quote(str(failure))}; ( exit {NO_ELIGIBLE_RETRY_EXIT} )",
        sentinel_id="998",
    )

    assert result.returncode == NO_ELIGIBLE_RETRY_EXIT, result.stdout + result.stderr
    marker = tmp_path / INCOMPLETE_MARKER_NAME
    assert not marker.exists() or NO_ELIGIBLE_RETRY_REASON not in marker.read_text()
    assert NO_ELIGIBLE_RETRY_REASON not in result.stdout
    assert _scancel_calls(tmp_path) == []
    assert _chain_end_entries(tmp_path) == []


def test_every_kind_whose_follow_up_is_kept_on_exit_76_has_a_stop_in_the_next_prologue():
    # The epilogue keeps the follow-up for these kinds because its prologue stops on each; a kind without a case
    # there would start the run again instead.
    prologue = render_resume_prologue(ResumeConfig())
    for kind in GUARDED_DRIVER_FAILURE_KINDS:
        assert f"\n        {kind})\n" in prologue, kind


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_driver_that_exits_76_with_a_kind_no_guard_handles_still_ends_its_own_chain(tmp_path, monkeypatch):
    # The follow-up's prologue writes no stop for this kind, so it would start the run again: the segment ends it.
    _segment_finishing_the_run(tmp_path, monkeypatch)
    failure = tmp_path / f"{DRIVER_FAILURE_NAME}.200"
    result = _run_prologue(
        tmp_path,
        ResumeConfig(),
        "TIMEOUT",
        driver=f"echo server_start_failed > {shlex.quote(str(failure))}; ( exit {NO_ELIGIBLE_RETRY_EXIT} )",
        sentinel_id="998",
    )

    assert result.returncode == NO_ELIGIBLE_RETRY_EXIT, result.stdout + result.stderr
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == NO_ELIGIBLE_RETRY_REASON + "\n"
    assert _scancel_calls(tmp_path) == ["998", "999"]
    [entry] = _chain_end_entries(tmp_path)
    assert entry["ended"] == NO_ELIGIBLE_RETRY


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_segment_that_leaves_the_run_incomplete_keeps_its_queued_jobs(tmp_path, monkeypatch):
    _segment_finishing_the_run(tmp_path, monkeypatch)
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", driver="true", sentinel_id="998")

    assert result.returncode == 0 and "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Run complete" not in result.stdout
    assert _scancel_calls(tmp_path) == []
    assert _chain_end_entries(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_segment_that_completes_the_run_without_its_export_receipt_writes_the_stop_and_cancels(
    tmp_path, monkeypatch
):
    # The sentinel would have started only to write this stop: the segment writes it instead.
    _segment_finishing_the_run(tmp_path, monkeypatch, receipt=None)
    result = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", driver=_FINISHES_THE_RUN, sentinel_id="998")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / INCOMPLETE_MARKER_NAME).read_text() == EXPORT_RECEIPT_MISSING_REASON + "\n"
    assert EXPORT_RECEIPT_MISSING_REASON in result.stdout and "Run complete" not in result.stdout
    assert _scancel_calls(tmp_path) == ["998", "999"]
    [entry] = _chain_end_entries(tmp_path)
    assert entry["ended"] == EXPORT_RECEIPT_MISSING


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "stop",
    [
        "Accumulated walltime reached the cap; continue with gym eval resubmit --recover.",
        # As a segment left it when its follow-up submission got no definite reply and the run then completed.
        f"{ACCOUNTING_UNKNOWN_PREFIX} Follow-up submission may have succeeded without an observed reply.",
        f"{FOLLOW_UP_SUBMISSION_REFUSED}: sbatch exited 1: sbatch: error: QOSMaxSubmitJobPerUserLimit",
    ],
    ids=["walltime-cap", "accounting-unknown", "follow-up-refused"],
)
def test_a_segment_that_completes_the_run_supersedes_an_earlier_stop(tmp_path, monkeypatch, stop):
    _segment_finishing_the_run(tmp_path, monkeypatch)
    result = _run_prologue(
        tmp_path,
        ResumeConfig(),
        "TIMEOUT",
        driver=f"{_FINISHES_THE_RUN}; echo {shlex.quote(stop)} > {INCOMPLETE_MARKER_NAME}",
        sentinel_id="998",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # A complete run leaves no incomplete-chain marker; the job log and the journal say what it superseded.
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()
    assert "Run complete" in result.stdout and f"The chain's earlier stop no longer applies: {stop}" in result.stdout
    assert _scancel_calls(tmp_path) == ["998", "999"]
    [entry] = _chain_end_entries(tmp_path)
    assert entry["ended"] == "run_complete_stop_superseded"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_sentinel_that_cannot_be_cancelled_is_journaled_and_the_segment_still_completes(tmp_path, monkeypatch):
    _segment_finishing_the_run(tmp_path, monkeypatch)
    result = _run_prologue(
        tmp_path, ResumeConfig(), "TIMEOUT", driver=_FINISHES_THE_RUN, sentinel_id="998", scancel_fails="998"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert _scancel_calls(tmp_path) == ["998", "999"]
    assert "WARNING: could not cancel this chain's queued sentinel 998" in result.stdout
    [entry] = _chain_end_entries(tmp_path)
    assert entry["cancelled"] == {
        "sentinel": {"job_id": "998", "result": "scancel_failed"},
        "follow_up": {"job_id": "999", "result": "cancelled"},
    }


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_segment_without_a_recorded_sentinel_cancels_only_its_follow_up(tmp_path, monkeypatch):
    # The sentinel was not queued (its sbatch failed), so nothing names one.
    _segment_finishing_the_run(tmp_path, monkeypatch)
    result = _run_prologue(
        tmp_path, ResumeConfig(), "TIMEOUT", driver=f"rm .gym_followup_sentinel.200; {_FINISHES_THE_RUN}"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert _scancel_calls(tmp_path) == ["999"]
    [entry] = _chain_end_entries(tmp_path)
    assert entry["cancelled"] == {"follow_up": {"job_id": "999", "result": "cancelled"}}


def _server_start_failed_before_any_rollout(tmp_path, prev_job_id: str = "100") -> None:
    # The previous segment's driver: its policy server could not bind its port, and nothing was saved.
    (tmp_path / f"{DRIVER_FAILURE_NAME}.{prev_job_id}").write_text("server_start_failed\n")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_server_that_failed_to_start_before_any_rollout_restarts_the_run_fresh_once(tmp_path):
    _server_start_failed_before_any_rollout(tmp_path)

    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "FAILED")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "restarting the run fresh as an infra retry" in result.stdout and "REACHED_WORK" in result.stdout
    assert "no eligible retries" not in result.stdout
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"
    assert (tmp_path / f"{FRESH_RESTART_MARKER_NAME}.200").exists()
    assert _sbatch_calls(tmp_path), "the follow-up is queued as for any infra retry"
    incomplete = tmp_path / INCOMPLETE_MARKER_NAME
    assert not incomplete.exists() or "no eligible retries" not in incomplete.read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("exhausted", ["restarted_once", "budget_spent"])
def test_a_server_start_failure_after_the_fresh_restart_stops_with_its_kind_and_a_fresh_resubmit(tmp_path, exhausted):
    _server_start_failed_before_any_rollout(tmp_path)
    if exhausted == "restarted_once":
        (tmp_path / f"{FRESH_RESTART_MARKER_NAME}.100").write_text("")
        resume = ResumeConfig(max_retries=2)
    else:
        resume = ResumeConfig(max_retries=0)

    result = _run_prologue(tmp_path, resume, "FAILED")

    assert result.returncode == 1, result.stdout + result.stderr
    stop = (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert "server_start_failed on job 100 before any rollout was saved; fresh restart exhausted" in stop
    assert "resubmit fresh:" in stop and "--recover" not in stop
    assert "REACHED_WORK" not in result.stdout and not _sbatch_calls(tmp_path)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_server_start_failure_with_saved_rollouts_keeps_the_saved_evidence_path(tmp_path):
    _server_start_failed_before_any_rollout(tmp_path)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    (tmp_path / "artifacts" / "rollouts_manifest.json").write_text("{}")

    result = _run_prologue(
        tmp_path, ResumeConfig(max_retries=2), "FAILED", admission_command="echo __GYM_CONTINUATION:stop"
    )

    assert "restarting the run fresh" not in result.stdout
    assert not (tmp_path / f"{FRESH_RESTART_MARKER_NAME}.200").exists()
    assert result.returncode == 1 and "no eligible retries" in result.stdout
