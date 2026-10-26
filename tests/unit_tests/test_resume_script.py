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

import os
import shutil
import subprocess

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.resume_script import (
    INCOMPLETE_MARKER_NAME,
    _walltime_to_seconds,
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
    assert "-gt 3" in out  # default max_retries


def test_prologue_with_max_walltime_adds_check():
    out = render_resume_prologue(ResumeConfig(max_walltime="48:00:00"))
    assert "_gym_accumulated >= 172800" in out


def test_prologue_uses_custom_max_retries():
    out = render_resume_prologue(ResumeConfig(max_retries=9))
    assert "-gt 9" in out
    assert "Infra retry limit (9)" in out


def test_prologue_queues_own_successor():
    out = render_resume_prologue(ResumeConfig())
    assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("resume", [ResumeConfig(), ResumeConfig(max_walltime="12:00:00")])
def test_rendered_prologue_is_valid_bash(resume, tmp_path):
    script = tmp_path / "prologue.sh"
    script.write_text(render_resume_prologue(resume))
    subprocess.run(["bash", "-n", str(script)], check=True)


# ---------------------------------------------------------------------------
# Running the prologue against fake sacct/sbatch
# ---------------------------------------------------------------------------

_FAKE_SACCT = """#!/bin/bash
echo call >> "$FAKE_LOG_DIR/sacct_calls"
if [[ "$*" == *ElapsedRaw* ]]; then echo "${FAKE_ELAPSED:-}"; else echo "${FAKE_STATE:-}"; fi
"""
_FAKE_SBATCH = """#!/bin/bash
echo "$*" >> "$FAKE_LOG_DIR/sbatch_calls"
if [[ -n "${FAKE_SBATCH_FAILS:-}" ]]; then echo "sbatch: error: Socket timed out"; exit 1; fi
# The queued follow-up waits on its dependency in this job dir, as squeue would list it. In a hidden partition
# squeue lists it only with --all.
_pending="$FAKE_LOG_DIR/pending${FAKE_HIDDEN_PARTITION:+_hidden}"
echo "999|${1#--dependency=}(unfulfilled)|$PWD" >> "$_pending"
# A lost reply: the job exists, but sbatch reports a failure.
if [[ -n "${FAKE_SBATCH_LOSES_REPLY:-}" ]]; then echo "sbatch: error: Socket timed out on send/recv operation"; exit 1; fi
echo "Submitted batch job 999"
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
    job_id: str = "200",
    prev_job_id: str = "100",
    sbatch_fails: bool = False,
    squeue_fails: bool = False,
    hidden_partition: bool = False,
    sbatch_loses_reply: bool = False,
) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (
        ("sacct", _FAKE_SACCT),
        ("sbatch", _FAKE_SBATCH),
        ("squeue", _FAKE_SQUEUE),
        ("sleep", "#!/bin/bash\n"),
    ):
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    script = tmp_path / "job.sh"
    script.write_text(render_resume_prologue(resume) + "\necho REACHED_WORK\n")
    args = ["bash", str(script)] + ([] if prev_state is None else [prev_job_id])
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SLURM_JOB_ID": job_id,
        "FAKE_LOG_DIR": str(tmp_path),
        "FAKE_STATE": prev_state or "",
        "FAKE_ELAPSED": "60" if prev_state else "",
        **({"FAKE_SBATCH_FAILS": "1"} if sbatch_fails else {}),
        **({"FAKE_SQUEUE_FAILS": "1"} if squeue_fails else {}),
        **({"FAKE_HIDDEN_PARTITION": "1"} if hidden_partition else {}),
        **({"FAKE_SBATCH_LOSES_REPLY": "1"} if sbatch_loses_reply else {}),
    }
    return subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_max_retries_allows_exactly_that_many_retries(tmp_path):
    resume = ResumeConfig(max_retries=3)
    for _ in range(3):
        result = _run_prologue(tmp_path, resume, "FAILED")
        assert result.returncode == 0, result.stdout
        assert "REACHED_WORK" in result.stdout

    result = _run_prologue(tmp_path, resume, "FAILED")
    assert result.returncode == 1
    assert "REACHED_WORK" not in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_empty_sacct_resumes_without_spending_a_retry(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=1), "")

    assert result.returncode == 0, result.stdout
    assert "REACHED_WORK" in result.stdout
    assert "state unknown" in result.stdout
    assert not (tmp_path / ".gym_infra_retries").exists()
    assert len((tmp_path / "sacct_calls").read_text().splitlines()) > 2


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_successor_is_killed_if_its_dependency_can_never_be_met(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(), None)

    assert result.returncode == 0, result.stdout
    sbatch_args = (tmp_path / "sbatch_calls").read_text()
    assert "--dependency=afternotok:200" in sbatch_args
    assert "--kill-on-invalid-dep=yes" in sbatch_args


def _job_dir(tmp_path) -> str:
    # The prologue's OUTPUT_DIR is the shell's $PWD, the physical path of the job dir.
    return os.path.realpath(tmp_path)


def _sbatch_calls(tmp_path) -> list[str]:
    calls = tmp_path / "sbatch_calls"
    return calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("prev_state", [None, "TIMEOUT"])
def test_a_requeued_job_does_not_queue_a_second_follow_up(tmp_path, prev_state):
    """Slurm reruns job.sh under the same job id on a requeue (node failure, preemption with --requeue)."""
    first = _run_prologue(tmp_path, ResumeConfig(), prev_state)
    requeued = _run_prologue(tmp_path, ResumeConfig(), prev_state)

    assert "REACHED_WORK" in first.stdout and "REACHED_WORK" in requeued.stdout
    assert _sbatch_calls(tmp_path) == [_sbatch_calls(tmp_path)[0]]
    assert "--dependency=afternotok:200" in _sbatch_calls(tmp_path)[0]
    assert "already queued" in requeued.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("claim", ["file", "directory"])
def test_a_claim_left_by_an_interrupted_submission_is_replaced(tmp_path, claim):
    """The shell died between taking the claim and sbatch answering: no follow-up exists, so queue one."""
    marker = tmp_path / ".gym_followup_queued.200"
    marker.mkdir() if claim == "directory" else marker.touch()  # a directory is #4840's marker format

    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert "REACHED_WORK" in requeued.stdout
    assert "submission interrupted" in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 1
    assert "--dependency=afternotok:200" in _sbatch_calls(tmp_path)[0]
    assert marker.read_text() == "999\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_the_claim_records_the_follow_up_and_a_requeue_finds_it_pending(tmp_path):
    _run_prologue(tmp_path, ResumeConfig(), None)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert (tmp_path / ".gym_followup_queued.200").read_text() == "999\n"
    assert "already queued: 999" in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 1


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

    assert len(_sbatch_calls(tmp_path)) == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_claim_is_kept_when_squeue_fails(tmp_path):
    (tmp_path / ".gym_followup_queued.200").touch()

    requeued = _run_prologue(tmp_path, ResumeConfig(), None, squeue_fails=True)

    assert "REACHED_WORK" in requeued.stdout
    assert "squeue failed" in requeued.stdout
    assert _sbatch_calls(tmp_path) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_each_resumed_segment_still_queues_its_own_follow_up(tmp_path):
    _run_prologue(tmp_path, ResumeConfig(), None, job_id="200")
    _run_prologue(tmp_path, ResumeConfig(), None, job_id="200")  # requeue of the first segment
    resumed = _run_prologue(tmp_path, ResumeConfig(), "TIMEOUT", job_id="999", prev_job_id="200")

    assert "REACHED_WORK" in resumed.stdout
    assert [call.split()[0] for call in _sbatch_calls(tmp_path)] == [
        "--dependency=afternotok:200",
        "--dependency=afternotok:999",
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_requeue_retries_a_follow_up_that_failed_to_submit(tmp_path):
    failed = _run_prologue(tmp_path, ResumeConfig(), None, sbatch_fails=True)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert "Failed to submit auto-resume follow-up" in failed.stdout
    assert "Auto-resume follow-up queued: 999" in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 2


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_stopped_at_the_walltime_cap_exits_non_zero_and_marks_the_run_incomplete(tmp_path):
    (tmp_path / ".gym_accumulated_walltime").write_text("3590")

    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), "TIMEOUT")

    assert result.returncode == 1, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert "walltime cap with rollouts missing; resubmit to continue" in result.stdout
    assert "walltime cap" in (tmp_path / INCOMPLETE_MARKER_NAME).read_text()
    assert not (tmp_path / "sbatch_calls").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("state", ["CANCELLED", "CANCELLED by 1234"])
def test_a_chain_stopped_after_a_cancel_exits_non_zero_and_marks_the_run_incomplete(tmp_path, state):
    result = _run_prologue(tmp_path, ResumeConfig(), state)

    assert result.returncode == 1, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert "was cancelled. Chain stopped with rollouts missing; resubmit to continue." in result.stdout
    assert (tmp_path / INCOMPLETE_MARKER_NAME).exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_whose_previous_job_completed_still_exits_zero(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="01:00:00"), "COMPLETED")

    assert result.returncode == 0, result.stdout
    assert "REACHED_WORK" not in result.stdout
    assert not (tmp_path / INCOMPLETE_MARKER_NAME).exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_chain_that_continues_clears_an_earlier_incomplete_marker(tmp_path):
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

    assert "REACHED_WORK" in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 1
    assert "--dependency=afternotok:200" in _sbatch_calls(tmp_path)[0]
    assert marker.read_text() == "999\n"


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

    assert "REACHED_WORK" in requeued.stdout
    assert "already queued: 999" in requeued.stdout
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

    assert len(_sbatch_calls(job_dir)) == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_pending_in_a_hidden_partition_is_found(tmp_path):
    """squeue lists jobs in a hidden partition, ours included, only with --all."""
    _run_prologue(tmp_path, ResumeConfig(), None, hidden_partition=True)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None, hidden_partition=True)

    assert "already queued: 999" in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_a_follow_up_whose_sbatch_reply_was_lost_is_not_queued_again(tmp_path):
    """sbatch failed after creating the job: the claim stays, and the requeue finds the job pending."""
    failed = _run_prologue(tmp_path, ResumeConfig(), None, sbatch_loses_reply=True)
    requeued = _run_prologue(tmp_path, ResumeConfig(), None)

    assert "Failed to submit auto-resume follow-up" in failed.stdout
    assert "already queued: 999" in requeued.stdout
    assert len(_sbatch_calls(tmp_path)) == 1
