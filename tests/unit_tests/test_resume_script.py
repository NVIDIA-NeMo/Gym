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
if [[ "$*" == *ElapsedRaw* ]]; then echo "${FAKE_ELAPSED:-}";
elif [[ "$*" == *ExitCode* ]]; then echo "${FAKE_EXIT:-1:0}"; else echo "${FAKE_STATE:-}"; fi
"""
_FAKE_SBATCH = """#!/bin/bash
echo "$*" >> "$FAKE_LOG_DIR/sbatch_calls"
if [[ -n "${FAKE_SBATCH_FAILS:-}" ]]; then echo "sbatch: error: Socket timed out"; exit 1; fi
echo "Submitted batch job 999"
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
) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("sacct", _FAKE_SACCT), ("sbatch", _FAKE_SBATCH), ("sleep", "#!/bin/bash\n")):
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
        "FAKE_EXIT": exit_code,
        "FAKE_ELAPSED": "60" if prev_state else "",
        **({"FAKE_SBATCH_FAILS": "1"} if sbatch_fails else {}),
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


@pytest.mark.parametrize("exit_code,continues", [("75:0", True), ("76:0", False)])
def test_incomplete_evaluation_chain_obeys_retryability(tmp_path, exit_code, continues):
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=0), "FAILED", exit_code=exit_code)
    assert result.returncode == 0, result.stdout
    assert ("REACHED_WORK" in result.stdout) is continues
    assert (tmp_path / "sbatch_calls").exists() is continues
    assert not (tmp_path / ".gym_infra_retries").exists()


def test_retryable_evaluation_still_obeys_walltime_cap(tmp_path):
    result = _run_prologue(tmp_path, ResumeConfig(max_walltime="00:01:00"), "FAILED", exit_code="75:0")
    assert result.returncode == 1
    assert "REACHED_WORK" not in result.stdout
    assert not (tmp_path / "sbatch_calls").exists()


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
