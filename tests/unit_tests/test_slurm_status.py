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
from pathlib import Path

import pytest

from nemo_gym.orchestration.executors.capabilities import Inspectable
from nemo_gym.orchestration.executors.resume_script import INCOMPLETE_MARKER_NAME, JOB_CHAIN_NAME
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm_status import render_status_script, state_from_sacct
from nemo_gym.orchestration.jobs import (
    CHAIN_INCOMPLETE_FILE,
    EVAL_EXIT_FILE,
    BenchmarkJob,
    JobRef,
    JobState,
    SubmissionRecord,
)


@pytest.fixture
def slurm_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`squeue` and `sacct` that answer from files and log how they were called.

    `squeue_rows` and `sacct_rows` hold what each prints; a `squeue_rc` or `sacct_rc` file makes it fail.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("squeue", "sacct"):
        script = bin_dir / tool
        script.write_text(
            f'#!/bin/bash\nd="$(dirname "$0")"\necho "$*" >> "$d/{tool}_calls"\n'
            f'if [ -f "$d/{tool}_rc" ]; then echo "{tool}: error: cannot reach the controller" >&2; '
            f'exit "$(cat "$d/{tool}_rc")"; fi\ncat "$d/{tool}_rows" 2>/dev/null\nexit 0\n'
        )
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _job(tmp_path: Path, name: str, *, job_id: str | None = "100", error: str | None = None) -> JobRef:
    job_dir = tmp_path / "run" / name
    job_dir.mkdir(parents=True)
    record = SubmissionRecord(
        gym_job_id="gym-job-x",
        gym_version="0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=str(tmp_path / "run"),
        cluster="cluster",
        executor="slurm",
        submitted_by="me",
        benchmarks=[BenchmarkJob(benchmark=name, job_dir=str(job_dir), job_id=job_id, error=error)],
    )
    return JobRef(record, record.benchmarks[0])


def _query(*refs: JobRef):
    return SlurmExecutor().query_states(list(refs))


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


# --- What Slurm says -----------------------------------------------------------------------------------------


def test_a_running_job_is_found_by_its_directory_and_its_chain_is_listed(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa", job_id="8123402")
    job_dir = Path(ref.job.job_dir)
    (job_dir / JOB_CHAIN_NAME).write_text("8123402\n8123455\n")
    (slurm_bin / "squeue_rows").write_text(f"8123455|RUNNING|{job_dir}\n")
    (slurm_bin / "sacct_rows").write_text("8123402|FAILED|1:0\n8123455|RUNNING|0:0\n")

    observed = _query(ref)[ref.key]

    assert observed.state is JobState.RUNNING
    assert observed.job_id == "8123455"
    assert observed.job_ids == ["8123402", "8123455"]


def test_a_job_directory_reached_through_a_symlink_is_still_the_same_directory(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa")
    real = Path(ref.job.job_dir)
    link = tmp_path / "link"
    link.symlink_to(real)
    ref = JobRef(ref.record, ref.job.model_copy(update={"job_dir": str(link)}))
    (slurm_bin / "squeue_rows").write_text(f"777|RUNNING|{real}\n")

    assert _query(ref)[ref.key].state is JobState.RUNNING


def test_a_queued_job_is_queued_and_a_running_one_wins_over_a_queued_follow_up(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa")
    job_dir = Path(ref.job.job_dir)
    (slurm_bin / "squeue_rows").write_text(f"900|PENDING|{job_dir}\n")
    assert _query(ref)[ref.key].state is JobState.QUEUED

    (slurm_bin / "squeue_rows").write_text(f"900|PENDING|{job_dir}\n800|RUNNING|{job_dir}\n")
    observed = _query(ref)[ref.key]
    assert (observed.state, observed.job_id) == (JobState.RUNNING, "800")


def test_a_job_started_outside_gym_is_seen_and_a_job_of_another_directory_is_not(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa", job_id="100")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (slurm_bin / "squeue_rows").write_text(f"777|RUNNING|{ref.job.job_dir}\n888|RUNNING|{other}\n")

    observed = _query(ref)[ref.key]

    assert (observed.state, observed.job_id) == (JobState.RUNNING, "777")
    assert observed.job_ids == ["100", "777"]


@pytest.mark.parametrize(
    ("accounting", "state", "exit_code"),
    [
        ("100|COMPLETED|0:0", JobState.SUCCEEDED, 0),
        ("100|TIMEOUT|0:0", JobState.FAILED, 0),
        ("100|OUT_OF_MEMORY|0:125", JobState.FAILED, 0),
        ("100|FAILED|3:0", JobState.FAILED, 3),
        ("100|CANCELLED by 4242|0:0", JobState.CANCELLED, 0),
    ],
)
def test_a_finished_job_is_read_from_accounting(tmp_path, slurm_bin, accounting, state, exit_code):
    ref = _job(tmp_path, "gpqa", job_id="100")
    (slurm_bin / "sacct_rows").write_text(accounting + "\n")

    observed = _query(ref)[ref.key]

    assert observed.state is state
    assert observed.exit_code == exit_code
    assert observed.detail == accounting.split("|")[1]
    assert (observed.reason is not None) is (state is JobState.FAILED)


def test_the_newest_job_of_a_resumed_run_says_how_it_ended(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa", job_id="100")
    (Path(ref.job.job_dir) / JOB_CHAIN_NAME).write_text("100\n101\n")
    (slurm_bin / "sacct_rows").write_text("100|TIMEOUT|0:0\n101|COMPLETED|0:0\n")

    observed = _query(ref)[ref.key]

    assert (observed.state, observed.job_id) == (JobState.SUCCEEDED, "101")


def test_one_squeue_and_one_sacct_serve_every_job_of_the_batch(tmp_path, slurm_bin):
    first, second = _job(tmp_path, "a", job_id="30"), _job(tmp_path, "b", job_id="10")
    (Path(first.job.job_dir) / JOB_CHAIN_NAME).write_text("30\n20\n")
    (Path(second.job.job_dir) / JOB_CHAIN_NAME).write_text("10\n20\n")
    (slurm_bin / "sacct_rows").write_text("10|COMPLETED|0:0\n20|COMPLETED|0:0\n30|COMPLETED|0:0\n")

    answered = _query(first, second)

    assert set(answered) == {first.key, second.key}
    assert len(_lines(slurm_bin / "squeue_calls")) == 1
    sacct_calls = _lines(slurm_bin / "sacct_calls")
    assert len(sacct_calls) == 1
    assert "-j 10,20,30 " in sacct_calls[0] + " "


def test_sacct_is_not_asked_when_there_is_no_job_id_to_ask_about(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa", job_id=None, error="sbatch: error: bad account")

    observed = _query(ref)[ref.key]

    assert not (slurm_bin / "sacct_calls").exists()
    assert (observed.state, observed.detail) == (JobState.FAILED, "sbatch: error: bad account")


def test_a_job_slurm_never_heard_of_is_unknown(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa", job_id="100")

    observed = _query(ref)[ref.key]

    assert observed.state is JobState.UNKNOWN
    assert "no record" in observed.detail


def test_a_scheduler_that_answers_with_an_error_is_unknown_with_the_reason(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa")
    (slurm_bin / "squeue_rc").write_text("1")

    observed = _query(ref)[ref.key]

    assert observed.state is JobState.UNKNOWN
    assert "squeue failed" in observed.detail
    assert "cannot reach the controller" in observed.detail


def test_without_accounting_the_files_the_job_wrote_still_come_back(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa")
    (Path(ref.job.job_dir) / EVAL_EXIT_FILE).write_text("0\n")
    (slurm_bin / "sacct_rc").write_text("1")

    observed = _query(ref)[ref.key]

    assert observed.state is JobState.UNKNOWN
    assert "sacct failed" in observed.detail
    assert observed.files == {EVAL_EXIT_FILE: "0\n"}


# --- The state files -----------------------------------------------------------------------------------------


def test_state_files_survive_quotes_newlines_and_unicode(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa")
    reason = "Chain stopped at the walltime cap:\n\"resubmit\" to continue; it's 'ok' — $HOME `x`\n"
    (Path(ref.job.job_dir) / CHAIN_INCOMPLETE_FILE).write_text(reason)

    assert _query(ref)[ref.key].files == {CHAIN_INCOMPLETE_FILE: reason}


def test_a_state_file_that_is_missing_is_absent(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa")

    assert _query(ref)[ref.key].files == {}


def test_a_job_directory_with_spaces_and_quotes_is_quoted(tmp_path, slurm_bin):
    ref = _job(tmp_path, "it's a dir")
    (Path(ref.job.job_dir) / EVAL_EXIT_FILE).write_text("7\n")
    (slurm_bin / "squeue_rows").write_text(f"55|RUNNING|{ref.job.job_dir}\n")

    observed = _query(ref)[ref.key]

    assert (observed.state, observed.files) == (JobState.RUNNING, {EVAL_EXIT_FILE: "7\n"})


def test_a_recorded_job_id_that_is_not_a_number_never_reaches_the_script(tmp_path):
    ref = _job(tmp_path, "gpqa", job_id="1; touch /tmp/pwned")

    script = "\n".join(render_status_script([ref]))

    assert "pwned" not in script


def test_the_names_of_the_files_asked_for_are_the_ones_the_job_script_writes():
    assert CHAIN_INCOMPLETE_FILE == INCOMPLETE_MARKER_NAME


# --- Reading files and the interface -------------------------------------------------------------------------


def test_run_files_are_read_and_a_missing_one_is_none(tmp_path, slurm_bin):
    ref = _job(tmp_path, "gpqa")
    present = tmp_path / "resolved-config.yaml"
    present.write_text("driver:\n  container: gym:latest\n")

    contents = SlurmExecutor().read_run_files(ref.record, [present, tmp_path / "absent.yaml"])

    assert contents == {present: "driver:\n  container: gym:latest\n", tmp_path / "absent.yaml": None}


def test_slurm_is_inspectable_and_groups_records_by_the_host_they_are_reached_through():
    executor = SlurmExecutor()
    local = SubmissionRecord(
        gym_job_id="a",
        gym_version="0",
        submitted_at="x",
        run_dir="/r",
        cluster="c",
        executor="slurm",
        submitted_by="u",
        benchmarks=[],
    )

    assert isinstance(executor, Inspectable)
    assert executor.connection_key(local) == "local"
    assert executor.connection_key(local.model_copy(update={"hostname": "login-01"})) == "login-01"


def test_asking_about_no_jobs_asks_nothing(slurm_bin):
    assert SlurmExecutor().query_states([]) == {}
    assert not (slurm_bin / "squeue_calls").exists()


@pytest.mark.parametrize(
    ("native", "state"),
    [
        ("COMPLETED", JobState.SUCCEEDED),
        ("CANCELLED", JobState.CANCELLED),
        ("CANCELLED by 1234", JobState.CANCELLED),
        ("CANCELLED+", JobState.CANCELLED),
        ("PENDING", JobState.QUEUED),
        ("REQUEUED", JobState.QUEUED),
        ("RUNNING", JobState.RUNNING),
        ("COMPLETING", JobState.RUNNING),
        ("FAILED", JobState.FAILED),
        ("TIMEOUT", JobState.FAILED),
        ("OUT_OF_MEMORY", JobState.FAILED),
        ("NODE_FAIL", JobState.FAILED),
        ("PREEMPTED", JobState.FAILED),
        ("SOMETHING_NEW", JobState.UNKNOWN),
        ("", JobState.UNKNOWN),
    ],
)
def test_slurm_states_map_onto_job_states(native, state):
    assert state_from_sacct(native) is state
