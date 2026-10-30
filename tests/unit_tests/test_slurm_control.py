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

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.base import UnsupportedOperationError
from nemo_gym.orchestration.executors.capabilities import LogReadable, Terminable
from nemo_gym.orchestration.executors.resume_script import JOB_CHAIN_NAME, render_resume_prologue
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm_control import FOLLOWUP_CLAIM_PREFIX, render_terminate_script
from nemo_gym.orchestration.jobs import BenchmarkJob, LogSource, SubmissionRecord


@pytest.fixture
def slurm_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`squeue` and `scancel` that answer from files and log how they were called.

    `squeue_rows` holds what squeue prints; a `squeue_rc` or `scancel_rc` file makes that tool fail.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("squeue", "scancel"):
        script = bin_dir / tool
        script.write_text(
            f'#!/bin/bash\nd="$(dirname "$0")"\necho "$*" >> "$d/{tool}_calls"\n'
            f'if [ -f "$d/{tool}_rc" ]; then echo "{tool}: error: Access/permission denied" >&2; '
            f'exit "$(cat "$d/{tool}_rc")"; fi\ncat "$d/{tool}_rows" 2>/dev/null\nexit 0\n'
        )
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _job(tmp_path: Path, name: str = "gpqa", *, job_id: str | None = "100") -> tuple[SubmissionRecord, BenchmarkJob]:
    job_dir = tmp_path / "run" / name
    (job_dir / "logs").mkdir(parents=True)
    record = SubmissionRecord(
        gym_job_id="gym-job-x",
        gym_version="0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=str(tmp_path / "run"),
        cluster="cluster",
        executor="slurm",
        submitted_by="me",
        benchmarks=[BenchmarkJob(benchmark=name, job_dir=str(job_dir), job_id=job_id)],
    )
    return record, record.benchmarks[0]


def _terminate(record: SubmissionRecord, job: BenchmarkJob) -> None:
    SlurmExecutor().terminate(record, job)


def _calls(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


# --- terminate -----------------------------------------------------------------------------------------------


def test_a_running_job_is_found_by_its_directory_and_cancelled(tmp_path, slurm_bin):
    record, job = _job(tmp_path)
    (slurm_bin / "squeue_rows").write_text(f"100|RUNNING|{job.job_dir}\n")

    _terminate(record, job)

    assert _calls(slurm_bin / "scancel_calls") == ["100"]


def test_a_pending_follow_up_is_cancelled_before_the_running_job_so_it_cannot_start_in_between(tmp_path, slurm_bin):
    record, job = _job(tmp_path)
    (slurm_bin / "squeue_rows").write_text(f"800|RUNNING|{job.job_dir}\n900|PENDING|{job.job_dir}\n")

    _terminate(record, job)

    assert _calls(slurm_bin / "scancel_calls") == ["900", "800"]


def test_a_job_of_another_directory_is_left_alone(tmp_path, slurm_bin):
    record, job = _job(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (slurm_bin / "squeue_rows").write_text(f"555|RUNNING|{other}\n")

    _terminate(record, job)

    assert not (slurm_bin / "scancel_calls").exists()


def test_a_directory_reached_through_a_symlink_is_still_the_same_directory(tmp_path, slurm_bin):
    record, job = _job(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(job.job_dir)
    job = job.model_copy(update={"job_dir": str(link)})
    (slurm_bin / "squeue_rows").write_text(f"777|RUNNING|{tmp_path / 'run' / 'gpqa'}\n")

    _terminate(record, job)

    assert _calls(slurm_bin / "scancel_calls") == ["777"]


def test_the_ids_the_chain_and_the_follow_up_claims_name_are_cancelled_wherever_squeue_says_they_run(
    tmp_path, slurm_bin
):
    record, job = _job(tmp_path)
    job_dir = Path(job.job_dir)
    (job_dir / JOB_CHAIN_NAME).write_text("100\n101\n")
    (job_dir / f"{FOLLOWUP_CLAIM_PREFIX}101").write_text("102\n")
    (slurm_bin / "squeue_rows").write_text("102|PENDING|/some/other/path\n999|PENDING|/some/other/path\n")

    _terminate(record, job)

    assert _calls(slurm_bin / "scancel_calls") == ["102"]


def test_a_job_that_already_left_the_queue_is_not_handed_to_scancel(tmp_path, slurm_bin):
    record, job = _job(tmp_path)
    (Path(job.job_dir) / JOB_CHAIN_NAME).write_text("100\n101\n")

    _terminate(record, job)

    assert not (slurm_bin / "scancel_calls").exists()


def test_a_refusal_from_scancel_is_raised_with_what_slurm_said(tmp_path, slurm_bin):
    record, job = _job(tmp_path)
    (slurm_bin / "squeue_rows").write_text(f"100|RUNNING|{job.job_dir}\n")
    (slurm_bin / "scancel_rc").write_text("1")

    with pytest.raises(RuntimeError, match="Could not stop gpqa.*permission denied"):
        _terminate(record, job)


def test_an_unreadable_queue_is_an_error_and_nothing_is_cancelled(tmp_path, slurm_bin):
    record, job = _job(tmp_path)
    (slurm_bin / "squeue_rc").write_text("1")

    with pytest.raises(RuntimeError, match="Could not stop gpqa.*squeue failed"):
        _terminate(record, job)

    assert not (slurm_bin / "scancel_calls").exists()


def test_a_job_directory_with_spaces_and_quotes_is_quoted(tmp_path, slurm_bin):
    record, job = _job(tmp_path, "it's a dir")
    (slurm_bin / "squeue_rows").write_text(f"100|RUNNING|{job.job_dir}\n")

    _terminate(record, job)

    assert _calls(slurm_bin / "scancel_calls") == ["100"]


def test_a_recorded_job_id_that_is_not_a_number_never_reaches_the_script(tmp_path):
    _, job = _job(tmp_path, job_id="1; touch /tmp/pwned")

    assert "pwned" not in "\n".join(render_terminate_script(job))


def test_the_follow_up_claim_files_are_the_ones_the_job_script_writes():
    prologue = render_resume_prologue(ResumeConfig())

    assert f'"$OUTPUT_DIR/{FOLLOWUP_CLAIM_PREFIX}$SLURM_JOB_ID"' in prologue


# --- logs ----------------------------------------------------------------------------------------------------


def _source(location: str = "logs/driver-100.log") -> LogSource:
    return LogSource(name="driver", kind="driver", location=location)


def _read(record: SubmissionRecord, job: BenchmarkJob, location: str = "logs/driver-100.log", **kwargs):
    kwargs = {"tail": None, "follow": False, **kwargs}
    return list(SlurmExecutor().stream_logs(record, job, source=_source(location), **kwargs))


def test_a_log_is_read_whole_or_by_its_last_lines(tmp_path):
    record, job = _job(tmp_path)
    (Path(job.job_dir) / "logs" / "driver-100.log").write_text("one\ntwo\nthree\nfour\n")

    assert _read(record, job) == ["one", "two", "three", "four"]
    assert _read(record, job, tail=2) == ["three", "four"]
    assert _read(record, job, tail=0) == []


def test_a_log_that_is_not_there_says_the_job_may_not_have_reached_that_step(tmp_path):
    record, job = _job(tmp_path)

    with pytest.raises(FileNotFoundError, match="logs/driver-100.log does not exist.*may not have reached"):
        _read(record, job)


def test_a_log_with_bytes_that_are_not_utf8_is_read_with_replacements(tmp_path):
    record, job = _job(tmp_path)
    (Path(job.job_dir) / "logs" / "driver-100.log").write_bytes(b"fine\nbad \xff byte\n")

    assert _read(record, job) == ["fine", "bad � byte"]


def test_a_log_that_contains_the_words_of_the_protocol_is_still_just_a_log(tmp_path):
    record, job = _job(tmp_path)
    (Path(job.job_dir) / "logs" / "driver-100.log").write_text("__GYM_CONTROL:missing\n__GYM_CONTROL:ok\nend\n")

    assert _read(record, job) == ["__GYM_CONTROL:missing", "__GYM_CONTROL:ok", "end"]


def test_a_log_in_a_directory_with_spaces_and_quotes_is_read(tmp_path):
    record, job = _job(tmp_path, "it's a dir")
    (Path(job.job_dir) / "logs" / "driver-100.log").write_text("hello\n")

    assert _read(record, job) == ["hello"]


def test_following_a_log_is_not_supported_yet_and_a_negative_tail_is_refused(tmp_path):
    record, job = _job(tmp_path)

    with pytest.raises(UnsupportedOperationError, match="cannot follow a log yet; use --tail"):
        _read(record, job, follow=True)
    with pytest.raises(ValueError, match="--tail must be 0 or more"):
        _read(record, job, tail=-1)


def test_slurm_can_stop_jobs_and_read_logs():
    executor = SlurmExecutor()

    assert isinstance(executor, Terminable)
    assert isinstance(executor, LogReadable)
