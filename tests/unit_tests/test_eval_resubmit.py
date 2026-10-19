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

from nemo_gym.orchestration.jobs import MANIFEST_NAME, BenchmarkJob, SubmissionRecord
from nemo_gym.orchestration.submit import resubmit


def _staged_run(tmp_path: Path, *, executor: str = "slurm") -> Path:
    run_dir = tmp_path / "gym-job-x"
    job_dir = run_dir / "gsm8k"
    job_dir.mkdir(parents=True)
    (job_dir / "job.sh").write_text("#!/bin/bash\n")
    (job_dir / ".gym_job_chain").write_text("100\n")
    record = SubmissionRecord(
        gym_job_id=run_dir.name,
        gym_version="0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=str(run_dir),
        cluster="c",
        executor=executor,
        submitted_by="me",
        benchmarks=[BenchmarkJob(benchmark="gsm8k", job_dir=str(job_dir), job_id="100")],
    )
    (run_dir / MANIFEST_NAME).write_text(record.dumps())
    return job_dir


@pytest.fixture
def slurm_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """squeue prints the `squeue_workdirs` file; sbatch records its cwd and answers 4242."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text('#!/bin/bash\ncat "$(dirname "$0")/squeue_workdirs" 2>/dev/null\n')
    (bin_dir / "sbatch").write_text('#!/bin/bash\necho "$PWD" > "$(dirname "$0")/cwd"\necho 4242\n')
    for stub in ("squeue", "sbatch"):
        (bin_dir / stub).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def test_resubmit_queues_job_sh_from_its_job_dir_and_appends_the_id(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (slurm_bin / "squeue_workdirs").write_text(f"{tmp_path / 'other'}\n")

    assert resubmit(str(job_dir)) == "4242"

    assert (slurm_bin / "cwd").read_text().strip() == str(job_dir)
    assert (job_dir / ".gym_job_chain").read_text() == "100\n4242\n"


def test_resubmit_starts_a_fresh_attempt_budget_and_retry_count(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (job_dir / "artifacts").mkdir()
    failures = '{"_ng_task_index": 0, "_ng_rollout_index": 0}\n{"_ng_task_index": 1, "_ng_rollout_index": 0}\n'
    (job_dir / "artifacts" / "rollouts_failures.jsonl").write_text(failures)
    (job_dir / ".gym_infra_retries").write_text("3\n")
    (job_dir / "job.sh").write_text("#!/bin/bash\n# --- Auto-resume chain ---\n")
    (slurm_bin / "squeue_workdirs").write_text(f"{tmp_path / 'other'}\n")

    assert resubmit(str(job_dir)) == "4242"

    assert (job_dir / "artifacts" / "rollouts_failures.jsonl").read_text() == failures
    assert (job_dir / "artifacts" / "rollouts_resubmit_offset").read_text().strip() == "2"
    assert not (job_dir / ".gym_infra_retries").exists()
    # The resumable job.sh chains itself when it starts, so the resubmit does not.
    assert (job_dir / ".gym_job_chain").read_text() == "100\n"


def test_resubmit_refuses_while_a_job_runs_in_the_job_dir(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (slurm_bin / "squeue_workdirs").write_text(f"{job_dir}\n")

    with pytest.raises(RuntimeError, match="still pending or running"):
        resubmit(str(job_dir))

    assert not (slurm_bin / "cwd").exists()
    assert (job_dir / ".gym_job_chain").read_text() == "100\n"


def test_resubmit_refuses_a_job_dir_no_record_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(ValueError, match="No submission record"):
        resubmit(str(tmp_path / "nowhere" / "gsm8k"))


def test_resubmit_refuses_a_non_slurm_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Slurm executor only"):
        resubmit(str(_staged_run(tmp_path, executor="kubernetes")))
