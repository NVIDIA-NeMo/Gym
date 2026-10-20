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
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import MANIFEST_NAME, RESOLVED_CONFIG_NAME, BenchmarkJob, SubmissionRecord
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
    (bin_dir / "squeue").write_text('#!/bin/bash\ncat "$(dirname "$0")/squeue_workdirs" 2>/dev/null; exit 0\n')
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


def _run_args(script: str, job_dir: Path) -> list[str]:
    """The `gym eval run` arguments job.sh builds, run through bash from its job dir."""
    start = script.index("GYM_CMD=(")
    end = script.index("; fi\n", start) + len("; fi\n")
    snippet = script[start:end] + 'printf "%s\\n" "${GYM_CMD[@]}"\n'
    return subprocess.run(
        ["bash", "-c", snippet], cwd=job_dir, text=True, capture_output=True, check=True
    ).stdout.split()


@pytest.mark.parametrize("run", [{}, {"resume_from_cache": False}])
def test_a_resubmitted_job_sh_resumes_even_without_cache_resume(tmp_path: Path, run: dict) -> None:
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "a", "hostname": "foo"}},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {"run": run}}},
            "job": {"output_path": str(tmp_path)},
        }
    )
    job_dir = tmp_path / "gsm8k"
    (job_dir / "artifacts").mkdir(parents=True)
    script = build_sbatch_script(
        config, "gsm8k", config.driver.benchmarks["gsm8k"], next(iter(config.compute.values())), job_dir
    )
    assert "++resume_from_cache=True" not in _run_args(script, job_dir)

    # A resubmit writes the offset; the run then keeps its rollouts instead of clearing them as a fresh run.
    (job_dir / "artifacts" / "rollouts_resubmit_offset").write_text("0\n")
    assert _run_args(script, job_dir)[-1] == "++resume_from_cache=True"


def test_resubmit_refuses_a_populated_run_whose_job_sh_starts_fresh(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (job_dir / "artifacts").mkdir()
    rollouts = job_dir / "artifacts" / "rollouts.jsonl"
    rollouts.write_text('{"reward": 1.0}\n')
    # A job.sh staged before resubmits forced resume: `gym eval run` would delete the rollouts above.
    (job_dir / "job.sh").write_text("#!/bin/bash\nGYM_CMD=(\n    gym eval run\n    +resume_from_cache=False\n)\n")
    (slurm_bin / "squeue_workdirs").write_text("")

    with pytest.raises(RuntimeError, match="would delete its rollouts"):
        resubmit(str(job_dir))

    assert not (slurm_bin / "cwd").exists()
    assert not (job_dir / "artifacts" / "rollouts_resubmit_offset").exists()
    assert rollouts.read_text() == '{"reward": 1.0}\n'


def test_two_concurrent_resubmits_queue_one_job(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    # squeue lists what sbatch queued; sbatch is slow, so both resubmits pass an unlocked squeue check first.
    (slurm_bin / "squeue").write_text('#!/bin/bash\ncat "$(dirname "$0")/queued" 2>/dev/null; exit 0\n')
    (slurm_bin / "sbatch").write_text('#!/bin/bash\nsleep 1\necho "$PWD" >> "$(dirname "$0")/queued"\necho 4242\n')
    barrier = threading.Barrier(2)

    def attempt() -> str:
        barrier.wait()
        try:
            return resubmit(str(job_dir))
        except RuntimeError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))

    assert sorted(outcome == "4242" for outcome in outcomes) == [False, True]
    assert re.search("another resubmit holds|still pending or running", next(o for o in outcomes if o != "4242"))
    assert (slurm_bin / "queued").read_text() == f"{job_dir}\n"
    assert not (job_dir / ".gym_resubmit.lock").exists()


def _save_otel(job_dir: Path, otel: dict) -> None:
    (job_dir.parent / RESOLVED_CONFIG_NAME).write_text(yaml.safe_dump({"otel": otel}))


def test_resubmit_forwards_the_saved_telemetry_token(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _staged_run(tmp_path)
    _save_otel(job_dir, {"enabled": True, "token_env": "MY_INGEST_TOKEN"})
    (slurm_bin / "sbatch").write_text('#!/bin/bash\necho "$MY_INGEST_TOKEN" > "$(dirname "$0")/token"\necho 4242\n')
    monkeypatch.setenv("MY_INGEST_TOKEN", "s3cret value")
    # The commands run in a shell without the caller's environment, as they do over SSH.
    clean_env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/")}
    monkeypatch.setattr(
        LocalConnection,
        "run",
        lambda self, commands: subprocess.run(
            ["bash", "-s"], input="\n".join(commands), env=clean_env, text=True, capture_output=True, check=True
        ).stdout,
    )

    assert resubmit(str(job_dir)) == "4242"

    assert (slurm_bin / "token").read_text() == "s3cret value\n"


def test_resubmit_without_the_telemetry_token_queues_nothing(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _staged_run(tmp_path)
    _save_otel(job_dir, {"enabled": True, "token_env": "MY_INGEST_TOKEN"})
    monkeypatch.delenv("MY_INGEST_TOKEN", raising=False)

    with pytest.raises(ValueError, match="MY_INGEST_TOKEN"):
        resubmit(str(job_dir))

    assert not (slurm_bin / "cwd").exists()


def test_resubmit_with_telemetry_off_needs_no_token(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _staged_run(tmp_path)
    _save_otel(job_dir, {"enabled": False})
    monkeypatch.delenv("OTEL_TOKEN", raising=False)

    assert resubmit(str(job_dir)) == "4242"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_resubmit_queues_nothing_when_the_offset_cannot_be_written(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    artifacts = job_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_failures.jsonl").write_text('{"_ng_task_index": 0, "_ng_rollout_index": 0}\n')
    (job_dir / ".gym_infra_retries").write_text("3\n")
    artifacts.chmod(0o555)
    try:
        with pytest.raises(RuntimeError, match="cannot write .*rollouts_resubmit_offset"):
            resubmit(str(job_dir))
    finally:
        artifacts.chmod(0o755)

    assert not (slurm_bin / "cwd").exists()
    assert (job_dir / ".gym_infra_retries").read_text() == "3\n"
    assert (job_dir / ".gym_job_chain").read_text() == "100\n"
