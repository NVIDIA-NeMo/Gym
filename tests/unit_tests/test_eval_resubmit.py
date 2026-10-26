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

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pytest

from nemo_gym.cli.main import COMMANDS, _eval_resubmit
from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor, UnsupportedOperationError
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.resume_script import RETRY_FILE_NAME, render_resume_prologue
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import (
    MANIFEST_NAME,
    BenchmarkJob,
    SubmissionRecord,
    load_submission_record,
    local_index_dir,
)
from nemo_gym.orchestration.submit import _EXECUTORS_BY_NAME, resubmit


# Rendered by this Gym, so it carries the line a resubmitted job uses to resume.
_JOB_SH = "#!/bin/bash\nGYM_CMD=(\n    gym eval run\n)\nif [ x = 1 ]; then GYM_CMD+=(++resubmit=True); fi\n"


def _config(tmp_path: Path, *, otel: dict | None = None, forward_env: list[str] | None = None) -> SubmitConfig:
    compute = {"type": "slurm", "account": "a", "hostname": None, "forward_env": forward_env or []}
    return SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": compute},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}, "mmlu": {}}},
            "job": {"output_path": str(tmp_path)},
            "otel": otel or {"enabled": False},
        }
    )


@pytest.fixture(autouse=True)
def _local_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def _staged_run(tmp_path: Path, *, executor: str = "slurm", config: SubmitConfig | None = None) -> Path:
    """A submitted run as `persist()` leaves it: local index, manifest and resolved config."""
    run_dir = tmp_path / "gym-job-x"
    job_dir = run_dir / "gsm8k"
    job_dir.mkdir(parents=True)
    (job_dir / "job.sh").write_text(_JOB_SH)
    record = SubmissionRecord(
        gym_job_id=run_dir.name,
        gym_version="0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=str(run_dir),
        cluster="cluster",
        executor=executor,
        submitted_by="me",
        benchmarks=[
            BenchmarkJob(benchmark="gsm8k", job_dir=str(job_dir), job_id="100"),
            BenchmarkJob(benchmark="mmlu", job_dir=str(run_dir / "mmlu"), job_id="101"),
        ],
    )
    SlurmExecutor().persist(record, config or _config(tmp_path), LocalConnection().write_text)
    return job_dir


def _handle(job_dir: Path) -> tuple[str, str]:
    """The CLI's executor-neutral handle for a staged job dir: the submission's id and the benchmark."""
    return job_dir.parent.name, job_dir.name


def _record(job_dir: Path) -> SubmissionRecord:
    return SubmissionRecord.load(json.loads((job_dir.parent / MANIFEST_NAME).read_text()))


@pytest.fixture
def slurm_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """squeue prints the `squeue_workdirs` file; sbatch records its cwd and environment and answers 4242."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text('#!/bin/bash\ncat "$(dirname "$0")/squeue_workdirs" 2>/dev/null; exit 0\n')
    (bin_dir / "sbatch").write_text(
        '#!/bin/bash\necho "$PWD" > "$(dirname "$0")/cwd"\nenv > "$(dirname "$0")/env"\necho 4242\n'
    )
    for stub in ("squeue", "sbatch"):
        (bin_dir / stub).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _sbatch_env(slurm_bin: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in (slurm_bin / "env").read_text().splitlines() if "=" in line)


@pytest.fixture
def ssh_like_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the executor's commands in a shell without the caller's environment, as they run over SSH."""
    clean_env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/")}
    monkeypatch.setattr(
        LocalConnection,
        "run",
        lambda self, commands: subprocess.run(
            ["bash", "-s"], input="\n".join(commands), env=clean_env, text=True, capture_output=True, check=True
        ).stdout,
    )


# --- Executor contract, record and CLI handle ---------------------------------------------------------------


def test_resubmit_queues_job_sh_from_its_job_dir_and_records_the_new_id(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)

    job = resubmit(*_handle(job_dir))

    assert job == BenchmarkJob(benchmark="gsm8k", job_dir=str(job_dir), job_id="4242")
    assert (slurm_bin / "cwd").read_text().strip() == str(job_dir)
    # Both stores `persist()` writes now name the new job; the other benchmark is untouched.
    index = SubmissionRecord.load(json.loads((local_index_dir() / f"{job_dir.parent.name}.json").read_text()))
    for record in (_record(job_dir), index):
        assert [(b.benchmark, b.job_id) for b in record.benchmarks] == [("gsm8k", "4242"), ("mmlu", "101")]


def test_the_record_is_found_by_its_id_and_benchmark_through_the_local_index(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    # The run dir is not on this machine (a remote cluster): only the local index knows the submission.
    (job_dir.parent / MANIFEST_NAME).unlink()

    assert load_submission_record(job_dir.parent.name).benchmark_job("gsm8k").job_id == "100"
    with pytest.raises(ValueError, match="has no benchmark 'nope'; it has: gsm8k, mmlu"):
        resubmit(job_dir.parent.name, "nope")
    assert not (slurm_bin / "cwd").exists()


def test_the_cli_takes_a_gym_job_id_and_a_benchmark(
    tmp_path: Path, slurm_bin: Path, capsys: pytest.CaptureFixture
) -> None:
    job_dir = _staged_run(tmp_path)

    _eval_resubmit(argparse.Namespace(gym_job_id=job_dir.parent.name, benchmark="gsm8k"), [])

    assert capsys.readouterr().out == "4242\n"
    assert "Slurm" not in COMMANDS["eval resubmit"].summary


@pytest.mark.parametrize(("executor", "supported"), [(SlurmExecutor, True), (KubernetesExecutor, False)])
def test_which_executors_support_resubmit(tmp_path: Path, slurm_bin: Path, executor: type, supported: bool) -> None:
    job_dir = _staged_run(tmp_path, executor=executor.name)

    if supported:
        assert resubmit(*_handle(job_dir)).job_id == "4242"
    else:
        with pytest.raises(UnsupportedOperationError, match="Kubernetes executor does not support resubmit"):
            resubmit(*_handle(job_dir))
    assert "resubmit" in BaseExecutor.__abstractmethods__
    assert set(_EXECUTORS_BY_NAME) == {"slurm", "kubernetes"}


def test_resubmit_refuses_while_a_job_runs_in_the_job_dir(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (slurm_bin / "squeue_workdirs").write_text(f"{job_dir}\n")

    with pytest.raises(RuntimeError, match="still pending or running"):
        resubmit(*_handle(job_dir))

    assert not (slurm_bin / "cwd").exists()
    assert _record(job_dir).benchmark_job("gsm8k").job_id == "100"


def test_resubmit_refuses_an_unknown_submission(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No submission 'gym-job-nope'"):
        resubmit("gym-job-nope", "gsm8k")


def test_resubmit_refuses_an_unknown_executor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown executor 'local'"):
        resubmit(*_handle(_staged_run(tmp_path, executor="local")))


def test_two_concurrent_resubmits_queue_one_job(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    # Hold sbatch until the competing resubmit returns, so the lock is tested without a timing guess.
    (slurm_bin / "squeue").write_text('#!/bin/bash\ncat "$(dirname "$0")/queued" 2>/dev/null; exit 0\n')
    (slurm_bin / "sbatch").write_text(
        '#!/bin/bash\nstub_dir=$(dirname "$0")\n'
        'while [ ! -f "$stub_dir/release" ]; do sleep 0.01; done\n'
        'echo "$PWD" >> "$stub_dir/queued"\necho 4242\n'
    )

    def attempt() -> str:
        try:
            return resubmit(*_handle(job_dir)).job_id
        except RuntimeError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = [pool.submit(attempt) for _ in range(2)]
        try:
            done, pending = wait(attempts, timeout=10, return_when=FIRST_COMPLETED)
            assert len(done) == len(pending) == 1, "one resubmit must refuse while the other holds the lock"
            assert "another resubmit holds" in next(iter(done)).result()
        finally:
            (slurm_bin / "release").touch()
        outcomes = [attempt.result(timeout=10) for attempt in attempts]

    assert sorted(outcome == "4242" for outcome in outcomes) == [False, True]
    assert (slurm_bin / "queued").read_text() == f"{job_dir}\n"
    assert not (job_dir / ".gym_resubmit.lock").exists()


# --- The resubmit only queues; the job resets its own run state ---------------------------------------------


def test_resubmit_only_queues_and_asks_the_job_to_reset_its_state(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (job_dir / "artifacts").mkdir()
    failures = '{"_ng_task_index": 0, "_ng_rollout_index": 0}\n'
    (job_dir / "artifacts" / "rollouts_failures.jsonl").write_text(failures)
    (job_dir / RETRY_FILE_NAME).write_text("3\n")

    assert resubmit(*_handle(job_dir)).job_id == "4242"

    # Nothing in the run dir is rewritten by the transport; the queued job is told it is a resubmit.
    assert sorted(p.name for p in (job_dir / "artifacts").iterdir()) == ["rollouts_failures.jsonl"]
    assert (job_dir / RETRY_FILE_NAME).read_text() == "3\n"
    assert _sbatch_env(slurm_bin)["GYM_RESUBMIT"] == "1"


def _run_prologue(prologue: str, job_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """The prologue up to the follow-up sbatch, whose environment it records."""
    bin_dir = job_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "sbatch").write_text('#!/bin/bash\necho "GYM_RESUBMIT=${GYM_RESUBMIT:-}" > follow_up_env\necho 7\n')
    (bin_dir / "sbatch").chmod(0o755)
    script = prologue + 'echo "_gym_resubmit=$_gym_resubmit"\n'
    return subprocess.run(
        ["bash", "-c", script],
        cwd=job_dir,
        env={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "SLURM_JOB_ID": "5", **env},
        text=True,
        capture_output=True,
        check=True,
    )


def test_a_resubmitted_job_resets_its_retry_count_once(tmp_path: Path) -> None:
    prologue = render_resume_prologue(ResumeConfig(max_retries=3))
    (tmp_path / RETRY_FILE_NAME).write_text("3\n")

    first = _run_prologue(prologue, tmp_path, {"GYM_RESUBMIT": "1"})

    assert not (tmp_path / RETRY_FILE_NAME).exists()
    assert "_gym_resubmit=1" in first.stdout
    # The chain's next hop inherits this environment, but it is an ordinary hop, not a second resubmit.
    assert (tmp_path / "follow_up_env").read_text() == "GYM_RESUBMIT=\n"

    (tmp_path / RETRY_FILE_NAME).write_text("2\n")
    _run_prologue(prologue, tmp_path, {})
    assert (tmp_path / RETRY_FILE_NAME).read_text() == "2\n"


def _run_args(script: str, job_dir: Path, env: dict[str, str]) -> list[str]:
    """The `gym eval run` arguments job.sh builds, run through bash from its job dir."""
    start = script.index("GYM_CMD=(")
    end = script.index("; fi\n", start) + len("; fi\n")
    snippet = script[start:end] + 'printf "%s\\n" "${GYM_CMD[@]}"\n'
    return subprocess.run(
        ["bash", "-c", snippet], cwd=job_dir, env={"PATH": os.environ["PATH"], **env}, text=True, capture_output=True
    ).stdout.split()


@pytest.mark.parametrize("run", [{}, {"resume_from_cache": False}])
@pytest.mark.parametrize("env", [{"GYM_RESUBMIT": "1"}, {"_gym_resubmit": "1"}])
def test_a_resubmitted_job_sh_tells_rollout_collection(tmp_path: Path, run: dict, env: dict) -> None:
    config = _config(tmp_path)
    config.driver.benchmarks["gsm8k"].run = run
    job_dir = tmp_path / "gsm8k"
    job_dir.mkdir()
    script = build_sbatch_script(
        config, "gsm8k", config.driver.benchmarks["gsm8k"], next(iter(config.compute.values())), job_dir
    )

    assert "++resubmit=True" not in _run_args(script, job_dir, {})
    # Set directly (no prologue), or moved into `_gym_resubmit` by the auto-resume prologue.
    assert _run_args(script, job_dir, env)[-1] == "++resubmit=True"


def test_resubmit_refuses_a_populated_run_whose_job_sh_predates_resubmit(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (job_dir / "artifacts").mkdir()
    rollouts = job_dir / "artifacts" / "rollouts.jsonl"
    rollouts.write_text('{"reward": 1.0}\n')
    # Without the resubmit line, a job.sh with resume off would start fresh and delete the rollouts above.
    (job_dir / "job.sh").write_text("#!/bin/bash\nGYM_CMD=(\n    gym eval run\n    +resume_from_cache=False\n)\n")

    with pytest.raises(RuntimeError, match="predates resubmit"):
        resubmit(*_handle(job_dir))

    assert not (slurm_bin / "cwd").exists()
    assert rollouts.read_text() == '{"reward": 1.0}\n'


def test_resubmit_queues_a_populated_command_benchmark(tmp_path: Path, slurm_bin: Path) -> None:
    config = _config(tmp_path)
    config.driver.benchmarks["gsm8k"].command = "python harness.py"
    job_dir = _staged_run(tmp_path, config=config)
    # A `command` benchmark's job.sh runs its own command, never `gym eval run`, so it has no resubmit line.
    (job_dir / "job.sh").write_text(
        build_sbatch_script(
            config, "gsm8k", config.driver.benchmarks["gsm8k"], next(iter(config.compute.values())), job_dir
        )
    )
    (job_dir / "artifacts").mkdir()
    (job_dir / "artifacts" / "rollouts.jsonl").write_text('{"reward": 1.0}\n')

    assert resubmit(*_handle(job_dir)).job_id == "4242"
    assert (slurm_bin / "cwd").read_text().strip() == str(job_dir)


def test_a_failed_record_update_names_the_queued_jobs(tmp_path: Path) -> None:
    job_dir = _staged_run(tmp_path)
    record = _record(job_dir).with_benchmark(BenchmarkJob(benchmark="gsm8k", job_dir=str(job_dir), job_id="4242"))

    def _explode(path, text, **_):
        raise OSError("remote is read-only")

    with pytest.raises(RuntimeError, match="Already queued: gsm8k=4242, mmlu=101. Record these by hand"):
        SlurmExecutor().persist_record(record, _explode)
    # The local index is written first, so it has the new job id although the manifest does not.
    index = SubmissionRecord.load(json.loads((local_index_dir() / f"{job_dir.parent.name}.json").read_text()))
    assert index.benchmark_job("gsm8k").job_id == "4242"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_resubmit_queues_nothing_when_preparation_fails(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    # The lock cannot be taken in a read-only job dir, so the script stops before sbatch.
    job_dir.chmod(0o555)
    try:
        with pytest.raises(RuntimeError, match="another resubmit holds"):
            resubmit(*_handle(job_dir))
    finally:
        job_dir.chmod(0o755)

    assert not (slurm_bin / "cwd").exists()
    assert _record(job_dir).benchmark_job("gsm8k").job_id == "100"


# --- The resubmit repeats what submit did -------------------------------------------------------------------


def test_resubmit_forwards_what_submit_forwards(
    tmp_path: Path, slurm_bin: Path, ssh_like_shell: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    otel = {"enabled": True, "token_env": "MY_INGEST_TOKEN", "endpoint": "https://otel.example", "service_name": "s"}
    job_dir = _staged_run(tmp_path, config=_config(tmp_path, otel=otel, forward_env=["MY_KEY"]))
    monkeypatch.setenv("MY_INGEST_TOKEN", "s3cret value")
    monkeypatch.setenv("MY_KEY", "k'ey")

    assert resubmit(*_handle(job_dir)).job_id == "4242"

    env = _sbatch_env(slurm_bin)
    assert (env["MY_INGEST_TOKEN"], env["MY_KEY"]) == ("s3cret value", "k'ey")


@pytest.mark.parametrize(
    ("config_kwargs", "unset", "message"),
    [
        ({"forward_env": ["MY_KEY"]}, "MY_KEY", "compute.forward_env variables are unset: MY_KEY"),
        (
            {"otel": {"enabled": True, "token_env": "MY_INGEST_TOKEN", "endpoint": "https://o", "service_name": "s"}},
            "MY_INGEST_TOKEN",
            "'MY_INGEST_TOKEN' is not set",
        ),
    ],
)
def test_resubmit_queues_nothing_when_a_forwarded_value_is_unset(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch, config_kwargs: dict, unset: str, message: str
) -> None:
    job_dir = _staged_run(tmp_path, config=_config(tmp_path, **config_kwargs))
    monkeypatch.delenv(unset, raising=False)

    with pytest.raises(ValueError, match=re.escape(message)):
        resubmit(*_handle(job_dir))

    assert not (slurm_bin / "cwd").exists()


def test_resubmit_with_telemetry_off_needs_no_token(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _staged_run(tmp_path)
    monkeypatch.delenv("OTEL_TOKEN", raising=False)

    assert resubmit(*_handle(job_dir)).job_id == "4242"
