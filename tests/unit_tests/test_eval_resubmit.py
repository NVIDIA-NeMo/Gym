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
import yaml
from omegaconf import OmegaConf

from nemo_gym.cli.main import COMMANDS, _eval_resubmit
from nemo_gym.orchestration.api import SAVED_CONFIG, ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor, UnsupportedOperationError
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.otel import render_collector_config
from nemo_gym.orchestration.executors.resume_script import RETRY_FILE_NAME, render_resume_prologue
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import (
    MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    BenchmarkJob,
    SubmissionRecord,
    load_submission_record,
    local_index_dir,
)
from nemo_gym.orchestration.submit import _EXECUTORS_BY_NAME, resubmit


# Rendered by this Gym, so it carries the line a resubmitted job uses to resume.
_JOB_SH = (
    "#!/bin/bash\nGYM_CMD=(\n    gym eval run\n)\n"
    "if [ x = 1 ]; then GYM_CMD+=(++resubmit=True ++resume_from_cache=True); fi\n"
)


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
    config = config or _config(tmp_path)
    conn = LocalConnection()
    if config.otel.enabled:
        conn.write_text(job_dir / "otel" / "collector.yaml", render_collector_config(config, "gsm8k", job_dir))
    SlurmExecutor().persist(record, config, conn.write_text)
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
    # A job in a hidden partition is listed only with --all.
    (bin_dir / "squeue").write_text(
        '#!/bin/bash\ncat "$(dirname "$0")/squeue_workdirs" 2>/dev/null\n'
        '[[ " $* " == *" --all "* ]] && cat "$(dirname "$0")/squeue_workdirs_hidden" 2>/dev/null\nexit 0\n'
    )
    (bin_dir / "sbatch").write_text(
        '#!/bin/bash\necho "$PWD" > "$(dirname "$0")/cwd"\nenv > "$(dirname "$0")/env"\necho 4242\n'
    )
    (bin_dir / "sacct").write_text('#!/bin/bash\ncat "$(dirname "$0")/accounting"\n')
    (bin_dir / "accounting").write_text("100|FAILED|7200\n101|FAILED|60\n")
    for stub in ("squeue", "sbatch", "sacct"):
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
        lambda self, commands: (
            subprocess.run(
                ["bash", "-s"], input="\n".join(commands), env=clean_env, text=True, capture_output=True, check=True
            ).stdout
        ),
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


def test_resubmit_refuses_while_a_job_runs_in_the_job_dir_in_a_hidden_partition(
    tmp_path: Path, slurm_bin: Path
) -> None:
    job_dir = _staged_run(tmp_path)
    (slurm_bin / "squeue_workdirs_hidden").write_text(f"{job_dir}\n")

    with pytest.raises(RuntimeError, match="still pending or running"):
        resubmit(*_handle(job_dir))

    assert not (slurm_bin / "cwd").exists()


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


def test_held_lock_refuses_a_resubmit_even_when_mkdir_is_not_exclusive(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir = _staged_run(tmp_path)
    (job_dir / ".gym_resubmit.lock").touch()
    # uutils mkdir (Ubuntu 26.04's default) can report success for a directory another process just created.
    (slurm_bin / "mkdir").write_text("#!/bin/bash\nexit 0\n")
    (slurm_bin / "mkdir").chmod(0o755)

    with pytest.raises(RuntimeError, match="another resubmit holds"):
        resubmit(*_handle(job_dir))

    assert not (slurm_bin / "cwd").exists()
    assert (job_dir / ".gym_resubmit.lock").exists()


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
    (bin_dir / "scontrol").write_text('#!/bin/bash\necho "JobId=$SLURM_JOB_ID TimeLimit=00:30:00"\n')
    (bin_dir / "scontrol").chmod(0o755)
    (bin_dir / "squeue").write_text('#!/bin/bash\necho "7|afternotok:5(unfulfilled)|$PWD"\n')
    (bin_dir / "squeue").chmod(0o755)
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


def test_a_resubmitted_job_starts_a_fresh_walltime_budget(tmp_path: Path) -> None:
    """A chain stopped at the walltime cap says to resubmit; the resubmitted chain gets the whole budget again."""
    prologue = render_resume_prologue(ResumeConfig(max_walltime="01:00:00"))
    walltime = tmp_path / ".gym_accumulated_walltime"
    walltime.write_text("3700\n")

    _run_prologue(prologue, tmp_path, {"GYM_RESUBMIT": "1"})

    assert not walltime.exists()

    walltime.write_text("1200\n")
    _run_prologue(prologue, tmp_path, {})
    assert walltime.read_text() == "1200\n"


def _run_args(script: str, job_dir: Path, env: dict[str, str]) -> list[str]:
    """The `gym eval run` arguments job.sh builds, run through bash from its job dir."""
    start = script.index("GYM_CMD=(")
    end = script.index("; fi\n", start) + len("; fi\n")
    snippet = script[start:end] + 'printf "%s\\n" "${GYM_CMD[@]}"\n'
    return subprocess.run(
        ["bash", "-c", snippet], cwd=job_dir, env={"PATH": os.environ["PATH"], **env}, text=True, capture_output=True
    ).stdout.split()


def _recorded_tracking_ids(args: list[str]) -> dict[str, str | None]:
    from nemo_gym.artifacts.exporters.mlflow import MLflowExporter
    from nemo_gym.artifacts.exporters.wandb import WandbExporter

    config = OmegaConf.from_dotlist([arg.lstrip("+") for arg in args if "=" in arg])
    output = Path(config.output_jsonl_fpath)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for exporter_class in (MLflowExporter, WandbExporter):
        marker = output.with_name(f"{output.stem}_{exporter_class.name}_run_id")
        if not marker.exists():
            marker.write_text("existing-run")
        result[exporter_class.name] = exporter_class(config)._recorded_run_id()
    return result


@pytest.mark.parametrize("run", [{}, {"resume_from_cache": False}])
@pytest.mark.parametrize("env", [{"GYM_RESUBMIT": "1"}, {"_gym_resubmit": "1"}])
def test_a_resubmitted_job_normalizes_resume_before_exporters_open(tmp_path: Path, run: dict, env: dict) -> None:
    config = _config(tmp_path)
    config.driver.benchmarks["gsm8k"].run = run
    job_dir = tmp_path / "gsm8k"
    job_dir.mkdir()
    script = build_sbatch_script(
        config, "gsm8k", config.driver.benchmarks["gsm8k"], next(iter(config.compute.values())), job_dir
    )

    fresh_args = _run_args(script, job_dir, {})
    assert "++resubmit=True" not in fresh_args
    assert _recorded_tracking_ids(fresh_args) == {"mlflow": None, "wandb": None}
    # Set directly (no prologue), or moved into `_gym_resubmit` by the auto-resume prologue.
    resumed_args = _run_args(script, job_dir, env)
    assert resumed_args[-2:] == ["++resubmit=True", "++resume_from_cache=True"]
    assert _recorded_tracking_ids(resumed_args) == {"mlflow": "existing-run", "wandb": "existing-run"}


@pytest.mark.parametrize("run", [{}, {"resume_from_cache": False}])
def test_resubmit_normalizes_an_older_saved_job_script(tmp_path: Path, slurm_bin: Path, run: dict) -> None:
    config = _config(tmp_path)
    config.driver.benchmarks["gsm8k"].run = run
    job_dir = _staged_run(tmp_path, config=config)
    script = build_sbatch_script(
        config, "gsm8k", config.driver.benchmarks["gsm8k"], next(iter(config.compute.values())), job_dir
    )
    # Older submitted jobs carry only the attempt-budget flag and are reused rather than rendered again.
    script = script.replace("++resubmit=True ++resume_from_cache=True", "++resubmit=True")
    job_sh = job_dir / "job.sh"
    job_sh.write_text(script)
    artifacts = job_dir / "artifacts"
    artifacts.mkdir()
    rollouts = artifacts / "rollouts.jsonl"
    rollouts.write_text('{"reward": 1.0}\n')
    markers = [artifacts / f"rollouts_{name}_run_id" for name in ("mlflow", "wandb")]
    for marker in markers:
        marker.write_text("existing-run")

    resubmit(*_handle(job_dir))

    assert [marker.read_text() for marker in markers] == ["existing-run", "existing-run"]
    updated = job_sh.read_text()
    args = _run_args(updated, job_dir, {"GYM_RESUBMIT": "1"})
    assert args[-2:] == ["++resubmit=True", "++resume_from_cache=True"]
    assert _recorded_tracking_ids(args) == {"mlflow": "existing-run", "wandb": "existing-run"}
    assert rollouts.read_text() == '{"reward": 1.0}\n'
    resubmit(*_handle(job_dir))
    assert job_sh.read_text() == updated


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


# --- The saved config round trip: what submit persists, resubmit loads -----------------------------------------

_ENV = {"HF_HOME": "lit:/cache/huggingface", "API_KEY": "host:GYM_TEST_SECRET", "LATE": "runtime:LATE"}


def _policy_config(tmp_path: Path, env: dict[str, str]) -> dict:
    """A submit config as written by hand: every env form, and a policy wired by `driver.policy_model`."""
    return {
        "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model", "env": env}},
        "compute": {"cluster": {"type": "slurm", "account": "a", "hostname": None}},
        "driver": {
            "container": "python:3.12",
            "policy_model": "vllm_model",
            "benchmarks": {"gsm8k": {}, "mmlu": {}},
            "env": env,
        },
        "job": {"output_path": str(tmp_path)},
        "otel": {"enabled": False},
    }


def _saved_config(job_dir: Path) -> SubmitConfig:
    """Load the run's saved config the way `gym eval resubmit` does."""
    return SlurmExecutor._saved_config(LocalConnection(), job_dir.parent)


def test_the_config_submit_saves_loads_back_as_resubmit_loads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GYM_TEST_SECRET", "s3cr3t-value")
    config = SubmitConfig.model_validate(_policy_config(tmp_path, _ENV))
    job_dir = _staged_run(tmp_path, config=config)
    saved_text = (job_dir.parent / RESOLVED_CONFIG_NAME).read_text()

    reloaded = _saved_config(job_dir)

    # The record keeps every prefix, and the secret only as its reference.
    assert "s3cr3t-value" not in saved_text
    saved = yaml.safe_load(saved_text)
    assert saved["driver"]["env"] == saved["services"]["vllm_model"]["env"] == _ENV
    # Resubmit gets what submit had: resolved env and the injected policy wiring.
    assert reloaded.driver.env == config.driver.env
    assert reloaded.services["vllm_model"].env == config.services["vllm_model"].env
    assert reloaded.driver.benchmarks == config.driver.benchmarks
    # Saving the reloaded config again writes the same record.
    _staged_run(tmp_path / "again", config=reloaded)
    assert (tmp_path / "again" / "gym-job-x" / RESOLVED_CONFIG_NAME).read_text() == saved_text


def test_resubmit_queues_a_run_submitted_with_lit_env_and_a_policy_model(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GYM_TEST_SECRET", "s3cr3t-value")
    job_dir = _staged_run(tmp_path, config=SubmitConfig.model_validate(_policy_config(tmp_path, _ENV)))

    assert resubmit(*_handle(job_dir)).job_id == "4242"


def test_a_config_saved_with_bare_literals_still_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configs saved before the dump kept `lit:` hold `HF_HOME: /cache/huggingface`."""
    monkeypatch.setenv("GYM_TEST_SECRET", "s3cr3t-value")
    job_dir = _staged_run(tmp_path, config=SubmitConfig.model_validate(_policy_config(tmp_path, _ENV)))
    resolved = job_dir.parent / RESOLVED_CONFIG_NAME
    saved = yaml.safe_load(resolved.read_text())
    for env in (saved["driver"]["env"], saved["services"]["vllm_model"]["env"]):
        env["HF_HOME"] = "/cache/huggingface"
    resolved.write_text(yaml.safe_dump(saved, sort_keys=False))

    reloaded = _saved_config(job_dir)

    assert reloaded.driver.env["HF_HOME"] == "/cache/huggingface"
    assert reloaded.services["vllm_model"].env["HF_HOME"] == "/cache/huggingface"


def test_a_bare_env_value_is_still_refused_in_a_config_written_by_hand(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must start with one of the prefixes"):
        SubmitConfig.model_validate(_policy_config(tmp_path, {"HF_HOME": "/cache/huggingface"}))


def test_a_run_with_a_policy_model_and_no_env_loads_back(tmp_path: Path) -> None:
    """Submit writes the derived policy keys into each benchmark's run, and the saved config keeps them."""
    config = SubmitConfig.model_validate(_policy_config(tmp_path, {}))
    job_dir = _staged_run(tmp_path, config=config)

    reloaded = _saved_config(job_dir)

    assert reloaded.driver.benchmarks["gsm8k"].run == {
        "policy_base_url": "http://localhost:8000/v1",
        "policy_model_name": "org/model",
        "policy_api_key": "dummy",  # pragma: allowlist secret
    }
    assert reloaded.driver.benchmarks == config.driver.benchmarks


def test_a_saved_per_node_policy_url_list_loads_back(tmp_path: Path) -> None:
    """A `server_per_node` policy writes one URL per node; the saved list is the one the policy model derives again."""
    raw = _policy_config(tmp_path, {})
    raw["services"]["vllm_model"].update({"node_pool": "gpu", "tensor_parallel_size": 4, "server_per_node": True})
    raw["compute"]["cluster"]["node_pools"] = {
        "gpu": {"partition": "batch", "nodes": 2, "ntasks_per_node": 1, "gpus_per_node": 4}
    }
    config = SubmitConfig.model_validate(raw)
    job_dir = _staged_run(tmp_path, config=config)

    reloaded = _saved_config(job_dir)

    assert reloaded.driver.benchmarks["gsm8k"].run["policy_base_url"] == [
        "http://${oc.env:GYM_POOL_GPU_NODE_0}:8000/v1",
        "http://${oc.env:GYM_POOL_GPU_NODE_1}:8000/v1",
    ]
    assert reloaded.driver.benchmarks == config.driver.benchmarks


def test_a_saved_policy_key_that_differs_from_the_policy_model_is_still_refused(tmp_path: Path) -> None:
    raw = _policy_config(tmp_path, {})
    raw["driver"]["benchmarks"]["gsm8k"] = {"run": {"policy_model_name": "another/model"}}

    with pytest.raises(ValueError, match=r"already sets \['policy_model_name'\]"):
        SubmitConfig.model_validate(raw, context={SAVED_CONFIG: True})


def test_policy_keys_written_by_hand_still_conflict_with_the_policy_model(tmp_path: Path) -> None:
    raw = _policy_config(tmp_path, {})
    raw["driver"]["benchmarks"]["gsm8k"] = {"run": {"policy_model_name": "org/model"}}

    with pytest.raises(ValueError, match="driver.policy_model is also set"):
        SubmitConfig.model_validate(raw)


# --- A command benchmark reads the policy from its run keys, as NEMO_GYM_POLICY_* -----------------------------

_DERIVED_POLICY_RUN = {
    "policy_base_url": "http://localhost:8000/v1",
    "policy_model_name": "org/model",
    "policy_api_key": "dummy",  # pragma: allowlist secret
}


def _command_policy_config(tmp_path: Path, run: dict | None = None) -> dict:
    raw = _policy_config(tmp_path, {})
    raw["driver"]["benchmarks"]["gsm8k"] = {"command": "true", **({"run": run} if run else {})}
    return raw


def test_a_saved_command_benchmark_with_a_policy_model_loads_back_and_resubmits(
    tmp_path: Path, slurm_bin: Path
) -> None:
    config = SubmitConfig.model_validate(_command_policy_config(tmp_path))
    job_dir = _staged_run(tmp_path, config=config)

    reloaded = _saved_config(job_dir)

    assert reloaded.driver.benchmarks["gsm8k"].command == "true"
    assert reloaded.driver.benchmarks["gsm8k"].run == config.driver.benchmarks["gsm8k"].run == _DERIVED_POLICY_RUN
    assert reloaded.driver.benchmarks == config.driver.benchmarks
    assert resubmit(*_handle(job_dir)).job_id == "4242"


@pytest.mark.parametrize(
    ("run", "message"),
    [
        ({**_DERIVED_POLICY_RUN, "policy_model_name": "another/model"}, r"already sets \['policy_model_name'\]"),
        ({**_DERIVED_POLICY_RUN, "split": "test"}, "sets both `command` and `run`"),
    ],
    ids=["differing_policy_value", "extra_run_key"],
)
def test_a_saved_command_with_other_run_settings_is_still_refused(tmp_path: Path, run: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SubmitConfig.model_validate(_command_policy_config(tmp_path, run), context={SAVED_CONFIG: True})


def test_a_saved_command_without_a_policy_model_still_refuses_run(tmp_path: Path) -> None:
    raw = _command_policy_config(tmp_path, dict(_DERIVED_POLICY_RUN))
    raw["driver"]["policy_model"] = None

    with pytest.raises(ValueError, match="sets both `command` and `run`"):
        SubmitConfig.model_validate(raw, context={SAVED_CONFIG: True})


def test_a_hand_written_command_with_run_is_still_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sets both `command` and `run`"):
        SubmitConfig.model_validate(_command_policy_config(tmp_path, dict(_DERIVED_POLICY_RUN)))


def test_resubmit_without_auth_updates_the_collector_and_can_restore_remote_export(
    tmp_path: Path, slurm_bin: Path, ssh_like_shell: None, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    otel = {"token_env": "NVAUTH_TOKEN", "endpoint": "https://otel.example", "service_name": "s"}
    job_dir = _staged_run(tmp_path, config=_config(tmp_path, otel=otel))
    collector_path = job_dir / "otel" / "collector.yaml"
    assert "otlp_http/managed" in collector_path.read_text()
    monkeypatch.delenv("NVAUTH_TOKEN", raising=False)
    original_job_sh = (job_dir / "job.sh").read_text()

    assert resubmit(*_handle(job_dir)).job_id == "4242"

    collector = yaml.safe_load(collector_path.read_text())
    assert set(collector["exporters"]) == {"file/metrics", "file/traces", "file/logs"}
    assert "NVAUTH_TOKEN" not in _sbatch_env(slurm_bin)
    assert "otel.remote_export=false" in caplog.text
    assert (job_dir / "job.sh").read_text() == original_job_sh

    # The saved requested config still permits remote export on a later authenticated resubmit.
    monkeypatch.setenv("NVAUTH_TOKEN", "new-token")
    assert resubmit(*_handle(job_dir)).job_id == "4242"
    assert "otlp_http/managed" in collector_path.read_text()
    assert _sbatch_env(slurm_bin)["NVAUTH_TOKEN"] == "new-token"


def test_resubmit_missing_auth_does_not_change_an_active_runs_collector(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    otel = {"token_env": "NVAUTH_TOKEN", "endpoint": "https://otel.example", "service_name": "s"}
    job_dir = _staged_run(tmp_path, config=_config(tmp_path, otel=otel))
    collector_path = job_dir / "otel" / "collector.yaml"
    original = collector_path.read_text()
    monkeypatch.delenv("NVAUTH_TOKEN", raising=False)
    (slurm_bin / "squeue_workdirs").write_text(f"{job_dir}\n")

    with pytest.raises(RuntimeError, match="still pending or running"):
        resubmit(*_handle(job_dir))

    assert collector_path.read_text() == original
    assert not (slurm_bin / "cwd").exists()


# --- Recovery keeps the spent budget; a plain resubmit starts a fresh one ---------------------------------------


def _resumable_run(tmp_path: Path) -> tuple[Path, str, ResumeConfig]:
    """A staged resumable run whose job.sh is rendered by this Gym, with spent retries, walltime and one missing row."""
    resume = ResumeConfig(max_retries=3, max_walltime="16:00:00")
    config = _config(tmp_path)
    config.driver.benchmarks["gsm8k"].resumable = resume
    job_dir = _staged_run(tmp_path, config=config)
    script = build_sbatch_script(
        config, "gsm8k", config.driver.benchmarks["gsm8k"], next(iter(config.compute.values())), job_dir
    )
    (job_dir / "job.sh").write_text(script)
    artifacts = job_dir / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": t, "_ng_rollout_index": 0} for t in range(2)]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (artifacts / "rollouts.jsonl").write_text(json.dumps(rows[0] | {"reward": 1.0}) + "\n")
    (job_dir / ".gym_job_chain").write_text("100\n101\n")
    (job_dir / RETRY_FILE_NAME).write_text("2\n")
    (job_dir / ".gym_accumulated_walltime").write_text("7200\n")
    return job_dir, script, resume


def _start_queued_job(script: str, resume: ResumeConfig, job_dir: Path, slurm_bin: Path) -> list[str]:
    """Start the job sbatch queued: the rendered prologue and `gym eval run` arguments, in the environment it got."""
    # Keep the generated installed-runtime admission command rather than substituting a bare prologue.
    admission_start = script.index("_gym_admission_output=$(") + len("_gym_admission_output=$(")
    admission_end = script.index(") || _gym_stop_incomplete", admission_start)
    prologue = render_resume_prologue(resume, admission_command=script[admission_start:admission_end])
    assert prologue in script
    queued = {k: v for k, v in _sbatch_env(slurm_bin).items() if k.startswith("GYM_RESUBMIT")}
    start = script.index("GYM_CMD=(")
    end = script.index("; fi\n", start) + len("; fi\n")
    body = prologue + script[start:end] + 'printf "ARG %s\\n" "${GYM_CMD[@]}"\n'
    out = _run_prologue(body, job_dir, queued).stdout
    return [line[4:] for line in out.splitlines() if line.startswith("ARG ")]


def test_recover_resubmit_keeps_the_spent_retries_walltime_and_attempts(tmp_path: Path, slurm_bin: Path) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir, script, resume = _resumable_run(tmp_path)

    out = recover(*_handle(job_dir))

    assert out.dispatched == "resubmit" and out.job_id == "4242"
    args = _start_queued_job(script, resume, job_dir, slurm_bin)
    assert (job_dir / RETRY_FILE_NAME).read_text() == "3\n"
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "7260\n"
    # It resumes without `++resubmit=True`, which would start a fresh per-rollout attempt budget.
    assert args[-1] == "++resume_from_cache=True" and "++resubmit=True" not in args
    # The follow-up the prologue queues is an ordinary hop.
    assert (job_dir / "follow_up_env").read_text() == "GYM_RESUBMIT=\n"


def test_plain_resubmit_of_the_same_run_still_starts_a_fresh_budget(tmp_path: Path, slurm_bin: Path) -> None:
    job_dir, script, resume = _resumable_run(tmp_path)

    resubmit(*_handle(job_dir))

    args = _start_queued_job(script, resume, job_dir, slurm_bin)
    assert not (job_dir / RETRY_FILE_NAME).exists()
    assert not (job_dir / ".gym_accumulated_walltime").exists()
    assert args[-2:] == ["++resubmit=True", "++resume_from_cache=True"]


def test_recover_refuses_a_job_sh_that_would_reset_the_budget(tmp_path: Path, slurm_bin: Path) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir, script, _ = _resumable_run(tmp_path)
    # A job.sh rendered before budget-preserving recovery: its prologue resets on any resubmit.
    (job_dir / "job.sh").write_text(
        re.sub(r" && \"\$_gym_keep_budget\" != 1", "", script).replace("GYM_RESUBMIT_KEEP_BUDGET", "UNUSED")
    )

    with pytest.raises(RuntimeError, match="predates budget-preserving recovery"):
        recover(*_handle(job_dir))

    assert not (slurm_bin / "env").exists()
    assert (job_dir / RETRY_FILE_NAME).read_text() == "2\n"


def _provider_failures(task: int, attempts: int, cap: int) -> str:
    rows = [
        {
            "_ng_task_index": task,
            "_ng_rollout_index": 0,
            "_ng_attempt_index": attempt,
            "failure_kind": "provider_unavailable",
            "_ng_failure_class": "provider_unavailable",
            "_ng_failure_terminal": False,
            "_ng_failure_max_attempts": cap,
            "_ng_failure_id": f"f{task}-{attempt}",
            "_ng_failure_events": {
                f"f{task}-{attempt}": {
                    "event_id": f"f{task}-{attempt}",
                    "boundary": "model",
                    "failure_kind": "provider_unavailable",
                    "operation": "model_request",
                    "operation_id": f"op{task}",
                    "attempt_id": f"a{task}-{attempt}",
                    "cause_id": None,
                }
            },
        }
        for attempt in range(attempts)
    ]
    return "".join(json.dumps(row) + "\n" for row in rows)


def _core_pending(job_dir: Path, monkeypatch: pytest.MonkeyPatch, cap: int) -> list[tuple[int, int]]:
    """The inputs rollout collection's resume would run (`_load_from_cache`), as a resume without resubmit loads it."""
    from types import SimpleNamespace

    from nemo_gym.rollout_collection import RolloutCollectionHelper

    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", str(cap))
    artifacts = job_dir / "artifacts"
    config = SimpleNamespace(
        materialized_jsonl_fpath=artifacts / "rollouts_materialized_inputs.jsonl",
        output_jsonl_fpath=artifacts / "rollouts.jsonl",
        retry_terminal_timeouts=False,
        dispatch_longest_first=False,
    )
    rows = RolloutCollectionHelper._load_from_cache(None, config, retain_results_in_memory=False)[0]
    return [(row["_ng_task_index"], row["_ng_rollout_index"]) for row in rows]


def test_recover_queues_nothing_when_every_unscored_identity_used_its_attempts(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir, _, _ = _resumable_run(tmp_path)
    (job_dir / "artifacts" / "rollouts_failures.jsonl").write_text(_provider_failures(1, attempts=2, cap=2))

    out = recover(*_handle(job_dir))

    assert out.decision.action == "stop" and out.decision.identities["exhausted"] == [(1, 0)]
    assert out.dispatched is None and not (slurm_bin / "env").exists()
    assert _core_pending(job_dir, monkeypatch, cap=2) == []


def test_recover_queues_a_mixed_run_and_the_resume_runs_only_the_eligible_identity(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir, _, _ = _resumable_run(tmp_path)
    artifacts = job_dir / "artifacts"
    rows = [{"_ng_task_index": t, "_ng_rollout_index": 0} for t in range(3)]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (artifacts / "rollouts_failures.jsonl").write_text(_provider_failures(1, attempts=2, cap=2))

    out = recover(*_handle(job_dir))

    assert out.dispatched == "resubmit" and _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
    assert out.decision.identities["missing"] == [(2, 0)] and out.decision.identities["exhausted"] == [(1, 0)]
    assert _core_pending(job_dir, monkeypatch, cap=2) == [(2, 0)]


@pytest.mark.parametrize(
    "accounting",
    [
        "",
        "100|FAILED|7200\n",
        "100|FAILED|7200\n101|RUNNING|60\n",
        "100|FAILED|7200\n101|FAILED|50400\n",
    ],
)
def test_recovery_never_queues_unaccounted_or_exhausted_chain(tmp_path, slurm_bin, accounting):
    from nemo_gym.orchestration.submit import recover

    job_dir, _, _ = _resumable_run(tmp_path)
    (slurm_bin / "accounting").write_text(accounting)
    before = (job_dir / ".gym_accumulated_walltime").read_bytes()
    out = recover(*_handle(job_dir))
    assert out.decision.action == "stop" and out.dispatched is None
    assert not (slurm_bin / "env").exists()
    assert (job_dir / ".gym_accumulated_walltime").read_bytes() == before


def test_recovery_does_not_recount_the_last_allocation_when_it_is_already_stored(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir, _, _ = _resumable_run(tmp_path)
    (job_dir / ".gym_accumulated_walltime").write_text("7260\n")
    out = recover(*_handle(job_dir))
    assert out.dispatched == "resubmit"
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "7260\n"


def test_recovery_does_not_reconstruct_an_unknown_manual_budget_epoch(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir, _, _ = _resumable_run(tmp_path)
    (job_dir / ".gym_accumulated_walltime").write_text("0\n")
    out = recover(*_handle(job_dir))
    assert out.decision.action == "stop" and out.dispatched is None
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "0\n"


def test_recover_refuses_unbounded_saved_prologue_before_dispatch(tmp_path, slurm_bin):
    from nemo_gym.orchestration.executors.resume_script import BOUNDED_BUDGET_MARKER
    from nemo_gym.orchestration.submit import recover

    job_dir, script, _ = _resumable_run(tmp_path)
    (job_dir / "job.sh").write_text(script.replace(BOUNDED_BUDGET_MARKER, ""))
    with pytest.raises(RuntimeError, match="predates bounded recovery budgets"):
        recover(*_handle(job_dir))
    assert not (slurm_bin / "env").exists()
    assert (job_dir / RETRY_FILE_NAME).read_text() == "2\n"


@pytest.mark.parametrize("recover_budget", [True, False])
def test_resubmit_refuses_a_chain_without_scheduler_cause_admission_before_dispatch(
    tmp_path: Path, slurm_bin: Path, recover_budget: bool
) -> None:
    # A job.sh rendered before scheduler-cause admission retries FAILED, OUT_OF_MEMORY and DEADLINE jobs.
    from nemo_gym.orchestration.executors.resume_script import SCHEDULER_CAUSE_MARKER
    from nemo_gym.orchestration.submit import recover

    job_dir, script, _ = _resumable_run(tmp_path)
    old = script.replace(SCHEDULER_CAUSE_MARKER + "\n", "")
    assert old != script
    (job_dir / "job.sh").write_text(old)
    with pytest.raises(RuntimeError, match="predates scheduler-cause retry admission"):
        recover(*_handle(job_dir)) if recover_budget else resubmit(*_handle(job_dir))
    assert not (slurm_bin / "env").exists()
    assert (job_dir / "job.sh").read_text() == old
    assert (job_dir / RETRY_FILE_NAME).read_text() == "2\n"
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "7200\n"


def test_recover_keeps_an_eligible_infrastructure_retry_and_spent_budgets(tmp_path: Path, slurm_bin: Path) -> None:
    from nemo_gym.orchestration.submit import recover

    job_dir, _, _ = _resumable_run(tmp_path)
    (slurm_bin / "accounting").write_text("100|PREEMPTED|7200\n101|NODE_FAIL|60\n")
    out = recover(*_handle(job_dir))
    assert out.dispatched == "resubmit" and _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
    assert (job_dir / RETRY_FILE_NAME).read_text() == "2\n"


@pytest.mark.parametrize("recover_budget", [True, False])
def test_resubmit_refuses_old_allocation_accounting_capability(tmp_path, slurm_bin, recover_budget):
    from nemo_gym.orchestration.executors.resume_script import ALLOCATION_ACCOUNTING_MARKER
    from nemo_gym.orchestration.submit import recover

    job_dir, script, _ = _resumable_run(tmp_path)
    old = script.replace(ALLOCATION_ACCOUNTING_MARKER + "\n", "")
    (job_dir / "job.sh").write_text(old)
    with pytest.raises(RuntimeError, match="predates durable allocation accounting"):
        recover(*_handle(job_dir)) if recover_budget else resubmit(*_handle(job_dir))
    assert not (slurm_bin / "env").exists()
    assert (job_dir / RETRY_FILE_NAME).read_text() == "2\n"
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "7200\n"


@pytest.mark.parametrize("state", ["durable_unknown", "empty_claim"])
def test_manual_resubmit_does_not_queue_with_unknown_prior_accounting(tmp_path, slurm_bin, state):
    job_dir, _, _ = _resumable_run(tmp_path)
    if state == "durable_unknown":
        (job_dir / ".gym_chain_incomplete").write_text("GYM_ACCOUNTING_UNKNOWN: unobserved follower\n")
    else:
        (job_dir / ".gym_followup_queued.101").touch()
    with pytest.raises(RuntimeError, match="unknown"):
        resubmit(*_handle(job_dir))
    assert not (slurm_bin / "env").exists()
    assert (job_dir / RETRY_FILE_NAME).read_text() == "2\n"
    assert (job_dir / ".gym_accumulated_walltime").read_text() == "7200\n"


@pytest.mark.parametrize("reply", ["success", "lost", "malformed"])
def test_started_worker_waits_for_real_resubmit_wrapper_acknowledgement(tmp_path, slurm_bin, reply):
    """Real local executor shell and generated worker, with scheduler commands stubbed and no workloads."""
    import time

    job_dir, _, resume = _resumable_run(tmp_path)
    worker = render_resume_prologue(resume) + "\necho REACHED_WORK > reached_work\n"
    (job_dir / "worker.sh").write_text(worker)
    (slurm_bin / "scontrol").write_text('#!/bin/bash\necho "JobId=$SLURM_JOB_ID TimeLimit=00:30:00"\n')
    (slurm_bin / "scontrol").chmod(0o755)
    response = {"success": "echo 4242", "lost": "echo lost-reply; exit 1", "malformed": "echo unknown-response"}[reply]
    (slurm_bin / "sbatch").write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == --dependency=* ]]; then echo "Submitted batch job 999"; exit 0; fi\n'
        '(SLURM_JOB_ID=4242 bash "$PWD/worker.sh"; echo $? > "$PWD/worker_exit") > "$PWD/worker.log" 2>&1 &\n'
        "sleep 0.2\n" + response + "\n"
    )
    before = {name: (job_dir / name).read_bytes() for name in (RETRY_FILE_NAME, ".gym_accumulated_walltime")}
    if reply == "success":
        assert resubmit(*_handle(job_dir)).job_id == "4242"
    else:
        with pytest.raises(RuntimeError, match="Resubmit"):
            resubmit(*_handle(job_dir))
    for _ in range(120):
        if (job_dir / "worker_exit").exists():
            break
        time.sleep(0.05)
    assert (job_dir / "worker_exit").exists(), (job_dir / "worker.log").read_text()
    if reply == "success":
        assert (job_dir / "worker_exit").read_text().strip() == "0"
        assert (job_dir / "reached_work").exists()
    else:
        assert (job_dir / "worker_exit").read_text().strip() == "1"
        assert not (job_dir / "reached_work").exists()
        assert all((job_dir / name).read_bytes() == value for name, value in before.items())
        assert (job_dir / ".gym_chain_incomplete").read_text().startswith("GYM_ACCOUNTING_UNKNOWN:")
    assert "4242" in (job_dir / ".gym_job_chain").read_text().split()


@pytest.mark.parametrize(
    "change", ["none", "chain", "claim", "retries", "walltime", "incomplete", "config", "offset", "config_before_lock"]
)
def test_recover_revalidates_the_collected_accounting_inside_the_resubmit_lock(
    tmp_path: Path, slurm_bin: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    # An allocation can end between evidence collection and the locked resubmit: squeue no longer lists it,
    # so only the accounting files it changed show that the collected decision is stale.
    from nemo_gym.orchestration.submit import recover

    job_dir, _, _ = _resumable_run(tmp_path)
    collect = SlurmExecutor.recovery_evidence

    def collect_then_change(self, *args, **kwargs):
        result = collect(self, *args, **kwargs)
        if change == "chain":
            (job_dir / ".gym_job_chain").write_text("100\n101\n102\n")
        elif change == "claim":
            (job_dir / ".gym_followup_queued.102").write_text("102\n")
        elif change == "retries":
            (job_dir / RETRY_FILE_NAME).write_text("3\n")
        elif change == "walltime":
            (job_dir / ".gym_accumulated_walltime").write_text("7300\n")
        elif change == "incomplete":
            (job_dir / ".gym_chain_incomplete").write_text("follow-up queued\n")
        elif change == "config":
            resolved = job_dir.parent / RESOLVED_CONFIG_NAME
            resolved.write_text(resolved.read_text().replace("16:00:00", "17:00:00"))
        elif change == "offset":
            (job_dir / "artifacts" / "rollouts_resubmit_offset").write_text("5\n")
        counters.update(
            {name: (job_dir / name).read_bytes() for name in (RETRY_FILE_NAME, ".gym_accumulated_walltime")}
        )
        return result

    original_run = LocalConnection.run

    def run_with_late_config_change(self, commands, *args, **kwargs):
        if change == "config_before_lock" and any(".gym_resubmit.lock" in command for command in commands):
            resolved = job_dir.parent / RESOLVED_CONFIG_NAME
            resolved.write_text(resolved.read_text().replace("16:00:00", "17:00:00"))
        return original_run(self, commands, *args, **kwargs)

    monkeypatch.setattr(LocalConnection, "run", run_with_late_config_change)
    counters: dict[str, bytes] = {}
    monkeypatch.setattr(SlurmExecutor, "recovery_evidence", collect_then_change)
    job_sh = (job_dir / "job.sh").read_text()
    if change == "none":
        out = recover(*_handle(job_dir))
        assert out.dispatched == "resubmit" and _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
        # The one eligible submit records the sacct-accounted chain walltime once.
        assert (job_dir / ".gym_accumulated_walltime").read_text() == "7260\n"
        return
    with pytest.raises(RuntimeError, match="changed since the recovery evidence was collected"):
        recover(*_handle(job_dir))
    assert not (slurm_bin / "env").exists()
    assert (job_dir / "job.sh").read_text() == job_sh
    assert {name: (job_dir / name).read_bytes() for name in counters} == counters
    assert not (job_dir / ".gym_resubmit.lock").exists()
    assert (job_dir / ".gym_chain_incomplete").exists() == (change == "incomplete")
    assert _record(job_dir).benchmark_job("gsm8k").job_id == "100"
