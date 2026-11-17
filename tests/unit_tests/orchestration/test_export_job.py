# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_gym.orchestration.executors.export_job import render_export_job


def test_export_job_uses_cpu_partition_without_gpu_request():
    script = render_export_job(
        bench_dir=Path("/jobs/example/gpqa"),
        partition="cpu",
        account="acct",
        walltime="00:30:00",
        export_command="echo export",
    )

    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --gpus" not in script
    assert "echo export" in script


@pytest.mark.parametrize(
    "partition",
    ["", "cpu;echo unsafe", "cpu\n#SBATCH --gpus=8"],
)
def test_invalid_partition_is_rejected(partition):
    with pytest.raises(ValueError, match="partition"):
        render_export_job(
            bench_dir=Path("/jobs/example/gpqa"),
            partition=partition,
            account="acct",
            walltime="00:30:00",
            export_command="echo export",
        )


def test_empty_export_command_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        render_export_job(
            bench_dir=Path("/jobs/example/gpqa"),
            partition="cpu",
            account="acct",
            walltime="00:30:00",
            export_command="",
        )


@pytest.mark.parametrize("walltime", ["00:99:00", "00:00:99", "1:00:00\n#SBATCH --gpus=8"])
def test_invalid_walltime_is_rejected(walltime):
    with pytest.raises(ValueError, match="walltime"):
        render_export_job(
            bench_dir=Path("/jobs/example/gpqa"),
            partition="cpu",
            account="acct",
            walltime=walltime,
            export_command="true",
        )


@pytest.mark.parametrize(
    "extra_args", [{"gpus": "8"}, {"comment": "hello\n#SBATCH --gpus=8"}, {"dependency": "afterok:123"}]
)
def test_unsafe_or_reserved_directives_are_rejected(extra_args):
    with pytest.raises(ValueError, match="argument"):
        render_export_job(
            bench_dir=Path("/jobs/example/gpqa"),
            partition="cpu",
            account="acct",
            walltime="00:30:00",
            export_command="true",
            extra_args=extra_args,
        )


def test_safe_extra_directive():
    script = render_export_job(
        bench_dir=Path("/jobs/example/gpqa"),
        partition="cpu",
        account="acct",
        walltime="00:30:00",
        export_command="true",
        extra_args={"comment": "export"},
    )
    assert "#SBATCH --comment=export" in script


def test_export_submission_is_locked_and_persists_job_id(tmp_path):
    import os
    import subprocess

    from nemo_gym.orchestration.executors.export_job import render_export_submission

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    fake.write_text("#!/bin/sh\necho 12345\n")
    fake.chmod(0o755)
    script = render_export_submission(bench_dir=tmp_path / "benchmark", job_script="#!/bin/bash\necho exported\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    for _ in range(2):
        result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert (tmp_path / "benchmark" / "exports.job_id").read_text() == "12345\n"
    assert "echo exported" in (tmp_path / "benchmark" / ".exports.sbatch").read_text()


def test_failed_submission_requires_reconciliation_and_is_not_retried(tmp_path):
    import os
    import subprocess

    from nemo_gym.orchestration.executors.export_job import render_export_submission

    bench = tmp_path / "benchmark"
    bench.mkdir()
    bins = tmp_path / "bin"
    bins.mkdir()

    calls = tmp_path / "sbatch_calls"
    sbatch = bins / "sbatch"
    sbatch.write_text(f'#!/bin/bash\necho called >> "{calls}"\nexit 1\n')
    sbatch.chmod(0o755)

    script = render_export_submission(
        bench_dir=bench,
        job_script="#!/bin/bash\ntrue\n",
    )
    env = {**os.environ, "PATH": f"{bins}:{os.environ['PATH']}"}

    for _ in range(2):
        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    assert calls.exists(), f"Fake sbatch was not called: {result.stderr}\nFake script: {sbatch.read_text()}"
    assert calls.read_text().splitlines() == ["called"]
    assert (bench / ".exports.submission_unknown").exists()
    assert not (bench / "exports.job_id").exists()


def test_successful_submission_is_not_repeated(tmp_path):
    import os
    import subprocess

    from nemo_gym.orchestration.executors.export_job import render_export_submission

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    counter = tmp_path / "calls"
    fake.write_text(f"#!/bin/sh\necho call >> '{counter}'\necho 12345\n")
    fake.chmod(0o755)
    script = render_export_submission(bench_dir=tmp_path / "benchmark", job_script="#!/bin/bash\ntrue\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    for _ in range(3):
        result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert counter.read_text().splitlines() == ["call"]


@pytest.mark.parametrize(
    "directive",
    [
        "wrap",
        "gpus-per-socket",
        "gpu-bind",
        "mem",
        "ntasks",
        "dependency",
        "chdir",
    ],
)
def test_export_job_rejects_unsupported_directives(directive):
    with pytest.raises(ValueError, match="Unsupported Slurm export extra argument"):
        render_export_job(
            bench_dir=Path("/tmp/gym-export-test"),
            partition="cpu",
            account="research",
            walltime="00:30:00",
            export_command="true",
            extra_args={directive: "1"},
        )


@pytest.mark.parametrize(
    "directive",
    [
        "wrap",
        "gpus-per-socket",
        "gpu-bind",
        "mem",
        "ntasks",
        "dependency",
        "chdir",
    ],
)
def test_export_job_rejects_unsupported_directives(directive):
    with pytest.raises(ValueError, match="Unsupported Slurm export extra argument"):
        render_export_job(
            bench_dir=Path("/tmp/gym-export-test"),
            partition="cpu",
            account="research",
            walltime="00:30:00",
            export_command="true",
            extra_args={directive: "1"},
        )
