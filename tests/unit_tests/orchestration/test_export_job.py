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


def test_failed_submission_does_not_fail_eval_and_can_retry(tmp_path):
    import os
    import subprocess

    from nemo_gym.orchestration.executors.export_job import render_export_submission

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    bench = tmp_path / "benchmark"
    script = render_export_submission(bench_dir=bench, job_script="#!/bin/bash\ntrue\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    first = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert first.returncode == 0
    assert not (bench / "exports.job_id").exists()
    assert not (bench / ".exports.submit.lock").exists()

    fake.write_text("#!/bin/sh\necho 45678\n")
    second = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert (bench / "exports.job_id").read_text() == "45678\n"


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
