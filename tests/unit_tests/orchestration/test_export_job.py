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
