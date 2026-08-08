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

import re
import shlex
import tempfile
from datetime import datetime
from pathlib import Path

import rich

from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.connection import get_connection
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script


_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


class SlurmExecutor(BaseExecutor):
    """Slurm executor for Pyxis-enabled clusters (https://github.com/NVIDIA/pyxis).

    Every service and the driver are launched via `srun --container-image` so they
    run inside the container specified in their config. Health checks run as plain
    bash inside the sbatch script (no container needed — they just poll HTTP).
    """

    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> None:
        compute = next(iter(config.compute.values()))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_run_dir = Path(config.job.output_path) / f"gym-job-{timestamp}"

        if dry_run:
            self._dry_run(config, compute, remote_run_dir)
            return

        with tempfile.TemporaryDirectory(prefix="gym-submit-") as staging_str:
            staging = self._stage(config, compute, remote_run_dir, Path(staging_str))
            with get_connection(compute.hostname) as conn:
                conn.copy(staging, remote_run_dir)
                output = conn.run(
                    [
                        f"sbatch {shlex.quote(str(remote_run_dir / name / 'job.sh'))}"
                        for name in config.driver.benchmarks
                    ]
                )

        benchmark_names = list(config.driver.benchmarks)
        job_ids = _SBATCH_JOB_ID_RE.findall(output)
        for name, job_id in zip(benchmark_names, job_ids):
            rich.print(f"[green]submitted[/green] {name} → Slurm job [bold]{job_id}[/bold]")
        for name in benchmark_names[len(job_ids) :]:
            rich.print(f"[green]submitted[/green] {name} (job ID unavailable)")

    def _dry_run(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path) -> None:
        print(f"[dry-run] remote run dir: {remote_run_dir}")
        for name, benchmark in config.driver.benchmarks.items():
            script = build_sbatch_script(config, name, benchmark, compute, remote_run_dir / name)
            print(f"\n{'=' * 60}")
            print(f"[dry-run] sbatch script for benchmark: {name}")
            print(f"{'=' * 60}")
            print(script)

    def _stage(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path, staging: Path) -> Path:
        for name, benchmark in config.driver.benchmarks.items():
            bench_dir = staging / name
            bench_dir.mkdir()
            (bench_dir / "logs").mkdir()
            (bench_dir / Path(benchmark.output_jsonl_fpath).parent).mkdir(parents=True, exist_ok=True)
            script = build_sbatch_script(config, name, benchmark, compute, remote_run_dir / name)
            (bench_dir / "job.sh").write_text(script)
        return staging
