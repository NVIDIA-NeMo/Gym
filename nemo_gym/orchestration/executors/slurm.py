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

        staging = self._stage(config, compute, remote_run_dir)
        with get_connection(compute.hostname) as conn:
            conn.copy(staging, remote_run_dir)
            output = conn.run([
                f"sbatch {shlex.quote(str(remote_run_dir / b.name / 'job.sh'))}"
                for b in config.driver.benchmarks
            ])

        benchmarks = [b.name for b in config.driver.benchmarks]
        job_ids = _SBATCH_JOB_ID_RE.findall(output)
        for benchmark, job_id in zip(benchmarks, job_ids):
            rich.print(f"[green]submitted[/green] {benchmark} → Slurm job [bold]{job_id}[/bold]")
        unmatched = benchmarks[len(job_ids):]
        for benchmark in unmatched:
            rich.print(f"[green]submitted[/green] {benchmark} (job ID unavailable)")

    def _dry_run(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path) -> None:
        print(f"[dry-run] remote run dir: {remote_run_dir}")
        for benchmark in config.driver.benchmarks:
            script = build_sbatch_script(config, benchmark, compute, remote_run_dir / benchmark.name)
            print(f"\n{'='*60}")
            print(f"[dry-run] sbatch script for benchmark: {benchmark.name}")
            print(f"{'='*60}")
            print(script)

    def _stage(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path) -> Path:
        staging = Path(tempfile.mkdtemp(prefix="gym-submit-"))
        for benchmark in config.driver.benchmarks:
            bench_dir = staging / benchmark.name
            bench_dir.mkdir()
            (bench_dir / "logs").mkdir()
            script = build_sbatch_script(config, benchmark, compute, remote_run_dir / benchmark.name)
            (bench_dir / "job.sh").write_text(script)
        return staging
