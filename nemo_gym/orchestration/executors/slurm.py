import shlex
import tempfile
from datetime import datetime
from pathlib import Path

from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.connection import get_connection
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script


class SlurmExecutor(BaseExecutor):
    """Slurm executor for Pyxis-enabled clusters (https://github.com/NVIDIA/pyxis).

    Every service and the driver are launched via `srun --container-image` so they
    run inside the container specified in their config. Health checks run as plain
    bash inside the sbatch script (no container needed — they just poll HTTP).
    """

    def run(self, config: SubmitConfig) -> None:
        compute = next(iter(config.compute.values()))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_run_dir = Path(config.job.output_path) / timestamp

        staging = self._stage(config, compute, remote_run_dir)
        with get_connection(compute.hostname) as conn:
            conn.copy(staging, remote_run_dir)
            conn.run([
                f"sbatch {shlex.quote(str(remote_run_dir / b.name / 'job.sh'))}"
                for b in config.driver.benchmarks
            ])

    def _stage(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path) -> Path:
        staging = Path(tempfile.mkdtemp(prefix="gym-submit-"))
        for benchmark in config.driver.benchmarks:
            bench_dir = staging / benchmark.name
            bench_dir.mkdir()
            (bench_dir / "logs").mkdir()
            script = build_sbatch_script(config, benchmark, compute, remote_run_dir / benchmark.name)
            (bench_dir / "job.sh").write_text(script)
        return staging
