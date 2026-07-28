import shlex
import tempfile
from pathlib import Path

from nemo_gym.orchestration.api import (
    BenchmarkRunConfig,
    RayServiceConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmServiceConfig,
)
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.connection import get_connection


def _build_vllm_command(service: VllmServiceConfig) -> str:
    cmd = f"vllm serve {shlex.quote(service.model)} --port {service.port} --tensor-parallel-size {service.tensor_parallel_size}"
    if service.trust_remote_code:
        cmd += " --trust-remote-code"
    return cmd


def _build_ray_command(_service: RayServiceConfig) -> str:
    return "ray start --head"


_BUILDERS = {
    VllmServiceConfig: _build_vllm_command,
    RayServiceConfig: _build_ray_command,
}


class SlurmExecutor(BaseExecutor):
    """Slurm executor for Pyxis-enabled clusters (https://github.com/NVIDIA/pyxis).

    Every service and the driver are launched via `srun --container-image` so they
    run inside the container specified in their config. Health checks run as plain
    bash inside the sbatch script (no container needed — they just poll HTTP).
    """

    def run(self, config: SubmitConfig) -> None:
        compute = next(iter(config.compute.values()))
        output_path = Path(config.job.output_path)

        staging = self._stage(config, compute)
        with get_connection(compute.hostname) as conn:
            conn.copy(staging, output_path)
            conn.run([
                f"sbatch {shlex.quote(str(output_path / b.name / 'job.sh'))}"
                for b in config.driver.benchmarks
            ])

    def _stage(self, config: SubmitConfig, compute: SlurmComputeConfig) -> Path:
        staging = Path(tempfile.mkdtemp(prefix="gym-submit-"))
        for benchmark in config.driver.benchmarks:
            bench_dir = staging / benchmark.name
            bench_dir.mkdir()
            (bench_dir / "logs").mkdir()
            script = self._build_job_script(config, benchmark, compute)
            (bench_dir / "job.sh").write_text(script)
        return staging

    def _build_job_script(
        self, config: SubmitConfig, benchmark: BenchmarkRunConfig, compute: SlurmComputeConfig
    ) -> str:
        lines = ["#!/bin/bash"]

        if compute.walltime:
            lines.append(f"#SBATCH --time={compute.walltime}")

        for name, service in config.services.items():
            builder = _BUILDERS[type(service)]
            lines.append(f"# service: {name}")
            lines.append(
                f"srun --container-image={shlex.quote(service.container)} "
                f"{builder(service)} &"
            )

        for name, service in config.services.items():
            if service.health_check:
                hc = service.health_check
                lines.append(
                    f"# health check: {name} (timeout {hc.timeout_seconds}s)\n"
                    f"timeout {hc.timeout_seconds} bash -c "
                    f"'until curl -sf http://localhost:{hc.port}{shlex.quote(hc.path)}; do sleep 2; done'"
                )

        lines.append(
            f"srun --container-image={shlex.quote(config.driver.container)} "
            f"gym eval run --benchmark {shlex.quote(benchmark.name)}"
        )
        return "\n".join(lines)
