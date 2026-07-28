from nemo_gym.orchestration.api import (
    BenchmarkRunConfig,
    ComputeConfig,
    RayServiceConfig,
    SubmitConfig,
    VllmServiceConfig,
)
from nemo_gym.orchestration.executors.base import BaseExecutor


def _build_vllm_command(service: VllmServiceConfig) -> str:
    cmd = f"vllm serve {service.model} --port {service.port} --tensor-parallel-size {service.tensor_parallel_size}"
    if service.trust_remote_code:
        cmd += " --trust-remote-code"
    return cmd


def _build_ray_command(service: RayServiceConfig) -> str:
    return "ray start --head"


_BUILDERS = {
    VllmServiceConfig: _build_vllm_command,
    RayServiceConfig: _build_ray_command,
}


class SlurmExecutor(BaseExecutor):
    def run(self, config: SubmitConfig) -> None:
        compute = next(iter(config.compute.values()))
        for benchmark in config.driver.benchmarks:
            script = self._build_job_script(config, benchmark, compute)
            self._sbatch(script, compute)

    def _build_job_script(
        self, config: SubmitConfig, benchmark: BenchmarkRunConfig, compute: ComputeConfig
    ) -> str:
        lines = ["#!/bin/bash"]

        if compute.walltime:
            lines.append(f"#SBATCH --time={compute.walltime}")

        for name, service in config.services.items():
            builder = _BUILDERS[type(service)]
            lines.append(f"# service: {name}")
            lines.append(builder(service) + " &")

        for name, service in config.services.items():
            if service.health_check:
                hc = service.health_check
                lines.append(
                    f"gym env wait --host localhost --port {hc.port} --path {hc.path} --timeout {hc.timeout_seconds}"
                )

        lines.append(f"gym eval run --benchmark {benchmark.name}")
        return "\n".join(lines)

    def _sbatch(self, script: str, compute: ComputeConfig) -> None:
        raise NotImplementedError  # pragma: no cover
