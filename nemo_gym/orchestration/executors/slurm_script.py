import shlex
from pathlib import Path

from nemo_gym.orchestration.executors.script_templates import bash_var, render_health_check
from nemo_gym.orchestration.executors.utils import flatten_run_args
from nemo_gym.orchestration.api import (
    BenchmarkRunConfig,
    NodePool,
    RayServiceConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmServiceConfig,  # used in _BUILDERS dispatch table
)

_SCRIPT_TEMPLATE = """\
#!/bin/bash
{directives}

{service_commands}

{health_checks}

{driver_command}
"""


def _render_directives(compute: SlurmComputeConfig, remote_bench_dir: Path, benchmark_name: str) -> str:
    lines = []
    lines.append(f"#SBATCH --job-name=gym-{benchmark_name}")
    lines.append(f"#SBATCH --account={compute.account}")
    if compute.walltime:
        lines.append(f"#SBATCH --time={compute.walltime}")
    lines.append(f"#SBATCH --chdir={remote_bench_dir}")
    for key, val in compute.extra_args.items():
        lines.append(f"#SBATCH --{key}={val}")
    for pool_name, pool in compute.node_pools.items():
        lines.extend(_render_pool_directives(pool_name, pool))
    return "\n".join(lines)


def _render_pool_directives(pool_name: str, pool: NodePool) -> list[str]:
    lines = [
        f"#SBATCH --partition={pool.partition}  # pool: {pool_name}",
        f"#SBATCH --nodes={pool.nodes}",
        f"#SBATCH --ntasks-per-node={pool.ntasks_per_node}",
    ]
    if pool.gpus_per_node is not None:
        lines.append(f"#SBATCH --gpus-per-node={pool.gpus_per_node}")
    for key, val in pool.extra_args.items():
        lines.append(f"#SBATCH --{key}={val}")
    return lines


def _render_service_command(name: str, container: str, command: str) -> str:
    var = bash_var(name)
    return (
        f"# service: {name}\n"
        f"srun --container-image={shlex.quote(container)} --output=logs/{name}.log {command} &\n"
        f"{var}_PID=$!"
    )


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


def build_sbatch_script(
    config: SubmitConfig,
    benchmark: BenchmarkRunConfig,
    compute: SlurmComputeConfig,
    remote_bench_dir: Path,
) -> str:
    directives = _render_directives(compute, remote_bench_dir, benchmark.name)

    service_commands = "\n\n".join(
        _render_service_command(name, service.container, _BUILDERS[type(service)](service))
        for name, service in config.services.items()
    )

    health_checks = "\n\n".join(
        render_health_check(name, service.health_check.port, service.health_check.timeout_seconds)
        for name, service in config.services.items()
        if service.health_check
    )

    run_args = " ".join(flatten_run_args(benchmark.run))
    driver_command = (
        f"srun --container-image={shlex.quote(config.driver.container)} "
        f"--output=logs/driver.log "
        f"gym eval run --benchmark {shlex.quote(benchmark.name)}"
        + (f" {run_args}" if run_args else "")
    )

    return _SCRIPT_TEMPLATE.format(
        directives=directives,
        service_commands=service_commands,
        health_checks=health_checks,
        driver_command=driver_command,
    )
