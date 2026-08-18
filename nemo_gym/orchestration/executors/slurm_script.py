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
from pathlib import Path

from nemo_gym.orchestration.api import (
    BenchmarkRunConfig,
    NodePool,
    RayServiceConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmServiceConfig,  # used in _BUILDERS dispatch table
)
from nemo_gym.orchestration.executors.script_templates import (
    bash_var,
    render_driver_entrypoint,
    render_gym_cmd,
    render_health_check,
)
from nemo_gym.orchestration.executors.utils import flatten_run_args


_SCRIPT_TEMPLATE = """\
#!/bin/bash
{directives}

{ray_prelude}

{service_commands}

{health_checks}

{prepare_command}

{driver_command}
"""


# Resolves the Ray head node from Slurm's node list, mirroring scripts/sbatch_base.sh.
# Runs once in the main sbatch shell (before any srun step) so $SLURM_JOB_NODELIST is available.
_RAY_PRELUDE = """\
# Resolve Ray head node IP for multi-node vLLM services (distributed_backend: ray).
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node_hostname" | awk '{print $1}')
RAY_HEAD_NODE_IP="$head_node_ip:6379"
echo "Ray head node IP address: $RAY_HEAD_NODE_IP\""""


def _render_directives(compute: SlurmComputeConfig, remote_bench_dir: Path, benchmark_name: str) -> str:
    lines = []
    lines.append(f"#SBATCH --job-name=gym-{benchmark_name}")
    lines.append(f"#SBATCH --account={compute.account}")
    if compute.walltime:
        lines.append(f"#SBATCH --time={compute.walltime}")
    # --chdir makes relative paths (logs/, artifacts/) resolve correctly inside the job.
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


_VALID_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_key(key: str) -> None:
    if not _VALID_ENV_KEY.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")


def _resolve_env(env: dict[str, str]) -> str:
    """Return an 'env K=V ...' prefix string (trailing space) scoped to a single command, or '' if empty."""
    if not env:
        return ""
    for k in env:
        _validate_env_key(k)
    pairs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"env {pairs} "


def _render_service_command(
    name: str,
    container: str,
    command: str,
    env: dict[str, str] | None = None,
    mounts: list[str] | None = None,
    nodes: int | None = None,
    ntasks: int | None = None,
) -> str:
    var = bash_var(name)
    env_prefix = _resolve_env(env) if env else ""
    node_flags = f" --nodes={nodes} --ntasks={ntasks}" if (nodes is not None and nodes > 1) else ""
    mounts_flag = f" --container-mounts={','.join(shlex.quote(m) for m in mounts)}" if mounts else ""
    # --overlap lets this step share the allocation with other concurrent steps (driver + services).
    # --no-container-mount-home avoids polluting the container with host home directory contents.
    # PID is captured so the health check can detect early service death.
    return (
        f"# service: {name}\n"
        f"{env_prefix}srun --overlap --no-container-mount-home{node_flags}{mounts_flag} --container-image={shlex.quote(container)} --output=logs/{name}.log {command} &\n"
        f"{var}_PID=$!"
    )


def _build_vllm_command(service: VllmServiceConfig) -> str:
    cmd = (
        f"vllm serve {shlex.quote(service.model)}"
        f" --port {service.port}"
        f" --tensor-parallel-size {service.tensor_parallel_size}"
    )
    if service.pipeline_parallel_size > 1:
        cmd += f" --pipeline-parallel-size {service.pipeline_parallel_size}"
    if service.number_of_instances > 1:
        cmd += f" --data-parallel-size {service.number_of_instances}"
    if service.trust_remote_code:
        cmd += " --trust-remote-code"
    return cmd


def _build_vllm_ray_command(service: VllmServiceConfig, total_nodes: int) -> str:
    # Uses vLLM's own Ray *core* executor to span nodes (--distributed-executor-backend ray).
    # This is not the ray.serve library - no Serve deployment/ingress/HTTP proxy is involved.
    inner_cmd = _build_vllm_command(service) + " --distributed-executor-backend ray"
    if service.number_of_instances > 1:
        inner_cmd += " --data-parallel-backend ray"
    resource_flags = (
        "--num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE}"
    )
    # ray symmetric-run starts/joins a Ray cluster across every task and runs the entrypoint
    # only on the elected head node, mirroring scripts/sbatch_base.sh. It requires Ray >= 2.50, so
    # containers with an older pin fall back to manually starting head/worker Ray processes, keyed
    # on Slurm's per-node task rank ($SLURM_NODEID). vLLM's Ray executor blocks on placement-group
    # scheduling until every node's GPUs join, so the fallback needs no separate cluster-ready wait.
    # Model-serving images (e.g. vllm/vllm-openai) don't necessarily bundle the ray CLI - vLLM only
    # needs ray as a runtime dependency when the ray executor backend is actually selected - so
    # install it on the fly if it's missing before relying on either code path above.
    return (
        "bash -lc '\n"
        '    command -v ray >/dev/null 2>&1 || pip install -q "ray[default]"\n'
        "    if ray symmetric-run --help >/dev/null 2>&1; then\n"
        "        ray symmetric-run \\\n"
        '            --address "$RAY_HEAD_NODE_IP" \\\n'
        f"            --min-nodes {total_nodes} \\\n"
        f"            {resource_flags} \\\n"
        f"            -- {inner_cmd}\n"
        '    elif [ "$SLURM_NODEID" = "0" ]; then\n'
        f"        ray start --head --port=6379 {resource_flags}\n"
        f"        {inner_cmd}\n"
        "    else\n"
        f'        ray start --address="$RAY_HEAD_NODE_IP" {resource_flags} --block\n'
        "    fi\n"
        "'"
    )


def _build_ray_command(_service: RayServiceConfig) -> str:
    return "ray start --head"


_BUILDERS = {
    VllmServiceConfig: _build_vllm_command,
    RayServiceConfig: _build_ray_command,
}


def _uses_ray_distributed_backend(service: VllmServiceConfig | RayServiceConfig) -> bool:
    return isinstance(service, VllmServiceConfig) and (
        service.distributed_backend is not None and service.distributed_backend.type == "ray"
    )


def _build_service_command(service: VllmServiceConfig | RayServiceConfig, total_nodes: int) -> str:
    if _uses_ray_distributed_backend(service):
        return _build_vllm_ray_command(service, total_nodes)
    return _BUILDERS[type(service)](service)


def _node_totals(compute: SlurmComputeConfig) -> tuple[int, int]:
    total_nodes = sum(pool.nodes for pool in compute.node_pools.values())
    total_ntasks = sum(pool.nodes * pool.ntasks_per_node for pool in compute.node_pools.values())
    return total_nodes, total_ntasks


def build_sbatch_script(
    config: SubmitConfig,
    benchmark_name: str,
    benchmark: BenchmarkRunConfig,
    compute: SlurmComputeConfig,
    remote_bench_dir: Path,
) -> str:
    directives = _render_directives(compute, remote_bench_dir, benchmark_name)

    total_nodes, total_ntasks = _node_totals(compute)
    is_multi_node = total_nodes > 1

    ray_prelude = _RAY_PRELUDE if any(_uses_ray_distributed_backend(s) for s in config.services.values()) else ""

    service_commands = "\n\n".join(
        _render_service_command(
            name,
            service.container,
            _build_service_command(service, total_nodes),
            service.env or None,
            service.mounts or None,
            nodes=total_nodes if is_multi_node else None,
            ntasks=total_ntasks if is_multi_node else None,
        )
        for name, service in config.services.items()
    )

    health_checks = "\n\n".join(
        render_health_check(
            name, service.health_check.port, service.health_check.path, service.health_check.timeout_seconds
        )
        for name, service in config.services.items()
        if service.health_check
    )

    gi = config.driver.gym_install

    prepare_cmd = None
    if benchmark.prepare:
        prepare_cmd = "gym eval prepare " + " ".join(flatten_run_args(benchmark.prepare))

    output_path = "+output_jsonl_fpath=artifacts/rollouts.jsonl"
    extra_flags = ["--model-type openai_model"] if config.driver.policy_model else []
    gym_cmd = render_gym_cmd("eval run", "GYM_CMD", [output_path] + extra_flags + flatten_run_args(benchmark.run))
    entrypoint = render_driver_entrypoint(
        repo=gi.repo if gi else None,
        ref=gi.ref if gi else None,
        prepare_cmd=prepare_cmd,
    )
    prepare_command = ""
    driver_env_prefix = _resolve_env(config.driver.env) if config.driver.env else ""
    driver_node_flags = " --nodes=1 --ntasks=1" if is_multi_node else ""
    driver_mounts_flag = (
        f" --container-mounts={','.join(shlex.quote(m) for m in config.driver.mounts)}" if config.driver.mounts else ""
    )
    driver_command = (
        f"{gym_cmd}\n"
        f"{driver_env_prefix}srun --overlap --no-container-mount-home{driver_node_flags}{driver_mounts_flag} --container-image={shlex.quote(config.driver.container)} "
        f"--output=logs/driver.log {entrypoint}"
    )

    return _SCRIPT_TEMPLATE.format(
        directives=directives,
        ray_prelude=ray_prelude,
        service_commands=service_commands,
        health_checks=health_checks,
        prepare_command=prepare_command,
        driver_command=driver_command,
    )
