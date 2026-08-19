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
    GymInstallConfig,
    NodePool,
    RayServiceConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmServiceConfig,  # used in _BUILDERS dispatch table
)
from nemo_gym.orchestration.executors.script_templates import (
    bash_var,
    render_driver_entrypoint,
    render_gym_clone_preamble,
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


# Resolves the head node from Slurm's node list, mirroring scripts/sbatch_base.sh.
# Runs once in the main sbatch shell (before any srun step) so $SLURM_JOB_NODELIST is available.
# Used by both multi-node code paths below: HEAD_NODE_IP for vLLM's own --data-parallel-address
# (data-parallel spanning nodes), RAY_HEAD_NODE_IP for `ray start`/`ray symmetric-run` (tensor/
# pipeline-parallel spanning nodes).
_RAY_PRELUDE = """\
# Resolve the head node IP for multi-node vLLM services (distributed_backend: ray).
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node_hostname" | awk '{print $1}')
export HEAD_NODE_IP="$head_node_ip"
export RAY_HEAD_NODE_IP="$head_node_ip:6379"
echo "Head node IP address: $HEAD_NODE_IP\""""


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


def _vllm_base_flags(service: VllmServiceConfig) -> str:
    cmd = (
        f"vllm serve {shlex.quote(service.model)}"
        f" --port {service.port}"
        f" --tensor-parallel-size {service.tensor_parallel_size}"
    )
    if service.pipeline_parallel_size > 1:
        cmd += f" --pipeline-parallel-size {service.pipeline_parallel_size}"
    return cmd


def _build_vllm_command(service: VllmServiceConfig) -> str:
    cmd = _vllm_base_flags(service)
    if service.number_of_instances > 1:
        cmd += f" --data-parallel-size {service.number_of_instances}"
    if service.trust_remote_code:
        cmd += " --trust-remote-code"
    return cmd


def _escape_for_single_quotes(s: str) -> str:
    """Escape s so it can be embedded literally inside a single-quoted bash string ('...').

    Uses the standard close-quote/escaped-quote/reopen-quote trick ('\\'') so callers can safely
    nest an already-quoted sub-command (e.g. `bash -c '...'`) inside another single-quoted string
    without its own quotes prematurely terminating the outer one.
    """
    return s.replace("'", "'\\''")


def _wrap_in_ray_cluster(entrypoint_cmd: str, total_nodes: int, ray_extras: str = "default") -> str:
    # Bootstraps a Ray cluster across every task and runs entrypoint_cmd only on the elected head
    # node, mirroring scripts/sbatch_base.sh.
    resource_flags = (
        "--num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE}"
    )
    # entrypoint_cmd is embedded inside this function's own single-quoted `bash -lc '...'` string
    # below, so any single quotes it contains (e.g. from a nested `bash -c '...'` wrapper) must be
    # escaped here or they'd prematurely close this outer quoting - this is a no-op for entrypoints
    # that don't themselves contain single quotes (e.g. the plain `vllm serve ...` case).
    entrypoint_cmd = _escape_for_single_quotes(entrypoint_cmd)
    # ray symmetric-run starts/joins a Ray cluster across every task and runs the entrypoint
    # only on the elected head node. It requires Ray >= 2.50, so containers with an older pin fall
    # back to manually starting head/worker Ray processes, keyed on Slurm's per-node task rank
    # ($SLURM_NODEID). Callers that block on cluster-wide scheduling (e.g. vLLM's Ray executor
    # waiting on placement groups, or Ray Serve replicas being placed) need no separate
    # cluster-ready wait in the fallback path.
    # Model-serving images (e.g. vllm/vllm-openai) don't necessarily bundle the ray CLI - it's only
    # needed as a runtime dependency when a ray-based distributed backend is actually selected - so
    # install it on the fly if it's missing before relying on either code path above.
    return (
        "bash -lc '\n"
        f'    command -v ray >/dev/null 2>&1 || pip install -q "ray[{ray_extras}]"\n'
        "    if ray symmetric-run --help >/dev/null 2>&1; then\n"
        "        ray symmetric-run \\\n"
        '            --address "$RAY_HEAD_NODE_IP" \\\n'
        f"            --min-nodes {total_nodes} \\\n"
        f"            {resource_flags} \\\n"
        f"            -- {entrypoint_cmd}\n"
        '    elif [ "$SLURM_NODEID" = "0" ]; then\n'
        f"        ray start --head --port=6379 {resource_flags}\n"
        f"        {entrypoint_cmd}\n"
        "    else\n"
        f'        ray start --address="$RAY_HEAD_NODE_IP" {resource_flags} --block\n'
        "    fi\n"
        "'"
    )


def _build_vllm_multi_node_tp_command(service: VllmServiceConfig, total_nodes: int) -> str:
    # A single instance's tensor/pipeline-parallel footprint spans nodes. Uses vLLM's own Ray
    # *core* executor (--distributed-executor-backend ray) - not the ray.serve library, no Serve
    # deployment/ingress/HTTP proxy is involved.
    inner_cmd = _build_vllm_command(service) + " --distributed-executor-backend ray"
    return _wrap_in_ray_cluster(inner_cmd, total_nodes)


def _build_vllm_multi_node_dp_command(service: VllmServiceConfig, total_nodes: int) -> str:
    # Data-parallel replicas span nodes. vLLM's Ray-based DP auto-placement doesn't spread ranks
    # across physical nodes - launching a single `vllm serve --data-parallel-size N` from one node
    # only sees that node's own GPUs when placing DP ranks. Real multi-node DP instead needs one
    # `vllm serve` invocation per node: the head node's serves the OpenAI API and coordinates,
    # worker nodes run `--headless` with a --data-parallel-start-rank offset. This is vLLM's
    # documented multi-node data-parallel deployment pattern and doesn't use Ray at all - each
    # node's tensor-parallel ranks stay local via vLLM's default (mp) executor backend.
    if service.number_of_instances % total_nodes != 0:
        raise ValueError(
            f"number_of_instances ({service.number_of_instances}) must be evenly divisible by the number of "
            f"nodes ({total_nodes}) for multi-node data-parallel deployment."
        )
    dp_size_local = service.number_of_instances // total_nodes
    common = _vllm_base_flags(service)
    dp_flags = (
        f" --data-parallel-size {service.number_of_instances}"
        f" --data-parallel-size-local {dp_size_local}"
        ' --data-parallel-address "$HEAD_NODE_IP"'
        " --data-parallel-rpc-port 13345"
    )
    trust_flag = " --trust-remote-code" if service.trust_remote_code else ""
    head_cmd = common + dp_flags + trust_flag
    worker_cmd = (
        common
        + dp_flags
        + trust_flag
        + " --headless"
        + f" --data-parallel-start-rank $(( SLURM_NODEID * {dp_size_local} ))"
    )
    return (
        "bash -lc '\n"
        '    if [ "$SLURM_NODEID" = "0" ]; then\n'
        f"        {head_cmd}\n"
        "    else\n"
        f"        {worker_cmd}\n"
        "    fi\n"
        "'"
    )


def _build_vllm_ray_command(service: VllmServiceConfig, total_nodes: int) -> str:
    if service.number_of_instances > 1:
        return _build_vllm_multi_node_dp_command(service, total_nodes)
    return _build_vllm_multi_node_tp_command(service, total_nodes)


def _build_vllm_ray_serve_command(
    service: VllmServiceConfig, total_nodes: int, gym_install: GymInstallConfig | None
) -> str:
    # Real ray.serve replicas: Ray Serve schedules/bin-packs number_of_instances replicas across
    # whatever nodes/GPUs are free cluster-wide, instead of the manual head/worker split used by
    # _build_vllm_multi_node_dp_command. Each replica's own TP/PP footprint stays within one node.
    #
    # Uses render_gym_clone_preamble (git-clone only, no `pip install -e .`) rather than
    # render_gym_install_preamble: ray_serve_vllm_app.py only needs stdlib at import time (vllm/ray
    # are imported lazily inside build_app itself), so putting the clone on PYTHONPATH is enough -
    # this avoids nemo_gym's own Python floor (see pyproject.toml), which model-serving images like
    # vllm/vllm-openai (Python 3.12) don't meet.
    gym_clone_preamble = render_gym_clone_preamble(
        gym_install.repo if gym_install else None, gym_install.ref if gym_install else None
    )
    pythonpath_preamble = ['export PYTHONPATH="$PWD:$PYTHONPATH"'] if gym_clone_preamble else []
    # `serve run`'s CLI flags for the HTTP proxy's host/port vary across Ray versions (and some
    # don't expose them at all for import-path apps), so build_app configures serve.start's
    # http_options itself from the `port` builder arg instead of relying on a `serve run` CLI flag.
    serve_run_cmd = (
        "serve run nemo_gym.orchestration.ray_serve_vllm_app:build_app"
        f" model={shlex.quote(service.model)}"
        f" port={service.port}"
        f" tensor_parallel_size={service.tensor_parallel_size}"
        f" pipeline_parallel_size={service.pipeline_parallel_size}"
        f" number_of_instances={service.number_of_instances}"
        f" trust_remote_code={service.trust_remote_code}"
    )
    body = "\n    ".join([*gym_clone_preamble, *pythonpath_preamble, serve_run_cmd])
    inner_cmd = f"bash -c '\n    {_escape_for_single_quotes(body)}\n'"
    return _wrap_in_ray_cluster(inner_cmd, total_nodes, ray_extras="serve,default")


def _build_ray_command(_service: RayServiceConfig) -> str:
    return "ray start --head"


_BUILDERS = {
    VllmServiceConfig: _build_vllm_command,
    RayServiceConfig: _build_ray_command,
}


def _distributed_backend_type(service: VllmServiceConfig | RayServiceConfig) -> str | None:
    if isinstance(service, VllmServiceConfig) and service.distributed_backend is not None:
        return service.distributed_backend.type
    return None


def _uses_ray_cluster_bootstrap(service: VllmServiceConfig | RayServiceConfig) -> bool:
    return _distributed_backend_type(service) in ("ray", "ray_serve")


def _build_service_command(
    service: VllmServiceConfig | RayServiceConfig, total_nodes: int, gym_install: GymInstallConfig | None = None
) -> str:
    backend_type = _distributed_backend_type(service)
    if backend_type == "ray_serve":
        return _build_vllm_ray_serve_command(service, total_nodes, gym_install)
    if backend_type == "ray":
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

    ray_prelude = _RAY_PRELUDE if any(_uses_ray_cluster_bootstrap(s) for s in config.services.values()) else ""

    service_commands = "\n\n".join(
        _render_service_command(
            name,
            service.container,
            _build_service_command(service, total_nodes, config.driver.gym_install),
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
