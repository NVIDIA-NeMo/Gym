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
    effective_ray_serve,
)
from nemo_gym.orchestration.executors.script_templates import (
    ENSURE_RAY_INSTALLED,
    bash_var,
    render_driver_entrypoint,
    render_gym_cmd,
    render_health_check,
    render_ray_prelude,
    render_vllm_ray_symmetric_run,
    render_write_file_from_base64,
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


def _build_vllm_single_instance_multi_node_command(service: VllmServiceConfig, total_nodes: int) -> str:
    # A single instance's tensor/pipeline-parallel footprint spans nodes. Uses vLLM's own Ray
    # *core* executor (--distributed-executor-backend ray) - not the ray.serve library, no Serve
    # deployment/ingress/HTTP proxy is involved.
    inner_cmd = _build_vllm_command(service) + " --distributed-executor-backend ray"
    resource_flags = (
        "--num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE}"
    )
    # Model-serving images (e.g. vllm/vllm-openai) don't necessarily bundle the ray CLI - vLLM only
    # needs ray as a runtime dependency when the ray executor backend is actually selected - so
    # render_vllm_ray_symmetric_run installs it on the fly if it's missing. vLLM's Ray executor
    # blocks on placement-group scheduling until every node's GPUs join, so the fallback path there
    # needs no separate cluster-ready wait.
    return render_vllm_ray_symmetric_run(inner_cmd, total_nodes, resource_flags)


def _build_vllm_multi_instance_multi_node_command(service: VllmServiceConfig, total_nodes: int) -> str:
    # Data-parallel replicas span nodes. vLLM's Ray-based DP auto-placement doesn't spread ranks
    # across physical nodes - launching a single `vllm serve --data-parallel-size N` from one node
    # only sees that node's own GPUs when placing DP ranks. Real multi-node DP instead needs one
    # `vllm serve` invocation per node: the head node's serves the OpenAI API and coordinates,
    # worker nodes run `--headless` with a --data-parallel-start-rank offset. This is vLLM's
    # documented multi-node data-parallel deployment pattern and doesn't use Ray at all - each
    # node's tensor-parallel ranks stay local via vLLM's default (mp) executor backend.
    # number_of_instances is guaranteed evenly divisible by total_nodes here - api.py's
    # SubmitConfig validation enforces this before build_sbatch_script is ever called.
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
        return _build_vllm_multi_instance_multi_node_command(service, total_nodes)
    return _build_vllm_single_instance_multi_node_command(service, total_nodes)


def _escape_for_double_quoted_bash(text: str) -> str:
    """Escape text for safe embedding inside a double-quoted bash string ("..."). Needed instead
    of shlex.quote (which produces single-quote-wrapped output) whenever the text is substituted
    inside an already-open single-quoted bash region - a literal `'` from shlex.quote would
    terminate that outer quoting early and corrupt the command."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


_RAY_SERVE_GATEWAY_SOURCE_PATH = Path(__file__).resolve().parent.parent / "ray_serve_gateway.py"


def _build_vllm_ray_serve_command(
    service: VllmServiceConfig, total_nodes: int, gpus_per_node_values: list[int]
) -> str:
    # Ray Serve (nemo_gym/orchestration/ray_serve_gateway.py) launches all number_of_instances
    # `vllm serve` processes itself and routes requests across them. Ray's own placement-group
    # scheduler only decides where each instance's *worker* ranks land - not where the `vllm serve`
    # driver process itself runs - so the gateway needs gpus_per_node to compute how many nodes
    # each instance's own footprint needs, to explicitly pin different instances' drivers to
    # different nodes. This is the only place NeMo Gym uses the `ray.serve` library, as opposed to
    # vLLM's own Ray core executor (see _build_vllm_single_instance_multi_node_command).
    gateway_args = (
        f"--model {shlex.quote(service.model)}"
        f" --port {service.port}"
        f" --tensor-parallel-size {service.tensor_parallel_size}"
        f" --pipeline-parallel-size {service.pipeline_parallel_size}"
        f" --number-of-instances {service.number_of_instances}"
    )
    if gpus_per_node_values:
        gateway_args += f" --gpus-per-node {max(gpus_per_node_values)}"
    if service.trust_remote_code:
        gateway_args += " --trust-remote-code"

    # ray_serve_gateway.py has no internal nemo_gym imports (stdlib + ray/fastapi/aiohttp only), so
    # its source is embedded directly (base64-encoded, see render_write_file_from_base64) into the
    # vLLM service's own container rather than git-cloning the whole nemo_gym repo or `pip install
    # -e .`-ing the package there - the vLLM container (e.g. vllm/vllm-openai) never has nemo_gym
    # installed, and a full package install there risks nemo_gym's own pinned deps (torch, ray, ...)
    # clobbering vLLM's already-working install. This also means using the Ray Serve gateway never
    # requires driver.gym_install to be configured.
    write_gateway = render_write_file_from_base64(_RAY_SERVE_GATEWAY_SOURCE_PATH.read_text(), "ray_serve_gateway.py")
    fetch_and_run = (
        f"{write_gateway}"
        " && pip install --quiet aiohttp"
        f" && ({ENSURE_RAY_INSTALLED})"
        f" && python3 ray_serve_gateway.py {gateway_args}"
    )
    if total_nodes <= 1:
        # No multi-node Ray cluster to join - the gateway starts its own local Ray instance and
        # launches all instances on this one node. Still needs double-quote escaping: fetch_and_run
        # contains shlex.quote(service.model), which wraps its output in literal single quotes
        # whenever the model name needs escaping (e.g. contains a space) - those would otherwise
        # terminate this bash -lc '...' wrapper early and corrupt the command.
        return f'bash -lc "{_escape_for_double_quoted_bash(fetch_and_run)}"'
    resource_flags = (
        "--num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE}"
    )
    # render_vllm_ray_symmetric_run splices inner_cmd, unquoted, into a `ray symmetric-run ... --
    # {inner_cmd}` statement that itself sits inside that template's own single-quoted `bash -lc
    # '...'` wrapper. Without protection, fetch_and_run's `&&` chain gets live-parsed as bash
    # operators once that outer bash -lc runs its script: `ray symmetric-run`'s entrypoint would
    # become just the first `&&`-segment, which succeeds and exits immediately, tearing down the
    # whole Ray cluster it just stood up before the gateway ever launches. Wrapping in `bash -c
    # "<escaped>"` makes the whole chain one opaque token immune to that live-parsing - double
    # quotes, not shlex.quote's single-quote style, because this text is substituted inside that
    # template's own single-quoted region: a literal `'` (which shlex.quote would introduce) would
    # terminate that outer quoting early.
    return render_vllm_ray_symmetric_run(
        f'bash -c "{_escape_for_double_quoted_bash(fetch_and_run)}"', total_nodes, resource_flags
    )


def _build_ray_command(_service: RayServiceConfig) -> str:
    return "ray start --head"


_BUILDERS = {
    VllmServiceConfig: _build_vllm_command,
    RayServiceConfig: _build_ray_command,
}


def _vllm_spans_multiple_nodes(service: VllmServiceConfig | RayServiceConfig, total_nodes: int) -> bool:
    # Node count alone determines this: multi-node compute always spans a vLLM service across
    # nodes via Ray, regardless of number_of_instances (single instance's TP/PP, or DP replicas).
    # Non-vLLM services (e.g. a plain Ray head) never span nodes this way.
    return isinstance(service, VllmServiceConfig) and total_nodes > 1


def _build_service_command(
    service: VllmServiceConfig | RayServiceConfig,
    total_nodes: int,
    gpus_per_node_values: list[int],
) -> str:
    if isinstance(service, VllmServiceConfig) and effective_ray_serve(service, total_nodes, gpus_per_node_values):
        return _build_vllm_ray_serve_command(service, total_nodes, gpus_per_node_values)
    if _vllm_spans_multiple_nodes(service, total_nodes):
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
    gpus_per_node_values = [
        pool.gpus_per_node for pool in compute.node_pools.values() if pool.gpus_per_node is not None
    ]

    ray_prelude = (
        render_ray_prelude()
        if any(_vllm_spans_multiple_nodes(s, total_nodes) for s in config.services.values())
        else ""
    )

    service_commands = "\n\n".join(
        _render_service_command(
            name,
            service.container,
            _build_service_command(service, total_nodes, gpus_per_node_values),
            service.env or None,
            service.mounts or None,
            # Only services that actually span multiple nodes need the whole allocation's --nodes/
            # --ntasks - not every service in a multi-node job (e.g. a plain Ray head service runs
            # on a single node regardless of how many nodes the overall job spans).
            nodes=total_nodes if _vllm_spans_multiple_nodes(service, total_nodes) else None,
            ntasks=total_ntasks if _vllm_spans_multiple_nodes(service, total_nodes) else None,
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
