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

import json
import re
import shlex
from pathlib import Path
from typing import Any

from nemo_gym.exporters import exporter_extras
from nemo_gym.global_config import MODEL_CALL_CAPTURE_DIR_KEY_NAME, OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.orchestration.api import (
    RUNTIME_ENV_PREFIX,
    BenchmarkRunConfig,
    NodePool,
    RayServiceConfig,
    ServiceConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmPDServiceConfig,
    VllmPDTierConfig,
    VllmServiceConfig,  # used in _BUILDERS dispatch table
    effective_ray_serve,
)
from nemo_gym.orchestration.executors.otel import (
    COLLECTOR_HEALTH_PORT,
    COLLECTOR_SERVICE_NAME,
    FINAL_SCRAPE_GRACE_SECONDS,
    GYM_TELEMETRY_EXTRA,
    SHUTDOWN_WAIT_SECONDS,
    collector_config_path,
    driver_telemetry_env,
    gym_telemetry_active,
    otel_active,
)
from nemo_gym.orchestration.executors.resume_script import render_resume_prologue
from nemo_gym.orchestration.executors.script_templates import (
    ENSURE_RAY_INSTALLED,
    bash_var,
    escape_for_single_quoted_block,
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

{resume_prologue}

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
    # --chdir sets the batch script's cwd on the HOST, so srun --output=logs/... resolves there.
    # Container-side cwd is set separately per step (see driver_workdir_flag).
    lines.append(f"#SBATCH --chdir={remote_bench_dir}")
    for key, val in compute.extra_args.items():
        lines.append(f"#SBATCH --{key}={val}")
    lines.extend(_render_pool_directives(compute.node_pools))
    return "\n".join(lines)


def _pool_directive(pools: dict[str, NodePool], attribute: str) -> Any:
    """The one value every pool agrees on for `attribute`.

    A plain sbatch job takes a single --partition/--ntasks-per-node/--gpus-per-node
    for the whole allocation, so pools that disagree cannot both be honoured. Slurm
    would silently apply whichever directive came last; say so instead.
    """
    values = {getattr(pool, attribute) for pool in pools.values()}
    if len(values) > 1:
        named = ", ".join(f"{name}={getattr(pool, attribute)!r}" for name, pool in pools.items())
        raise ValueError(
            f"Node pools disagree on {attribute} ({named}). One Slurm job takes a single value for the whole "
            "allocation; split the run or make the pools agree."
        )
    return next(iter(values))


def _render_pool_directives(pools: dict[str, NodePool]) -> list[str]:
    """One set of directives for the whole allocation, not one per pool.

    Pools divide an allocation between services (see _pool_offsets); they are not
    separate Slurm requests. Emitting --nodes per pool made every pool but the last
    a no-op, so a two-pool job asked for one pool's nodes while the rest of the
    executor sized itself on the sum.
    """
    if not pools:
        return []
    lines = [
        f"#SBATCH --partition={_pool_directive(pools, 'partition')}",
        f"#SBATCH --nodes={sum(pool.nodes for pool in pools.values())}",
        f"#SBATCH --ntasks-per-node={_pool_directive(pools, 'ntasks_per_node')}",
    ]
    gpus_per_node = _pool_directive(pools, "gpus_per_node")
    if gpus_per_node is not None:
        lines.append(f"#SBATCH --gpus-per-node={gpus_per_node}")
    extra_args: dict[str, str] = {}
    for name, pool in pools.items():
        for key, val in pool.extra_args.items():
            if extra_args.setdefault(key, val) != val:
                raise ValueError(
                    f"Node pool {name!r} sets extra_args[{key!r}]={val!r}, which conflicts with another pool's "
                    f"{extra_args[key]!r}. #SBATCH directives apply to the whole allocation."
                )
    lines.extend(f"#SBATCH --{key}={val}" for key, val in extra_args.items())
    return lines


_VALID_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_key(key: str) -> None:
    if not _VALID_ENV_KEY.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")


def _resolve_env(env: dict[str, str]) -> str:
    """Return an 'env K=V ...' prefix string (trailing space) scoped to a single command, or '' if empty.

    A `runtime:VAR` value (see resolve_env_dict in api.py) is emitted as an unquoted `K=$VAR`
    shell reference instead of a literal, so it's resolved from the job's own environment when
    the command actually runs on the compute node, rather than baked in at script-generation time.
    """
    if not env:
        return ""
    for k in env:
        _validate_env_key(k)
    pairs = " ".join(
        f"{k}=${{{v[len(RUNTIME_ENV_PREFIX) :]}}}" if v.startswith(RUNTIME_ENV_PREFIX) else f"{k}={shlex.quote(v)}"
        for k, v in env.items()
    )
    return f"env {pairs} "


def _render_service_command(
    name: str,
    container: str | None,
    command: str,
    env: dict[str, str] | None = None,
    mounts: list[str] | None = None,
    nodes: int | None = None,
    ntasks: int | None = None,
    pre_command: str = "",
    workdir: str | None = None,
    single_node: bool = False,
    nodelist: str | None = None,
) -> str:
    var = bash_var(name)
    env_prefix = _resolve_env(env) if env else ""
    if single_node:
        # Pin to one node of a multi-node allocation; without it srun fans the step out to every node.
        node_flags = " --nodes=1 --ntasks=1"
    else:
        node_flags = f" --nodes={nodes} --ntasks={ntasks}" if (nodes is not None and nodes > 1) else ""
    # --nodelist names the exact hosts. --relative is only a starting point Slurm may
    # move off when that node's resources are taken, landing a service on the wrong node.
    if nodelist is not None:
        node_flags = f' --nodelist="${{{nodelist}}}" --nodes={nodes} --ntasks={ntasks}'
    mounts_flag = f" --container-mounts={','.join(shlex.quote(m) for m in mounts)}" if mounts else ""
    workdir_flag = f" --container-workdir={shlex.quote(workdir)}" if workdir else ""
    if pre_command:
        # Wrapped in one shell so export/unset statements in pre_command are
        # visible to the exec'd command; shlex.quote keeps the whole thing one
        # word, so it can't interfere with --container-mounts/-image parsing
        # regardless of what pre_command contains.
        command = f"bash -c {shlex.quote(pre_command + chr(10) + 'exec ' + command)}"
    # --overlap lets this step share the allocation with other concurrent steps (driver + services).
    # --no-container-mount-home avoids polluting the container with host home directory contents.
    # PID is captured so the health check can detect early service death.
    # Without a container the command runs directly on the node: no image, mounts or workdir flags.
    # $SLURM_JOB_ID suffixes the log so a resumed job (same job dir, new job id) keeps the previous log.
    container_flags = (
        f" --no-container-mount-home{mounts_flag}{workdir_flag} --container-image={shlex.quote(container)}"
        if container is not None
        else ""
    )
    return (
        f"# service: {name}\n"
        f"{env_prefix}srun --overlap{node_flags}{container_flags} --output=logs/{name}-$SLURM_JOB_ID.log {command} &\n"
        f"{var}_PID=$!"
    )


def _vllm_base_flags(service: VllmServiceConfig) -> str:
    cmd = (
        f"vllm serve {shlex.quote(service.model)}"
        f" --port {service.port}"
        f" --tensor-parallel-size {service.tensor_parallel_size}"
    )
    if service.served_model_name:
        cmd += f" --served-model-name {shlex.quote(service.served_model_name)}"
    if service.pipeline_parallel_size > 1:
        cmd += f" --pipeline-parallel-size {service.pipeline_parallel_size}"
    if service.extra_args:
        cmd += " " + service.extra_args
    return cmd


def _kv_transfer_flag(service: VllmServiceConfig) -> str:
    """The --kv-transfer-config for one tier of a prefill/decode pair.

    Single-quoted because it is JSON: the value travels through the same
    single-quoted `bash -c` block as everything else, and the escaping helper
    handles it there.
    """
    if not isinstance(service, VllmPDTierConfig) or not service.kv_transfer_config:
        return ""
    config = json.dumps(service.kv_transfer_config, separators=(",", ":"))
    return f" --kv-transfer-config {shlex.quote(config)}"


def _build_vllm_command(service: VllmServiceConfig) -> str:
    cmd = _vllm_base_flags(service)
    if service.number_of_instances > 1:
        cmd += f" --data-parallel-size {service.number_of_instances}"
    if service.trust_remote_code:
        cmd += " --trust-remote-code"
    return cmd + _kv_transfer_flag(service)


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


# vLLM refuses `--api-server-count` in headless mode ("no API servers are started in headless
# mode") and exits before loading anything. The flag is legitimate on the head node and reaches us
# through a service's own extra_args, so it is stripped from the worker command rather than
# rejected: Gym decides which nodes run headless, so Gym keeps their command valid.
_HEADLESS_INCOMPATIBLE_FLAG = re.compile(r"\s--api-server-count(?:[= ]\S+)?")


def _strip_headless_incompatible_flags(cmd: str) -> str:
    return _HEADLESS_INCOMPATIBLE_FLAG.sub("", cmd)


def _build_vllm_multi_instance_multi_node_command(
    service: VllmServiceConfig, total_nodes: int, head: str = '"$HEAD_NODE_IP"'
) -> str:
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
    common = _vllm_base_flags(service) + _kv_transfer_flag(service)
    dp_flags = (
        f" --data-parallel-size {service.number_of_instances}"
        f" --data-parallel-size-local {dp_size_local}"
        f" --data-parallel-address {head}"
        f" --data-parallel-rpc-port {service.data_parallel_rpc_port}"
    )
    trust_flag = " --trust-remote-code" if service.trust_remote_code else ""
    head_cmd = common + dp_flags + trust_flag
    worker_cmd = (
        _strip_headless_incompatible_flags(common)
        + dp_flags
        + trust_flag
        + " --headless"
        + f" --data-parallel-start-rank $(( SLURM_NODEID * {dp_size_local} ))"
    )
    # Both branches go inside a single-quoted `bash -lc '...'`, and this service's
    # command carries JSON flags that are themselves single-quoted
    # (--hf-overrides, --limit-mm-per-prompt, --media-io-kwargs). Unescaped they
    # end the block early and the whole invocation word-splits; mmlu-prox died
    # that way with "/usr/bin/env: Argument list too long".
    return (
        "bash -lc '\n"
        '    if [ "$SLURM_NODEID" = "0" ]; then\n'
        f"        {escape_for_single_quoted_block(head_cmd)}\n"
        "    else\n"
        f"        {escape_for_single_quoted_block(worker_cmd)}\n"
        "    fi\n"
        "'"
    )


def _build_vllm_ray_command(service: VllmServiceConfig, total_nodes: int, head: str | None = None) -> str:
    if service.number_of_instances > 1:
        if head is None:
            return _build_vllm_multi_instance_multi_node_command(service, total_nodes)
        return _build_vllm_multi_instance_multi_node_command(service, total_nodes, head)
    return _build_vllm_single_instance_multi_node_command(service, total_nodes)


def _escape_for_double_quoted_bash(text: str) -> str:
    """Escape text for safe embedding inside a double-quoted bash string ("...")."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


_RAY_SERVE_GATEWAY_SOURCE_PATH = Path(__file__).resolve().parent.parent / "ray_serve_gateway.py"


def _build_vllm_ray_serve_command(
    service: VllmServiceConfig, total_nodes: int, gpus_per_node_values: list[int]
) -> str:
    # Launches ray_serve_gateway.py, which creates the instances and routes requests via ray.serve.
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
    if service.served_model_name:
        gateway_args += f" --served-model-name {shlex.quote(service.served_model_name)}"
    if service.extra_args:
        gateway_args += f" --extra-args {shlex.quote(service.extra_args)}"

    # Embeds the gateway's source directly rather than git-cloning/installing nemo_gym into the
    # vLLM container - no driver.gym_install needed for this path.
    write_gateway = render_write_file_from_base64(_RAY_SERVE_GATEWAY_SOURCE_PATH.read_text(), "ray_serve_gateway.py")
    fetch_and_run = (
        f"{write_gateway}"
        " && pip install --quiet aiohttp"
        f" && ({ENSURE_RAY_INSTALLED})"
        f" && python3 ray_serve_gateway.py {gateway_args}"
    )
    if total_nodes <= 1:
        # No multi-node Ray cluster to join - the gateway starts its own local Ray instance.
        return f'bash -lc "{_escape_for_double_quoted_bash(fetch_and_run)}"'
    resource_flags = (
        "--num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE}"
    )
    # Double-quote escaping keeps the whole &&-chain as one opaque token for ray symmetric-run's
    # entrypoint, immune to the outer bash -lc live-parsing its own && operators.
    return render_vllm_ray_symmetric_run(
        f'bash -c "{_escape_for_double_quoted_bash(fetch_and_run)}"', total_nodes, resource_flags
    )


# Resolves the allocation's node list so a router can address each tier's head by
# the pool offset the tier is pinned to. `scontrol show hostnames` expands the
# compact nodelist into one hostname per line, in allocation order -- the same
# order --relative counts in.
NODE_LIST_PRELUDE = 'gym_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))'


def _pool_head(offset: int) -> str:
    """Shell expansion for the first hostname of the pool starting at `offset`."""
    return f"${{gym_nodes[{offset}]}}"


def _nixl_pre_command(service: VllmServiceConfig) -> str:
    """Exports NIXL needs before a tier's `vllm serve` starts.

    The side-channel host is the node's own hostname and can only be known on the
    node, so it is a shell statement rather than an `env` entry.
    """
    if not isinstance(service, VllmPDTierConfig):
        return ""
    return (
        "export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)\n"
        f"export VLLM_NIXL_SIDE_CHANNEL_PORT={service.nixl_side_channel_port}"
    )


def _build_router_command(router: VllmPDServiceConfig, offsets: dict[str, tuple[int, int]]) -> str:
    """The vllm-router invocation fronting a prefill/decode pair.

    Both tiers are addressed at their pool's head node, which is where each tier's
    API rank runs (the remaining ranks in a tier are headless).
    """
    endpoints = ""
    for flag, tier in (("--prefill", router.prefill), ("--decode", router.decode)):
        start, count = offsets[tier.node_pool]
        # A per-node tier is one server per node; otherwise the tier's API rank is its head.
        for i in range(count if router.server_per_node else 1):
            endpoints += f' {flag} "http://{_pool_head(start + i)}:{tier.port}"'
    return (
        "vllm-router"
        f" --prefill-policy {shlex.quote(router.prefill_policy)}"
        f" --decode-policy {shlex.quote(router.decode_policy)}"
        " --vllm-pd-disaggregation"
        f"{endpoints}"
        # All interfaces: the driver and the health probe reach it over localhost.
        " --host 0.0.0.0"
        f" --port {router.port}"
        f" --intra-node-data-parallel-size {router.intra_node_data_parallel_size}"
        f" --request-timeout-secs {router.request_timeout_secs}"
        f" --log-level {shlex.quote(router.log_level)}"
    )


# Flags `ray start` refuses on a worker. They reach a worker through the service's
# shared extra_args; Gym decides which nodes are workers, so Gym drops them there.
_RAY_HEAD_ONLY_FLAG = re.compile(r"\s--(?:port|redis-shard-ports|include-dashboard)(?:[= ]\S+)?")


def _build_ray_command(
    service: RayServiceConfig, *, pool: str | None = None, address_var: str | None = None, worker: bool = False
) -> str:
    """`ray start` for one node of a ray service: the head, or with `worker` a node joining it.

    `address_var` names the env var holding the head's host:port (see ray_head_address_var).
    """
    # --block keeps the srun step alive. `ray start` daemonises and returns, so
    # without it the step exits the moment the node is up and Slurm tears the
    # service down again.
    cmd = "ray start --block"
    if worker:
        cmd += f' --address "${address_var}"'
    else:
        cmd += f" --head --port {service.port}"
        if address_var is not None:
            # Bind the head to the address the workers and the driver dial.
            cmd += f' --node-ip-address "${{{address_var}%:*}}"'
    if service.num_cpus is not None:
        cmd += f" --num-cpus {service.num_cpus}"
    if service.num_gpus is not None:
        cmd += f" --num-gpus {service.num_gpus}"
    resources = service.resources.get(pool, {}) if pool is not None else {}
    if resources:
        # Ray takes fractional custom resources, so the field is float-typed, but a
        # whole number is written as one: {"extra_gpu": 4}, not 4.0.
        resources = {k: int(v) if v.is_integer() else v for k, v in resources.items()}
        cmd += " --resources=" + shlex.quote(json.dumps(resources, sort_keys=True))
    extra_args = _RAY_HEAD_ONLY_FLAG.sub("", " " + service.extra_args).strip() if worker else service.extra_args
    if extra_args:
        cmd += " " + extra_args
    return cmd


_NODE_ARRAY = NODE_LIST_PRELUDE


def pool_nodes_var(pool: str) -> str:
    """The env var build_sbatch_script exports with a node pool's comma-separated hosts."""
    return f"GYM_POOL_{bash_var(pool)}_NODES"


def _places_services(config: SubmitConfig, is_multi_node: bool) -> bool:
    """Whether the script declares the allocation's host array to place services with."""
    return is_multi_node or any(s.node_pool for s in config.deployed_services.values())


def _render_pool_nodes(config: SubmitConfig, compute: SlurmComputeConfig, is_multi_node: bool) -> str:
    """Declare the allocation's hosts, and export each node pool's, for --nodelist."""
    if not _places_services(config, is_multi_node):
        return ""
    lines = [_NODE_ARRAY]
    for name, (start, count) in _pool_offsets(compute).items():
        if count:
            lines.append(f'export {pool_nodes_var(name)}="$(IFS=,; echo "${{gym_nodes[*]:{start}:{count}}}")"')
    return "\n".join(lines)


def ray_head_address_var(ray_service: str) -> str:
    """The env var build_sbatch_script exports with a ray service's head host:port."""
    return f"GYM_RAY_ADDRESS_{bash_var(ray_service)}"


def ray_workers_var(ray_service: str, pool: str) -> str:
    """The env var build_sbatch_script exports with a ray service's worker hosts in one pool."""
    return f"GYM_RAY_{bash_var(ray_service)}_{bash_var(pool)}_WORKERS"


def _ray_worker_ranges(
    service: RayServiceConfig, compute: SlurmComputeConfig, head_node: int
) -> dict[str, tuple[int, int]]:
    """Each spanned pool's worker nodes as (first index, count): every node but the head.

    The head is the driver's node, which is always the first node of some pool.
    """
    offsets = _pool_offsets(compute)
    ranges: dict[str, tuple[int, int]] = {}
    for pool in service.node_pools:
        start, count = offsets[pool]
        if start == head_node:
            start, count = start + 1, count - 1
        if count > 0:
            ranges[pool] = (start, count)
    return ranges


def _render_ray_head_addresses(config: SubmitConfig, compute: SlurmComputeConfig) -> str:
    """Export every ray service's head address, and its worker hosts, from the nodes Slurm gave it.

    The head runs beside the driver. Workers and the driver join it without knowing,
    when the config is written, which host the job will land on.
    """
    services = {n: s for n, s in config.deployed_services.items() if isinstance(s, RayServiceConfig)}
    if not services:
        return ""
    head_node = _driver_node(config, compute)
    lines = [] if _places_services(config, _node_totals(compute)[0] > 1) else [_NODE_ARRAY]
    for name, service in services.items():
        ip = f"$(getent hosts ${{gym_nodes[{head_node}]}} | awk '{{print $1}}')"
        lines.append(f'export {ray_head_address_var(name)}="{ip}:{service.port}"')
        for pool, (start, count) in _ray_worker_ranges(service, compute, head_node).items():
            lines.append(f'export {ray_workers_var(name, pool)}="$(IFS=,; echo "${{gym_nodes[*]:{start}:{count}}}")"')
    return "\n".join(lines)


def _pool_of(compute: SlurmComputeConfig, node: int) -> str | None:
    for name, (start, count) in _pool_offsets(compute).items():
        if start <= node < start + count:
            return name
    return None


def _render_ray_service(
    name: str,
    service: RayServiceConfig,
    config: SubmitConfig,
    compute: SlurmComputeConfig,
    driver_node: int | None,
) -> str:
    """A ray service's srun steps: the head beside the driver, then one step per spanned pool's workers.

    Every step uses the service's one container, env and mounts.
    """
    total_nodes, total_ntasks = _node_totals(compute)
    head_node = _driver_node(config, compute)
    head_pool = _pool_of(compute, head_node)
    address_var = ray_head_address_var(name)
    steps = [
        _render_service_command(
            name,
            service.container,
            _build_ray_command(
                service, pool=head_pool if head_pool in service.node_pools else None, address_var=address_var
            ),
            service.env or None,
            service.mounts or None,
            nodes=_srun_nodes(service, compute, total_nodes),
            ntasks=_srun_ntasks(service, compute, total_nodes, total_ntasks),
            pre_command=service.pre_command,
            nodelist=_service_nodelist(service, driver_node, total_nodes),
        )
    ]
    # A worker only joins a running head, and the head may still be installing.
    wait_for_head = f'until ray status --address "${address_var}" >/dev/null 2>&1; do sleep 5; done'
    for pool, (_, count) in _ray_worker_ranges(service, compute, head_node).items():
        steps.append(
            _render_service_command(
                f"{name}_{pool}_workers",
                service.container,
                _build_ray_command(service, pool=pool, address_var=address_var, worker=True),
                service.env or None,
                service.mounts or None,
                nodes=count,
                ntasks=count,
                pre_command=f"{service.pre_command.rstrip()}\n{wait_for_head}".lstrip(),
                nodelist=ray_workers_var(name, pool),
            )
        )
    return "\n\n".join(steps)


_BUILDERS = {
    VllmServiceConfig: _build_vllm_command,
    VllmPDTierConfig: _build_vllm_command,
    RayServiceConfig: _build_ray_command,
}


def _vllm_spans_multiple_nodes(service: ServiceConfig, total_nodes: int) -> bool:
    # Node count alone determines this: multi-node compute always spans a vLLM service across
    # nodes via Ray, regardless of number_of_instances (single instance's TP/PP, or DP replicas).
    # Non-vLLM services (e.g. a plain Ray head) never span nodes this way.
    return isinstance(service, VllmServiceConfig) and total_nodes > 1


def _build_service_command(
    service: ServiceConfig,
    total_nodes: int,
    gpus_per_node_values: list[int],
    offsets: dict[str, tuple[int, int]] | None = None,
) -> str:
    if isinstance(service, VllmPDServiceConfig):
        assert offsets is not None
        return _build_router_command(service, offsets)
    if isinstance(service, VllmServiceConfig) and effective_ray_serve(service, total_nodes, gpus_per_node_values):
        return _build_vllm_ray_serve_command(service, total_nodes, gpus_per_node_values)
    if _vllm_spans_multiple_nodes(service, total_nodes):
        # A pinned service's data-parallel ranks coordinate on its own pool's head.
        # Pointing them at the allocation's head node instead would make two tiers
        # of a prefill/decode pair try to coordinate through the same rank 0.
        head = None
        if service.node_pool is not None and offsets is not None:
            head = _pool_head(offsets[service.node_pool][0])
        return _build_vllm_ray_command(service, total_nodes, head)
    return _BUILDERS[type(service)](service)


def _render_collector_service(
    config: SubmitConfig, remote_bench_dir: Path, *, is_multi_node: bool, driver_node: int | None = None
) -> str:
    """The collector's srun step. Started before the model services so the scrape covers their
    startup.

    Runs on exactly one node, the batch host. That is the first node of the allocation, which is
    also where the Ray prelude puts the head of a multi-node vLLM service and where the driver runs,
    so `localhost:<port>` reaches the API server and the OTLP endpoints from the same node. In a
    container the job directory is mounted for the config and the local `otel/*.jsonl` output; on
    the node it is simply there, so no mounts or workdir are passed.
    """
    obs = config.otel
    command = f"{shlex.quote(obs.binary)} --config {shlex.quote(str(collector_config_path(remote_bench_dir)))}"
    container_kwargs = (
        {"mounts": [f"{remote_bench_dir}:{remote_bench_dir}"], "workdir": str(remote_bench_dir)}
        if obs.container is not None
        else {}
    )
    return _render_service_command(
        COLLECTOR_SERVICE_NAME,
        obs.container,
        command,
        env={
            obs.token_env: f"{RUNTIME_ENV_PREFIX}{obs.token_env}",
            "SLURM_JOB_ID": f"{RUNTIME_ENV_PREFIX}SLURM_JOB_ID",
        },
        single_node=is_multi_node,
        # Beside the driver: it scrapes the policy API on localhost, as the driver calls it.
        nodelist=f"gym_nodes[{driver_node}]" if driver_node is not None else None,
        nodes=1 if driver_node is not None else None,
        ntasks=1 if driver_node is not None else None,
        **container_kwargs,
    )


def _render_collector_health_check(config: SubmitConfig, driver_node: int | None = None) -> str:
    # The collector runs beside the driver (see _render_collector_service).
    return render_health_check(
        COLLECTOR_SERVICE_NAME,
        COLLECTOR_HEALTH_PORT,
        "/",
        config.otel.health_check_timeout_seconds,
        _probe_host(driver_node or 0),
    )


def _render_collector_shutdown(config: SubmitConfig, remote_bench_dir: Path) -> str:
    """Run after the driver: one more scrape interval so the final counters are seen, then a
    graceful stop, keeping the driver's exit code.

    The TERM goes to the collector process itself, matched by its unique `--config` path. Sent to
    `srun` instead, TERM makes Slurm kill the step outright and INT is treated as a console
    interrupt; neither reaches the collector, so its final batch would be lost.
    """
    pid = f"${bash_var(COLLECTOR_SERVICE_NAME)}_PID"
    # Anchored to the binary: the srun that launched it carries the same `--config <path>` on its
    # own command line, and a TERM to srun makes Slurm kill the step before the flush completes.
    binary = re.escape(config.otel.binary)
    pattern = shlex.quote(f"^{binary} --config {re.escape(str(collector_config_path(remote_bench_dir)))}")
    return (
        "DRIVER_RC=$?\n"
        f"sleep {FINAL_SCRAPE_GRACE_SECONDS}\n"
        f'pkill -TERM -u "$USER" -f -- {pattern} || true\n'
        f"for _i in $(seq 1 {SHUTDOWN_WAIT_SECONDS}); do kill -0 {pid} 2>/dev/null || break; sleep 1; done\n"
        f"kill -TERM {pid} 2>/dev/null || true\n"
        "exit $DRIVER_RC"
    )


def _pool_offsets(compute: SlurmComputeConfig) -> dict[str, tuple[int, int]]:
    """Each pool's (first node index, node count) within the allocation.

    Pools are laid out contiguously in declaration order, which is the order the
    single #SBATCH --nodes total is built from, so pool i owns the nodes after
    every pool before it.
    """
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for name, pool in compute.node_pools.items():
        offsets[name] = (start, pool.nodes)
        start += pool.nodes
    return offsets


def _service_nodes(service: ServiceConfig, compute: SlurmComputeConfig, total_nodes: int) -> int:
    """How many nodes this service actually runs on.

    A service pinned to a pool sees only that pool, so a single-node pool inside a
    ten-node job is a single-node deployment and must not be built as a multi-node
    Ray one.
    """
    if service.node_pool is None:
        return total_nodes
    return compute.node_pools[service.node_pool].nodes


def _driver_node(config: SubmitConfig, compute: SlurmComputeConfig) -> int:
    """The node the driver runs on in a multi-node job: the policy's first node.

    The driver reaches the policy on localhost. A multi-node policy serves its API
    from node 0, and a pinned one from its pool's first node.
    """
    policy = config.deployed_services.get(config.driver.policy_model or "")
    if policy is not None and policy.node_pool is not None:
        return _pool_offsets(compute)[policy.node_pool][0]
    return 0


def _service_nodelist(service: ServiceConfig, driver_node: int | None, total_nodes: int) -> str | None:
    """What a service's --nodelist names, as the shell variable it expands.

    A pinned service names its pool. In a multi-node job an unpinned service that
    fits on one node joins the driver, which reaches it on localhost; otherwise
    Slurm could start it on any node.
    """
    if service.node_pool is not None:
        return pool_nodes_var(service.node_pool)
    if driver_node is not None and not _vllm_spans_multiple_nodes(service, total_nodes):
        return f"gym_nodes[{driver_node}]"
    return None


def _build_nodes(service: ServiceConfig, compute: SlurmComputeConfig, total_nodes: int) -> int:
    """Nodes the service command is written for; a per-node server is a single-node command."""
    if isinstance(service, VllmPDTierConfig) and service.server_per_node:
        return 1
    return _service_nodes(service, compute, total_nodes)


def _srun_nodes(service: ServiceConfig, compute: SlurmComputeConfig, total_nodes: int) -> int | None:
    nodes = _service_nodes(service, compute, total_nodes)
    if service.node_pool is not None:
        return nodes
    if _vllm_spans_multiple_nodes(service, total_nodes):
        return total_nodes
    return 1 if total_nodes > 1 else None


def _srun_ntasks(
    service: ServiceConfig,
    compute: SlurmComputeConfig,
    total_nodes: int,
    total_ntasks: int,
) -> int | None:
    if service.node_pool is not None:
        pool = compute.node_pools[service.node_pool]
        return pool.nodes * pool.ntasks_per_node
    if _vllm_spans_multiple_nodes(service, total_nodes):
        return total_ntasks
    return 1 if total_nodes > 1 else None


def _node_totals(compute: SlurmComputeConfig) -> tuple[int, int]:
    total_nodes = sum(pool.nodes for pool in compute.node_pools.values())
    total_ntasks = sum(pool.nodes * pool.ntasks_per_node for pool in compute.node_pools.values())
    return total_nodes, total_ntasks


def _with_default_capture_dir(run: dict[str, Any], remote_bench_dir: Path) -> dict[str, Any]:
    """Enable capture for submitted evaluations unless explicitly disabled.

    Auto-derive model_call_capture_dir from this benchmark's own real output
    directory when observability is on and the caller didn't set one.

    Hydra interpolation resolves before remote_bench_dir exists (it's computed
    here, in build_sbatch_script, well after SubmitConfig validation), so
    there's no way for a YAML value to reference it -- this has to happen in
    Python, once the real path is known. An explicit model_call_capture_dir in
    run always wins over this default.
    """
    run = {OBSERVABILITY_ENABLED_KEY_NAME: True, **run}
    if run.get(OBSERVABILITY_ENABLED_KEY_NAME) and MODEL_CALL_CAPTURE_DIR_KEY_NAME not in run:
        return {**run, MODEL_CALL_CAPTURE_DIR_KEY_NAME: str(remote_bench_dir / "model-calls")}
    return run


def _command_env(benchmark: BenchmarkRunConfig, remote_bench_dir: Path) -> dict[str, str]:
    """Environment a `command` benchmark gets in place of `gym eval run` arguments.

    Values are already literal here (driver.env prefixes were resolved at
    validation time), so they are passed through as-is.
    """
    env = {"NEMO_GYM_BENCH_DIR": str(remote_bench_dir)}
    for key, name in (
        ("policy_base_url", "NEMO_GYM_POLICY_BASE_URL"),
        ("policy_model_name", "NEMO_GYM_POLICY_MODEL_NAME"),
        ("policy_api_key", "NEMO_GYM_POLICY_API_KEY"),
    ):
        value = benchmark.run.get(key)
        if value is not None:
            env[name] = str(value)
    return env


def _probe_host(node: int) -> str:
    """How the batch script, which runs on the allocation's first node, reaches `node`."""
    return "localhost" if node == 0 else _pool_head(node)


def _health_check_host(
    service: ServiceConfig,
    config: SubmitConfig,
    compute: SlurmComputeConfig,
    driver_node: int | None,
    total_nodes: int,
) -> str:
    """Where this service answers its health probe: the node it runs on, as _service_nodelist places it."""
    if isinstance(service, RayServiceConfig):
        return _probe_host(_driver_node(config, compute))
    if service.node_pool is not None:
        return _probe_host(_pool_offsets(compute)[service.node_pool][0])
    if driver_node is not None and not _vllm_spans_multiple_nodes(service, total_nodes):
        return _probe_host(driver_node)
    return "localhost"


def _service_pre_command(service: ServiceConfig) -> str:
    """The service's own pre_command, with any exports its role requires in front."""
    if not isinstance(service, VllmServiceConfig):
        return service.pre_command
    nixl = _nixl_pre_command(service)
    if not nixl:
        return service.pre_command
    return f"{nixl}\n{service.pre_command}" if service.pre_command else nixl


def _with_resume_flag(run: dict[str, Any], resumable: bool) -> dict[str, Any]:
    """Auto-enable `gym eval run`'s own cache-based resume when the auto-resume chain is on.

    A no-op on the chain's cold (first) run -- rollout_collection.py falls back to a fresh run
    when the cache files don't exist yet -- and a no-op if the caller already set the key.
    """
    if resumable and "resume_from_cache" not in run:
        return {**run, "resume_from_cache": True}
    return run


def build_sbatch_script(
    config: SubmitConfig,
    benchmark_name: str,
    benchmark: BenchmarkRunConfig,
    compute: SlurmComputeConfig,
    remote_bench_dir: Path,
) -> str:
    directives = _render_directives(compute, remote_bench_dir, benchmark_name)

    resume = benchmark.resume_config
    resume_prologue = render_resume_prologue(resume) if resume else ""

    total_nodes, total_ntasks = _node_totals(compute)
    is_multi_node = total_nodes > 1
    gpus_per_node_values = [
        pool.gpus_per_node for pool in compute.node_pools.values() if pool.gpus_per_node is not None
    ]

    ray_prelude = "\n".join(
        block
        for block in (
            render_ray_prelude()
            if any(_vllm_spans_multiple_nodes(s, total_nodes) for s in config.deployed_services.values())
            else "",
            _render_pool_nodes(config, compute, is_multi_node),
            _render_ray_head_addresses(config, compute),
        )
        if block
    )

    offsets = _pool_offsets(compute)
    observed = otel_active(config)
    instrumented = gym_telemetry_active(config)

    driver_node = _driver_node(config, compute) if is_multi_node else None
    service_commands = "\n\n".join(
        (
            [_render_collector_service(config, remote_bench_dir, is_multi_node=is_multi_node, driver_node=driver_node)]
            if observed
            else []
        )
        + [
            _render_service_command(
                name,
                service.container,
                _build_service_command(
                    service,
                    _build_nodes(service, compute, total_nodes),
                    gpus_per_node_values,
                    offsets,
                ),
                service.env or None,
                service.mounts or None,
                # Only services that actually span multiple nodes need --nodes/--ntasks - not every
                # service in a multi-node job (e.g. a plain Ray head service runs on a single node
                # regardless of how many nodes the overall job spans). A pinned service always gets
                # them, so the node list and the step size agree.
                nodes=_srun_nodes(service, compute, total_nodes),
                ntasks=_srun_ntasks(service, compute, total_nodes, total_ntasks),
                pre_command=_service_pre_command(service),
                nodelist=_service_nodelist(service, driver_node, total_nodes),
            )
            for name, service in config.deployed_services.items()
            if not isinstance(service, RayServiceConfig)
        ]
        + [
            _render_ray_service(name, service, config, compute, driver_node)
            for name, service in config.deployed_services.items()
            if isinstance(service, RayServiceConfig)
        ]
    )

    health_checks = "\n\n".join(
        ([_render_collector_health_check(config, driver_node)] if observed else [])
        + [
            render_health_check(
                name,
                service.health_check.port,
                service.health_check.path,
                service.health_check.timeout_seconds,
                _health_check_host(service, config, compute, driver_node, total_nodes),
            )
            for name, service in config.deployed_services.items()
            if service.health_check
        ]
    )

    gi = config.driver.gym_install

    prepare_cmd = None
    if benchmark.prepare:
        prepare_cmd = "gym eval prepare " + " ".join(flatten_run_args(benchmark.prepare))

    # ABSOLUTE, not relative. The driver `cd`s into the Gym checkout so that a
    # benchmark's own relative `prepare_script` / `jsonl_fpath` resolve, which
    # means a relative output path would write every artifact inside that
    # checkout instead of the job directory -- the run completes, exits 0, and
    # leaves nothing behind. Making the OUTPUT absolute is what keeps artifacts
    # in the job directory without constraining cwd.
    output_path = f"+output_jsonl_fpath={remote_bench_dir}/artifacts/rollouts.jsonl"
    policy_type = config.driver.policy_model_type
    extra_flags = [f"--model-type {shlex.quote(policy_type)}"] if config.driver.policy_model and policy_type else []
    driver_env = dict(config.driver.env)
    if instrumented:
        driver_env = {
            **driver_telemetry_env(
                remote_bench_dir.parent.name, config.otel.gym_span_groups, logs=config.otel.gym_logs
            ),
            **driver_env,
        }
    if benchmark.command is None:
        run_args = _with_default_capture_dir(benchmark.run, remote_bench_dir)
        run_args.setdefault("require_complete", True)
        run_args = _with_resume_flag(run_args, resume is not None)
        gym_cmd = render_gym_cmd("eval run", "GYM_CMD", [output_path] + extra_flags + flatten_run_args(run_args))
    else:
        # A command replaces `gym eval run`, so the run args it would have carried
        # have nowhere to go. What the script still needs is where to write and how
        # to reach the policy, which it gets as environment variables rather than
        # as a calling convention it would have to parse.
        gym_cmd = ""
        driver_env |= _command_env(benchmark, remote_bench_dir)
    entrypoint = render_driver_entrypoint(
        repo=gi.repo if gi else None,
        ref=gi.ref if gi else None,
        prepare_cmd=prepare_cmd,
        command=benchmark.command,
        extras=((GYM_TELEMETRY_EXTRA,) if instrumented else ()) + exporter_extras(benchmark.run),
    )
    prepare_command = ""
    driver_env_prefix = _resolve_env(driver_env) if driver_env else ""
    driver_node_flags = (
        f' --nodelist="${{gym_nodes[{driver_node}]}}" --nodes=1 --ntasks=1' if driver_node is not None else ""
    )
    # The driver writes everything relative to the job directory -- `output_path`
    # above is `artifacts/rollouts.jsonl`. `#SBATCH --chdir` sets the cwd of the
    # BATCH script on the host, but inside a Pyxis container the cwd is whatever
    # the image declares and the job directory is not visible at all unless it is
    # mounted. Without both of these the run completes cleanly, exits 0, and
    # writes every artifact into the container's ephemeral overlay, which is
    # discarded on exit: no rollouts, no metrics, no preprocessed data, and
    # nothing to say so. Logs survive only because srun resolves `--output` on
    # the host, which is what makes the loss so easy to miss.
    driver_mounts = [*config.driver.mounts, f"{remote_bench_dir}:{remote_bench_dir}"]
    driver_mounts_flag = f" --container-mounts={','.join(shlex.quote(m) for m in driver_mounts)}"
    driver_command = (f"{gym_cmd}\n" if gym_cmd else "") + (
        f"{driver_env_prefix}srun --overlap --no-container-mount-home{driver_node_flags}{driver_mounts_flag}"
        f" --container-image={shlex.quote(config.driver.container)} "
        f"--output=logs/driver-$SLURM_JOB_ID.log {entrypoint}"
    )
    if observed:
        driver_command += "\n" + _render_collector_shutdown(config, remote_bench_dir)

    return _SCRIPT_TEMPLATE.format(
        directives=directives,
        resume_prologue=resume_prologue,
        ray_prelude=ray_prelude,
        service_commands=service_commands,
        health_checks=health_checks,
        prepare_command=prepare_command,
        driver_command=driver_command,
    )
