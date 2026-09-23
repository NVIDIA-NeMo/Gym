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

from nemo_gym.global_config import MODEL_CALL_CAPTURE_DIR_KEY_NAME, OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.orchestration.api import (
    RUNTIME_ENV_PREFIX,
    BenchmarkRunConfig,
    NodePool,
    RayServiceConfig,
    RouterServiceConfig,
    ServiceConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmServiceConfig,  # used in _BUILDERS dispatch table
    effective_ray_serve,
)
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
    container: str,
    command: str,
    env: dict[str, str] | None = None,
    mounts: list[str] | None = None,
    nodes: int | None = None,
    ntasks: int | None = None,
    pre_command: str = "",
    relative: int | None = None,
) -> str:
    var = bash_var(name)
    env_prefix = _resolve_env(env) if env else ""
    node_flags = f" --nodes={nodes} --ntasks={ntasks}" if (nodes is not None and nodes > 1) else ""
    # --relative=N starts the step at node N of the allocation. With --overlap it is
    # what gives a pinned service nodes of its own while other steps run elsewhere.
    if relative is not None:
        node_flags = f" --relative={relative} --nodes={nodes} --ntasks={ntasks}"
    mounts_flag = f" --container-mounts={','.join(shlex.quote(m) for m in mounts)}" if mounts else ""
    if pre_command:
        # Wrapped in one shell so export/unset statements in pre_command are
        # visible to the exec'd command; shlex.quote keeps the whole thing one
        # word, so it can't interfere with --container-mounts/-image parsing
        # regardless of what pre_command contains.
        command = f"bash -c {shlex.quote(pre_command + chr(10) + 'exec ' + command)}"
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
    if service.kv_role is None:
        return ""
    config = json.dumps(
        {
            "kv_connector": service.kv_connector,
            "kv_role": f"kv_{service.kv_role}",
            "kv_load_failure_policy": service.kv_load_failure_policy,
        },
        separators=(",", ":"),
    )
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
    if service.kv_role is None:
        return ""
    return (
        "export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)\n"
        f"export VLLM_NIXL_SIDE_CHANNEL_PORT={service.nixl_side_channel_port}"
    )


def _build_router_command(
    router: RouterServiceConfig,
    services: dict[str, ServiceConfig],
    offsets: dict[str, tuple[int, int]],
) -> str:
    """The vllm-router invocation fronting a prefill/decode pair.

    Both tiers are addressed at their pool's head node, which is where each tier's
    API rank runs (the remaining ranks in a tier are headless).
    """
    prefill = services[router.prefill]
    decode = services[router.decode]
    assert isinstance(prefill, VllmServiceConfig) and isinstance(decode, VllmServiceConfig)
    assert prefill.node_pool is not None and decode.node_pool is not None
    prefill_head = _pool_head(offsets[prefill.node_pool][0])
    decode_head = _pool_head(offsets[decode.node_pool][0])
    return (
        "vllm-router"
        f" --prefill-policy {shlex.quote(router.prefill_policy)}"
        f" --decode-policy {shlex.quote(router.decode_policy)}"
        " --vllm-pd-disaggregation"
        f' --prefill "http://{prefill_head}:{prefill.port}"'
        f' --decode "http://{decode_head}:{decode.port}"'
        f" --host $(hostname)"
        f" --port {router.port}"
        f" --intra-node-data-parallel-size {router.intra_node_data_parallel_size}"
        f" --request-timeout-secs {router.request_timeout_secs}"
        f" --log-level {shlex.quote(router.log_level)}"
    )


def _build_ray_command(service: RayServiceConfig) -> str:
    # --block keeps the srun step alive. `ray start` daemonises and returns, so
    # without it the step exits the moment the node is up and Slurm tears the
    # service down again.
    cmd = "ray start --block"
    if service.mode == "head":
        cmd += f" --head --port {service.port}"
    else:
        cmd += f" --address {shlex.quote(str(service.address))}"
    if service.num_cpus is not None:
        cmd += f" --num-cpus {service.num_cpus}"
    if service.num_gpus is not None:
        cmd += f" --num-gpus {service.num_gpus}"
    if service.resources:
        # Ray takes fractional custom resources, so the field is float-typed, but a
        # whole number is written as one: {"extra_gpu": 4}, not 4.0, so the rendered
        # command reads the way the config does.
        resources = {k: int(v) if v.is_integer() else v for k, v in service.resources.items()}
        cmd += " --resources=" + shlex.quote(json.dumps(resources, sort_keys=True))
    return cmd


_BUILDERS = {
    VllmServiceConfig: _build_vllm_command,
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
    services: dict[str, ServiceConfig] | None = None,
    offsets: dict[str, tuple[int, int]] | None = None,
) -> str:
    if isinstance(service, RouterServiceConfig):
        assert services is not None and offsets is not None
        return _build_router_command(service, services, offsets)
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


def _srun_nodes(service: ServiceConfig, compute: SlurmComputeConfig, total_nodes: int) -> int | None:
    nodes = _service_nodes(service, compute, total_nodes)
    if service.node_pool is not None:
        return nodes
    return total_nodes if _vllm_spans_multiple_nodes(service, total_nodes) else None


def _srun_ntasks(
    service: ServiceConfig,
    compute: SlurmComputeConfig,
    total_nodes: int,
    total_ntasks: int,
) -> int | None:
    if service.node_pool is not None:
        pool = compute.node_pools[service.node_pool]
        return pool.nodes * pool.ntasks_per_node
    return total_ntasks if _vllm_spans_multiple_nodes(service, total_nodes) else None


def _node_totals(compute: SlurmComputeConfig) -> tuple[int, int]:
    total_nodes = sum(pool.nodes for pool in compute.node_pools.values())
    total_ntasks = sum(pool.nodes * pool.ntasks_per_node for pool in compute.node_pools.values())
    return total_nodes, total_ntasks


def _with_default_capture_dir(run: dict[str, Any], remote_bench_dir: Path) -> dict[str, Any]:
    """Auto-derive model_call_capture_dir from this benchmark's own real output
    directory when observability is on and the caller didn't set one.

    Hydra interpolation resolves before remote_bench_dir exists (it's computed
    here, in build_sbatch_script, well after SubmitConfig validation), so
    there's no way for a YAML value to reference it -- this has to happen in
    Python, once the real path is known. An explicit model_call_capture_dir in
    run always wins over this default.
    """
    if run.get(OBSERVABILITY_ENABLED_KEY_NAME) and MODEL_CALL_CAPTURE_DIR_KEY_NAME not in run:
        return {**run, MODEL_CALL_CAPTURE_DIR_KEY_NAME: str(remote_bench_dir / "model-calls")}
    return run


def _health_check_host(service: ServiceConfig, offsets: dict[str, tuple[int, int]]) -> str:
    """Where this service answers its health probe.

    The probe runs from the batch script, on the allocation's first node. A service
    pinned elsewhere answers on its own pool's head, so probing localhost would wait
    out the timeout on a service that is up.
    """
    if service.node_pool is None or offsets[service.node_pool][0] == 0:
        return "localhost"
    return _pool_head(offsets[service.node_pool][0])


def _service_pre_command(service: ServiceConfig) -> str:
    """The service's own pre_command, with any exports its role requires in front."""
    if not isinstance(service, VllmServiceConfig):
        return service.pre_command
    nixl = _nixl_pre_command(service)
    if not nixl:
        return service.pre_command
    return f"{nixl}\n{service.pre_command}" if service.pre_command else nixl


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
    # A router addresses each tier's head by pool offset, so it needs the
    # allocation's node list in allocation order.
    if any(isinstance(s, RouterServiceConfig) for s in config.services.values()):
        ray_prelude = f"{ray_prelude}\n{NODE_LIST_PRELUDE}" if ray_prelude else NODE_LIST_PRELUDE

    offsets = _pool_offsets(compute)
    service_commands = "\n\n".join(
        _render_service_command(
            name,
            service.container,
            _build_service_command(
                service,
                _service_nodes(service, compute, total_nodes),
                gpus_per_node_values,
                config.services,
                offsets,
            ),
            service.env or None,
            service.mounts or None,
            # Only services that actually span multiple nodes need --nodes/--ntasks - not every
            # service in a multi-node job (e.g. a plain Ray head service runs on a single node
            # regardless of how many nodes the overall job spans). A pinned service always gets
            # them, since --relative is meaningless without a node count.
            nodes=_srun_nodes(service, compute, total_nodes),
            ntasks=_srun_ntasks(service, compute, total_nodes, total_ntasks),
            pre_command=_service_pre_command(service),
            relative=offsets[service.node_pool][0] if service.node_pool else None,
        )
        for name, service in config.services.items()
    )

    health_checks = "\n\n".join(
        render_health_check(
            name,
            service.health_check.port,
            service.health_check.path,
            service.health_check.timeout_seconds,
            _health_check_host(service, offsets),
        )
        for name, service in config.services.items()
        if service.health_check
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
    run_args = _with_default_capture_dir(benchmark.run, remote_bench_dir)
    gym_cmd = render_gym_cmd("eval run", "GYM_CMD", [output_path] + extra_flags + flatten_run_args(run_args))
    entrypoint = render_driver_entrypoint(
        repo=gi.repo if gi else None,
        ref=gi.ref if gi else None,
        prepare_cmd=prepare_cmd,
    )
    prepare_command = ""
    driver_env_prefix = _resolve_env(config.driver.env) if config.driver.env else ""
    driver_node_flags = " --nodes=1 --ntasks=1" if is_multi_node else ""
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
    driver_command = (
        f"{gym_cmd}\n"
        f"{driver_env_prefix}srun --overlap --no-container-mount-home{driver_node_flags}{driver_mounts_flag}"
        f" --container-image={shlex.quote(config.driver.container)} "
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
