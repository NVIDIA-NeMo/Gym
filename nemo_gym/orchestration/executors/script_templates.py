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


_RAY_PRELUDE = """\
# Resolve the head node IP for multi-node vLLM services (spanning nodes via Ray).
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node_hostname" | awk '{print $1}')
export HEAD_NODE_IP="$head_node_ip"
export RAY_HEAD_NODE_IP="$head_node_ip:6379"
echo "Head node IP address: $HEAD_NODE_IP\""""


_VLLM_RAY_SYMMETRIC_RUN_BODY = """\
    command -v ray >/dev/null 2>&1 || pip install -q "ray[default]"
    if ray symmetric-run --help >/dev/null 2>&1; then
        ray symmetric-run \\
            --address "$RAY_HEAD_NODE_IP" \\
            --min-nodes {total_nodes} \\
            {resource_flags} \\
            -- {inner_cmd}
    elif [ "$SLURM_NODEID" = "0" ]; then
        ray start --head --port=6379 {resource_flags}
        {inner_cmd}
    else
        ray start --address="$RAY_HEAD_NODE_IP" {resource_flags} --block
    fi"""


_HAPROXY_INSTANCE_LAUNCH_SHARED_NODE = """\
    for i in $(seq 1 {instances_per_node}); do
        port=$(( {frontend_port} + i ))
        {vllm_cmd_no_port} --port $port &
    done"""


_HAPROXY_INSTANCE_LAUNCH_SPANS_NODES = """\
    group=$(( node_idx / {nodes_per_instance} ))
    group_head_hostname=${{nodes_array[$(( group * {nodes_per_instance} ))]}}
    group_head_ip=$(getent hosts "$group_head_hostname" | cut -d" " -f1)
    export RAY_HEAD_NODE_IP="$group_head_ip:6379"
    # Every node in the group runs the same symmetric-run bootstrap; ray symmetric-run
    # self-elects the head via rendezvous on RAY_HEAD_NODE_IP, so this is correct regardless of
    # which physical node the group lands on. (The legacy pre-symmetric-run fallback inside it
    # keys off the *global* $SLURM_NODEID and only self-elects correctly for group 0 - acceptable
    # since symmetric-run is the primary path on any reasonably current Ray.) Run as a
    # parenthesized subshell rather than the usual `bash -lc '...'` wrapper - this whole command
    # is itself embedded inside an outer `bash -c '...'`, and nesting single-quoted strings would
    # terminate that outer quote early.
    (
{vllm_ray_symmetric_run_body}
    ) &"""


_HAPROXY_BACKEND_LOOP = """\
        idx=0
        for backend_node_idx in "${{!nodes_array[@]}}"; do
            if [ $(( backend_node_idx % {nodes_per_instance} )) -eq 0 ]; then
                backend_node_ip=$(getent hosts "${{nodes_array[$backend_node_idx]}}" | cut -d" " -f1)
                for i in $(seq 1 {instances_per_node}); do
                    backend_port=$(( {frontend_port} + i ))
                    echo "    server srv_${{idx}} ${{backend_node_ip}}:${{backend_port}} check inter 5s fall 3 rise 2 httpchk GET {health_path}"
                    idx=$((idx+1))
                done
            fi
        done"""


_HAPROXY_MULTI_INSTANCE = """\
bash -c '
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    node_idx=$SLURM_NODEID

{instance_launch}

    if [ "$SLURM_NODEID" = "0" ]; then
        command -v haproxy >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq haproxy)
        {{
            echo "global"
            echo "    maxconn 4096"
            echo "defaults"
            echo "    mode http"
            echo "    timeout connect 5s"
            echo "    timeout client 300s"
            echo "    timeout server 300s"
            echo "frontend fe_{bash_name}"
            echo "    bind *:{frontend_port}"
            echo "    default_backend be_{bash_name}"
            echo "backend be_{bash_name}"
            echo "    balance leastconn"
{backend_loop}
        }} > haproxy_{bash_name}.cfg
        exec haproxy -f haproxy_{bash_name}.cfg -db
    else
        wait
    fi
'"""


_HEALTH_WAIT_MULTI = """\
# Wait for {name} (try multiple health endpoints)
echo "Waiting for {name} at {url}..."
{name_upper}_READY=0
for _i in $(seq 1 {max_attempts}); do
    if curl -sf "{url}{path}" > /dev/null 2>&1; then
        echo "  {name} ready."
        {name_upper}_READY=1
        break
    fi
    if [ -n "${{{name_upper}_PID:-}}" ] && ! kill -0 ${name_upper}_PID 2>/dev/null; then
        echo "  {name} died during startup."
        exit 1
    fi
    sleep 5
done
if [ ${name_upper}_READY -eq 0 ]; then
    echo "ERROR: {name} did not become healthy after {max_attempts} attempts."
    exit 1
fi
"""


def bash_var(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", name.upper())


def render_ray_prelude() -> str:
    return _RAY_PRELUDE


def render_vllm_ray_symmetric_run_body(inner_cmd: str, total_nodes: int, resource_flags: str) -> str:
    """Render just the Ray head/worker bootstrap statements (no surrounding `bash -lc '...'`), for
    embedding inside a caller-provided shell wrapper that may already be single-quoted.
    """
    return _VLLM_RAY_SYMMETRIC_RUN_BODY.format(
        total_nodes=total_nodes, resource_flags=resource_flags, inner_cmd=inner_cmd
    )


def render_vllm_ray_symmetric_run(inner_cmd: str, total_nodes: int, resource_flags: str) -> str:
    """Render the Ray head/worker bootstrap that wraps a single vLLM instance's TP/PP command so
    it spans multiple Slurm nodes.

    Uses `ray symmetric-run` when available (Ray >= 2.50), which starts/joins a Ray cluster across
    every task and runs the entrypoint only on the elected head node. Containers with an older Ray
    pin fall back to manually starting head/worker Ray processes, keyed on Slurm's per-node task
    rank ($SLURM_NODEID).
    """
    body = render_vllm_ray_symmetric_run_body(inner_cmd, total_nodes, resource_flags)
    return f"bash -lc '\n{body}\n'"


def render_haproxy_multi_instance_command(
    name: str,
    vllm_cmd_no_port: str,
    frontend_port: int,
    health_path: str,
    nodes_per_instance: int,
    instances_per_node: int,
) -> str:
    """Render N independent `vllm serve` processes fronted by an HAProxy reverse proxy.

    Exactly one of nodes_per_instance/instances_per_node is > 1 (see api.haproxy_topology):
    - instances_per_node > 1: every node launches that many local vllm serve processes on ports
      frontend_port+1..+instances_per_node.
    - nodes_per_instance > 1: every group of nodes runs a single vLLM instance spanning the group
      via vLLM's Ray core executor (reusing render_vllm_ray_symmetric_run), bound to
      frontend_port+1 on the group's head node.

    Only the global head node (SLURM_NODEID == 0) runs HAProxy, bound to frontend_port, load
    balancing across every instance/group across the whole allocation.
    """
    bash_name = bash_var(name)

    if nodes_per_instance > 1:
        resource_flags = (
            "--num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} "
            "--num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE}"
        )
        inner_cmd = f"{vllm_cmd_no_port} --port {frontend_port + 1} --distributed-executor-backend ray"
        vllm_ray_symmetric_run_body = render_vllm_ray_symmetric_run_body(inner_cmd, nodes_per_instance, resource_flags)
        instance_launch = _HAPROXY_INSTANCE_LAUNCH_SPANS_NODES.format(
            nodes_per_instance=nodes_per_instance,
            vllm_ray_symmetric_run_body=vllm_ray_symmetric_run_body,
        )
    else:
        instance_launch = _HAPROXY_INSTANCE_LAUNCH_SHARED_NODE.format(
            instances_per_node=instances_per_node,
            frontend_port=frontend_port,
            vllm_cmd_no_port=vllm_cmd_no_port,
        )

    backend_loop = _HAPROXY_BACKEND_LOOP.format(
        nodes_per_instance=nodes_per_instance,
        instances_per_node=instances_per_node,
        frontend_port=frontend_port,
        health_path=health_path,
    )

    return _HAPROXY_MULTI_INSTANCE.format(
        bash_name=bash_name,
        frontend_port=frontend_port,
        instance_launch=instance_launch,
        backend_loop=backend_loop,
    )


def render_health_check(name: str, port: int, path: str, timeout: int) -> str:
    return _HEALTH_WAIT_MULTI.format(
        name=name,
        name_upper=bash_var(name),
        url=f"http://localhost:{port}",
        path=path,
        max_attempts=timeout // 5,
    )


def render_gym_cmd(subcommand: str, var_name: str, args: list[str]) -> str:
    """Render a bash array with each argument on its own line."""
    entries = [f"gym {subcommand}", *args]
    return f"{var_name}=(\n    " + "\n    ".join(entries) + "\n)"


def render_driver_entrypoint(
    repo: str | None,
    ref: str | None,
    prepare_cmd: str | None,
) -> str:
    """Render the srun entrypoint for the driver step.

    When either gym_install or prepare is needed, wraps everything in a single
    bash -c so prepare and run happen in the same srun step and container.
    """
    preamble: list[str] = []

    if repo and ref:
        repo_name = repo.rstrip("/").split("/")[-1].removesuffix(".git")
        preamble += [
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            'source "$HOME/.local/bin/env"',
            f"git clone {shlex.quote(repo)}",
            f"cd {shlex.quote(repo_name)}",
            f"git checkout {shlex.quote(ref)}",
            "uv pip install -e . --system",
        ]

    if prepare_cmd:
        preamble.append(prepare_cmd)

    if not preamble:
        return '"${GYM_CMD[@]}"'

    preamble.append('exec "$@"')
    body = "\n    ".join(preamble)
    return f"bash -c '\n    {body}\n' -- \"${{GYM_CMD[@]}}\""
