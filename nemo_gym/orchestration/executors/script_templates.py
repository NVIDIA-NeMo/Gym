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


_VLLM_RAY_SYMMETRIC_RUN = """\
bash -lc '
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


def render_vllm_ray_symmetric_run(inner_cmd: str, total_nodes: int, resource_flags: str) -> str:
    """Render the Ray head/worker bootstrap that wraps a single vLLM instance's TP/PP command so
    it spans multiple Slurm nodes.

    Uses `ray symmetric-run` when available (Ray >= 2.50), which starts/joins a Ray cluster across
    every task and runs the entrypoint only on the elected head node. Containers with an older Ray
    pin fall back to manually starting head/worker Ray processes, keyed on Slurm's per-node task
    rank ($SLURM_NODEID).
    """
    return _VLLM_RAY_SYMMETRIC_RUN.format(total_nodes=total_nodes, resource_flags=resource_flags, inner_cmd=inner_cmd)


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
