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


def escape_for_single_quoted_block(body: str) -> str:
    """Make `body` safe to embed inside a single-quoted `bash -c '...'` block.

    POSIX shells do not nest single quotes: an inner quote ENDS the outer string
    rather than nesting in it. Every value that had to be quoted -- a Hydra
    override containing a space, or any of the JSON blobs vLLM flags take
    (`--hf-overrides '{"architectures":[...]}'`) -- would otherwise break out of
    the block and word-split. Both failure modes have been observed on real
    submissions: Hydra rejecting `+multistage.stages=[{num_tasks:` on its own,
    and `/usr/bin/env: Argument list too long` from a multi-node vLLM command
    whose JSON flags reopened the quoting.

    `'"'"'` is the standard end-quote / literal-quote / reopen-quote sequence.
    It leaves `$VAR` and `$(( ))` untouched, which matters: the inner shell is
    the one meant to expand them.
    """
    return body.replace("'", "'\"'\"'")


def render_vllm_ray_symmetric_run(inner_cmd: str, total_nodes: int, resource_flags: str) -> str:
    """Render the Ray head/worker bootstrap that wraps a single vLLM instance's TP/PP command so
    it spans multiple Slurm nodes.

    Uses `ray symmetric-run` when available (Ray >= 2.50), which starts/joins a Ray cluster across
    every task and runs the entrypoint only on the elected head node. Containers with an older Ray
    pin fall back to manually starting head/worker Ray processes, keyed on Slurm's per-node task
    rank ($SLURM_NODEID).
    """
    # Only the interpolated values are escaped; the template's own structure is
    # what the quoting is meant to preserve.
    return _VLLM_RAY_SYMMETRIC_RUN.format(
        total_nodes=total_nodes,
        resource_flags=escape_for_single_quoted_block(resource_flags),
        inner_cmd=escape_for_single_quoted_block(inner_cmd),
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
        # Clone outside the working directory, and never `cd` into it. Both
        # matter, and both were learned from a failed run:
        #
        # * The driver's cwd is the job directory, so cloning here would put a
        #   second copy of every built-in asset under it. Gym resolves named
        #   assets against cwd AND the install root, so `--model-type
        #   openai_model` then matches twice and the run aborts as ambiguous.
        # * `cd`-ing into the clone and staying there would silently redirect
        #   every relative output path -- `+output_jsonl_fpath=artifacts/...` --
        #   into the clone instead of the job directory, which is the same
        #   artifact loss the container workdir exists to prevent.
        #
        # `uv pip install -e <path>` installs from a path, so no `cd` is needed.
        preamble += [
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            'source "$HOME/.local/bin/env"',
            # Clone OUT of the working directory. The driver's cwd is the job
            # directory, so cloning into it would leave a full Gym checkout plus
            # its .venv inside every benchmark's rundir on lustre -- slow to
            # write and noise in the artifacts. /tmp is node-local and the
            # driver's output path is absolute, so nothing depends on the clone
            # being reachable afterwards.
            'GYM_SRC="$(mktemp -d /tmp/gym-install-XXXXXX)"',
            f'git clone {shlex.quote(repo)} "$GYM_SRC/gym"',
            f'git -C "$GYM_SRC/gym" checkout {shlex.quote(ref)}',
            # Run from the install root. A benchmark's prepare_script and
            # jsonl_fpath are relative and resolved against cwd, not through the
            # install-root search that config_paths gets, so from anywhere else
            # `gym eval prepare` reports the benchmark as "missing a valid
            # prepare script" for a file that is right there. A runtime image
            # bakes no Gym, so without this there is no cwd where it resolves.
            # It also puts cwd and the install root in the same directory, so a
            # named asset resolves once rather than ambiguously.
            'cd "$GYM_SRC/gym"',
            # A real venv, not --system: --system targets whatever
            # interpreter happens to be on the container's PATH, which
            # sidesteps uv's own project-aware Python selection entirely --
            # `uv venv` run inside this checkout instead reads
            # requires-python from its pyproject.toml and auto-downloads a
            # satisfying interpreter if the container's own Python doesn't
            # qualify (e.g. a container shipping Python 3.12 against a
            # nemo-gym pin requiring >=3.13.14 -- --system fails outright
            # there, this doesn't). A fresh venv is also never
            # EXTERNALLY-MANAGED (PEP 668), so this needs no
            # --break-system-packages override either.
            "uv venv --seed .venv",
            "source .venv/bin/activate",
            "uv pip install -e .",
        ]

    if prepare_cmd:
        preamble.append(prepare_cmd)

    if not preamble:
        return '"${GYM_CMD[@]}"'

    preamble.append('exec "$@"')
    body = "\n    ".join(["set -euo pipefail", *preamble])
    body = escape_for_single_quoted_block(body)
    return f"bash -c '\n    {body}\n' -- \"${{GYM_CMD[@]}}\""
