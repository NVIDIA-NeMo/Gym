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

import base64
import re
import shlex
from pathlib import Path

from nemo_gym.core.failure_kinds import DRIVER_INSTALL_FAILED
from nemo_gym.orchestration.jobs import EVAL_EXIT_FILENAME, EXPORTS_DIRNAME


# The driver step's environment names the file, <job dir>/.gym_driver_failure.<allocation>, where a driver failure
# that the next segment would repeat leaves its registered kind (see the guard in resume_script.py).
DRIVER_FAILURE_ENV = "GYM_DRIVER_FAILURE_FILE"

_RAY_PRELUDE = """\
# Resolve the head node IP for multi-node vLLM services (spanning nodes via Ray).
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node_hostname" | awk '{print $1}')
export HEAD_NODE_IP="$head_node_ip"
export RAY_HEAD_NODE_IP="$head_node_ip:6379"
echo "Head node IP address: $HEAD_NODE_IP\""""


# A container without the ray CLI installs a tested Ray with the Serve extra: the gateway imports ray.serve, whose
# dependencies `default` lacks, and an unpinned install takes whatever Ray was released last.
ENSURE_RAY_INSTALLED = 'command -v ray >/dev/null 2>&1 || pip install -q "ray[default,serve]==2.59.0"'

# Default node-join wait (30s) is too short for slow image pulls. Also bounds a worker's wait for the head.
RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT_S = 600

# Default replica queue-length RPC deadline (0.1s) is too tight for cross-node hops.
RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S = 1.0

# The head node's proxy prefers its local replica until that replica is full, and a gateway replica
# never fills (MAX_ONGOING_REQUESTS_PER_INSTANCE), so every other instance would sit idle.
RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING = 0


_VLLM_RAY_SYMMETRIC_RUN = (
    """\
bash -lc '
    export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT="""
    + str(RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT_S)
    + """
    export RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S="""
    + str(RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S)
    + """
    export RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING="""
    + str(RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING)
    + """
    """
    + ENSURE_RAY_INSTALLED
    # Ray applies the wait above only to the head waiting for workers: a worker gives up on a late
    # head after ~40s (GcsClient connect timeout), so workers first wait for the head's GCS port.
    # Node 0 is the head here as in the fallback below: the prelude takes RAY_HEAD_NODE_IP from it.
    + """
    if [ "${{SLURM_NODEID:-0}}" != "0" ]; then
        echo "Waiting up to ${{RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT}}s for the Ray head at $RAY_HEAD_NODE_IP..."
        deadline=$((SECONDS + RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT))
        until (: >"/dev/tcp/${{RAY_HEAD_NODE_IP%:*}}/${{RAY_HEAD_NODE_IP##*:}}") 2>/dev/null; do
            if [ "$SECONDS" -ge "$deadline" ]; then
                echo "ERROR: timed out waiting for the Ray head at $RAY_HEAD_NODE_IP." >&2
                exit 1
            fi
            sleep 1
        done
    fi
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
)


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


def render_health_check(name: str, port: int, path: str, timeout: int, host: str = "localhost") -> str:
    """`host` is emitted verbatim so it may be a shell expansion: a service pinned to
    a node pool answers on that pool's head, not on the node running this script."""
    return _HEALTH_WAIT_MULTI.format(
        name=name,
        name_upper=bash_var(name),
        url=f"http://{host}:{port}",
        path=path,
        max_attempts=timeout // 5,
    )


def render_gym_cmd(subcommand: str, var_name: str, args: list[str]) -> str:
    """Render a bash array with each argument on its own line."""
    entries = [f"gym {subcommand}", *args]
    return f"{var_name}=(\n    " + "\n    ".join(entries) + "\n)"


def render_repo_checkout(repo: str, ref: str, dest: str | None = None) -> str:
    """Render checkout steps that each fail immediately under the driver's ``set -e``.

    `dest` is emitted verbatim so it may be a shell expression (the driver passes
    `"$GYM_SRC/gym"` to clone outside the job directory); omit it to clone into a
    directory named after the repo in the current one.
    """
    repo_name = repo.rstrip("/").split("/")[-1].removesuffix(".git")
    target = dest if dest is not None else shlex.quote(repo_name)
    ensure_git = "command -v git >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git)"
    return (
        f"({ensure_git})\ngit clone --filter=blob:none --no-checkout {shlex.quote(repo)} {target}\n"
        f"cd {target}\ngit checkout {shlex.quote(ref)}"
    )


def render_write_file_from_base64(content: str, dest_path: str) -> str:
    """Render a command that reconstructs `content` at `dest_path` via a base64 round-trip."""
    encoded = base64.b64encode(content.encode()).decode()
    return f"printf '%s' '{encoded}' | base64 -d > {shlex.quote(dest_path)}"


# What the driver step runs for each export, with the exporter's name and its overrides as arguments.
EXPORT_STEP_MODULE = "nemo_gym.orchestration.export_step"

# Set for a driver whose script has exports: a file this attempt creates once its evaluation has succeeded. The
# job script watches the services only until then (see `_render_watched_driver`).
EVAL_SUCCEEDED_MARKER_ENV = "GYM_EVAL_SUCCEEDED_MARKER"


def render_export_tail(bench_dir: Path, exports: list[tuple[str, list[str]]]) -> str:
    """Render the part of the driver script that runs `gym eval run`, then each export of its rollouts.

    `exports` pairs an exporter name with its shell-quoted override arguments for `EXPORT_STEP_MODULE`. The run is a child, not
    `exec`'d, so the script can go on to the exports; TERM, which is what the driver gets at the time limit, is
    passed on to it, so an interrupted run behaves as it did when it was `exec`'d. The run's exit code goes to
    `<bench_dir>/eval.exit` and is the script's exit code: the exports are reported in their own result files
    and cannot change it. They run only after a successful run, so a partial run is never exported, and an
    export that already succeeded is not repeated (the export step decides that from its result file).
    """
    bench = shlex.quote(str(bench_dir))
    lines = [
        "set +e",
        '"$@" &',
        "gym_run_pid=$!",
        "trap 'kill -TERM \"$gym_run_pid\" 2>/dev/null' TERM INT",
        'wait "$gym_run_pid"',
        "gym_run_rc=$?",
        "# A signal ends the first wait early; wait again for the exit code of the run itself.",
        'while kill -0 "$gym_run_pid" 2>/dev/null; do wait "$gym_run_pid"; gym_run_rc=$?; done',
        "trap - TERM INT",
        "set -e",
        f"mkdir -p {bench}/{EXPORTS_DIRNAME} || true",
        f'echo "$gym_run_rc" > {bench}/{EVAL_EXIT_FILENAME} || true',
        'if [ "$gym_run_rc" -eq 0 ]; then',
        f'    [ -z "${{{EVAL_SUCCEEDED_MARKER_ENV}:-}}" ] || : > "${EVAL_SUCCEEDED_MARKER_ENV}" || true',
        *(
            f"    python3 -m {EXPORT_STEP_MODULE} {shlex.quote(name)} {' '.join(args)} || true"
            for name, args in exports
        ),
        "fi",
        'exit "$gym_run_rc"',
    ]
    return "\n".join(lines)


# `uv sync` errors on an extra the checkout does not define. A pin from before `ray` and `mlflow` became extras defines
# neither, so each requested `--extra NAME` is kept only when the checkout's pyproject.toml declares NAME.
_DECLARED_EXTRAS_FILTER = (
    "import sys, tomllib\n"
    'declared = set(tomllib.load(open("pyproject.toml", "rb")).get("project", {}).get("optional-dependencies", {}))\n'
    "requested = sys.argv[2::2]\n"
    'print(*(f"--extra {name}" for name in requested if name in declared))\n'
    "for name in requested:\n"
    "    if name not in declared:\n"
    '        print(f"gym: the checkout defines no {name!r} extra, installing without it", file=sys.stderr)\n'
)


def render_driver_entrypoint(
    repo: str | None,
    ref: str | None,
    prepare_cmd: str | None,
    command: str | None = None,
    *,
    extras: tuple[str, ...] = (),
    cleanup: bool = False,
    record_install_failure: bool = False,
    private_uv: bool = False,
    export_tail: str | None = None,
) -> str:
    """Render the srun entrypoint for the driver step.

    When either gym_install or prepare is needed, wraps everything in a single
    bash -c so prepare and run happen in the same srun step and container.

    `command` replaces the `gym eval run` invocation for a benchmark whose harness
    is not Gym's own runner. It is emitted verbatim, so it may be a multi-line
    script; everything before it (the install, and prepare) is unchanged.

    `extras` are optional-dependency groups installed with the checkout; the server
    venvs Gym builds copy their telemetry packages from this process, so the driver
    is where `telemetry` has to be installed for any server to have it. `ray` is
    always added: `gym` needs Ray to start a server that declares `ray_enabled = True`.
    An extra the checkout does not define is dropped with a notice, since uv errors on it.

    `record_install_failure` writes `driver_install_failed` to the file named by `$GYM_DRIVER_FAILURE_FILE` when
    the install exits with a failure status (1 to 128; a signal is not an install failure), for the guard of the
    next segment. It replaces the EXIT trap, so it cannot be combined with `cleanup`.

    `private_uv` gives the install its own uv binary inside its temporary directory (a copy of a working `uv`, else
    a fresh install there), keeps uv's cache on the host's local disk in this user's own directory only, and gives
    the install its own managed Pythons. Concurrent installs on other hosts replace the shared `~/.local/bin/uv` and
    cache entries on a network home, and a uv running from those files then dies with SIGBUS.

    `cleanup` removes the temporary checkout (`$GYM_SRC`) through an EXIT trap when the step exits.

    `export_tail` (see `render_export_tail`) takes the place of `exec "$@"`, so the exports run in the same step and
    container as the run, where `gym` is installed.
    """
    if record_install_failure and cleanup:
        raise ValueError("record_install_failure and cleanup both need the EXIT trap")
    on_exit = (
        f'_gym_rc=$?; if (( _gym_rc > 0 && _gym_rc <= 128 )) && [[ -n "${{{DRIVER_FAILURE_ENV}:-}}" ]]; then '
        f'echo {DRIVER_INSTALL_FAILED} > "${DRIVER_FAILURE_ENV}"; fi'
    )
    record_failure = [f"trap {shlex.quote(on_exit)} EXIT"] if record_install_failure else []
    preamble: list[str] = []
    # Double quotes: the whole preamble ends up inside a single-quoted `bash -c` block.
    sync_extras = "".join(f" --extra {shlex.quote(extra)}" for extra in dict.fromkeys((*extras, "ray")))

    if repo and ref:
        # Clone to /tmp rather than the job directory: a checkout plus its .venv
        # inside every benchmark's rundir is slow to write on lustre and noise in
        # the artifacts. `cd` into it because a benchmark's prepare_script and
        # jsonl_fpath resolve against cwd; the driver's output path is absolute,
        # so nothing depends on the clone being reachable afterwards.
        if private_uv:
            uv_setup = [
                'GYM_SRC="$(mktemp -d /tmp/gym-install-XXXXXX)"',
                *(["trap " + shlex.quote('rm -rf -- "$GYM_SRC"') + " EXIT"] if cleanup else []),
                'mkdir -p "$GYM_SRC/uv-bin"',
                'export PATH="$HOME/.local/bin:$PATH"',
                # Run a private copy, never the shared binary another host's installer may replace; a copy that
                # does not run is replaced by a private install. That install is unmanaged: it writes no user-level
                # receipt pointing at a directory this read deletes, and leaves PATH files alone.
                'if _gym_uv="$(command -v uv)" && "$_gym_uv" --version >/dev/null 2>&1 '
                '&& cp "$_gym_uv" "$GYM_SRC/uv-bin/uv" && "$GYM_SRC/uv-bin/uv" --version >/dev/null 2>&1; then :; '
                "else curl -LsSf https://astral.sh/uv/install.sh | "
                'env UV_UNMANAGED_INSTALL="$GYM_SRC/uv-bin" sh; fi',
                # The cache and managed Pythons stay on this host's local disk, reused by its later readers: a
                # rename there leaves a running uv's open files intact, where a network home does not. A shared
                # temporary-directory name is used only when it is this user's own directory, not a link and not
                # writable by others; otherwise another user could plant a Python or a cache entry there, so the
                # install keeps them in its own temporary directory instead.
                '_gym_own_dir() { mkdir -p -m 700 "$1" 2>/dev/null; if [ -d "$1" ] && [ ! -L "$1" ] && [ -O "$1" ] '
                '&& [ -z "$(find "$1" -maxdepth 0 -perm /022)" ]; then printf %s "$1"; else printf %s "$2"; fi; }',
                'export PATH="$GYM_SRC/uv-bin:$PATH" '
                'UV_CACHE_DIR="$(_gym_own_dir "${TMPDIR:-/tmp}/gym-uv-cache-$(id -u)" "$GYM_SRC/uv-cache")" '
                # Managed Pythons are per read: an interpreter file another process rewrites or a killed install
                # leaves half-written breaks every venv built on it, so no reader shares one. uv otherwise prefers
                # a compatible Python already on PATH, such as a link in the shared `~/.local/bin`, so only the
                # managed Pythons of this read are allowed.
                'UV_PYTHON_INSTALL_DIR="$GYM_SRC/uv-python" UV_PYTHON_PREFERENCE=only-managed',
            ]
        else:
            uv_setup = [
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                # Not `source "$HOME/.local/bin/env"`: the installer skips writing that file when
                # the directory is already on PATH, and the file only does this export anyway.
                'export PATH="$HOME/.local/bin:$PATH"',
                'GYM_SRC="$(mktemp -d /tmp/gym-install-XXXXXX)"',
                *(["trap " + shlex.quote('rm -rf -- "$GYM_SRC"') + " EXIT"] if cleanup else []),
            ]
        preamble += [
            *record_failure,
            *uv_setup,
            render_repo_checkout(repo, ref, dest='"$GYM_SRC/gym"'),
            # A real venv, not --system: --system targets whatever interpreter happens to be on
            # the container's PATH, sidestepping uv's own project-aware Python selection - `uv
            # venv` instead reads requires-python from pyproject.toml and auto-downloads a
            # satisfying interpreter if the container's own Python doesn't qualify. Also never
            # EXTERNALLY-MANAGED (PEP 668), so no --break-system-packages override needed either.
            "uv venv --seed .venv",
            "source .venv/bin/activate",
            f'_gym_declared_extras() {{ python -c {shlex.quote(_DECLARED_EXTRAS_FILTER)} "$@"; }}',
            f'_gym_sync_extras="$(_gym_declared_extras{sync_extras})"',
            'UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" uv sync --locked $_gym_sync_extras',
            *(["trap - EXIT"] if record_failure else []),
        ]

    if prepare_cmd:
        preamble.append(prepare_cmd)

    if command is None:
        if not preamble and export_tail is None:
            return '"${GYM_CMD[@]}"'
        preamble.append(export_tail if export_tail is not None else 'exec "$@"')
        body = "\n    ".join(["set -euo pipefail", *preamble])
        body = escape_for_single_quoted_block(body)
        return f"bash -c '\n    {body}\n' -- \"${{GYM_CMD[@]}}\""

    # No GYM_CMD array to hand over: the command IS the final statement, so there
    # is nothing to `exec "$@"` and no trailing `-- "${GYM_CMD[@]}"`.
    body = "\n    ".join(["set -euo pipefail", *preamble, command])
    body = escape_for_single_quoted_block(body)
    return f"bash -c '\n    {body}\n'"
