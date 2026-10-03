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
"""Run an agent on tasks in one command: serve, collect rollouts, and shut down.

A trampoline over the two-step flow ``gym env start ...`` + ``gym eval run --no-serve ...``: it starts
the servers in the background, waits for the head server's /readyz, runs the collection in the
foreground, then stops every server process it started, also on failure or Ctrl-C.

Without an environment flag (``--config``, ``--resources-server``, ``--environment``, ``--benchmark``)
it serves the codex_tools coding environment on the current directory's repository:

    cd /path/to/target/repo      # the harness cwd: the repository the coding agent works on
    python /path/to/Gym/examples/simple_agent.py \\
        --model-type openai_model --model gpt-6-sol \\
        --prompt "Fix the failing tests in tests/" --check-command "python -m pytest -q tests" \\
        --output /tmp/rollouts.jsonl

``--prompt`` (with an optional ``--check-command``) writes a one-task input file,
``<output stem>_task.jsonl``, in place of ``--input``. ``--echo pretty|json`` prints simple_agent's
items (input, model output, tool output) as they happen, also saved to ``<output stem>_items.*``;
it defaults to ``pretty`` with ``--prompt`` and ``off`` otherwise.

Run it with any Python environment that has NeMo Gym installed. Each server runs in its own
virtualenv, which Gym sets up with ``uv``. Existing ones are reused as they are
(``--skip-venv-if-present``, the default: ``+skip_venv_if_present=true``); only missing ones need
``uv``. ``--no-skip-venv-if-present`` sets them all up again.

Other arguments are ``gym eval run`` flags: those ``gym env start`` also accepts (``--config``,
``--model``, ...) start the servers, the rest (``--input``, ``--output``, ``--agent``,
``--num-repeats``, ...) go to the collection, and Hydra ``+key=value`` overrides go to both. Arguments after ``--`` go to the collection unchanged.
``--agent`` may be omitted when the served config has a single agent.

Two directories are kept apart:

- The harness cwd (``--harness-cwd``, default: where this script is run) is passed to the servers as
  ``+harness_cwd=<path>`` for configs that work on a repository (e.g. codex_tools' default
  ``repo_path``).
- Gym's own working directory (venvs, cache, results) is the Gym checkout for an editable install,
  otherwise ``--gym-workdir`` (default: ``~/.cache/nemo_gym/simple_agent``), so nothing Gym writes lands
  in the target repository.

Relative paths mean what they would for ``gym`` run here: ``--config``/``--input`` are looked up in
the current directory first, then among Gym's built-ins, and ``--output`` is relative to the current
directory. Server output goes to ``<output stem>_servers.log`` next to the rollouts file. See
examples/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import nemo_gym
from nemo_gym.cli.main import COMMANDS


PREFIX = "[simple_agent]"
HEAD_HOST = "127.0.0.1"
# Flags both commands need: logging, and extra search roots (the collection resolves -i through them).
_BOTH = frozenset({"verbose", "search_dir"})
DEFAULT_RESOURCES_SERVER = "codex_tools"
# Flags that choose what to serve; without any of them, the default resources server is served.
_ENVIRONMENT_DESTS = frozenset({"config", "resources_server", "environment", "benchmark"})


def log(message: str) -> None:
    print(f"{PREFIX} {message}", file=sys.stderr, flush=True)


# ---- Argument routing ----


def _command_actions(command_name: str) -> dict[str, argparse.Action]:
    """Option string -> argparse action for a `gym` command, from the CLI's own flag definitions."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    for flag in COMMANDS[command_name].flags:
        flag.register(parser)
    return dict(parser._option_string_actions)


@dataclass
class RoutedArgs:
    start: list[str] = field(default_factory=list)
    collect: list[str] = field(default_factory=list)
    start_dests: set[str] = field(default_factory=set)
    collect_dests: set[str] = field(default_factory=set)
    output: Optional[Path] = None


def route_args(argv: Sequence[str], *, cwd: Path) -> RoutedArgs:
    """Split `gym eval run`-style arguments between `gym env start` and `gym eval run --no-serve`."""
    start_actions = _command_actions("env start")
    collect_actions = _command_actions("eval run")
    routed = RoutedArgs()
    tokens = list(argv)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if token == "--":
            routed.collect += tokens[i:]
            for passthrough in tokens[i:]:
                action = collect_actions.get(passthrough.partition("=")[0])
                if action is not None:
                    routed.collect_dests.add(action.dest)
            break
        if not token.startswith("-"):
            # A Hydra override; harmless for whichever command does not use it.
            routed.start.append(token)
            routed.collect.append(token)
            continue
        option, has_inline_value, inline_value = token.partition("=")
        action = start_actions.get(option) or collect_actions.get(option)
        if action is None or option == "--no-serve" or action.nargs not in (None, 0):
            raise SystemExit(f"{PREFIX} unsupported argument: {token}")
        values: list[str] = []
        if has_inline_value:
            values = [inline_value]
        elif action.nargs is None:
            if i >= len(tokens):
                raise SystemExit(f"{PREFIX} {option} expects a value")
            values = [tokens[i]]
            i += 1
        if action.dest == "output":
            # A write path: relative to where the user ran the command, not Gym's working directory.
            routed.output = Path(os.path.normpath(cwd / values[0]))
            values = [str(routed.output)]
        routed_tokens = [option, *values]
        # Flags both commands accept (--config, --model, ...) configure the servers; the collection
        # reads the served config from the head server.
        if action.dest in _BOTH:
            routed.start += routed_tokens
            routed.collect += routed_tokens
        elif option in start_actions:
            routed.start += routed_tokens
            routed.start_dests.add(action.dest)
        else:
            routed.collect += routed_tokens
            routed.collect_dests.add(action.dest)
    return routed


def _has_override(tokens: Sequence[str], *keys: str) -> bool:
    return any(token.lstrip("+").split("=", 1)[0] in keys for token in tokens if not token.startswith("-"))


def selects_environment(routed: RoutedArgs) -> bool:
    return bool(routed.start_dests & _ENVIRONMENT_DESTS) or _has_override(routed.start, "config_paths")


def write_prompt_task(path: Path, prompt: str, check_command: Optional[str]) -> None:
    """Write a one-task codex_tools input file for --prompt."""
    from resources_servers.codex_tools.make_tasks import make_row

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_row(prompt, check_command=check_command), ensure_ascii=False) + "\n")


# ---- Directories and processes ----


def gym_workdir(requested: Optional[str]) -> Path:
    """Where Gym writes venvs, cache, and results: the checkout for editable installs (where Gym
    ignores the cwd anyway), else a dedicated directory rather than the target repository."""
    checkout = Path(nemo_gym.__file__).resolve().parent.parent
    if requested is None and (checkout / "pyproject.toml").exists():
        return checkout
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nemo_gym" / "simple_agent"
    path = Path(requested).expanduser().absolute() if requested else default
    path.mkdir(parents=True, exist_ok=True)
    return path


def gym_executable() -> str:
    candidate = Path(sys.executable).with_name("gym")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("gym")
    if found is None:
        raise SystemExit(f"{PREFIX} cannot find the `gym` command next to {sys.executable} or on PATH")
    return found


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HEAD_HOST, 0))
        return sock.getsockname()[1]


def _hydra_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def head_overrides(port: int) -> list[str]:
    return [f"++head_server.host={HEAD_HOST}", f"++head_server.port={port}"]


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_until_ready(start: subprocess.Popen, port: int, *, timeout_s: float, poll_s: float = 2.0) -> None:
    """Block until the head server reports every server and model endpoint ready (/readyz 200)."""
    deadline = time.monotonic() + timeout_s
    while True:
        if start.poll() is not None:
            raise RuntimeError(f"`gym env start` exited with code {start.returncode} before the servers were ready")
        try:
            if _http_get(f"http://{HEAD_HOST}:{port}/readyz", timeout=poll_s)[0] == 200:
                return
        except (OSError, urllib.error.URLError):
            pass  # The head server is not listening yet.
        if time.monotonic() > deadline:
            raise RuntimeError(f"servers were not ready after {timeout_s:.0f} s")
        time.sleep(poll_s)


def served_agents(port: int) -> list[str]:
    from omegaconf import OmegaConf

    status, body = _http_get(f"http://{HEAD_HOST}:{port}/global_config_dict_yaml")
    if status != 200:
        return []
    config = OmegaConf.to_container(OmegaConf.create(json.loads(body)), resolve=False)
    return sorted(
        name for name, value in config.items() if isinstance(value, dict) and "responses_api_agents" in value
    )


def stop_servers(start: subprocess.Popen, *, grace_s: float = 60.0) -> None:
    """Interrupt `gym env start` (which stops its servers and Ray), then kill whatever is left of its
    process group, including servers orphaned by a failed startup."""
    if start.poll() is None:
        start.send_signal(signal.SIGINT)
        try:
            start.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            log(f"`gym env start` did not stop within {grace_s:.0f} s; killing it")
    try:
        os.killpg(start.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    start.wait()


def wait_for_collection(collect: subprocess.Popen) -> int:
    """Wait for the collection; on Ctrl-C give it time to finish its own interrupt handling."""
    try:
        return collect.wait()
    except KeyboardInterrupt:
        for sig, wait_s in ((None, 10), (signal.SIGINT, 20), (signal.SIGKILL, None)):
            if sig is not None:
                collect.send_signal(sig)
            try:
                collect.wait(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        raise


def _tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


class FileFollower(threading.Thread):
    """Copy complete lines appended to a file to stdout until stopped, then drain what is left."""

    def __init__(self, path: Path, *, poll_s: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.poll_s = poll_s
        self._stop_requested = threading.Event()
        self._offset = 0
        self._partial = b""

    def _copy_new_lines(self) -> None:
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                data = handle.read()
        except FileNotFoundError:
            return
        self._offset += len(data)
        lines, _, self._partial = (self._partial + data).rpartition(b"\n")
        if lines:
            sys.stdout.write((lines + b"\n").decode("utf-8", "replace"))
            sys.stdout.flush()

    def run(self) -> None:
        while not self._stop_requested.wait(self.poll_s):
            self._copy_new_lines()
        self._copy_new_lines()

    def stop(self) -> None:
        self._stop_requested.set()
        self.join()


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--harness-cwd", help="Repository the harness works on (default: the current directory).")
    parser.add_argument("--gym-workdir", help="Gym's working directory for non-editable installs.")
    parser.add_argument(
        "--startup-timeout", type=float, default=1800.0, help="Seconds to wait for the servers (default: 1800)."
    )
    parser.add_argument(
        "--prompt", help="Task for a codex_tools coding agent; replaces --input with a one-task input file."
    )
    parser.add_argument(
        "--check-command",
        help="With --prompt: shell command run in the workspace after the agent finishes; exit status 0 = reward 1.",
    )
    parser.add_argument(
        "--echo",
        choices=("off", "pretty", "json"),
        help="Print simple_agent's items (input, model output, tool output) as they happen "
        "(default: pretty with --prompt, otherwise off).",
    )
    parser.add_argument(
        "--skip-venv-if-present",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing per-server virtualenvs instead of setting them up again with uv "
        "(+skip_venv_if_present; default: true). Missing virtualenvs are always set up, which needs uv.",
    )
    before, separator, after = list(argv), [], []
    if "--" in before:
        index = before.index("--")
        before, separator, after = before[:index], ["--"], before[index + 1 :]
    own, rest = parser.parse_known_args(before)
    return own, rest + separator + after


def main(argv: Optional[Sequence[str]] = None) -> int:
    own, gym_args = parse_args(sys.argv[1:] if argv is None else argv)
    cwd = Path.cwd()
    harness_cwd = Path(own.harness_cwd).expanduser().absolute() if own.harness_cwd else cwd
    routed = route_args(gym_args, cwd=cwd)
    has_input = "input" in routed.collect_dests or _has_override(routed.collect, "input_jsonl_fpath")
    if own.prompt is not None and has_input:
        raise SystemExit(f"{PREFIX} use either --prompt or -i/--input, not both")
    if own.check_command is not None and own.prompt is None:
        raise SystemExit(f"{PREFIX} --check-command requires --prompt (for input files, put it in the task rows)")
    if "output" not in routed.collect_dests and not _has_override(routed.collect, "output_jsonl_fpath"):
        raise SystemExit(f"{PREFIX} -o/--output is required")
    if own.prompt is None and not has_input:
        raise SystemExit(f"{PREFIX} -i/--input (or --prompt) is required")
    if not selects_environment(routed):
        routed.start += ["--resources-server", DEFAULT_RESOURCES_SERVER]
        log(f"no environment selected; serving --resources-server {DEFAULT_RESOURCES_SERVER}")

    workdir = gym_workdir(own.gym_workdir)
    output = routed.output
    log_path = output.with_name(f"{output.stem}_servers.log") if output else workdir / "simple_agent_servers.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if own.prompt is not None:
        task_path = output.with_name(f"{output.stem}_task.jsonl") if output else workdir / "simple_agent_task.jsonl"
        write_prompt_task(task_path, own.prompt, own.check_command)
        routed.collect += ["--input", str(task_path)]
        log(f"task written to {task_path}")
    echo = own.echo or ("pretty" if own.prompt is not None else "off")
    echo_path: Optional[Path] = None
    if echo != "off":
        suffix = "_items.jsonl" if echo == "json" else "_items.log"
        echo_path = output.with_name(f"{output.stem}{suffix}") if output else workdir / f"simple_agent{suffix}"
        echo_path.unlink(missing_ok=True)

    port = free_port()
    start_args = [*routed.start, *head_overrides(port)]
    if not _has_override(routed.start, "harness_cwd"):
        start_args.append(f"+harness_cwd={_hydra_string(str(harness_cwd))}")
    if echo_path is not None:
        # simple_agent reads these global keys; explicit overrides on the command line win.
        if not _has_override(routed.start, "simple_agent_echo_items"):
            start_args.append(f"+simple_agent_echo_items={echo}")
        if not _has_override(routed.start, "simple_agent_echo_file"):
            start_args.append(f"+simple_agent_echo_file={_hydra_string(str(echo_path))}")
    uv_found = shutil.which("uv", path=os.environ.get("PATH")) is not None
    if not _has_override(routed.start, "skip_venv_if_present"):
        skip = own.skip_venv_if_present
        start_args.append(f"+skip_venv_if_present={'true' if skip else 'false'}")
        if skip:
            log("reusing existing server venvs (--no-skip-venv-if-present to set them up again)")
        elif not uv_found:
            # Gym sets up every server venv with uv, so this will fail at startup.
            log("warning: --no-skip-venv-if-present sets up every server venv with uv, which is not on PATH")
    collect_args = [*routed.collect, *head_overrides(port)]
    env = dict(os.environ)
    # Gym runs from its own working directory; keep relative --config/-i lookups starting here.
    env[nemo_gym.NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME] = os.pathsep.join(
        filter(None, [str(cwd), env.get(nemo_gym.NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME)])
    )
    gym = gym_executable()

    # Children inherit ignored signals, and background jobs of non-interactive shells start with SIGINT
    # ignored; `gym env start` would then ignore the SIGINT that stops it gracefully.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    log(f"harness cwd: {harness_cwd}")
    log(f"gym workdir: {workdir}")
    log(f"starting servers (head server {HEAD_HOST}:{port}); output in {log_path}")
    with open(log_path, "ab") as server_log:
        start = subprocess.Popen(
            [gym, "env", "start", *start_args],
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # its own process group, so teardown can reach every descendant
        )
    try:
        try:
            wait_until_ready(start, port, timeout_s=own.startup_timeout)
        except RuntimeError as error:
            tail = _tail(log_path)
            log(f"{error}. Last lines of {log_path}:\n{tail}")
            if not uv_found and any(message in tail for message in ("uv: command not found", "uv: not found")):
                log(
                    "a server venv needs to be set up, which requires uv: install it "
                    "(e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`) or put it on PATH"
                )
            return 1
        log("servers ready")

        if "agent" not in routed.collect_dests and not _has_override(routed.collect, "agent_name", "agent_map"):
            agents = served_agents(port)
            if len(agents) == 1:
                log(f"using agent {agents[0]}")
                collect_args += ["--agent", agents[0]]

        command = [gym, "eval", "run", "--no-serve", *collect_args]
        log("collecting: " + " ".join(command[1:]))
        follower = FileFollower(echo_path) if echo_path is not None else None
        if follower is not None:
            log(f"echoing {echo} items (also in {echo_path})")
            follower.start()
        try:
            # With a json echo, stdout carries only the items (JSON Lines); the collection's own
            # output (progress, metrics) goes to stderr.
            collect_stdout = sys.stderr if echo == "json" else None
            collect = subprocess.Popen(command, cwd=workdir, env=env, stdout=collect_stdout)
            code = wait_for_collection(collect)
        finally:
            if follower is not None:
                follower.stop()
        log(f"collection exited with code {code}")
        return code
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    finally:
        log("stopping servers")
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not abandon the teardown
        stop_servers(start)
        log("servers stopped")


if __name__ == "__main__":
    sys.exit(main())
