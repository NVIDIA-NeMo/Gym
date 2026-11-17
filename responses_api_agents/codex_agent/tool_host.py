# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Codex's tool commands in a pid namespace nested inside the task sandbox.

Without it, the commands a model runs share one pid namespace with the harness: the process
supervisor, the Codex runner and the Codex CLI all run as the same user, so ``pkill -f
python``, ``pkill -f node`` or ``kill -9 -1`` from a tool command ends the episode.

Codex (0.144) can run every tool command (``exec_command``/``write_stdin`` sessions, the
shell tool, ``apply_patch`` and file reads) through an exec-server it spawns over stdio, named
by ``$CODEX_HOME/environments.toml``. The tool host is that exec-server, started under a
nested ``bwrap``: same filesystem, devices and network, but a new pid namespace with a fresh
``/proc`` in which the exec-server is pid 1. Tool commands can then see and signal only each
other and the tool host. As namespace init the tool host ignores SIGKILL and every signal it
does not handle from inside, and ``kill -1`` skips it.

A sandboxed root can create this namespace only if it can map uid 0, which needs
CAP_SETFCAP (Linux >= 5.12); see the bwrap provider's ``allow_nested_userns``.
"""

import json
import shlex


TOOL_HOST_ENVIRONMENT_ID = "nemo-gym-tools"
# Name of the exec-server process in the tool namespace: no generic word such as node or python.
TOOL_HOST_PROCESS_NAME = "nemo-gym-tool-host"
TOOL_HOST_BWRAP = "bwrap"
PROBE_VARIABLE = "NEMO_GYM_TOOL_HOST_PROBE"


def tool_host_paths(session_dir: str) -> tuple[str, str]:
    """The uploaded bwrap and the tool-host link to the native Codex binary, in the session's bin/."""
    return f"{session_dir}/bin/{TOOL_HOST_BWRAP}", f"{session_dir}/bin/{TOOL_HOST_PROCESS_NAME}"


def tool_namespace_args() -> list[str]:
    """bwrap options for the tool namespace: everything as the harness sees it, but the pid namespace."""
    return [
        "--unshare-user",
        "--unshare-pid",
        "--as-pid-1",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        # Recursive, keeping each mount's flags: read-only stays read-only, /dev keeps its nodes.
        "--dev-bind",
        "/",
        "/",
        "--proc",
        "/proc",
    ]


def tool_host_argv(session_dir: str) -> list[str]:
    bwrap, tool_host = tool_host_paths(session_dir)
    return [bwrap, *tool_namespace_args(), "--", tool_host, "exec-server", "--listen", "stdio"]


def environments_toml(session_dir: str) -> str:
    """``$CODEX_HOME/environments.toml``: one stdio exec-server environment, no local one."""
    program, *args = tool_host_argv(session_dir)
    return (
        f"default = {json.dumps(TOOL_HOST_ENVIRONMENT_ID)}\n"
        "include_local = false\n\n"
        "[[environments]]\n"
        f"id = {json.dumps(TOOL_HOST_ENVIRONMENT_ID)}\n"
        f"program = {json.dumps(program)}\n"
        f"args = [{', '.join(json.dumps(arg) for arg in args)}]\n"
    )


def codex_config_overrides() -> dict:
    """config.toml entries that keep the model-visible tools identical to the local environment.

    Without a local environment Codex adds an orchestrator ``skills`` tool namespace.
    """
    return {"orchestrator": {"skills": {"enabled": False}}}


def tool_host_setup_command(session_dir: str, runtime: str) -> str:
    """Link the tool host to the installed native Codex binary, then prove the namespace works.

    The probe must run as pid 1 of a new pid namespace and see itself as pid 1 in /proc (a
    /proc left over from the sandbox would show the sandbox's pid 1), then run the tool host.
    """
    bwrap, tool_host = tool_host_paths(session_dir)
    probe = (
        f'[ "$$" = 1 ] || {{ echo "tool namespace: not pid 1" >&2; exit 1; }}; '
        f'tr "\\0" "\\n" < /proc/1/environ | grep -qx {PROBE_VARIABLE}=1 '
        f'|| {{ echo "tool namespace: /proc is not its own" >&2; exit 1; }}; '
        f"exec {shlex.quote(tool_host)} --version"
    )
    return (
        "set -eu; "
        f"host=$(find {shlex.quote(runtime + '/codex/node_modules')} -path '*/vendor/*-unknown-linux-*/bin/codex' "
        "-type f | head -n 1); "
        f'[ -n "$host" ] || {{ echo "native Codex binary not found under {runtime}/codex" >&2; exit 1; }}; '
        f"chmod 0755 {shlex.quote(bwrap)}; "
        f'ln -sfn "$host" {shlex.quote(tool_host)}; '
        f"{shlex.join([bwrap, *tool_namespace_args(), '--setenv', PROBE_VARIABLE, '1', '--', '/bin/sh', '-c', probe])}"
    )


def setup_failure_hint(stderr: str) -> str:
    if "uid map" in stderr:
        return (
            " The sandbox's root cannot map uid 0 into a nested user namespace: keep CAP_SETFCAP in the"
            " sandbox (bwrap provider: allow_nested_userns: true)."
        )
    if "proc" in stderr:
        return " The sandbox's /proc cannot be remounted in a nested pid namespace (is part of it covered?)."
    return ""
