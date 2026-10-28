# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode filesystem isolation for OpenCode and its image tools."""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class EvalSandboxConfig(BaseModel):
    """Preinstalled, read-only dependencies required by an isolated episode."""

    bwrap_command: list[str] = Field(min_length=1)
    python_env: Path
    plugin_dependencies: Path
    workspace_tmp: bool = Field(
        default=False,
        description="Bind the episode's workspace/tmp at /tmp so relative and absolute temporary paths agree.",
    )


def sandbox_command(
    runtime: EvalSandboxConfig, workspace: Path, opencode: Path, env: dict[str, str], command: list[str]
) -> list[str]:
    if runtime.plugin_dependencies is None:
        raise ValueError("All OpenCode episodes require shared plugin dependencies")
    package = runtime.plugin_dependencies / "node_modules/@opencode-ai/plugin/package.json"
    if not package.is_file():
        raise ValueError(f"Missing preinstalled OpenCode plugin dependencies: {package}")
    args = list(runtime.bwrap_command)
    args += [
        "--unshare-all",
        "--share-net",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for source in (
        "/usr",
        "/bin",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/alternatives",
        "/etc/ssl",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
    ):
        if Path(source).exists():
            args += ["--ro-bind", source, source]
    python_root = (runtime.python_env / "bin/python").resolve().parents[1]
    if python_root.parent.name == "python":
        python_root = python_root.parent
    python_roots = [runtime.python_env, python_root]
    venv_config = runtime.python_env / "pyvenv.cfg"
    if venv_config.is_file():
        configuration = dict(line.split("=", 1) for line in venv_config.read_text().splitlines() if "=" in line)
        home = next(
            (value.strip() for key, value in configuration.items() if key.strip() == "home"),
            None,
        )
        if home:
            # Preserve the lexical path too: symlinked runtime directories
            # may differ from the canonical interpreter mount above.
            python_home = Path(home).absolute()
            if python_home.name != "bin":
                raise ValueError(f"Expected a Linux Python bin directory: {home}")
            python_roots.append(python_home.parent)
    for root in dict.fromkeys(python_roots):
        if not root.is_relative_to("/usr"):
            args += ["--ro-bind", str(root), str(root)]
    (workspace / ".passwd").write_text(f"agent:x:{os.getuid()}:{os.getgid()}:Agent:/home/agent:/bin/bash\n")
    (workspace / ".group").write_text(f"agent:x:{os.getgid()}:\n")
    args += [
        "--ro-bind",
        str(opencode),
        "/agent-bin/opencode",
        "--ro-bind",
        str(workspace / ".passwd"),
        "/etc/passwd",
        "--ro-bind",
        str(workspace / ".group"),
        "/etc/group",
        "--bind",
        str(workspace),
        "/testbed",
        # Pyxis masks parts of procfs, so nested user namespaces cannot mount
        # a fresh instance. Share its read-only view; /tmp and /testbed remain
        # episode-local. This prevents filename collisions, not hostile access
        # through another process's procfs entries.
        "--ro-bind",
        "/proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    if runtime.workspace_tmp:
        # Match the sandbox path contract without sharing files between episodes.
        temporary_dir = workspace / "tmp"
        temporary_dir.mkdir(exist_ok=True)
        args += ["--bind", str(temporary_dir), "/tmp"]
    else:
        args += ["--tmpfs", "/tmp"]
    args += [
        "--dir",
        "/home/agent",
        "--chdir",
        "/testbed",
        "--clearenv",
    ]
    # Stock prompts also initialize this installation. Share it for every
    # episode so a default-prompt run cannot exhaust the filesystem's inodes.
    args += [
        "--ro-bind",
        str(runtime.plugin_dependencies.resolve()),
        "/testbed/.config/opencode",
    ]
    environment = {
        "HOME": "/home/agent",
        "PATH": f"{runtime.python_env}/bin:/agent-bin:/usr/local/bin:/usr/bin:/bin",
        "SHELL": "/bin/bash",
        "PWD": "/testbed",
        "XDG_DATA_HOME": "/testbed/.opencode-data",
        "XDG_CONFIG_HOME": "/testbed/.config",
        "XDG_CACHE_HOME": "/testbed/.cache",
        "XDG_STATE_HOME": "/testbed/.state",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
        "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"],
        "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"],
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    if runtime.workspace_tmp:
        environment.update(TMPDIR="/testbed/tmp", TMP="/testbed/tmp", TEMP="/testbed/tmp")
    for key, value in environment.items():
        args += ["--setenv", key, value]
    return args + command
