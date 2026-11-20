# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers shared by the built-in sandbox providers.

These were previously duplicated verbatim across the enroot, apptainer, and
opensandbox providers. They live here so there is a single implementation.
"""

import asyncio
import contextlib
import os
import posixpath
import signal
from collections.abc import Mapping, Sequence
from typing import Any

import anyio

from nemo_gym.sandbox.providers.base import SandboxExecArgvUnsupportedError


# How long a cancelled CLI command's process may take to be reaped after it is killed.
REAP_TIMEOUT_S = 5.0


def coerce_config(value: Any, config_cls: type[Any]) -> Any:
    """Accept either a config dataclass instance or a plain mapping (Hydra YAML)."""
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


def path_under_mount(mount_point: str, path: str) -> str | None:
    """If `path` is inside the mount, return its path relative to the mount; else None."""
    if not path.startswith("/"):
        return None
    mp = posixpath.normpath(mount_point.rstrip("/") or "/")
    normalized = posixpath.normpath(path)
    if normalized == mp:
        return ""
    try:
        if posixpath.commonpath([mp, normalized]) != mp:
            return None
    except ValueError:
        return None
    if mp == "/":
        return normalized.lstrip("/")
    return normalized[len(mp) + 1 :]


async def kill_cancelled_command(proc: asyncio.subprocess.Process) -> None:
    """Kill the process group of a CLI command whose wait was cancelled, then reap it.

    A provider runner calls this when the await on its command is cancelled, for example by the time limit of
    :meth:`AsyncSandbox.stop`, and then re-raises the cancellation. Without it, only the runner's own command
    timeout kills the process, so a cancelled call leaves the command running. The reap is shielded from the
    cancellation that is still in force and bounded by ``REAP_TIMEOUT_S``.
    """
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with anyio.CancelScope(shield=True), anyio.move_on_after(REAP_TIMEOUT_S), contextlib.suppress(Exception):
        await proc.wait()


def require_shell_text(command: str | Sequence[str], *, provider: str) -> str:
    """``command`` as the shell text a provider without ``SupportsSandboxExecArgv`` runs.

    An argument list raises ``SandboxExecArgvUnsupportedError`` instead of being joined into shell text: its
    caller relies on no shell running. ``AsyncSandbox.exec`` refuses one first; this covers direct provider calls.
    """
    if isinstance(command, str):
        return command
    raise SandboxExecArgvUnsupportedError(
        f"Sandbox provider {provider!r} cannot run an argument list without a shell; pass shell text instead"
    )
