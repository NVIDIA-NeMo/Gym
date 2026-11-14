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
"""A stand-in ``docker`` package so the upstream BaxBench code runs unchanged without a Docker daemon.

Upstream builds an image per sample and drives a container through the Docker SDK (exec, archives, stats).
Here the app runs as a process inside the grading sandbox, so ``containers.get(id)`` returns a
``LocalContainer`` that does the same operations on the local filesystem and process table.
``install`` must run before any upstream module is imported: ``env/base.py`` calls ``docker.from_env()``
at import time. This file is uploaded into the sandbox next to ``sandbox_runner.py`` and stays stdlib-only.
"""

import io
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import types
from collections import namedtuple
from pathlib import Path
from typing import Iterator


ExecResult = namedtuple("ExecResult", "exit_code,output")


class APIError(Exception):
    pass


class NotFound(APIError):
    pass


class DockerException(Exception):
    pass


def session_pids(sid: int) -> list[int]:
    """PIDs of every live process in session ``sid`` (the app was started as a session leader)."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        # Fields after the parenthesised command: state ppid pgrp session ...
        fields = stat[stat.rfind(")") + 2 :].split()
        if fields[0] != "Z" and int(fields[3]) == sid:
            pids.append(int(entry))
    return pids


def session_rss_bytes(sid: int) -> int:
    total = 0
    for pid in session_pids(sid):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except OSError:
            continue
    return total


def kill_session(sid: int) -> None:
    for pid in session_pids(sid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def live_pids() -> set[int]:
    """PIDs of every process that has not exited (zombies left out)."""
    pids = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        if stat[stat.rfind(")") + 2] != "Z":
            pids.add(int(entry))
    return pids


def kill_new_processes(baseline: set[int], *, list_pids=live_pids, passes: int = 3) -> None:
    """Kill every process missing from ``baseline`` except this one. Only for a single-use grading sandbox,
    where nothing but the runner and what it started can be new; repeat to catch late forks."""
    for _ in range(passes):
        new = list_pids() - baseline - {os.getpid()}
        if not new:
            return
        for pid in new:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


# Set in the environment of every build step and of the app, so processes that leave the app's session
# (a daemon started with setsid) can still be found and stopped before the result is written.
MARKER_VARIABLE = "BAXBENCH_GRADE_TOKEN"


def kill_marked(token: str, *, passes: int = 3) -> None:
    """Kill every process whose environment carries ``MARKER_VARIABLE=token``; repeat to catch late forks."""
    needle = f"{MARKER_VARIABLE}={token}".encode()
    for _ in range(passes):
        marked = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) == os.getpid():
                continue
            try:
                environ = Path(f"/proc/{entry}/environ").read_bytes()
            except OSError:
                continue
            if needle in environ.split(b"\0"):
                marked.append(int(entry))
        if not marked:
            return
        for pid in marked:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class LocalContainer:
    """The Docker SDK ``Container`` calls upstream makes, applied to the app process and the local filesystem."""

    def __init__(self, container_id: str, *, app_sid: int, workdir: str, env: dict[str, str], log_path: Path) -> None:
        self.id = container_id
        self.name = container_id
        self.app_sid = app_sid
        self.workdir = workdir
        self.env = env
        self.log_path = log_path

    def running(self) -> bool:
        return bool(session_pids(self.app_sid))

    def exec_run(self, cmd: str | list[str], stdout: bool = True, stderr: bool = True, **_: object) -> ExecResult:
        # The Docker SDK splits a string command with shlex and runs it without a shell, in the image WORKDIR.
        argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        try:
            proc = subprocess.run(
                argv,
                cwd=self.workdir,
                env=self.env,
                stdout=subprocess.PIPE if stdout else subprocess.DEVNULL,
                stderr=(subprocess.STDOUT if stdout else subprocess.PIPE) if stderr else subprocess.DEVNULL,
                timeout=600,
            )
        except FileNotFoundError as e:
            # Docker reports a missing executable as exit 127 with the runtime's message on the stream.
            return ExecResult(127, f"exec failed: {e}".encode())
        return ExecResult(proc.returncode, proc.stdout if stdout else proc.stderr or b"")

    def put_archive(self, path: str, data: bytes) -> bool:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            tar.extractall(path, filter="fully_trusted")
        return True

    def get_archive(self, path: str) -> tuple[Iterator[bytes], dict[str, object]]:
        target = Path(path)
        if not target.exists():
            raise NotFound(f"Could not find the file {path} in container {self.id}")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.add(str(target), arcname=target.name)
        return iter([buffer.getvalue()]), {"name": target.name, "size": target.stat().st_size}

    def stats(self, stream: bool = False, **_: object) -> dict[str, object]:
        # A stopped container reports no memory usage; upstream reads the missing key as a crash.
        if not self.running():
            return {"memory_stats": {}}
        return {"memory_stats": {"usage": session_rss_bytes(self.app_sid)}}

    def logs(self, stdout: bool = True, stderr: bool = True, follow: bool = False, **_: object) -> bytes:
        try:
            return self.log_path.read_bytes()
        except OSError:
            return b""

    def remove(self, force: bool = False, **_: object) -> None:
        kill_session(self.app_sid)

    def kill(self, **_: object) -> None:
        kill_session(self.app_sid)


class _Containers:
    def __init__(self, registry: dict[str, LocalContainer]) -> None:
        self._registry = registry

    def get(self, container_id: str) -> LocalContainer:
        try:
            return self._registry[container_id]
        except KeyError as e:
            raise NotFound(f"No such container: {container_id}") from e


class LocalClient:
    def __init__(self) -> None:
        self.registry: dict[str, LocalContainer] = {}
        self.containers = _Containers(self.registry)


class _NoDaemonClient:
    """Lets upstream modules import for prompt building; any container operation fails loudly."""

    @property
    def containers(self) -> object:
        raise DockerException("BaxBench prompt building has no Docker daemon")

    @property
    def images(self) -> object:
        raise DockerException("BaxBench prompt building has no Docker daemon")


def install(client: object | None = None) -> object:
    """Register the ``docker`` modules upstream imports; ``docker.from_env()`` returns ``client``."""
    client = client if client is not None else _NoDaemonClient()
    docker = types.ModuleType("docker")
    errors = types.ModuleType("docker.errors")
    models = types.ModuleType("docker.models")
    containers = types.ModuleType("docker.models.containers")
    errors.APIError = APIError
    errors.NotFound = NotFound
    errors.DockerException = DockerException
    containers.Container = LocalContainer
    containers.ExecResult = ExecResult
    docker.errors = errors
    docker.models = models
    models.containers = containers
    docker.from_env = lambda **_: client
    docker.DockerClient = type(client)
    sys.modules.update(
        {"docker": docker, "docker.errors": errors, "docker.models": models, "docker.models.containers": containers}
    )
    return client
