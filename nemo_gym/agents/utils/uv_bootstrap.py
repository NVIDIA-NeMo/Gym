# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``uv`` binary a harness uploads into a task sandbox, chosen for the sandbox's platform.

An agent host and its task sandboxes need not share a CPU architecture or a C library: an aarch64
login node drives x86_64 pods, and an Alpine image runs musl. A harness that installs a Python
runtime inside the sandbox with ``uv`` therefore asks the sandbox what it is and picks a build that
runs there. The static musl build of ``uv`` runs on glibc and musl alike, so it is the one
downloaded; the host's own binary is reused only when it is known to run in the sandbox.

``uv_for_sandbox`` is the entry point; everything else here is implementation detail. Shared by the
harnesses that install a Python runtime in the sandbox (Hermes today). Harnesses on
Node, such as Codex and Pi, do not acquire a ``uv`` dependency through this module.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


LOG = logging.getLogger(__name__)
# One lock per cache path, so concurrent cold-cache callers in a process share one download.
_DOWNLOAD_LOCKS: dict[Path, asyncio.Lock] = {}

DEFAULT_UV_CACHE_DIR = "~/.cache/nemo_gym/uv"
UV_RELEASE_URL = "https://github.com/astral-sh/uv/releases/download/{version}/uv-{target}.tar.gz"
# One shell line: kernel, machine, C library. musl images ship the loader under /lib/ld-musl-*.so.1.
_PLATFORM_PROBE = (
    "uname -s && uname -m && if ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; else echo gnu; fi"
)
_MACHINES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}


class _Sandbox(Protocol):
    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float | None = None): ...


@dataclass(frozen=True)
class _SandboxPlatform:
    """What ``uname -s``, ``uname -m`` and the C library probe reported inside the sandbox."""

    system: str
    machine: str
    libc: str

    @property
    def arch(self) -> str | None:
        return _MACHINES.get(self.machine)


async def _probe_sandbox_platform(
    sandbox: _Sandbox, workdir: str | None, *, timeout_s: float = 30
) -> _SandboxPlatform:
    """Ask the sandbox for its kernel, machine and C library."""
    result = await sandbox.exec(_PLATFORM_PROBE, cwd=workdir, timeout_s=timeout_s)
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if getattr(result, "return_code", 1) != 0 or len(lines) < 3:
        raise RuntimeError(
            f"Could not determine the sandbox platform: {result.stderr or result.stdout or 'no output'}"
        )
    return _SandboxPlatform(system=lines[0], machine=lines[1], libc=lines[2])


def _uv_release_target(sandbox_platform: _SandboxPlatform) -> str:
    """The uv release target to download for a sandbox: the static musl build for its architecture."""
    if sandbox_platform.system != "Linux":
        raise RuntimeError(f"uv bootstrap supports Linux sandboxes only, not {sandbox_platform.system!r}")
    if sandbox_platform.arch is None:
        raise RuntimeError(f"No uv build is known for sandbox architecture {sandbox_platform.machine!r}")
    return f"{sandbox_platform.arch}-unknown-linux-musl"


def _host_uv_runs_in(sandbox_platform: _SandboxPlatform) -> bool:
    """Whether the host's own ``uv`` runs in the sandbox: a Linux host, the same architecture, and a glibc
    sandbox (the host binary is a gnu build; the musl download covers musl sandboxes)."""
    return (
        platform.system() == "Linux"
        and sandbox_platform.system == "Linux"
        and _MACHINES.get(platform.machine()) == sandbox_platform.arch
        and sandbox_platform.libc == "gnu"
    )


def _host_uv_version(uv_path: str) -> str:
    """``uv --version`` -> ``0.12.3``."""
    output = subprocess.run([uv_path, "--version"], check=True, capture_output=True, text=True).stdout
    parts = output.split()
    if len(parts) < 2 or parts[0] != "uv":
        raise RuntimeError(f"Unexpected uv version output: {output!r}")
    return parts[1]


def _download_uv(version: str, target: str, destination: Path) -> None:
    """Fetch the uv release for ``target`` into ``destination`` (atomic, idempotent)."""
    url = UV_RELEASE_URL.format(version=version, target=target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nemo-gym-uv-", dir=destination.parent) as tmp:
        archive = Path(tmp) / "uv.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        except OSError as exc:
            raise RuntimeError(f"Could not download uv {version} for {target} from {url}: {exc}") from exc
        with tarfile.open(archive, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.isfile() and m.name.endswith("/uv")), None)
            if member is None:
                raise RuntimeError(f"uv release archive {url} has no uv binary")
            tar.extract(member, tmp, filter="data")
            extracted = Path(tmp) / member.name
        extracted.chmod(0o755)
        os.replace(extracted, destination)


async def uv_for_sandbox(sandbox: _Sandbox, workdir: str | None, cache_dir: str = DEFAULT_UV_CACHE_DIR) -> Path:
    """The uv binary to upload into ``sandbox``.

    The host's binary when it is known to run there, else the static musl build of the host's uv
    version for the sandbox's architecture, fetched once into ``cache_dir``.
    """
    host_uv = shutil.which("uv")
    if host_uv is None:
        raise RuntimeError("Installing a sandbox runtime requires uv on the agent host")
    try:
        sandbox_platform = await _probe_sandbox_platform(sandbox, workdir)
    except RuntimeError as exc:
        # A sandbox that cannot answer the probe gets the host's binary, as before the probe existed.
        LOG.warning("%s; uploading the host's uv", exc)
        return Path(host_uv)
    target = _uv_release_target(sandbox_platform)
    if _host_uv_runs_in(sandbox_platform):
        return Path(host_uv)
    version = await asyncio.to_thread(_host_uv_version, host_uv)
    cached = Path(cache_dir).expanduser() / version / target / "uv"
    if cached.is_file():
        return cached
    # Many rollouts start at once on a cold cache: the first caller downloads, the rest wait and
    # find the file. Atomic replacement in _download_uv covers other processes filling the same path.
    async with _DOWNLOAD_LOCKS.setdefault(cached, asyncio.Lock()):
        if not cached.is_file():
            await asyncio.to_thread(_download_uv, version, target, cached)
    return cached
