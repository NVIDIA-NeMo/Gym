# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Python harness's private runtime, built on the Agent Server host and uploaded into the task sandbox.

The sandbox needs no network, no ``uv`` and no Python of its own. For each sandbox platform, the host downloads a
pinned python-build-standalone CPython and checks its SHA-256, then installs the harness's requirements for that
platform with ``uv`` into a separate site-packages archive. Packages are pinned to the versions installed in this
Agent Server's environment, so the sandbox runs the versions the adapter is tested with. Requirements given as
direct URLs, such as a harness pinned to a commit, are built from source on the host; every other package must
publish a wheel for the sandbox's platform. Both archives are cached on the host.

In the sandbox, ``install_python_runtime.sh`` unpacks the archives in a private directory, checks that the runtime
imports the harness, and renames it into place, so a concurrent seed never sees a partial runtime. The runtime
lives under ``/tmp``, outside the task's working directory and ``HOME``, and every command runs from ``/``.
"""

import asyncio
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shlex
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from aiohttp import ClientTimeout
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import request


LOG = logging.getLogger(__name__)

PYTHON_VERSION = "3.13.14"
_PYTHON_RELEASE = "20260805"
# Published with the immutable python-build-standalone release:
# https://github.com/astral-sh/python-build-standalone/releases/tag/20260805
_PYTHON_SHA256 = {
    "x86_64/gnu": "39e82d05926bdcd206732026bcd878d9f00e288cd227d91c1fabf379b6ea4fa5",  # pragma: allowlist secret
    "aarch64/gnu": "f1cda716f5595a97e79f760f81da7150e4f6b4ae9dc49968b4ce5d6d64bba058",  # pragma: allowlist secret
    "x86_64/musl": "c6b77767b2c97ce77c21496909396b6e056b96220b2cda45f2c93b654376dd93",  # pragma: allowlist secret
    "aarch64/musl": "960738ebde53abeceb5452b71c2217d50452023e3baf2439dc57290d07930637",  # pragma: allowlist secret
}
_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/{release}/"
    "cpython-{version}%2B{release}-{arch}-unknown-linux-{libc}-install_only_stripped.tar.gz"
)
_SITE_PACKAGES = f"python/lib/python{PYTHON_VERSION.rsplit('.', 1)[0]}/site-packages"
CACHE_DIR = Path("~/.cache/nemo_gym/sandbox_python").expanduser()
_INSTALLER = Path(__file__).with_name("install_python_runtime.sh")
# Kernel, machine, then "musl" or the glibc version. musl images ship the loader under /lib/ld-musl-*.so.1.
_PLATFORM_PROBE = (
    "uname -s; uname -m; if ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; "
    "else getconf GNU_LIBC_VERSION 2>/dev/null || ldd --version 2>&1 | head -n 1; fi"
)
_MACHINES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
# glibc 2.17 is the oldest python-build-standalone supports; uv names these manylinux levels in --python-platform.
_MANYLINUX_LEVELS = (40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 28, 17)
# One lock per cached archive, so concurrent cold-cache seeds in this process share one build.
_BUILD_LOCKS: dict[Path, asyncio.Lock] = {}


@dataclass(frozen=True)
class SandboxPlatform:
    """The sandbox's CPU architecture and C library, which select the Python build and the wheels."""

    arch: str
    libc: str
    glibc_minor: int | None = None

    @property
    def wheel_platform(self) -> str:
        """The ``uv pip --python-platform`` value whose wheels run in the sandbox."""
        if self.libc == "musl":
            return f"{self.arch}-unknown-linux-musl"
        # Without a reported version, assume the oldest glibc the interpreter runs on.
        minor = 17 if self.glibc_minor is None else self.glibc_minor
        level = next((level for level in _MANYLINUX_LEVELS if level <= minor), None)
        if level is None:
            raise RuntimeError(f"The sandbox's glibc 2.{minor} is older than 2.17, which the sandbox Python needs")
        return f"{self.arch}-manylinux_2_{level}"


async def probe_sandbox_platform(sandbox: AsyncSandbox) -> SandboxPlatform:
    """Ask the sandbox for its kernel, machine and C library."""
    result = await sandbox.exec(_PLATFORM_PROBE, cwd="/", timeout_s=30)
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.return_code != 0 or result.error_type is not None or len(lines) < 3:
        raise RuntimeError(
            f"Could not determine the sandbox platform: {result.error_type or result.stderr or result.stdout}"
        )
    system, machine, libc = lines[:3]
    arch = _MACHINES.get(machine)
    if system != "Linux" or arch is None:
        raise RuntimeError(f"No sandbox Python runtime is available for {system} on {machine}")
    if libc == "musl":
        return SandboxPlatform(arch=arch, libc="musl")
    versions = re.findall(r"\b2\.(\d+)\b", libc)
    return SandboxPlatform(arch=arch, libc="gnu", glibc_minor=int(versions[-1]) if versions else None)


@functools.cache
def _agent_environment_pins() -> tuple[str, ...]:
    """``name==version`` for each package this server's environment installed from an index.

    Passed to uv as constraints, which keep the extras that dependencies request (an override would replace
    ``lxml[html_clean]`` with plain ``lxml``). Packages installed from a URL or a path, including editable ones,
    are left free.
    """
    pins: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name and distribution.read_text("direct_url.json") is None:
            pins.setdefault(canonicalize_name(name), f"{name}=={distribution.version}")
    return tuple(sorted(pins.values()))


async def _cached(path: Path, build: Callable[[], Awaitable[None]]) -> Path:
    """``path``, built once per host. Builders write atomically, so other processes see it whole or not at all."""
    async with _BUILD_LOCKS.setdefault(path, asyncio.Lock()):
        if not path.is_file():
            await build()
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


async def _python_archive(platform: SandboxPlatform) -> Path:
    """The pinned python-build-standalone archive for ``platform``, checked against its published SHA-256."""
    expected = _PYTHON_SHA256.get(f"{platform.arch}/{platform.libc}")
    if expected is None:
        raise RuntimeError(f"No pinned Python build for {platform.arch}/{platform.libc}")
    url = _PYTHON_URL.format(release=_PYTHON_RELEASE, version=PYTHON_VERSION, arch=platform.arch, libc=platform.libc)
    path = CACHE_DIR / url.rsplit("/", 1)[1].replace("%2B", "+")

    async def download() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{uuid4().hex}")
        try:
            async with await request("GET", url, timeout=ClientTimeout(total=600)) as response:
                response.raise_for_status()
                partial.write_bytes(await response.read())
            if await asyncio.to_thread(_sha256, partial) != expected:
                raise RuntimeError(f"Checksum mismatch for {url}")
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)

    await _cached(path, download)
    # The cache outlives this process, so check the bytes that are about to be uploaded.
    if await asyncio.to_thread(_sha256, path) != expected:
        raise RuntimeError(f"Cached Python build {path} does not match its pinned checksum; delete it to retry")
    return path


def _source_built(requirements: Iterable[str]) -> list[str]:
    """Names of the requirements given as direct URLs; uv builds these from source on the host."""
    return [parsed.name for parsed in map(Requirement, requirements) if parsed.url]


async def _run(argv: list[str], *, cwd: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode != 0:
        output = (stderr or stdout).decode(errors="replace")
        raise RuntimeError(f"{argv[0]} exited with {process.returncode}: {output[-4000:]}")


def _write_archive(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="python")


@dataclass(frozen=True)
class SandboxPythonRuntime:
    """A harness's private Python in the sandbox: the pinned CPython with ``requirements`` installed.

    ``import_check`` is Python source that runs without error only in a usable runtime, for example
    ``"import run_agent, mcp"``. ``overrides`` are the uv overrides the adapter's own environment is installed with,
    such as the lines of its ``overrides.txt``.
    """

    harness: str
    requirements: tuple[str, ...]
    import_check: str
    overrides: tuple[str, ...] = ()

    @functools.cached_property
    def directory(self) -> str:
        """Sandbox directory keyed by everything that goes into the runtime.

        A runtime found there, left by an earlier session in this sandbox or baked into the image, was built from the
        same inputs and is reused. The adapter may keep its own harness files here, beside ``python/``.
        """
        inputs = [PYTHON_VERSION, _PYTHON_RELEASE, self.requirements, self.overrides, _agent_environment_pins()]
        key = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:16]
        return f"/tmp/nemo-gym-{self.harness}-runtime-{key}"

    @property
    def python(self) -> str:
        return f"{self.directory}/python/bin/python3"

    async def is_installed(self, sandbox: AsyncSandbox) -> bool:
        """Whether the runtime in the sandbox passes ``import_check``. The existence of the directory proves nothing.

        Like the installer's checks, it runs with ``HOME`` in the runtime directory, because some harnesses write
        their state under ``HOME`` when imported.
        """
        home = shlex.quote(f"{self.directory}/home")
        check = await sandbox.exec(
            f"mkdir -p {home} && HOME={home} {shlex.quote(self.python)} -I -c {shlex.quote(self.import_check)}",
            cwd="/",
            timeout_s=120,
        )
        return check.return_code == 0 and check.error_type is None

    async def install(self, sandbox: AsyncSandbox, *, timeout_s: float) -> None:
        """Build the runtime for the sandbox's platform on this host if needed, then upload and publish it.

        ``timeout_s`` bounds the whole installation, including a build that another seed started.
        """
        async with asyncio.timeout(timeout_s):
            platform = await probe_sandbox_platform(sandbox)
            python_archive = await _python_archive(platform)
            site_packages_archive = await self._site_packages_archive(platform)
            setup = f"{self.directory}/.setup-{uuid4().hex}"
            prepared = await sandbox.exec(f"mkdir -p {shlex.quote(setup)}", cwd="/", timeout_s=30)
            if prepared.return_code != 0 or prepared.error_type is not None:
                raise RuntimeError(prepared.stderr or prepared.stdout or f"Could not create {setup} in the sandbox")
            await sandbox.upload(python_archive, f"{setup}/python.tar.gz")
            await sandbox.upload(site_packages_archive, f"{setup}/site-packages.tar.gz")
            await sandbox.upload(_INSTALLER, f"{setup}/{_INSTALLER.name}")
            command = shlex.join(["bash", f"{setup}/{_INSTALLER.name}", self.directory, setup, self.import_check])
            result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s)
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(
                f"Installing the {self.harness} Python runtime failed with {result.error_type or result.return_code}: "
                f"{(result.stderr or '')[-2000:]}{(result.stdout or '')[-2000:]}"
            )

    async def _site_packages_archive(self, platform: SandboxPlatform) -> Path:
        path = CACHE_DIR / f"{Path(self.directory).name}-{platform.wheel_platform}.tar.gz"
        return await _cached(path, lambda: self._build_site_packages(platform, path))

    async def _build_site_packages(self, platform: SandboxPlatform, destination: Path) -> None:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("Building a sandbox Python runtime requires uv on the Agent Server host")
        LOG.info("Building the %s sandbox runtime for %s into %s", self.harness, platform.wheel_platform, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".build-", dir=destination.parent) as scratch:
            scratch = Path(scratch)
            constraints = scratch / "constraints.txt"
            constraints.write_text("".join(f"{pin}\n" for pin in _agent_environment_pins()))
            argv = [
                uv,
                "pip",
                "install",
                # Configuration files never shape the install; the host's UV_* environment, such as an index mirror,
                # still applies.
                "--no-config",
                "--python",
                sys.executable,
                "--python-version",
                PYTHON_VERSION,
                "--python-platform",
                platform.wheel_platform,
                "--target",
                str(scratch / _SITE_PACKAGES),
                "--constraints",
                str(constraints),
                "--only-binary",
                ":all:",
            ]
            if self.overrides:
                overrides = scratch / "overrides.txt"
                overrides.write_text("".join(f"{line}\n" for line in self.overrides))
                argv += ["--overrides", str(overrides)]
            for name in _source_built(self.requirements):
                argv += ["--no-binary", name]
            await _run([*argv, *self.requirements], cwd=scratch)
            archive = scratch / "site-packages.tar.gz"
            await asyncio.to_thread(_write_archive, scratch / "python", archive)
            os.replace(archive, destination)
