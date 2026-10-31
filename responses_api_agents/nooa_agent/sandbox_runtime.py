# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package and stage a private NOOA runtime for the target sandbox."""

import asyncio
import gzip
import hashlib
import io
import json
import logging
import re
import shlex
import shutil
import tarfile
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from uuid import uuid4
from weakref import WeakKeyDictionary

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from nemo_gym.sandbox import AsyncSandbox


LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")
_IO_LIMITS: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()


async def _runtime_io(function: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    """Bound archive I/O and join cancelled workers before removing their files."""
    limit = _IO_LIMITS.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(2))
    async with limit:
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            joined = asyncio.gather(worker, return_exceptions=True)
            while not joined.done():
                try:
                    await asyncio.shield(joined)
                except asyncio.CancelledError:
                    continue
            raise


def _stage_installer(source: bytes, local: Path) -> Path:
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
        member = archive.extractfile(_INSTALLER_PATH)
        assert member is not None
        installer = local / "install_nooa_runtime.sh"
        installer.write_bytes(member.read())
    return installer


def _stage_runtime_artifacts(
    local: Path,
    *,
    source: bytes,
    python_source: Path,
    python_sha256: str,
    wheels_source: Path | None,
    wheels_sha256: str | None,
) -> str:
    from responses_api_agents.nooa_agent.offline_runtime import file_sha256

    python_archive = local / "python.tar.gz"
    shutil.copyfile(python_source, python_archive)
    if file_sha256(python_archive) != python_sha256:
        raise RuntimeError("NOOA Python artifact checksum mismatch before sandbox upload")
    if wheels_source is not None:
        wheels_archive = local / "wheels.tar.gz"
        shutil.copyfile(wheels_source, wheels_archive)
        if file_sha256(wheels_archive) != wheels_sha256:
            raise RuntimeError("NOOA wheelhouse changed before sandbox upload")
    (local / "source.tar.gz").write_bytes(source)
    return hashlib.sha256(source).hexdigest()


_RUNTIME_ROOT = "/opt/nemo-gym-nooa"
_PYTHON_VERSION = "3.13.14"
_PYTHON_RELEASE = "20260805"
_INSTALLER_PATH = "responses_api_agents/nooa_agent/install_nooa_runtime.sh"
# Expected digests published with the immutable Astral PBS 20260805 release.
# https://github.com/astral-sh/python-build-standalone/releases/tag/20260805
_PYTHON_SHA256 = {
    "x86_64/gnu": "9f9ebf776b2d4913fe5a91508ab42e6ce17d266138f76a7782c05fa1c3cba1d1",  # pragma: allowlist secret
    "aarch64/gnu": "4777d7df2edb47b96e53abad5e1b9df1b2a1a9b2f7bdba12b5c0122163b3fed9",  # pragma: allowlist secret
    "x86_64/musl": "0c3e9eb834452e6442800ab9cabf88616886c4f1d55163a4da675ec3bcb4f27c",  # pragma: allowlist secret
    "aarch64/musl": "038e5e8ce956a39fed3cb81471258503e3a9d87cb4f8b7c0dc00eb8a5031a8f6",  # pragma: allowlist secret
}
_IMPORT_CHECK = (
    "import sys; assert sys.version_info[:3] == (3,13,14); "
    "import nooa, nooa_cli; from nooa_bench.bench_agent import BenchAgent; "
    "import responses_api_agents.nooa_agent.sandbox_entrypoint; "
    "from responses_api_agents.nooa_agent.resource_tools import ResourceToolDispatcher; "
    "from responses_api_agents.nooa_agent.observability import GymTraceHooks"
)


def _runtime_archive(root: Path) -> tuple[bytes, str]:
    """Include current source, never repository metadata, caches, or user artifacts."""
    files = {name: (root / name).read_bytes() for name in ("README.md", "LICENSE")}
    template = (root / "responses_api_agents/nooa_agent/runtime/pyproject.toml").read_text()
    runtime_config = tomllib.loads(template)["tool"]["nemo-gym-runtime"]
    core_dependencies = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    core_by_name = {canonicalize_name(Requirement(dependency).name): dependency for dependency in core_dependencies}
    try:
        dependencies = [core_by_name[canonicalize_name(name)] for name in runtime_config["core-dependencies"]]
    except KeyError as error:
        raise ValueError(
            f"NOOA runtime dependency {error.args[0]!r} is missing from the root pyproject.toml"
        ) from error
    dependencies.extend(runtime_config["additional-dependencies"])
    files["pyproject.toml"] = template.replace(
        "dependencies = []", f"dependencies = {json.dumps(dependencies)}", 1
    ).encode()
    for path in (root / "nemo_gym").rglob("*.py"):
        if not path.is_symlink() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    agent_dir = root / "responses_api_agents/nooa_agent"
    files[_INSTALLER_PATH] = (root / _INSTALLER_PATH).read_bytes()
    for path in agent_dir.glob("*.py"):
        if not path.is_symlink():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    requirements = (agent_dir / "requirements.txt").read_text()
    runtime_requirements = []
    for line in requirements.splitlines():
        if not line.strip() or line.startswith(("#", "-e ")):
            continue
        # Same immutable source pin, without a dependency on the task image's git.
        line = re.sub(
            r"git\+https://github\.com/([^@ ]+)\.git@([0-9a-f]{40})$",
            r"https://github.com/\1/archive/\2.tar.gz",
            line,
        )
        runtime_requirements.append(line)
    files["runtime-requirements.txt"] = ("\n".join(runtime_requirements) + "\n").encode()
    digest = hashlib.sha256(
        json.dumps([_PYTHON_VERSION, _PYTHON_RELEASE, _PYTHON_SHA256, _IMPORT_CHECK], sort_keys=True).encode()
    )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, contents in sorted(files.items()):
                digest.update(name.encode() + b"\0" + contents + b"\0")
                entry = tarfile.TarInfo(name)
                entry.size = len(contents)
                entry.mode = 0o644
                archive.addfile(entry, io.BytesIO(contents))
    return output.getvalue(), digest.hexdigest()


def _python_download_url(target: str) -> str:
    arch, libc = target.split("/")
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{_PYTHON_RELEASE}/cpython-{_PYTHON_VERSION}%2B{_PYTHON_RELEASE}-"
        f"{arch}-unknown-linux-{libc}-install_only.tar.gz"
    )


async def _check_libc(sandbox: AsyncSandbox, libc: str, manifest: dict[str, object]) -> None:
    """Refuse a bundle whose wheels were built for a newer C library than the task image carries."""
    kind = "glibc" if libc == "gnu" else "musl"
    required = str(manifest.get("libc_version") or manifest.get("glibc") or "")
    if libc == "gnu":
        result = await sandbox.exec("getconf GNU_LIBC_VERSION", cwd="/", timeout_s=30)
        fields = (result.stdout or "").strip().split()
        found = fields[1] if len(fields) == 2 and fields[0] == "glibc" else None
    else:
        # musl prints its version on stderr: "musl libc (x86_64)\nVersion 1.2.5\n...". Alpine's snapshot builds
        # append the git date ("Version 1.2.4_git20230717"); the release they derive from is what matters.
        result = await sandbox.exec("ldd --version", cwd="/", timeout_s=30)
        match = re.search(
            r"^Version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?:[_-]\S*)?$",
            f"{result.stdout or ''}\n{result.stderr or ''}",
            re.M,
        )
        found = match.group(1) if match and "musl" in f"{result.stdout or ''}{result.stderr or ''}" else None
    if (
        found is None
        or re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", required) is None
        or tuple(map(int, found.split("."))) < tuple(map(int, required.split(".")))
    ):
        raise RuntimeError(
            f"Offline NOOA runtime requires {kind} >= {required}; got {found or (result.stdout or '').strip().split()}"
        )


async def prepare_nooa_runtime(
    sandbox: AsyncSandbox, *, wheelhouse_path: Path | None = None, musl_wheelhouse_path: Path | None = None
) -> str:
    """Stage verified artifacts and return the target's healthy private Python.

    Cold setup requires Linux x86_64/aarch64 with glibc or musl, bash, tar, gzip,
    sha256sum, flock, tee, mv with -T support, and writable /opt/nemo-gym-nooa
    and /tmp. No root or system
    package manager is invoked. The controller supplies Python; the sandbox still
    needs outbound package/build dependency access unless a wheelhouse selects
    a source-bound bundle prepared outside the task: ``wheelhouse_path`` for glibc
    images, ``musl_wheelhouse_path`` for musl images (Alpine and BusyBox based
    task images). When a bundle is configured for one C library and the task
    image carries the other, setup fails rather than installing from the network.
    Offline setup installs only local wheels, without package indexes, URL
    requirements or build isolation. A healthy preinstalled or mounted runtime is
    reused without writing into it.
    """
    root = Path(__file__).resolve().parents[2]
    source, fingerprint = await _runtime_io(_runtime_archive, root)
    scratch = f"/tmp/nemo-gym-nooa-setup-{uuid4().hex}"
    script = scratch + ".sh"
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix="nooa-runtime-") as directory:
            local = Path(directory)
            # Execute precisely the installer included in this source fingerprint.
            installer = await _runtime_io(_stage_installer, source, local)
            await sandbox.upload(installer, script)
            probe = await sandbox.exec(shlex.join(["bash", script, "probe"]), cwd="/", timeout_s=30)
            target = (probe.stdout or "").strip()
            if probe.return_code != 0 or target not in _PYTHON_SHA256:
                raise RuntimeError(
                    f"Unsupported NOOA sandbox target: {target!r}; exit={probe.return_code}; "
                    f"stderr={(probe.stderr or '')[-2000:]}"
                )
            arch, libc = target.split("/")
            wheelhouse = None
            bundles = {"gnu": wheelhouse_path, "musl": musl_wheelhouse_path}
            if any(bundles.values()) and bundles[libc] is None:
                setting = "runtime_wheelhouse_musl" if libc == "musl" else "runtime_wheelhouse"
                raise RuntimeError(
                    f"Offline NOOA runtime bundle for {target} is not configured ({setting}); the task image uses "
                    f"{'musl' if libc == 'musl' else 'glibc'} and this agent does not install from the network"
                )
            wheelhouse_path = bundles[libc]
            if wheelhouse_path is not None:
                from responses_api_agents.nooa_agent.offline_runtime import load_runtime_wheelhouse

                wheelhouse = await _runtime_io(
                    load_runtime_wheelhouse, wheelhouse_path, target=target, source_fingerprint=fingerprint
                )
                await _check_libc(sandbox, libc, wheelhouse)
                # Dependency resolution can change independently of source; never reuse
                # a runtime installed from different wheel bytes.
                fingerprint = hashlib.sha256(f"{fingerprint}:{wheelhouse['wheels_sha256']}".encode()).hexdigest()
            runtime = f"{_RUNTIME_ROOT}/{fingerprint}-{arch}-{libc}"
            python = f"{runtime}/python/bin/python3"
            check = shlex.join(["bash", script, "check", runtime, fingerprint, _IMPORT_CHECK])
            cached = await sandbox.exec(check, cwd="/", timeout_s=60)
            if cached.return_code == 0:
                completed = True
                return python

            expected = _PYTHON_SHA256[target]
            python_archive = local / "python.tar.gz"
            wheels_sha256 = None
            if wheelhouse is not None:
                assert wheelhouse_path is not None
                python_source = wheelhouse_path / "python.tar.gz"
                wheels_sha256 = str(wheelhouse["wheels_sha256"])
            else:
                # Reuse mini-SWE's checksum-aware, atomic controller download cache.
                from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download

                python_source = await cached_download(_python_download_url(target), sha256=expected)
            # Check the private upload snapshot too, so cache replacement cannot
            # change the bytes between verification and sandbox transfer.
            source_sha256 = await _runtime_io(
                _stage_runtime_artifacts,
                local,
                source=source,
                python_source=python_source,
                python_sha256=expected,
                wheels_source=wheelhouse_path / "wheels.tar.gz" if wheelhouse_path is not None else None,
                wheels_sha256=wheels_sha256,
            )
            if wheels_sha256 is not None:
                await sandbox.upload(local / "wheels.tar.gz", f"{scratch}/wheels.tar.gz")
            source_archive = local / "source.tar.gz"
            await sandbox.upload(source_archive, f"{scratch}/source.tar.gz")
            await sandbox.upload(python_archive, f"{scratch}/python.tar.gz")
            command = shlex.join(
                [
                    "bash",
                    script,
                    "install",
                    runtime,
                    fingerprint,
                    source_sha256,
                    expected,
                    _IMPORT_CHECK,
                    *([wheels_sha256] if wheels_sha256 is not None else []),
                ]
            )
            result = await sandbox.exec(command, cwd="/", timeout_s=1800)
            if result.return_code != 0:
                raise RuntimeError(
                    f"NOOA runtime preparation failed: exit={result.return_code}; command={command}; "
                    f"diagnostics={scratch}; stdout={(result.stdout or '')[-2000:]} "
                    f"stderr={(result.stderr or '')[-2000:]}"
                )
            completed = True
            return python
    finally:
        # Failed/cancelled setup retains its unique uploads and installer logs.
        # It may still be executing remotely; a retry never modifies those paths.
        if completed:
            try:
                cleanup = await sandbox.exec(
                    f"rm -rf -- {shlex.quote(scratch)} {shlex.quote(script)}", cwd="/", timeout_s=30
                )
                if cleanup.return_code != 0:
                    LOGGER.warning("Could not remove NOOA setup files at %s", scratch)
            except Exception:
                # The runtime is already healthy; a scratch cleanup transport
                # failure must not turn successful installation into seed failure.
                LOGGER.warning("Could not remove NOOA setup files at %s", scratch, exc_info=True)
