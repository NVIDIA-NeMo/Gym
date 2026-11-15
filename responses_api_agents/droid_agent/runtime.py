# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache Droid's npm platform packages on the Gym host and transfer them to the task sandbox."""

import asyncio
import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from uuid import uuid4

from aiohttp import ClientTimeout

from nemo_gym.agents import AgentSeedRejectedError
from nemo_gym.client import request
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult


CACHE_DIR = Path.home() / ".cache" / "nemo_gym" / "droid"
REGISTRY = "https://registry.npmjs.org"
# Droid publishes one native binary per platform as @factory/cli-<target>. The baseline x64 build runs
# on CPUs without AVX2. There is no musl build.
TARGETS = frozenset({"linux-x64", "linux-x64-baseline", "linux-arm64"})
# npm integrity of the default release's packages, so they are checked independently of registry metadata.
PINNED_INTEGRITY = {
    (
        "1.1.0",
        "linux-x64",
    ): "sha512-9Hx1dy/vKTDWNGrlPkJwCsMg4z4k+Nza1suEhTMfIZ3+Gg1gyzA1txAaC/eotupdp1TWHhhe/Pk0uhyZcgL7bA==",  # pragma: allowlist secret
    (
        "1.1.0",
        "linux-x64-baseline",
    ): "sha512-6sHX/mMspih52TBqbN9cmywU2T29F55t08N2lRvQ59zyKd8WE63J11HDwxkAYyl977EvDeB0ST1A1LrIJUF2gw==",  # pragma: allowlist secret
    (
        "1.1.0",
        "linux-arm64",
    ): "sha512-oNx51G1tWLydrpWWxdKKW98ws3g+pOKsibUkSaSq8hjOXVjyRrgsU1PHqn/a/CifdjxL2NLbBASuxwUhn5FZyw==",  # pragma: allowlist secret
}
# Musl tooling can put musl's loader into a glibc image, so the image counts as musl only when glibc is missing.
PLATFORM_PROBE = (
    "uname -s && uname -m"
    " && if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then echo gnu;"
    " elif ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; else echo gnu; fi"
    " && if grep -qw avx2 /proc/cpuinfo 2>/dev/null; then echo avx2; else echo baseline; fi"
)
_CACHE_LOCKS: dict[Path, asyncio.Lock] = {}
_VERIFIED: dict[Path, tuple[tuple[int, ...], str, str]] = {}
# Registry metadata per (version, target): a published npm version never changes, so it is fetched once.
_SOURCES: dict[tuple[str, str], tuple[str, str]] = {}
_UPLOAD_SLOTS = asyncio.Semaphore(4)


def package_target(probe: str) -> str:
    """Map the sandbox platform probe to the npm package that runs there.

    A platform Droid publishes no build for is refused as final; unreadable probe output stays retryable.
    """
    lines = probe.strip().splitlines()
    if len(lines) != 4:
        raise ValueError(f"Unexpected Droid sandbox platform probe output: {probe!r}")
    system, architecture, libc, cpu = lines
    if system != "Linux":
        raise AgentSeedRejectedError(f"Unsupported Droid sandbox platform: {system}")
    if libc != "gnu":
        raise AgentSeedRejectedError(
            "Droid publishes glibc Linux builds only; the sandbox image has musl and no glibc"
        )
    if architecture == "x86_64":
        return "linux-x64" if cpu == "avx2" else "linux-x64-baseline"
    if architecture == "aarch64":
        return "linux-arm64"
    raise AgentSeedRejectedError(f"Unsupported Droid sandbox architecture: {architecture}")


def digests(path: Path) -> tuple[str, str]:
    """Return the npm integrity (sha512, base64) and the SHA-256 of a package file in one read."""
    sha512, sha256 = hashlib.sha512(), hashlib.sha256()
    with path.open("rb") as package:
        for chunk in iter(lambda: package.read(1024 * 1024), b""):
            sha512.update(chunk)
            sha256.update(chunk)
    return "sha512-" + base64.b64encode(sha512.digest()).decode(), sha256.hexdigest()


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise PermissionError(f"Droid cache must be owned by the current user without shared write access: {path}")
    path.chmod(0o700)


def _fingerprint(path: Path) -> tuple[int, ...]:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


async def _verified(path: Path, integrity: str) -> str | None:
    """SHA-256 of ``path`` when its bytes match ``integrity``; verification is memoized per file identity."""
    identity = _fingerprint(path)
    cached = _VERIFIED.get(path)
    if cached is None or cached[0] != identity:
        actual, sha256 = await asyncio.to_thread(digests, path)
        if _fingerprint(path) != identity:
            raise ValueError(f"Droid package changed during verification: {path}")
        cached = _VERIFIED[path] = (identity, actual, sha256)
    return cached[2] if cached[1] == integrity else None


async def _package_source(version: str, target: str) -> tuple[str, str]:
    """Tarball URL and npm integrity, from the pinned table or the registry's metadata for other versions."""
    name = f"cli-{target}"
    pinned = PINNED_INTEGRITY.get((version, target))
    if pinned is not None:
        return f"{REGISTRY}/@factory/{name}/-/{name}-{version}.tgz", pinned
    if (version, target) in _SOURCES:
        return _SOURCES[version, target]
    async with await request("GET", f"{REGISTRY}/@factory/{name}/{version}", timeout=ClientTimeout(total=60)) as reply:
        if reply.status == 404:
            # An unpublished version fails every retry the same way; other registry errors may pass.
            raise AgentSeedRejectedError(f"Droid {version} is not published as @factory/{name} on npm")
        reply.raise_for_status()
        metadata = json.loads(await reply.text())
    dist = metadata["dist"]
    if not str(dist.get("integrity", "")).startswith("sha512-"):
        raise ValueError(f"npm metadata for @factory/{name}@{version} has no sha512 integrity")
    _SOURCES[version, target] = dist["tarball"], dist["integrity"]
    return _SOURCES[version, target]


async def cached_package(version: str, target: str) -> tuple[Path, str]:
    """Download a platform package on the host once and return it with its SHA-256."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) or target not in TARGETS:
        raise ValueError("Droid runtime requires an exact version and a supported Linux target")
    _private_directory(CACHE_DIR)
    directory = CACHE_DIR / version
    _private_directory(directory)
    path = directory / f"cli-{target}.tgz"
    async with _CACHE_LOCKS.setdefault(path, asyncio.Lock()):
        url, integrity = await _package_source(version, target)
        if path.is_file() and (checksum := await _verified(path, integrity)):
            return path, checksum
        temporary = directory / (".download-" + uuid4().hex)
        try:
            async with await request("GET", url, timeout=ClientTimeout(total=600)) as reply:
                reply.raise_for_status()
                with temporary.open("wb") as output:
                    async for chunk in reply.content.iter_chunked(1024 * 1024):
                        await asyncio.to_thread(output.write, chunk)
            if not await _verified(temporary, integrity):
                raise ValueError(f"Droid package integrity mismatch for @factory/cli-{target}@{version}")
            temporary.replace(path)
            _VERIFIED[path] = (_fingerprint(path), *_VERIFIED.pop(temporary)[1:])
        finally:
            _VERIFIED.pop(temporary, None)
            temporary.unlink(missing_ok=True)
        return path, _VERIFIED[path][2]


def _check(result: SandboxExecResult) -> None:
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"Droid runtime setup failed (exit={result.return_code}, error={result.error_type}): {details[-16000:]}"
        )


async def _upload_package(sandbox: AsyncSandbox, package: Path, directory: str, timeout: float) -> str:
    """Stream bounded pieces with a process-wide limit on staged bytes and uploads."""
    destination = directory + "/droid-package.tgz"
    # A piece write cancelled in its thread can outlive the directory; that must not replace the caller's error.
    with TemporaryDirectory(prefix="droid-upload-", ignore_cleanup_errors=True) as staging:

        def uploaded(part: Path) -> None:
            part.unlink(missing_ok=True)
            _UPLOAD_SLOTS.release()

        parts = []
        async with asyncio.TaskGroup() as tasks:
            with package.open("rb") as source:
                while True:
                    await _UPLOAD_SLOTS.acquire()
                    try:
                        data = await asyncio.to_thread(source.read, 1024 * 1024)
                        if not data:
                            _UPLOAD_SLOTS.release()
                            break
                        part = Path(staging) / f"droid-upload-{len(parts):04d}.part"
                        await asyncio.to_thread(part.write_bytes, data)
                        del data
                    except BaseException:
                        _UPLOAD_SLOTS.release()
                        raise
                    parts.append(part)
                    upload = tasks.create_task(sandbox.upload(part, directory + "/" + part.name))
                    # A done callback also runs for a piece the group cancels before it starts, so no slot leaks.
                    upload.add_done_callback(lambda _, part=part: uploaded(part))
        if not parts:
            raise ValueError("Cannot upload an empty Droid package")
        sources = " ".join(quote(directory + "/" + part.name) for part in parts)
        _check(await sandbox.exec(f"cat {sources} > {quote(destination)} && rm -f {sources}", timeout_s=timeout))
    return destination


async def install_sandbox_runtime(
    sandbox: AsyncSandbox, *, version: str, directory: str, runtime: str, timeout: float
) -> None:
    """Reuse a verified runtime or upload the host-cached package; the sandbox needs no network."""
    probe = await sandbox.exec(PLATFORM_PROBE, timeout_s=30)
    _check(probe)
    package, checksum = await cached_package(version, package_target(probe.stdout or ""))
    command = ["bash", directory + "/install_droid_runtime.sh", "--version", version, "--runtime", runtime]
    command += ["--sha256", checksum]
    existing = await sandbox.exec(" ".join(map(quote, command + ["--check-only"])), timeout_s=timeout)
    if existing.return_code == 0 and not existing.error_type:
        return
    if existing.error_type or existing.return_code != 3:
        _check(existing)
    source = await _upload_package(sandbox, package, directory, timeout)
    _check(await sandbox.exec(" ".join(map(quote, command + ["--source", source])), timeout_s=timeout))
