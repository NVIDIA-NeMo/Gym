# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache native Linux binaries on the Gym host and transfer them to offline sandboxes."""

import asyncio
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from uuid import uuid4

from aiohttp import ClientTimeout

from nemo_gym.client import request
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult


CACHE_DIR = Path.home() / ".cache" / "nemo_gym" / "claude"
_CACHE_LOCKS: dict[Path, asyncio.Lock] = {}
_HASH_LOCKS: dict[Path, asyncio.Lock] = {}
_VERIFIED: dict[Path, tuple[tuple[int, ...], str]] = {}
_UPLOAD_SLOTS = asyncio.Semaphore(4)


def sha256_file(path: Path) -> str:
    """Hash a Claude binary in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise PermissionError(f"Claude cache must be owned by the current user without shared write access: {path}")
    path.chmod(0o700)


def _fingerprint(path: Path) -> tuple[int, ...]:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


async def _verified_sha256(path: Path) -> str:
    # Rehash replacements or edits, but share verification across warm concurrent seeds.
    async with _HASH_LOCKS.setdefault(path, asyncio.Lock()):
        identity = _fingerprint(path)
        cached = _VERIFIED.get(path)
        if cached is not None and cached[0] == identity:
            return cached[1]
        digest = await asyncio.to_thread(sha256_file, path)
        if _fingerprint(path) != identity:
            raise ValueError(f"Claude artifact changed during checksum verification: {path}")
        _VERIFIED[path] = (identity, digest)
        return digest


def _target(probe: str) -> str:
    lines = probe.strip().splitlines()
    if len(lines) != 3 or lines[0] != "Linux":
        raise ValueError(f"Unsupported Claude sandbox platform: {probe!r}")
    _, architecture, libc = lines
    architectures = {"x86_64": "x64", "aarch64": "arm64"}
    if architecture not in architectures or libc not in {"gnu", "musl"}:
        raise ValueError(f"Unsupported Claude sandbox platform: {architecture}/{libc}")
    return "linux-" + architectures[architecture] + ("-musl" if libc == "musl" else "")


async def cached_binary(version: str, target: str, *, expected: str | None = None) -> tuple[Path, str]:
    """Download on the host once per version/platform and verify the cached bytes."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) or target not in {
        "linux-x64",
        "linux-arm64",
        "linux-x64-musl",
        "linux-arm64-musl",
    }:
        raise ValueError("Claude runtime requires an exact version and supported Linux target")
    _private_directory(CACHE_DIR)
    directory = CACHE_DIR / version
    async with _CACHE_LOCKS.setdefault(directory, asyncio.Lock()):
        _private_directory(directory)
        base = "https://downloads.claude.ai/claude-code-releases/" + version
        if expected is None:
            manifest_path = directory / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text())
                expected = manifest["platforms"][target]["checksum"]
            except (OSError, ValueError, KeyError, TypeError):
                async with await request("GET", base + "/manifest.json", timeout=ClientTimeout(total=60)) as response:
                    response.raise_for_status()
                    manifest = await response.json()
                expected = manifest["platforms"][target]["checksum"]
                temporary = directory / ("manifest-" + uuid4().hex)
                try:
                    temporary.write_text(json.dumps(manifest))
                    temporary.replace(manifest_path)
                finally:
                    temporary.unlink(missing_ok=True)
        checksum = expected.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Claude executable requires a SHA-256 checksum")
        path = directory / target / "claude"
        _private_directory(path.parent)
        if path.is_file() and await _verified_sha256(path) == checksum:
            return path, checksum
        temporary = path.with_name(".claude-" + uuid4().hex)
        try:
            async with await request(
                "GET", base + "/" + target + "/claude", timeout=ClientTimeout(total=180)
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        await asyncio.to_thread(output.write, chunk)
            if await _verified_sha256(temporary) != checksum:
                raise ValueError("Claude executable SHA-256 mismatch")
            temporary.replace(path)
            _VERIFIED[path] = (_fingerprint(path), checksum)
        finally:
            _VERIFIED.pop(temporary, None)
            _HASH_LOCKS.pop(temporary, None)
            temporary.unlink(missing_ok=True)
        return path, checksum


def _compress(source: Path, destination: Path) -> None:
    with source.open("rb") as input_file, gzip.open(destination, "wb", compresslevel=6) as output:
        shutil.copyfileobj(input_file, output)


async def compressed_binary(source: Path, checksum: str) -> Path:
    """Cache a smaller transfer artifact while retaining verification of the native bytes."""
    _private_directory(CACHE_DIR)
    directory = CACHE_DIR / "compressed"
    _private_directory(directory)
    path = directory / (checksum + ".gz")
    async with _CACHE_LOCKS.setdefault(path, asyncio.Lock()):
        receipt = path.with_suffix(".sha256")
        if path.is_file() and receipt.is_file() and await _verified_sha256(path) == receipt.read_text().strip():
            return path
        temporary = directory / (".gzip-" + uuid4().hex)
        try:
            await asyncio.to_thread(_compress, source, temporary)
            digest = await _verified_sha256(temporary)
            temporary.replace(path)
            _VERIFIED[path] = (_fingerprint(path), digest)
            receipt.write_text(digest)
        finally:
            _VERIFIED.pop(temporary, None)
            _HASH_LOCKS.pop(temporary, None)
            temporary.unlink(missing_ok=True)
        return path


def _check(result: SandboxExecResult) -> None:
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"Claude runtime setup failed (exit={result.return_code}, error={result.error_type}): {details[-16000:]}"
        )


async def _upload_archive(sandbox: AsyncSandbox, archive: Path, directory: str, timeout: float) -> str:
    """Stream bounded pieces with a process-wide limit on staged bytes and uploads."""
    destination = directory + "/claude-upload.gz"
    with TemporaryDirectory(prefix="claude-upload-") as staging:

        async def upload(part: Path) -> None:
            try:
                await sandbox.upload(part, directory + "/" + part.name)
            finally:
                part.unlink(missing_ok=True)
                _UPLOAD_SLOTS.release()

        parts = []
        async with asyncio.TaskGroup() as tasks:
            with archive.open("rb") as source:
                while True:
                    await _UPLOAD_SLOTS.acquire()
                    try:
                        data = await asyncio.to_thread(source.read, 1024 * 1024)
                        if not data:
                            _UPLOAD_SLOTS.release()
                            break
                        part = Path(staging) / f"claude-upload-{len(parts):04d}.part"
                        await asyncio.to_thread(part.write_bytes, data)
                        del data
                    except BaseException:
                        _UPLOAD_SLOTS.release()
                        raise
                    parts.append(part)
                    tasks.create_task(upload(part))
        if not parts:
            raise ValueError("Cannot upload an empty Claude archive")
        sources = " ".join(quote(directory + "/" + part.name) for part in parts)
        result = await sandbox.exec(f"cat {sources} > {quote(destination)} && rm -f {sources}", timeout_s=timeout)
        _check(result)
    return destination


async def install_sandbox_runtime(
    sandbox: AsyncSandbox,
    *,
    version: str,
    directory: str,
    runtime: str,
    timeout: float,
    source: str | None = None,
    local_source: str | None = None,
    expected: str | None = None,
) -> None:
    """Reuse a verified runtime or upload host-cached bytes; never fetch from the sandbox."""
    if (source or local_source) and not expected:
        raise ValueError("A supplied Claude binary requires a trusted SHA-256")
    command = ["bash", directory + "/install_claude_runtime.sh", "--version", version, "--runtime", runtime]
    if source:
        command += ["--source", source, "--sha256", expected]
    else:
        if local_source:
            local = Path(local_source).expanduser()
            checksum = await _verified_sha256(local)
            if checksum != expected.lower():
                raise ValueError("Local Claude executable SHA-256 mismatch")
        else:
            probe = await sandbox.exec(
                "uname -s && uname -m && if ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; else echo gnu; fi",
                timeout_s=30,
            )
            _check(probe)
            local, checksum = await cached_binary(version, _target(probe.stdout or ""), expected=expected)
        command += ["--sha256", checksum]
        cached = await sandbox.exec(" ".join(map(quote, command + ["--check-only"])), timeout_s=timeout)
        if cached.return_code == 0 and not cached.error_type:
            return
        if cached.error_type or cached.return_code != 3:
            _check(cached)
        archive = await compressed_binary(local, checksum)
        destination = await _upload_archive(sandbox, archive, directory, timeout)
        command += ["--source", destination, "--gzip"]
    installed = await sandbox.exec(" ".join(map(quote, command)), timeout_s=timeout)
    _check(installed)
