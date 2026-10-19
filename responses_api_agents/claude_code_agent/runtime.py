# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache native Linux binaries on the Gym host and transfer them to offline sandboxes."""

import asyncio
import gzip
import json
import re
import shutil
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory, gettempdir
from uuid import uuid4

from aiohttp import ClientTimeout

from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.server_utils import request
from responses_api_agents.claude_code_agent.install_claude_runtime import sha256_file


CACHE_DIR = Path(gettempdir()) / "nemo-gym-claude-assets"
_DOWNLOAD_LOCK = asyncio.Lock()


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
    async with _DOWNLOAD_LOCK:
        directory = CACHE_DIR / version
        directory.mkdir(parents=True, exist_ok=True)
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
        path = directory / target / "claude"
        if path.is_file() and await asyncio.to_thread(sha256_file, path) == checksum:
            return path, checksum
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".claude-" + uuid4().hex)
        try:
            async with await request(
                "GET", base + "/" + target + "/claude", timeout=ClientTimeout(total=180)
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        await asyncio.to_thread(output.write, chunk)
            if await asyncio.to_thread(sha256_file, temporary) != checksum:
                raise ValueError("Claude executable SHA-256 mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path, checksum


def _compress(source: Path, destination: Path) -> None:
    with source.open("rb") as input_file, gzip.open(destination, "wb", compresslevel=6) as output:
        shutil.copyfileobj(input_file, output)


async def compressed_binary(source: Path, checksum: str) -> Path:
    """Cache a smaller transfer artifact while retaining verification of the native bytes."""
    async with _DOWNLOAD_LOCK:
        directory = CACHE_DIR / "compressed"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (checksum + ".gz")
        receipt = path.with_suffix(".sha256")
        if (
            path.is_file()
            and receipt.is_file()
            and await asyncio.to_thread(sha256_file, path) == receipt.read_text().strip()
        ):
            return path
        temporary = directory / (".gzip-" + uuid4().hex)
        try:
            await asyncio.to_thread(_compress, source, temporary)
            digest = await asyncio.to_thread(sha256_file, temporary)
            temporary.replace(path)
            receipt.write_text(digest)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def _check(result: SandboxExecResult) -> None:
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"Claude runtime setup failed (exit={result.return_code}, error={result.error_type}): {details[-16000:]}"
        )


async def _upload_archive(sandbox: AsyncSandbox, archive: Path, directory: str, timeout: float) -> str:
    """Keep each upload below sandbox proxy request-size limits."""
    destination = directory + "/claude-upload.gz"
    with TemporaryDirectory(prefix="claude-upload-") as staging:

        def split() -> list[Path]:
            parts = []
            with archive.open("rb") as source:
                while data := source.read(1024 * 1024):
                    path = Path(staging) / f"claude-upload-{len(parts):04d}.part"
                    path.write_bytes(data)
                    parts.append(path)
            return parts

        parts = await asyncio.to_thread(split)
        semaphore = asyncio.Semaphore(4)

        async def upload(part: Path) -> None:
            async with semaphore:
                await sandbox.upload(part, directory + "/" + part.name)

        async with asyncio.TaskGroup() as tasks:
            for part in parts:
                tasks.create_task(upload(part))
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
    command = ["python3", "-I", directory + "/install_claude_runtime.py", "--version", version, "--runtime", runtime]
    if source:
        command += ["--source", source, "--sha256", expected]
    else:
        if local_source:
            local = Path(local_source).expanduser()
            checksum = await asyncio.to_thread(sha256_file, local)
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
