# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode and ripgrep executables that the agent server host downloads once and uploads into task sandboxes.

With ``sandbox_runtime_upload``, the host downloads the pinned release archive for the sandbox's platform, never its
own, checks its SHA-256 and unpacks the executable. The archives are cached under ``sandbox_opencode`` in Gym's
``cache_dir``; a host that cannot reach GitHub can be given them there, and the checksums still apply. The sandbox
checks the uploaded executable against the SHA-256 computed here, so it needs no download tool, archive tool, root
or package manager.
"""

import asyncio
import functools
import hashlib
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from nemo_gym.agents import SandboxPlatform, download_pinned, sandbox_cache_dir
from responses_api_agents.opencode_agent.runtime import OPENCODE_RELEASE_SHA256


_OPENCODE_DOWNLOADS = "https://github.com/anomalyco/opencode/releases/download"
# The release asset the sandbox installer downloads for each CPU architecture; a musl sandbox adds "-musl".
_OPENCODE_ARCHITECTURES = {"x86_64": "x64-baseline", "aarch64": "arm64"}

# The ripgrep OpenCode 1.19.6 and 2.3.2 download for themselves (packages/core/src/ripgrep/binary.ts).
RIPGREP_VERSION = "15.1.0"
_RIPGREP_DOWNLOADS = "https://github.com/BurntSushi/ripgrep/releases/download"
# The build for each sandbox platform and the SHA-256 that release publishes for its archive. The x86_64 build is
# statically linked, so it also runs on musl. The release has no aarch64 build for musl.
_RIPGREP_BUILDS = {
    "x86_64/gnu": (
        "x86_64-unknown-linux-musl",
        "1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599",  # pragma: allowlist secret
    ),
    "x86_64/musl": (
        "x86_64-unknown-linux-musl",
        "1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599",  # pragma: allowlist secret
    ),
    "aarch64/gnu": (
        "aarch64-unknown-linux-gnu",
        "2b661c6ef508e902f388e9098d9c4c5aca72c87b55922d94abdba830b4dc885e",  # pragma: allowlist secret
    ),
}
# The executable this process unpacked from each archive, so no later seed hashes the archive again, and the
# unpacking in progress for each, so concurrent seeds share one download.
_UNPACKED: dict[Path, "HostExecutable"] = {}
_UNPACKING: dict[Path, asyncio.Task["HostExecutable"]] = {}


@dataclass(frozen=True)
class HostExecutable:
    """An executable on this host, unpacked from a release archive whose SHA-256 is pinned."""

    path: Path
    sha256: str


def opencode_release_asset(platform: SandboxPlatform) -> str:
    """The name OpenCode's releases give the build for ``platform``, such as ``x64-baseline-musl``."""
    return _OPENCODE_ARCHITECTURES[platform.arch] + ("-musl" if platform.libc == "musl" else "")


async def opencode_executable(version: str, platform: SandboxPlatform) -> HostExecutable:
    """The OpenCode ``version`` executable for ``platform``. Raises ``RuntimeError`` without a pinned SHA-256."""
    asset = opencode_release_asset(platform)
    sha256 = OPENCODE_RELEASE_SHA256.get(version, {}).get(asset)
    if sha256 is None:
        raise RuntimeError(f"No pinned SHA-256 for opencode-linux-{asset}.tar.gz {version}")
    return await _executable(
        f"{_OPENCODE_DOWNLOADS}/v{version}/opencode-linux-{asset}.tar.gz",
        # Release assets carry no version in their name.
        sandbox_cache_dir("sandbox_opencode") / f"opencode-{version}-linux-{asset}.tar.gz",
        sha256,
        member="opencode",
    )


async def ripgrep_executable(platform: SandboxPlatform) -> HostExecutable | None:
    """The pinned ripgrep executable for ``platform``, or None where the release has no build that runs there."""
    build = _RIPGREP_BUILDS.get(f"{platform.arch}/{platform.libc}")
    if build is None:
        return None
    name = f"ripgrep-{RIPGREP_VERSION}-{build[0]}"
    return await _executable(
        f"{_RIPGREP_DOWNLOADS}/{RIPGREP_VERSION}/{name}.tar.gz",
        sandbox_cache_dir("sandbox_opencode") / f"{name}.tar.gz",
        build[1],
        member=f"{name}/rg",
    )


def _unpacked(archive: Path, unpacking: asyncio.Task[HostExecutable]) -> None:
    _UNPACKING.pop(archive, None)
    # Every waiter may have been cancelled; the outcome is still consumed here. A failure is retried by the next seed.
    if not unpacking.cancelled() and unpacking.exception() is None:
        _UNPACKED[archive] = unpacking.result()


async def _executable(url: str, archive: Path, sha256: str, *, member: str) -> HostExecutable:
    """``member`` of the archive at ``url``, unpacked once per process from a download that matches ``sha256``.

    Concurrent callers share one download and its outcome, and a caller that is cancelled leaves it running for the
    others. An executable that is gone from the cache, for example because the cache was cleared, is unpacked again.
    """
    executable = _UNPACKED.get(archive)
    if executable is not None and executable.path.is_file():
        return executable
    unpacking = _UNPACKING.get(archive)
    if unpacking is None:
        unpacking = _UNPACKING[archive] = asyncio.create_task(_download_and_unpack(url, archive, sha256, member))
        unpacking.add_done_callback(functools.partial(_unpacked, archive))
    return await asyncio.shield(unpacking)


async def _download_and_unpack(url: str, archive: Path, sha256: str, member: str) -> HostExecutable:
    await download_pinned(url, archive, sha256)
    return await asyncio.to_thread(_unpack, archive, member)


def _unpack(archive: Path, member: str) -> HostExecutable:
    """Write ``member`` of the verified ``archive`` beside it, and return it with the SHA-256 of the bytes written.

    Every process unpacks again instead of trusting an executable an earlier one left in the cache: only the
    archive's SHA-256 is pinned. The file is renamed into place, so a process that is uploading the earlier copy,
    which holds the same bytes, is not disturbed.
    """
    executable = archive.with_name(archive.name.removesuffix(".tar.gz"))
    partial = executable.with_name(f".{executable.name}.{uuid4().hex}")
    digest = hashlib.sha256()
    try:
        with tarfile.open(archive) as tar:
            try:
                source = tar.extractfile(member)
            except KeyError:
                source = None
            if source is None:
                raise RuntimeError(f"{archive} holds no file {member}")
            with source, partial.open("wb") as output:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
                    output.write(block)
        partial.chmod(0o755)
        os.replace(partial, executable)
    finally:
        partial.unlink(missing_ok=True)
    return HostExecutable(path=executable, sha256=digest.hexdigest())
