# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode and ripgrep executables that the agent server host downloads and uploads into task sandboxes.

With ``sandbox_runtime_upload``, the host downloads the pinned release archive for the sandbox's platform, never its
own, checks its SHA-256 and unpacks the executable. The archives are cached under ``sandbox_opencode`` in Gym's
``cache_dir``; a host that cannot reach GitHub can be given them there, and the checksums still apply. The sandbox
checks the upload against the SHA-256 computed here, so it needs no download tool, archive tool, root or package
manager.
"""

import asyncio
import hashlib
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from nemo_gym.agents import AgentSeedRejectedError, SandboxPlatform, download_pinned, sandbox_cache_dir
from responses_api_agents.opencode_agent.runtime import OPENCODE_RELEASE_SHA256


_OPENCODE_DOWNLOADS = "https://github.com/anomalyco/opencode/releases/download"
# The release asset the sandbox installer downloads for each CPU architecture; a musl sandbox adds "-musl".
_OPENCODE_ARCHITECTURES = {"x86_64": "x64-baseline", "aarch64": "arm64"}
# The ripgrep OpenCode 1.19.6 and 2.3.2 download for themselves (packages/core/src/ripgrep/binary.ts), and the
# SHA-256 its release publishes for each archive. The x86_64 build is statically linked, so it also runs on musl;
# the release has no aarch64 build for musl.
RIPGREP_VERSION = "15.1.0"
_RIPGREP_DOWNLOADS = "https://github.com/BurntSushi/ripgrep/releases/download"
_RIPGREP_BUILDS = {"x86_64": "x86_64-unknown-linux-musl", "aarch64": "aarch64-unknown-linux-gnu"}
_RIPGREP_SHA256 = {
    "x86_64-unknown-linux-musl": "1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599",  # pragma: allowlist secret
    "aarch64-unknown-linux-gnu": "2b661c6ef508e902f388e9098d9c4c5aca72c87b55922d94abdba830b4dc885e",  # pragma: allowlist secret
}
# The executable unpacked from each archive in this process, or the unpacking in progress: concurrent seeds share one
# download, and later seeds neither hash the archive nor unpack it again.
_UNPACKED: dict[Path, asyncio.Task["HostExecutable"]] = {}


@dataclass(frozen=True)
class HostExecutable:
    """An executable on this host, unpacked from a release archive whose SHA-256 is pinned."""

    path: Path
    sha256: str


async def opencode_executable(version: str, platform: SandboxPlatform) -> HostExecutable:
    """The OpenCode ``version`` executable for ``platform``.

    Raises ``AgentSeedRejectedError`` when the release has no pinned SHA-256 for the platform's build, which every
    attempt on its image would repeat, and ``RuntimeError`` when the download fails or does not match.
    """
    asset = _OPENCODE_ARCHITECTURES[platform.arch] + ("-musl" if platform.libc == "musl" else "")
    sha256 = OPENCODE_RELEASE_SHA256.get(version, {}).get(asset)
    if sha256 is None:
        raise AgentSeedRejectedError(
            f"sandbox_runtime_upload: OpenCode {version} has no pinned SHA-256 for opencode-linux-{asset}.tar.gz, "
            "the build for this sandbox's platform"
        )
    return await _executable(
        f"{_OPENCODE_DOWNLOADS}/v{version}/opencode-linux-{asset}.tar.gz",
        # Release assets carry no version in their name.
        sandbox_cache_dir("sandbox_opencode") / f"opencode-{version}-linux-{asset}.tar.gz",
        sha256,
        member="opencode",
    )


async def ripgrep_executable(platform: SandboxPlatform) -> HostExecutable | None:
    """The pinned ripgrep executable for ``platform``, or None where the release has no build that runs there."""
    if platform.arch == "aarch64" and platform.libc == "musl":
        return None
    name = f"ripgrep-{RIPGREP_VERSION}-{_RIPGREP_BUILDS[platform.arch]}"
    return await _executable(
        f"{_RIPGREP_DOWNLOADS}/{RIPGREP_VERSION}/{name}.tar.gz",
        sandbox_cache_dir("sandbox_opencode") / f"{name}.tar.gz",
        _RIPGREP_SHA256[_RIPGREP_BUILDS[platform.arch]],
        member=f"{name}/rg",
    )


def _forget_failed(archive: Path, unpacking: asyncio.Task[HostExecutable]) -> None:
    # Every waiter may have been cancelled; the outcome is still consumed here. A failure is retried by the next seed.
    if unpacking.cancelled() or unpacking.exception() is not None:
        _UNPACKED.pop(archive, None)


async def _executable(url: str, archive: Path, sha256: str, *, member: str) -> HostExecutable:
    """``member`` of the archive at ``url``, unpacked once per process from a download that matches ``sha256``.

    A caller that is cancelled, such as a seed that timed out, leaves the download running for the others.
    """
    unpacking = _UNPACKED.get(archive)
    if unpacking is None:
        unpacking = _UNPACKED[archive] = asyncio.create_task(_download_and_unpack(url, archive, sha256, member))
        unpacking.add_done_callback(lambda done: _forget_failed(archive, done))
    return await asyncio.shield(unpacking)


async def _download_and_unpack(url: str, archive: Path, sha256: str, member: str) -> HostExecutable:
    await download_pinned(url, archive, sha256)
    return await asyncio.to_thread(_unpack, archive, member)


def _unpack(archive: Path, member: str) -> HostExecutable:
    """Write ``member`` of the verified ``archive`` beside it, and return it with the SHA-256 of the bytes written.

    Every process unpacks again instead of trusting an executable an earlier one left in the cache: only the
    archive's SHA-256 is pinned. The file is renamed into place, so another process that is uploading the earlier
    copy, which holds the same bytes, is not disturbed.
    """
    executable = archive.with_name(archive.name.removesuffix(".tar.gz"))
    partial = executable.with_name(f".{executable.name}.{uuid4().hex}")
    digest = hashlib.sha256()
    try:
        with tarfile.open(archive) as tar, tar.extractfile(member) as source, partial.open("wb") as output:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
                output.write(block)
        partial.chmod(0o755)
        os.replace(partial, executable)
    finally:
        partial.unlink(missing_ok=True)
    return HostExecutable(path=executable, sha256=digest.hexdigest())
