# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode and ripgrep executables the agent server host downloads for a sandbox's platform."""

import asyncio
import hashlib
import io
import tarfile
from types import SimpleNamespace

import pytest

from nemo_gym.agents import AgentSeedRejectedError, SandboxPlatform
from responses_api_agents.opencode_agent import host_runtime
from responses_api_agents.opencode_agent.runtime import OPENCODE_RELEASE_SHA256, OPENCODE_VERSION


def archive(member: str, body: bytes) -> bytes:
    """A release archive: a gzipped tar holding ``member``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        entry = tarfile.TarInfo(member)
        entry.size = len(body)
        tar.addfile(entry, io.BytesIO(body))
    return buffer.getvalue()


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """The checked download, which here writes ``archive`` and records the URL and the SHA-256 it was asked to check."""
    made = SimpleNamespace(calls=[], archive=None, cache=tmp_path / "sandbox_opencode")

    async def download_pinned(url, path, sha256):
        made.calls.append((url, sha256))
        await asyncio.sleep(0)
        if made.archive is None:
            raise RuntimeError(f"Downloading {url} failed with HTTP 503")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(made.archive)
        return path

    monkeypatch.setattr(host_runtime, "download_pinned", download_pinned)
    monkeypatch.setattr(host_runtime, "sandbox_cache_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(host_runtime, "_UNPACKED", {})
    return made


@pytest.mark.parametrize(
    "platform, asset",
    [
        (SandboxPlatform("x86_64", "gnu", 36), "x64-baseline"),
        (SandboxPlatform("x86_64", "musl"), "x64-baseline-musl"),
        (SandboxPlatform("aarch64", "gnu", 17), "arm64"),
        (SandboxPlatform("aarch64", "musl"), "arm64-musl"),
    ],
)
async def test_the_opencode_build_is_the_pinned_one_for_the_sandbox_platform(downloads, platform, asset) -> None:
    downloads.archive = archive("opencode", b"opencode")
    # An executable an earlier process left in the cache is not trusted: only the archive's SHA-256 is pinned.
    downloads.cache.mkdir()
    (downloads.cache / f"opencode-{OPENCODE_VERSION}-linux-{asset}").write_bytes(b"left in the cache")

    # Concurrent seeds share one download.
    executable, *others = await asyncio.gather(
        *(host_runtime.opencode_executable(OPENCODE_VERSION, platform) for _ in range(3))
    )

    # The sandbox's architecture and C library select the asset, whatever this host runs on.
    url = f"https://github.com/anomalyco/opencode/releases/download/v{OPENCODE_VERSION}/opencode-linux-{asset}.tar.gz"
    assert downloads.calls == [(url, OPENCODE_RELEASE_SHA256[OPENCODE_VERSION][asset])]
    assert others == [executable, executable]
    assert executable.path == downloads.cache / f"opencode-{OPENCODE_VERSION}-linux-{asset}"
    assert executable.path.read_bytes() == b"opencode"
    assert executable.sha256 == hashlib.sha256(b"opencode").hexdigest()
    assert executable.path.stat().st_mode & 0o777 == 0o755


async def test_a_failed_download_is_retried_by_the_next_seed(downloads) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 36)
    with pytest.raises(RuntimeError, match="failed with HTTP 503"):
        await host_runtime.opencode_executable(OPENCODE_VERSION, platform)

    downloads.archive = archive("opencode", b"opencode")
    assert (await host_runtime.opencode_executable(OPENCODE_VERSION, platform)).path.read_bytes() == b"opencode"
    assert len(downloads.calls) == 2


async def test_a_release_without_a_pinned_digest_for_the_platform_refuses_the_seed(downloads, monkeypatch) -> None:
    # Nothing a retry could change: the refusal is final, unlike a failed download.
    monkeypatch.setitem(host_runtime.OPENCODE_RELEASE_SHA256, "9.9.9", {"x64-baseline": "0" * 64})
    with pytest.raises(AgentSeedRejectedError, match="no pinned SHA-256 for opencode-linux-arm64-musl.tar.gz"):
        await host_runtime.opencode_executable("9.9.9", SandboxPlatform("aarch64", "musl"))
    assert downloads.calls == []


@pytest.mark.parametrize(
    "platform, build",
    [
        (SandboxPlatform("x86_64", "gnu", 36), "x86_64-unknown-linux-musl"),
        # The x86_64 build is static, so it also serves musl; the release has no aarch64 build for musl.
        (SandboxPlatform("x86_64", "musl"), "x86_64-unknown-linux-musl"),
        (SandboxPlatform("aarch64", "gnu", 36), "aarch64-unknown-linux-gnu"),
        (SandboxPlatform("aarch64", "musl"), None),
    ],
)
async def test_the_ripgrep_build_is_the_pinned_one_for_the_sandbox_platform(downloads, platform, build) -> None:
    if build is None:
        assert await host_runtime.ripgrep_executable(platform) is None and downloads.calls == []
        return
    downloads.archive = archive(f"ripgrep-15.1.0-{build}/rg", b"ripgrep")

    executable = await host_runtime.ripgrep_executable(platform)

    url = f"https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-{build}.tar.gz"
    assert downloads.calls == [(url, host_runtime._RIPGREP_SHA256[build])]
    assert executable.path == downloads.cache / f"ripgrep-15.1.0-{build}"
    assert executable.path.read_bytes() == b"ripgrep"
