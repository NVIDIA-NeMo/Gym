# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode and ripgrep executables the agent server host downloads for a sandbox's platform."""

import asyncio
import hashlib
import io
import tarfile
from types import SimpleNamespace

import pytest

from nemo_gym.agents import SandboxPlatform
from nemo_gym.agents.utils import sandbox_python
from responses_api_agents.opencode_agent import host_runtime
from responses_api_agents.opencode_agent.runtime import OPENCODE_RELEASE_SHA256, OPENCODE_VERSION


VERSION = "9.9.9"
OPENCODE = b"#!/bin/sh\necho opencode\n"
RIPGREP = b"#!/bin/sh\necho ripgrep\n"


def archive(member: str, body: bytes) -> bytes:
    """A release archive: a gzipped tar holding ``member``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        entry = tarfile.TarInfo(member)
        entry.size = len(body)
        tar.addfile(entry, io.BytesIO(body))
    return buffer.getvalue()


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class _Response:
    def __init__(self, body: bytes, status: int) -> None:
        self.content = SimpleNamespace(iter_chunked=self.chunks)
        self.body, self.status, self.ok = body, status, status < 400

    async def chunks(self, size: int):
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None


@pytest.fixture
def releases(monkeypatch, tmp_path):
    """GitHub, serving ``bodies`` by URL, and an empty host cache. The OpenCode pins are those of the served archives."""
    opencode = archive("opencode", OPENCODE)
    served = SimpleNamespace(bodies={}, urls=[], cache=tmp_path / "sandbox_opencode", status=200)
    for asset in ("x64-baseline", "x64-baseline-musl", "arm64", "arm64-musl"):
        served.bodies[
            f"https://github.com/anomalyco/opencode/releases/download/v{VERSION}/opencode-linux-{asset}.tar.gz"
        ] = opencode
    for build in ("x86_64-unknown-linux-musl", "aarch64-unknown-linux-gnu"):
        name = f"ripgrep-15.1.0-{build}"
        served.bodies[f"https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/{name}.tar.gz"] = archive(
            f"{name}/rg", RIPGREP
        )

    async def request(method: str, url: str, **kwargs) -> _Response:
        served.urls.append(url)
        await asyncio.sleep(0)
        return _Response(served.bodies[url], served.status)

    monkeypatch.setattr(sandbox_python, "request", request)
    monkeypatch.setattr(sandbox_python, "maybe_get_global_config_dict", lambda: {"cache_dir": str(tmp_path)})
    monkeypatch.setattr(host_runtime, "_UNPACKED", {})
    monkeypatch.setattr(host_runtime, "_UNPACKING", {})
    monkeypatch.setitem(
        host_runtime.OPENCODE_RELEASE_SHA256,
        VERSION,
        dict.fromkeys(OPENCODE_RELEASE_SHA256["1.19.6"], sha256(opencode)),
    )
    monkeypatch.setattr(
        host_runtime,
        "_RIPGREP_BUILDS",
        {
            platform: (build, sha256(served.bodies[next(url for url in served.bodies if f"{build}.tar.gz" in url)]))
            for platform, (build, _) in host_runtime._RIPGREP_BUILDS.items()
        },
    )
    return served


@pytest.mark.parametrize(
    "platform, asset",
    [
        (SandboxPlatform("x86_64", "gnu", 36), "x64-baseline"),
        (SandboxPlatform("x86_64", "musl"), "x64-baseline-musl"),
        (SandboxPlatform("aarch64", "gnu", 17), "arm64"),
        (SandboxPlatform("aarch64", "musl"), "arm64-musl"),
    ],
)
async def test_the_opencode_build_is_the_one_for_the_sandbox_platform(releases, platform, asset) -> None:
    executable = await host_runtime.opencode_executable(VERSION, platform)

    # The sandbox's architecture and C library select the asset, whatever this host runs on.
    assert releases.urls == [
        f"https://github.com/anomalyco/opencode/releases/download/v{VERSION}/opencode-linux-{asset}.tar.gz"
    ]
    assert executable.path == releases.cache / f"opencode-{VERSION}-linux-{asset}"
    assert executable.path.read_bytes() == OPENCODE
    assert executable.sha256 == sha256(OPENCODE)
    assert executable.path.stat().st_mode & 0o777 == 0o755


def test_every_pinned_release_has_a_digest_for_every_sandbox_platform() -> None:
    platforms = [SandboxPlatform(arch, libc) for arch in ("x86_64", "aarch64") for libc in ("gnu", "musl")]
    assert OPENCODE_VERSION in OPENCODE_RELEASE_SHA256
    for version, digests in OPENCODE_RELEASE_SHA256.items():
        assert {host_runtime.opencode_release_asset(platform) for platform in platforms} == set(digests), version


async def test_the_archive_is_downloaded_once_and_shared_by_concurrent_seeds(releases) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 36)

    executables = await asyncio.gather(*(host_runtime.opencode_executable(VERSION, platform) for _ in range(3)))

    assert len(releases.urls) == 1
    assert len(set(executables)) == 1
    # Later seeds in this process neither download nor unpack again.
    executables[0].path.write_bytes(b"not unpacked again")
    assert await host_runtime.opencode_executable(VERSION, platform) == executables[0]
    assert executables[0].path.read_bytes() == b"not unpacked again"
    assert len(releases.urls) == 1


async def test_a_download_that_does_not_match_its_pin_is_not_used_and_the_next_seed_retries(releases) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 36)
    url = next(url for url in releases.bodies if url.endswith("opencode-linux-x64-baseline.tar.gz"))
    pinned, releases.bodies[url] = releases.bodies[url], archive("opencode", b"other bytes")

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        await host_runtime.opencode_executable(VERSION, platform)
    assert not releases.cache.exists() or list(releases.cache.iterdir()) == []

    releases.bodies[url] = pinned
    assert (await host_runtime.opencode_executable(VERSION, platform)).sha256 == sha256(OPENCODE)
    assert releases.urls == [url, url]


async def test_a_cached_archive_is_checked_before_another_process_unpacks_it(releases) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 36)
    first = await host_runtime.opencode_executable(VERSION, platform)
    cached = releases.cache / f"opencode-{VERSION}-linux-x64-baseline.tar.gz"
    # A new process has no memory of the first one's result; it finds the cache as the first left it.
    host_runtime._UNPACKED.clear()
    first.path.write_bytes(b"changed in the cache")

    second = await host_runtime.opencode_executable(VERSION, platform)

    # The executable comes from the checked archive again, never from the copy left in the cache.
    assert second == first and second.path.read_bytes() == OPENCODE
    assert len(releases.urls) == 1

    host_runtime._UNPACKED.clear()
    cached.write_bytes(archive("opencode", b"changed in the cache"))
    with pytest.raises(RuntimeError, match="does not match its pinned checksum"):
        await host_runtime.opencode_executable(VERSION, platform)


async def test_an_executable_removed_from_the_cache_is_unpacked_again(releases) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 36)
    first = await host_runtime.opencode_executable(VERSION, platform)

    # The cache is cleared while the agent server runs: first the executable, then the archive as well.
    first.path.unlink()
    assert await host_runtime.opencode_executable(VERSION, platform) == first
    assert first.path.read_bytes() == OPENCODE and len(releases.urls) == 1

    for path in releases.cache.iterdir():
        path.unlink()
    assert await host_runtime.opencode_executable(VERSION, platform) == first
    assert first.path.read_bytes() == OPENCODE and len(releases.urls) == 2


async def test_a_failed_download_is_not_cached(releases) -> None:
    platform = SandboxPlatform("aarch64", "gnu", 36)
    releases.status = 503

    with pytest.raises(RuntimeError, match="failed with HTTP 503"):
        await host_runtime.opencode_executable(VERSION, platform)

    releases.status = 200
    assert (await host_runtime.opencode_executable(VERSION, platform)).path.read_bytes() == OPENCODE


async def test_a_version_without_pinned_digests_is_not_downloaded(releases) -> None:
    with pytest.raises(RuntimeError, match=r"No pinned SHA-256 for opencode-linux-x64-baseline.tar.gz 1\.2\.3"):
        await host_runtime.opencode_executable("1.2.3", SandboxPlatform("x86_64", "gnu", 36))
    assert releases.urls == []


async def test_an_archive_without_the_executable_is_refused(releases) -> None:
    url = next(url for url in releases.bodies if url.endswith("opencode-linux-arm64.tar.gz"))
    releases.bodies[url] = archive("README.md", b"no executable")
    host_runtime.OPENCODE_RELEASE_SHA256[VERSION]["arm64"] = sha256(releases.bodies[url])

    with pytest.raises(RuntimeError, match="holds no file opencode"):
        await host_runtime.opencode_executable(VERSION, SandboxPlatform("aarch64", "gnu", 36))
    assert [path.name for path in releases.cache.iterdir()] == [f"opencode-{VERSION}-linux-arm64.tar.gz"]


@pytest.mark.parametrize(
    "platform, build",
    [
        (SandboxPlatform("x86_64", "gnu", 36), "x86_64-unknown-linux-musl"),
        # The x86_64 build is static, so it also serves musl.
        (SandboxPlatform("x86_64", "musl"), "x86_64-unknown-linux-musl"),
        (SandboxPlatform("aarch64", "gnu", 36), "aarch64-unknown-linux-gnu"),
    ],
)
async def test_the_ripgrep_build_is_the_one_for_the_sandbox_platform(releases, platform, build) -> None:
    executable = await host_runtime.ripgrep_executable(platform)

    assert releases.urls == [
        f"https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-{build}.tar.gz"
    ]
    assert executable.path == releases.cache / f"ripgrep-15.1.0-{build}"
    assert executable.path.read_bytes() == RIPGREP
    assert executable.sha256 == sha256(RIPGREP)


async def test_a_platform_without_a_ripgrep_build_gets_none(releases) -> None:
    assert await host_runtime.ripgrep_executable(SandboxPlatform("aarch64", "musl")) is None
    assert releases.urls == []
