# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from nemo_gym.agents import AgentSeedRejectedError
from responses_api_agents.droid_agent import runtime


PACKAGE = b"fake droid package bytes"
INTEGRITY = "sha512-" + base64.b64encode(hashlib.sha512(PACKAGE).digest()).decode()
SHA256 = hashlib.sha256(PACKAGE).hexdigest()


class Reply:
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status
        self.content = SimpleNamespace(iter_chunked=self.chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise ClientResponseError(SimpleNamespace(real_url="https://registry.npmjs.org"), (), status=self.status)

    async def text(self):
        return self.body.decode()

    async def chunks(self, size):
        for start in range(0, len(self.body), 4):
            yield self.body[start : start + 4]


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """Serve npm metadata and one tarball; record each requested URL."""
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runtime, "_VERIFIED", {})
    monkeypatch.setattr(runtime, "_SOURCES", {})
    served = {"package": PACKAGE, "urls": [], "metadata_status": 200}

    async def request(method, url, **kwargs):
        served["urls"].append(url)
        if url.endswith(".tgz"):
            return Reply(served["package"])
        tarball = "https://registry.npmjs.org/@factory/cli-linux-x64/-/cli-linux-x64-9.9.9.tgz"
        metadata = json.dumps({"dist": {"tarball": tarball, "integrity": INTEGRITY}}).encode()
        return Reply(metadata, served["metadata_status"])

    monkeypatch.setattr(runtime, "request", request)
    return served


@pytest.mark.parametrize(
    "probe,target",
    [
        ("Linux\nx86_64\ngnu\navx2\n", "linux-x64"),
        ("Linux\nx86_64\ngnu\nbaseline\n", "linux-x64-baseline"),
        ("Linux\naarch64\ngnu\nbaseline\n", "linux-arm64"),
    ],
)
def test_platform_probe_selects_the_matching_package(probe, target):
    assert runtime.package_target(probe) == target


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="the platform probe is a POSIX shell script")
@pytest.mark.parametrize(
    "glibc,musl_loader,target",
    [
        (True, False, "linux-x64"),
        # Musl tooling can add musl's loader to a glibc image; Droid still runs there.
        (True, True, "linux-x64"),
        (False, True, None),
    ],
)
def test_platform_probe_counts_musl_only_without_glibc(tmp_path, glibc, musl_loader, target):
    """Run the probe script itself, with stub commands standing in for the image's tools."""
    commands = {
        "uname": 'case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac',
        "getconf": "exit 0" if glibc else "exit 1",
        "ls": "exit 0" if musl_loader else "exit 2",
        "grep": "exit 0",
    }
    for name, script in commands.items():
        (tmp_path / name).write_text(f"#!/bin/sh\n{script}\n")
        (tmp_path / name).chmod(0o755)
    probe = subprocess.run(
        ["/bin/sh", "-c", runtime.PLATFORM_PROBE],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if target is None:
        with pytest.raises(AgentSeedRejectedError, match="musl and no glibc"):
            runtime.package_target(probe)
    else:
        assert runtime.package_target(probe) == target


@pytest.mark.parametrize(
    "probe,message",
    [
        ("Linux\nx86_64\nmusl\navx2\n", "glibc"),
        ("Linux\nriscv64\ngnu\nbaseline\n", "architecture"),
        ("Darwin\narm64\ngnu\nbaseline\n", "platform"),
        ("Linux\nx86_64\n", "platform"),
    ],
)
def test_unsupported_platforms_are_named(probe, message):
    with pytest.raises(ValueError, match=message) as error:
        runtime.package_target(probe)
    # A platform Droid has no build for is refused as final; unreadable probe output stays retryable.
    assert isinstance(error.value, AgentSeedRejectedError) is (len(probe.splitlines()) == 4)


def test_default_version_has_pinned_integrity_for_every_target():
    assert {target for version, target in runtime.PINNED_INTEGRITY if version == "1.1.0"} == runtime.TARGETS


@pytest.mark.asyncio
async def test_package_is_verified_cached_and_reused(registry):
    path, checksum = await runtime.cached_package("9.9.9", "linux-x64")
    assert path.read_bytes() == PACKAGE and checksum == SHA256
    assert path.parent.stat().st_mode & 0o777 == 0o700
    again, _ = await runtime.cached_package("9.9.9", "linux-x64")
    assert again == path
    assert sum(url.endswith(".tgz") for url in registry["urls"]) == 1


@pytest.mark.asyncio
async def test_registry_metadata_is_fetched_once_per_version(registry):
    for _ in range(3):
        await runtime.cached_package("9.9.9", "linux-x64")
    assert registry["urls"] == [
        "https://registry.npmjs.org/@factory/cli-linux-x64/9.9.9",
        "https://registry.npmjs.org/@factory/cli-linux-x64/-/cli-linux-x64-9.9.9.tgz",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,final", [(404, True), (503, False)])
async def test_unpublished_version_is_refused_as_final(registry, status, final):
    registry["metadata_status"] = status
    with pytest.raises(Exception) as error:
        await runtime.cached_package("9.9.9", "linux-x64")
    # A missing version is the configuration's fault; a registry outage may pass on a retry.
    assert isinstance(error.value, AgentSeedRejectedError) is final
    assert final or isinstance(error.value, ClientResponseError)


@pytest.mark.asyncio
async def test_integrity_mismatch_leaves_no_cached_package(registry):
    registry["package"] = b"tampered"
    with pytest.raises(ValueError, match="integrity mismatch"):
        await runtime.cached_package("9.9.9", "linux-x64")
    assert not list((runtime.CACHE_DIR / "9.9.9").iterdir())


@pytest.mark.asyncio
async def test_pinned_version_does_not_trust_registry_metadata(registry):
    # The pinned integrity of 1.1.0 does not match the fake bytes, whatever the registry says.
    with pytest.raises(ValueError, match="integrity mismatch"):
        await runtime.cached_package("1.1.0", "linux-x64")
    assert registry["urls"] == ["https://registry.npmjs.org/@factory/cli-linux-x64/-/cli-linux-x64-1.1.0.tgz"]


@pytest.mark.asyncio
@pytest.mark.parametrize("version,target", [("1.1", "linux-x64"), ("1.1.0", "linux-x64-musl")])
async def test_inexact_version_or_unknown_target_is_rejected(version, target):
    with pytest.raises(ValueError, match="exact version"):
        await runtime.cached_package(version, target)


def result(code=0, stdout=""):
    return SimpleNamespace(return_code=code, error_type=None, stdout=stdout, stderr="installer said no")


@pytest.mark.asyncio
async def test_present_runtime_is_reused_without_upload(monkeypatch, tmp_path):
    package = tmp_path / "droid.tgz"
    package.write_bytes(PACKAGE)
    monkeypatch.setattr(runtime, "cached_package", AsyncMock(return_value=(package, SHA256)))
    sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=[result(stdout="Linux\nx86_64\ngnu\navx2\n"), result()]), upload=AsyncMock()
    )
    await runtime.install_sandbox_runtime(
        sandbox, version="1.1.0", directory="/tmp/s", runtime="/tmp/nemo-gym-droid-runtime-1.1.0", timeout=5
    )
    runtime.cached_package.assert_awaited_once_with("1.1.0", "linux-x64")
    check = sandbox.exec.await_args_list[1].args[0]
    assert "--check-only" in check and f"--sha256 {SHA256}" in check
    sandbox.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_absent_runtime_uploads_the_package_in_pieces(monkeypatch, tmp_path):
    package = tmp_path / "droid.tgz"
    package.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    monkeypatch.setattr(runtime, "cached_package", AsyncMock(return_value=(package, SHA256)))
    sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=[result(stdout="Linux\nx86_64\ngnu\navx2\n"), result(3), result(), result()]),
        upload=AsyncMock(),
    )
    await runtime.install_sandbox_runtime(
        sandbox, version="1.1.0", directory="/tmp/s", runtime="/tmp/nemo-gym-droid-runtime-1.1.0", timeout=5
    )
    assert sandbox.upload.await_count == 3
    assembled, installed = (call.args[0] for call in sandbox.exec.await_args_list[2:])
    assert assembled.endswith(
        "> /tmp/s/droid-package.tgz && rm -f /tmp/s/droid-upload-0000.part "
        "/tmp/s/droid-upload-0001.part /tmp/s/droid-upload-0002.part"
    )
    assert installed.endswith("--source /tmp/s/droid-package.tgz") and "--check-only" not in installed


@pytest.mark.asyncio
async def test_failed_check_or_install_reports_the_installer_output(monkeypatch, tmp_path):
    package = tmp_path / "droid.tgz"
    package.write_bytes(PACKAGE)
    monkeypatch.setattr(runtime, "cached_package", AsyncMock(return_value=(package, SHA256)))
    sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=[result(stdout="Linux\nx86_64\ngnu\navx2\n"), result(2)]), upload=AsyncMock()
    )
    with pytest.raises(RuntimeError, match="installer said no"):
        await runtime.install_sandbox_runtime(sandbox, version="1.1.0", directory="/tmp/s", runtime="/r", timeout=5)
    sandbox.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_cached_package_is_downloaded_again(registry):
    cached = runtime.CACHE_DIR / "9.9.9" / "cli-linux-x64.tgz"
    cached.parent.mkdir(parents=True)
    for directory in (runtime.CACHE_DIR, cached.parent):
        directory.chmod(0o700)
    cached.write_bytes(b"truncated")
    path, checksum = await runtime.cached_package("9.9.9", "linux-x64")
    assert path == cached and path.read_bytes() == PACKAGE and checksum == SHA256
    assert sum(url.endswith(".tgz") for url in registry["urls"]) == 1


@pytest.mark.asyncio
async def test_failed_upload_releases_every_upload_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_UPLOAD_SLOTS", asyncio.Semaphore(4))

    async def to_thread(function, *args):
        # In the loop, so the failure lands while the group still holds pieces it has not started.
        await asyncio.sleep(0)
        return function(*args)

    monkeypatch.setattr(runtime.asyncio, "to_thread", to_thread)
    package = tmp_path / "droid.tgz"
    package.write_bytes(b"x" * (6 * 1024 * 1024))

    async def upload(source, destination):
        await asyncio.sleep(0)
        if destination.endswith("-0001.part"):
            raise ConnectionError("upload lost")
        await asyncio.sleep(0.01)

    sandbox = SimpleNamespace(upload=upload, exec=AsyncMock(return_value=result()))
    with pytest.raises(ExceptionGroup) as error:
        await runtime._upload_package(sandbox, package, "/tmp/s", timeout=5)
    assert error.group_contains(ConnectionError)
    sandbox.exec.assert_not_awaited()
    # The limit is process-wide: a slot a failed upload kept would stall every later seed once four leaked.
    for _ in range(4):
        await asyncio.wait_for(runtime._UPLOAD_SLOTS.acquire(), timeout=1)
