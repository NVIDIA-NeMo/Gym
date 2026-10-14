# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The uv build a harness uploads is chosen for the sandbox's OS, architecture and C library."""

import asyncio
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.agent_utils import uv_bootstrap
from nemo_gym.agent_utils.uv_bootstrap import (
    _PLATFORM_PROBE,
    _download_uv,
    _host_uv_runs_in,
    _host_uv_version,
    _probe_sandbox_platform,
    _SandboxPlatform,
    _uv_release_target,
    uv_for_sandbox,
)


def _sandbox(system="Linux", machine="x86_64", libc="gnu", return_code=0):
    sandbox = AsyncMock()
    sandbox.exec.return_value = MagicMock(return_code=return_code, stdout=f"{system}\n{machine}\n{libc}\n", stderr="")
    return sandbox


def _host(monkeypatch, *, system="Linux", machine="aarch64", uv="/usr/bin/uv", version="0.12.3"):
    monkeypatch.setattr(uv_bootstrap.platform, "system", lambda: system)
    monkeypatch.setattr(uv_bootstrap.platform, "machine", lambda: machine)
    monkeypatch.setattr(uv_bootstrap.shutil, "which", lambda name: uv)
    monkeypatch.setattr(uv_bootstrap, "_host_uv_version", lambda path: version)


async def test_probe_reports_kernel_machine_and_libc():
    sandbox = _sandbox("Linux", "aarch64", "musl")
    assert await _probe_sandbox_platform(sandbox, "/app") == _SandboxPlatform("Linux", "aarch64", "musl")
    sandbox.exec.assert_awaited_once_with(_PLATFORM_PROBE, cwd="/app", timeout_s=30)
    with pytest.raises(RuntimeError, match="Could not determine the sandbox platform"):
        await _probe_sandbox_platform(_sandbox(return_code=1), "/app")


@pytest.mark.parametrize(
    "platform_, target",
    [
        (_SandboxPlatform("Linux", "x86_64", "gnu"), "x86_64-unknown-linux-musl"),
        (_SandboxPlatform("Linux", "amd64", "musl"), "x86_64-unknown-linux-musl"),
        (_SandboxPlatform("Linux", "aarch64", "musl"), "aarch64-unknown-linux-musl"),
        (_SandboxPlatform("Linux", "arm64", "gnu"), "aarch64-unknown-linux-musl"),
    ],
)
def test_release_target_is_the_static_musl_build_for_the_architecture(platform_, target):
    # The musl build is static, so one download serves glibc and musl sandboxes alike.
    assert _uv_release_target(platform_) == target


def test_release_target_rejects_other_kernels_and_unknown_machines():
    with pytest.raises(RuntimeError, match="Linux sandboxes only"):
        _uv_release_target(_SandboxPlatform("Darwin", "arm64", "gnu"))
    with pytest.raises(RuntimeError, match="No uv build is known"):
        _uv_release_target(_SandboxPlatform("Linux", "riscv64", "gnu"))


def test_host_uv_is_reused_only_on_a_matching_glibc_linux_sandbox(monkeypatch):
    _host(monkeypatch, machine="x86_64")
    assert _host_uv_runs_in(_SandboxPlatform("Linux", "x86_64", "gnu"))
    assert _host_uv_runs_in(_SandboxPlatform("Linux", "amd64", "gnu"))
    assert not _host_uv_runs_in(_SandboxPlatform("Linux", "x86_64", "musl"))  # host build is gnu
    assert not _host_uv_runs_in(_SandboxPlatform("Linux", "aarch64", "gnu"))
    _host(monkeypatch, system="Darwin", machine="arm64")
    assert not _host_uv_runs_in(_SandboxPlatform("Linux", "aarch64", "gnu"))


async def test_same_platform_uses_host_uv(monkeypatch, tmp_path):
    _host(monkeypatch, machine="aarch64")
    assert await uv_for_sandbox(_sandbox("Linux", "aarch64", "gnu"), "/app", str(tmp_path)) == Path("/usr/bin/uv")
    assert not any(tmp_path.iterdir())


async def test_other_platforms_download_the_matching_musl_build_once(monkeypatch, tmp_path):
    _host(monkeypatch, machine="aarch64")
    downloads: list[tuple[str, str]] = []

    def fake_download(version, target, destination):
        downloads.append((version, target))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"uv")

    monkeypatch.setattr(uv_bootstrap, "_download_uv", fake_download)
    first = await uv_for_sandbox(_sandbox("Linux", "x86_64", "gnu"), "/app", str(tmp_path))
    second = await uv_for_sandbox(_sandbox("Linux", "x86_64", "gnu"), "/app", str(tmp_path))
    musl = await uv_for_sandbox(_sandbox("Linux", "aarch64", "musl"), "/app", str(tmp_path))
    assert first == second == tmp_path / "0.12.3" / "x86_64-unknown-linux-musl" / "uv"
    assert musl == tmp_path / "0.12.3" / "aarch64-unknown-linux-musl" / "uv"
    assert downloads == [("0.12.3", "x86_64-unknown-linux-musl"), ("0.12.3", "aarch64-unknown-linux-musl")]


async def test_concurrent_cold_cache_calls_share_one_download(monkeypatch, tmp_path):
    _host(monkeypatch, machine="aarch64")
    started = threading.Event()
    release = threading.Event()
    downloads: list[tuple[str, str]] = []

    def slow_download(version, target, destination):
        # Runs in the worker thread like the real download. It is held open until every other caller
        # has queued behind the lock, so none of them can observe a cold cache after it finishes.
        downloads.append((version, target))
        started.set()
        assert release.wait(timeout=10)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"uv")

    monkeypatch.setattr(uv_bootstrap, "_download_uv", slow_download)
    tasks = [
        asyncio.create_task(uv_for_sandbox(_sandbox("Linux", "x86_64", "gnu"), "/app", str(tmp_path)))
        for _ in range(8)
    ]
    while not started.is_set():
        await asyncio.sleep(0.01)
    for _ in range(10):  # let every other task reach the lock
        await asyncio.sleep(0)
    release.set()
    paths = await asyncio.gather(*tasks)
    assert downloads == [("0.12.3", "x86_64-unknown-linux-musl")]
    assert set(paths) == {tmp_path / "0.12.3" / "x86_64-unknown-linux-musl" / "uv"}
    assert paths[0].read_bytes() == b"uv"


async def test_missing_host_uv_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(uv_bootstrap.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="requires uv"):
        await uv_for_sandbox(_sandbox(), "/app", str(tmp_path))


def test_host_uv_version_parses(monkeypatch):
    monkeypatch.setattr(
        uv_bootstrap.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="uv 0.12.3 (aarch64-unknown-linux-gnu)\n"),
    )
    assert _host_uv_version("/usr/bin/uv") == "0.12.3"
    monkeypatch.setattr(uv_bootstrap.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="nope"))
    with pytest.raises(RuntimeError, match="Unexpected uv version"):
        _host_uv_version("/usr/bin/uv")


def test_download_uv_extracts_the_binary(monkeypatch, tmp_path):
    release_dir = tmp_path / "0.12.3"
    release_dir.mkdir()
    archive = release_dir / "uv-x86_64-unknown-linux-musl.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        binary = tmp_path / "uv"
        binary.write_bytes(b"#!/bin/sh\necho uv\n")
        tar.add(binary, arcname="uv-x86_64-unknown-linux-musl/uv")
    monkeypatch.setattr(uv_bootstrap, "UV_RELEASE_URL", tmp_path.as_uri() + "/{version}/uv-{target}.tar.gz")
    destination = tmp_path / "cache" / "0.12.3" / "x86_64-unknown-linux-musl" / "uv"

    _download_uv("0.12.3", "x86_64-unknown-linux-musl", destination)

    assert destination.read_bytes().startswith(b"#!/bin/sh")
    assert destination.stat().st_mode & 0o111
    with pytest.raises(RuntimeError, match="Could not download uv"):
        _download_uv("9.9.9", "x86_64-unknown-linux-musl", tmp_path / "cache" / "9.9.9" / "uv")


async def test_a_sandbox_that_cannot_answer_the_probe_gets_the_host_uv(monkeypatch, tmp_path, caplog):
    _host(monkeypatch, machine="aarch64")
    assert await uv_for_sandbox(_sandbox(return_code=1), "/app", str(tmp_path)) == Path("/usr/bin/uv")
    assert "uploading the host's uv" in caplog.text
