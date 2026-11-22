# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned host-side delivery for sandboxes whose network permits only the model."""

import asyncio
import hashlib
import io
import platform
import sys
import tarfile
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from responses_api_agents.cline_agent import sandbox as adapter
from responses_api_agents.cline_agent.tests.test_sessions import local_state, result


VERSION = "3.0.55"
BINARY = b'#!/bin/sh\n[ "$1" = --version ] && echo 3.0.55\n'


def archive_bytes(*, symlink=False):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        entry = tarfile.TarInfo("package/bin/cline")
        if symlink:
            entry.type, entry.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(entry)
        else:
            entry.size = len(BINARY)
            archive.addfile(entry, io.BytesIO(BINARY))
        archive.addfile(tarfile.TarInfo("package/unused-dashboard"), io.BytesIO())
    return data.getvalue()


def registry(monkeypatch, data, *, digest=None):
    async def chunks(size):
        for offset in range(0, len(data), 7):
            yield data[offset : offset + 7]

    response = SimpleNamespace(content=SimpleNamespace(iter_chunked=chunks), release=MagicMock())
    fetch = AsyncMock(return_value=response)
    monkeypatch.setattr(adapter, "request", fetch)
    monkeypatch.setattr(adapter, "raise_for_status", AsyncMock())
    monkeypatch.setitem(
        adapter.CLINE_RUNTIME_SHA512,
        VERSION,
        dict.fromkeys(("x64", "arm64"), digest or hashlib.sha512(data).hexdigest()),
    )
    return fetch, response


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture", ["x64", "arm64"])
async def test_host_download_verifies_archive_and_extracts_only_binary(monkeypatch, tmp_path, architecture):
    fetch, response = registry(monkeypatch, archive_bytes())
    destination = tmp_path / "cline"
    digest = await adapter.download_cline_binary(version=VERSION, architecture=architecture, destination=destination)
    assert destination.read_bytes() == BINARY
    assert digest == hashlib.sha512(BINARY).hexdigest()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cline", "cline.tgz"]
    assert fetch.await_args.args == (
        "GET",
        f"https://registry.npmjs.org/@cline/cli-linux-{architecture}/-/cli-linux-{architecture}-{VERSION}.tgz",
    )
    assert fetch.await_args.kwargs == {"_max_num_tries": 3}
    response.release.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("symlink", [False, True])
async def test_host_download_rejects_wrong_checksum_or_nonregular_binary(monkeypatch, tmp_path, symlink):
    _, response = registry(monkeypatch, archive_bytes(symlink=symlink), digest=None if symlink else "0" * 128)
    destination = tmp_path / "cline"
    with pytest.raises(RuntimeError, match="regular file" if symlink else "SHA-512 verification"):
        await adapter.download_cline_binary(version=VERSION, architecture="x64", destination=destination)
    assert not destination.exists()
    response.release.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"),
    reason="The installer targets Linux x86_64/aarch64 sandboxes",
)
async def test_offline_session_install_reuses_verified_runtime_without_second_download(monkeypatch, tmp_path):
    state, _, _ = local_state(tmp_path)
    fetch, _ = registry(monkeypatch, archive_bytes())
    await state.prepare(timeout=30, allow_package_install=False)
    await state.install_runtime(version=VERSION, timeout=30, download_on_host=True)
    assert (tmp_path / "runtime" / "cline").read_bytes() == BINARY
    assert sorted(p.name for p in (tmp_path / "runtime").iterdir()) == ["cline", "ready"]
    await state.install_runtime(version=VERSION, timeout=30, download_on_host=True)
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_restricted_prerequisite_refusal_does_not_attempt_package_install(tmp_path):
    state, _, _ = local_state(tmp_path)
    state.session.sandbox = SimpleNamespace(exec=AsyncMock(return_value=result("missing bash", code=1)))
    with pytest.raises(adapter.AgentSeedRejectedError, match="restricted sandbox preparation failed"):
        await state.prepare(timeout=30, allow_package_install=False)
    command = state.session.sandbox.exec.await_args.args[0]
    assert "preinstall Bash and coreutils" in command
    assert "apt-get" not in command and "dnf" not in command and "yum" not in command


@pytest.mark.asyncio
async def test_cancelled_download_joins_file_io_before_temporary_files_can_be_removed(tmp_path):
    started, released = threading.Event(), threading.Event()
    destination = tmp_path / "cline"

    def write_binary():
        started.set()
        assert released.wait(timeout=10)
        destination.write_bytes(BINARY)

    task = asyncio.create_task(adapter._complete_file_io(write_binary))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # Repeated cancellation must not detach a still-writing worker.
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert destination.read_bytes() == BINARY
