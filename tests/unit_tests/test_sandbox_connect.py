# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ConnectableProvider capability and the serialize/connect facade.

Hermetic: a fake in-process provider stands in for a reattachable backend, so
these exercise the facade and protocol without any sandbox server.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from nemo_gym.sandbox import (
    AsyncSandbox,
    ConnectableProvider,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


_STORE: dict[str, dict[str, Any]] = {}


class FakeConnectableProvider:
    """In-memory provider that supports connect; boxes live in a global store
    keyed by id, so a second instance can reconnect by id."""

    name = "fake_connectable"

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        sid = f"fake-{uuid4().hex[:8]}"
        _STORE[sid] = {"files": {}, "closed": False}
        return SandboxHandle(sandbox_id=sid, provider_name=self.name, raw=sid)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        box = _STORE.get(handle.sandbox_id)
        if box is None or box["closed"]:
            return SandboxExecResult(stdout=None, stderr="no such sandbox", return_code=1)
        if command.startswith("cat "):
            data = box["files"].get(command[4:].strip())
            if data is None:
                return SandboxExecResult(stdout=None, stderr="no such file", return_code=1)
            return SandboxExecResult(stdout=data.decode(), stderr=None, return_code=0)
        return SandboxExecResult(stdout=f"ran: {command}", stderr=None, return_code=0)

    async def upload_file(self, handle, source_path, target_path) -> None:
        _STORE[handle.sandbox_id]["files"][target_path] = Path(source_path).read_bytes()

    async def download_file(self, handle, source_path, target_path) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(_STORE[handle.sandbox_id]["files"][source_path])

    async def status(self, handle) -> SandboxStatus:
        box = _STORE.get(handle.sandbox_id)
        if box is None:
            return SandboxStatus.UNKNOWN
        return SandboxStatus.STOPPED if box["closed"] else SandboxStatus.RUNNING

    async def close(self, handle) -> None:
        box = _STORE.get(handle.sandbox_id)
        if box is not None:
            box["closed"] = True

    async def aclose(self) -> None:
        return None

    async def serialize_handle(self, handle, *, scope=None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor) -> SandboxHandle:
        sid = str(descriptor["sandbox_id"])
        if sid not in _STORE:
            raise RuntimeError(f"no such sandbox {sid!r}")
        return SandboxHandle(sandbox_id=sid, provider_name=self.name, raw=sid)


class _OpsOnlyProvider:
    """A provider without the connect capability (negative case)."""

    name = "ops_only"

    async def create(self, spec):
        return SandboxHandle(sandbox_id="x", provider_name=self.name, raw="x")

    async def exec(self, *a, **k):
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, *a, **k):
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


def test_connectable_provider_isinstance() -> None:
    assert isinstance(FakeConnectableProvider(), ConnectableProvider)
    assert not isinstance(_OpsOnlyProvider(), ConnectableProvider)


def test_serialize_then_connect_round_trip(tmp_path: Path) -> None:
    async def _run() -> None:
        sandbox = await AsyncSandbox(FakeConnectableProvider(), SandboxSpec(workdir="/w")).start()
        payload = tmp_path / "f.txt"
        payload.write_text("hello-connect")
        await sandbox.upload(payload, "/w/f.txt")

        descriptor = await sandbox.serialize()
        assert "sandbox_id" in descriptor

        # A separate provider instance rebuilds a working handle from the descriptor.
        reattached = await AsyncSandbox.connect(descriptor, provider=FakeConnectableProvider())
        result = await reattached.exec("cat /w/f.txt")
        assert result.return_code == 0
        assert result.stdout == "hello-connect"

    asyncio.run(_run())


def test_serialize_requires_connect_capability() -> None:
    async def _run() -> None:
        sandbox = await AsyncSandbox(_OpsOnlyProvider(), SandboxSpec()).start()
        with pytest.raises(RuntimeError, match="serialize"):
            await sandbox.serialize()

    asyncio.run(_run())


def test_connect_requires_connect_capability() -> None:
    async def _run() -> None:
        with pytest.raises(RuntimeError, match="serialize"):
            await AsyncSandbox.connect({"sandbox_id": "x"}, provider=_OpsOnlyProvider())

    asyncio.run(_run())
