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

"""Standalone tests for the sandbox server and the connect capability.

These are hermetic: a fake in-process provider backs everything, so no docker,
cloud, or external OpenSandbox server is needed. Three things are covered:

1. ``SandboxRef`` serialization.
2. Direct reattach with a ``ConnectableProvider`` (``serialize``/``connect``) and
   no sandbox server.
3. The full HTTP path: the real ``SandboxServer`` app driven in-process (via
   httpx ASGI transport) through ``RemoteSandboxProvider``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest

from nemo_gym.sandbox import (
    SCOPE_OPERATE,
    SCOPE_OWNER,
    AsyncSandbox,
    SandboxExecResult,
    SandboxHandle,
    SandboxRef,
    SandboxSpec,
    SandboxStatus,
    register_provider,
)
from nemo_gym.sandbox.providers.remote import RemoteSandboxProvider
from nemo_gym.server_utils import ServerClient


# --- load the sandbox server app (lives outside the nemo_gym package) --------

_APP_PATH = Path(__file__).resolve().parents[2] / "sandbox_servers" / "sandbox_server" / "app.py"
_spec = importlib.util.spec_from_file_location("sandbox_server_app", _APP_PATH)
sandbox_server_app = importlib.util.module_from_spec(_spec)
# Register before executing so pydantic can resolve the config's annotations
# (the module uses ``from __future__ import annotations``, so forward refs are
# looked up via ``sys.modules[cls.__module__]``).
sys.modules["sandbox_server_app"] = sandbox_server_app
_spec.loader.exec_module(sandbox_server_app)
SandboxServer = sandbox_server_app.SandboxServer
SandboxServerConfig = sandbox_server_app.SandboxServerConfig


# --- a fake provider: an in-memory "external control plane" ------------------

_FAKE_STORE: dict[str, dict[str, Any]] = {}


class FakeProvider:
    """In-memory provider that supports the connect capability.

    Boxes live in a process-global store keyed by id, so a second provider
    instance (a different process, conceptually) can reconnect by id.
    """

    name = "fake"

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        sid = f"fake-{uuid4().hex[:8]}"
        _FAKE_STORE[sid] = {"files": {}, "closed": False, "workdir": spec.workdir}
        return SandboxHandle(sandbox_id=sid, provider_name=self.name, raw=sid)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        box = _FAKE_STORE.get(handle.sandbox_id)
        if box is None or box["closed"]:
            return SandboxExecResult(stdout=None, stderr="no such sandbox", return_code=1)
        if command.startswith("cat "):
            data = box["files"].get(command[4:].strip())
            if data is None:
                return SandboxExecResult(stdout=None, stderr="no such file", return_code=1)
            return SandboxExecResult(stdout=data.decode(), stderr=None, return_code=0)
        if command.startswith("echo "):
            return SandboxExecResult(stdout=command[len("echo ") :], stderr=None, return_code=0)
        return SandboxExecResult(stdout=f"ran: {command}", stderr=None, return_code=0)

    async def upload_file(self, handle, source_path, target_path) -> None:
        _FAKE_STORE[handle.sandbox_id]["files"][target_path] = Path(source_path).read_bytes()

    async def download_file(self, handle, source_path, target_path) -> None:
        data = _FAKE_STORE[handle.sandbox_id]["files"][source_path]
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(data)

    async def status(self, handle) -> SandboxStatus:
        box = _FAKE_STORE.get(handle.sandbox_id)
        if box is None:
            return SandboxStatus.UNKNOWN
        return SandboxStatus.STOPPED if box["closed"] else SandboxStatus.RUNNING

    async def close(self, handle) -> None:
        box = _FAKE_STORE.get(handle.sandbox_id)
        if box is not None:
            box["closed"] = True

    async def aclose(self) -> None:
        return None

    async def serialize_handle(self, handle, *, scope=None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor) -> SandboxHandle:
        sid = str(descriptor["sandbox_id"])
        if sid not in _FAKE_STORE:
            raise RuntimeError(f"no such sandbox {sid!r}")
        return SandboxHandle(sandbox_id=sid, provider_name=self.name, raw=sid)


class _OpsOnlyProvider:
    """A provider without the connect capability (for the negative case)."""

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


# --- httpx ASGI transport adapter (aiohttp-shaped responses) -----------------


class _Resp:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status = response.status_code

    async def json(self) -> Any:
        return self._response.json()


class ASGITransportAdapter:
    """A SandboxHttpTransport that drives an ASGI app in-process, exposing the
    aiohttp ``ClientResponse`` shape the remote provider expects."""

    def __init__(self, app: Any) -> None:
        self._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")

    async def request(self, method, url, *, json=None, params=None, headers=None) -> _Resp:
        response = await self._client.request(method, url, json=json, params=params, headers=headers)
        return _Resp(response)

    async def raise_for_status(self, response: _Resp) -> None:
        response._response.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


def _build_server(monkeypatch: pytest.MonkeyPatch) -> SandboxServer:
    register_provider("fake", FakeProvider, override=True)
    # The server resolves its provider against the global config; an inline
    # single-key mapping does not need it, so stub it to stay hermetic.
    monkeypatch.setattr(sandbox_server_app, "get_global_config_dict", lambda: {})
    config = SandboxServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="sbx",
        sandbox_provider={"fake": {}},
        max_concurrent=4,
        default_ttl_s=None,
    )
    return SandboxServer(config=config, server_client=MagicMock(spec=ServerClient))


# --- tests -------------------------------------------------------------------


def test_sandbox_ref_roundtrip_and_scope() -> None:
    ref = SandboxRef(
        server_url="http://host:8080",
        sandbox_id="abc",
        lease_token="tok",
        provider_name="fake",
        scope=SCOPE_OWNER,
        workdir="/work",
    )
    restored = SandboxRef.from_dict(ref.to_dict())
    assert restored == ref
    assert restored.can_close is True
    assert SandboxRef.from_dict({"server_url": "u", "sandbox_id": "s", "lease_token": "t"}).scope == SCOPE_OPERATE
    assert SandboxRef.from_dict({"server_url": "u", "sandbox_id": "s", "lease_token": "t"}).can_close is False


def test_direct_reattach_without_server(tmp_path: Path) -> None:
    async def _run() -> None:
        owner_provider = FakeProvider()
        sandbox = await AsyncSandbox(owner_provider, SandboxSpec(workdir="/w")).start()

        payload = tmp_path / "f.txt"
        payload.write_text("hello-direct")
        await sandbox.upload(payload, "/w/f.txt")

        descriptor = await sandbox.serialize()
        assert "sandbox_id" in descriptor

        # A separate provider instance reconnects by id, no sandbox server.
        other_provider = FakeProvider()
        reattached = await AsyncSandbox.connect(descriptor, provider=other_provider)
        result = await reattached.exec("cat /w/f.txt")
        assert result.return_code == 0
        assert result.stdout == "hello-direct"

    asyncio.run(_run())


def test_serialize_requires_connect_capability() -> None:
    async def _run() -> None:
        sandbox = await AsyncSandbox(_OpsOnlyProvider(), SandboxSpec()).start()
        with pytest.raises(RuntimeError, match="serialize"):
            await sandbox.serialize()

    asyncio.run(_run())


def test_server_end_to_end_over_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def _run() -> None:
        server = _build_server(monkeypatch)
        app = server.setup_webserver()
        transport = ASGITransportAdapter(app)
        try:
            provider = RemoteSandboxProvider(server_url="http://testserver", transport=transport)
            sandbox = await AsyncSandbox(provider, SandboxSpec(image="fake-image", workdir="/w")).start()

            # exec
            result = await sandbox.exec("echo hi")
            assert result.return_code == 0
            assert result.stdout == "hi"

            # upload + download round-trip through the server
            payload = tmp_path / "in.txt"
            payload.write_text("through-the-server")
            await sandbox.upload(payload, "/w/in.txt")
            out = tmp_path / "out.txt"
            await sandbox.download("/w/in.txt", out)
            assert out.read_text() == "through-the-server"

            # status
            assert await sandbox.status() == SandboxStatus.RUNNING

            # owner ref serializes to itself; an operate co-lease is a fresh ref
            owner_ref = SandboxRef.from_dict(await sandbox.serialize())
            assert owner_ref.scope == SCOPE_OWNER
            operate_ref = SandboxRef.from_dict(await sandbox.serialize(scope=SCOPE_OPERATE))
            assert operate_ref.scope == SCOPE_OPERATE
            assert operate_ref.sandbox_id == owner_ref.sandbox_id
            assert operate_ref.lease_token != owner_ref.lease_token

            # a second server reattaches with the co-lease and operates the box
            co_provider = RemoteSandboxProvider(server_url="http://testserver", transport=transport)
            co_sandbox = await AsyncSandbox.connect(operate_ref, provider=co_provider)
            co_result = await co_sandbox.exec("echo from-colease")
            assert co_result.stdout == "from-colease"

            # a bad lease token is rejected
            bad_ref = SandboxRef(
                server_url="http://testserver",
                sandbox_id=owner_ref.sandbox_id,
                lease_token="not-a-valid-token",
                scope=SCOPE_OPERATE,
            )
            bad_provider = RemoteSandboxProvider(server_url="http://testserver", transport=transport)
            bad_sandbox = await AsyncSandbox.connect(bad_ref, provider=bad_provider)
            with pytest.raises(httpx.HTTPStatusError):
                await bad_sandbox.exec("echo nope")

            # releasing the co-lease does not destroy the box; owner close does
            await co_sandbox.stop()
            assert await sandbox.status() == SandboxStatus.RUNNING
            await sandbox.stop()
        finally:
            await transport.aclose()

    asyncio.run(_run())
