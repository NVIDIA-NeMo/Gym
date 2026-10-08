# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.agentenv import AgentEnvProvider


pytestmark = pytest.mark.sandbox


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://agentenv.example",
        transport=httpx.MockTransport(handler),
        headers={"X-API-Key": "test-key"},
    )


async def test_agentenv_create_endpoint_fork_and_close() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/sandboxes":
            assert json.loads(request.content) == {"templateID": "osworld-template", "timeout": 600}
            return httpx.Response(201, json={"sandboxID": "parent"})
        if request.method == "POST" and request.url.path == "/sandboxes/parent/fork":
            assert json.loads(request.content) == {"count": 2, "timeout": 600}
            return httpx.Response(
                201,
                json=[
                    {"sandbox": {"sandboxID": "child-1"}},
                    {"sandbox": {"sandboxID": "child-2"}},
                ],
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    provider = AgentEnvProvider(
        connection={"endpoint": "https://agentenv.example", "api_key": "test-key"},
        create={"template": "osworld-template", "timeout_s": 600},
    )
    provider._client = _client(handler)
    sandbox = AsyncSandbox(provider, SandboxSpec(ports=(5000, 9222)))
    await sandbox.start()

    endpoint = await sandbox.endpoint(5000)
    assert endpoint.endpoint == "https://agentenv.example"
    assert endpoint.headers == {
        "X-API-Key": "test-key",
        "E2b-Sandbox-Id": "parent",
        "E2b-Sandbox-Port": "5000",
    }

    children = await sandbox.fork(2)
    assert [child._require_handle().sandbox_id for child in children] == ["child-1", "child-2"]
    assert children[0]._provider is not children[1]._provider
    await sandbox.stop()
    assert any(request.method == "DELETE" and request.url.path == "/sandboxes/parent" for request in requests)


async def test_agentenv_status_and_validation(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sandboxes/running":
            return httpx.Response(200, json={"state": "running"})
        if request.url.path == "/sandboxes/missing":
            return httpx.Response(404)
        raise AssertionError(request.url.path)

    provider = AgentEnvProvider(connection={"endpoint": "https://agentenv.example", "api_key": "test-key"})
    provider._client = _client(handler)
    running = provider._handle("running")
    missing = provider._handle("missing")
    assert await provider.status(running) == SandboxStatus.RUNNING
    assert await provider.status(missing) == SandboxStatus.STOPPED

    with pytest.raises(ValueError, match="absolute http"):
        AgentEnvProvider(connection={"endpoint": "agentenv.example"})
    missing_ca = tmp_path / "missing.pem"
    with pytest.raises(FileNotFoundError, match="TLS CA"):
        AgentEnvProvider(
            connection={
                "endpoint": "https://agentenv.example",
                "api_key": "test-key",
                "tls_ca": str(missing_ca),
            }
        )._build_client()


async def test_agentenv_unsupported_generic_operations() -> None:
    provider = AgentEnvProvider(connection={"endpoint": "https://agentenv.example", "api_key": "test-key"})
    handle = provider._handle("sandbox")
    with pytest.raises(NotImplementedError, match="generic exec"):
        await provider.exec(handle, "true")
