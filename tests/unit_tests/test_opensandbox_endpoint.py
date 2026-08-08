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

import asyncio
from types import SimpleNamespace

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    SandboxEndpoint,
    SandboxHandle,
    SandboxSpec,
    SupportsSandboxEndpoint,
)
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


def _provider() -> opensandbox_provider.OpenSandboxProvider:
    return opensandbox_provider.OpenSandboxProvider(probe={"command": None})


def _handle(raw: object) -> SandboxHandle:
    return SandboxHandle(sandbox_id="sbx-1", provider_name="opensandbox", raw=raw)


class _RawWithEndpoint:
    def __init__(
        self, endpoint: str = "http://sandbox.example/v1/sandboxes/sbx-1/proxy/6000", headers: dict | None = None
    ):
        self._endpoint = endpoint
        self._headers = {"OPEN-SANDBOX-API-KEY": "secret"} if headers is None else headers
        self.requested_ports: list[int] = []

    async def get_endpoint(self, port: int) -> SimpleNamespace:
        self.requested_ports.append(port)
        return SimpleNamespace(endpoint=self._endpoint, headers=self._headers)


def test_opensandbox_provider_satisfies_the_endpoint_protocol() -> None:
    assert isinstance(_provider(), SupportsSandboxEndpoint)


def test_endpoint_returns_the_sdk_url_and_auth_headers() -> None:
    raw = _RawWithEndpoint()
    resolved = asyncio.run(_provider().endpoint(_handle(raw), 6000))
    assert isinstance(resolved, SandboxEndpoint)
    assert resolved.endpoint == "http://sandbox.example/v1/sandboxes/sbx-1/proxy/6000"
    assert resolved.headers == {"OPEN-SANDBOX-API-KEY": "secret"}
    assert raw.requested_ports == [6000]


def test_endpoint_requires_a_recent_sdk() -> None:
    class RawWithoutEndpoint:
        pass

    with pytest.raises(NotImplementedError, match="get_endpoint"):
        asyncio.run(_provider().endpoint(_handle(RawWithoutEndpoint()), 6000))


def test_endpoint_rejects_an_empty_url() -> None:
    with pytest.raises(RuntimeError, match="empty endpoint"):
        asyncio.run(_provider().endpoint(_handle(_RawWithEndpoint(endpoint="")), 6000))


def test_endpoint_headers_default_to_empty_dict_without_configured_key() -> None:
    resolved = asyncio.run(_provider().endpoint(_handle(_RawWithEndpoint(headers={})), 6000))
    assert resolved.headers == {}


def test_schemeless_endpoint_is_absolutized_and_key_header_injected() -> None:
    """The SDK returns proxy endpoints without a scheme and with empty headers (observed on
    a production cluster, SDK 0.1.15); the provider must produce an absolute URL and carry the API key."""
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "http://sandbox.example",
            "api_key": "secret",  # pragma: allowlist secret
            "use_server_proxy": True,
        },
        probe={"command": None},
    )
    raw = _RawWithEndpoint(endpoint="sandbox.example/v1/sandboxes/sbx-1/proxy/6000", headers={})
    resolved = asyncio.run(provider.endpoint(_handle(raw), 6000))
    assert resolved.endpoint == "http://sandbox.example/v1/sandboxes/sbx-1/proxy/6000"
    assert resolved.headers == {"OPEN-SANDBOX-API-KEY": "secret"}  # pragma: allowlist secret


def test_direct_mode_endpoint_never_carries_the_key() -> None:
    """A direct endpoint terminates at the sandbox, which runs untrusted code; handing it
    the API key would leak the key to the workload. Mirrors the provider's proxy-only rule."""
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "http://sandbox.example", "api_key": "secret"},  # pragma: allowlist secret
        probe={"command": None},
    )
    raw = _RawWithEndpoint(endpoint="http://pod.example:6000", headers={})
    resolved = asyncio.run(provider.endpoint(_handle(raw), 6000))
    assert resolved.headers == {}


def test_sdk_supplied_headers_are_never_overridden() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "http://sandbox.example", "api_key": "secret"},
        probe={"command": None},
    )
    raw = _RawWithEndpoint(headers={"X-Route-Token": "t"})
    resolved = asyncio.run(provider.endpoint(_handle(raw), 6000))
    assert resolved.headers == {"X-Route-Token": "t"}


def test_async_sandbox_endpoint_flows_through_the_provider() -> None:
    """AsyncSandbox.endpoint() must accept the opensandbox provider once the port is declared."""

    async def main() -> SandboxEndpoint:
        provider = _provider()
        sandbox = AsyncSandbox(provider=provider, spec=SandboxSpec(image="img", ports=(6000,)))
        sandbox._handle = _handle(_RawWithEndpoint())
        sandbox._stopped = False
        return await sandbox.endpoint(6000)

    resolved = asyncio.run(main())
    assert resolved.endpoint.endswith("/proxy/6000")


def test_async_sandbox_endpoint_still_rejects_undeclared_ports() -> None:
    async def main() -> None:
        provider = _provider()
        sandbox = AsyncSandbox(provider=provider, spec=SandboxSpec(image="img", ports=(6000,)))
        sandbox._handle = _handle(_RawWithEndpoint())
        sandbox._stopped = False
        await sandbox.endpoint(8080)

    with pytest.raises(ValueError, match="not declared"):
        asyncio.run(main())
