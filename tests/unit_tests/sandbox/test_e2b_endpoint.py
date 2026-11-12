# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2BProvider.endpoint: the two ways a deployment publishes sandbox ports.

The gateway expectations below are the shapes observed against a live AgentENV
deployment on 2026-09-02 with e2b 2.46.0: ``sandbox_headers`` carried
``X-Access-Token``, ``E2b-Sandbox-Id`` and ``E2b-Sandbox-Port: 49983``, and
``get_sandbox_url`` returned the configured gateway origin verbatim.
"""

from __future__ import annotations

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.e2b import E2BProvider


@pytest.fixture(autouse=True)
def _no_ambient_sandbox_url(monkeypatch):
    """Pin the routing input these tests are about.

    ``auto`` deliberately reads E2B_SANDBOX_URL, the same signal the SDK uses,
    so any process that exports it -- a runner sourcing a credential file, for
    instance -- flips the default from hostname to gateway. Tests asserting
    hostname routing must say so rather than depend on the variable's absence.
    """
    monkeypatch.delenv("E2B_SANDBOX_URL", raising=False)


ENVD_PORT = 49983
SANDBOX_ID = "01a06156-bb35-7c00-a322-47e691e634f0"
GATEWAY = "http://10.57.212.63:8000"


class FakeConnectionConfig:
    """The subset of e2b's ConnectionConfig that endpoint() reads."""

    def __init__(self, *, sandbox_url: str | None = None, debug: bool = False) -> None:
        self._sandbox_url = sandbox_url
        self.debug = debug

    def get_host(self, sandbox_id: str, sandbox_domain: str, port: int) -> str:
        if self.debug:
            return f"localhost:{port}"
        return f"{port}-{sandbox_id}.{sandbox_domain}"

    def get_sandbox_url(self, sandbox_id: str, sandbox_domain: str) -> str:
        if self._sandbox_url:
            return self._sandbox_url
        return f"https://{self.get_host(sandbox_id, sandbox_domain, ENVD_PORT)}"

    @property
    def sandbox_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "e2b-python-sdk/2.46.0",
            "X-Access-Token": "1f83b3bf5178deadbeef",  # pragma: allowlist secret
            "E2b-Sandbox-Id": SANDBOX_ID,
            "E2b-Sandbox-Port": str(ENVD_PORT),
        }


class FakeSandbox:
    def __init__(self, connection_config: FakeConnectionConfig) -> None:
        self.connection_config = connection_config
        self.sandbox_id = SANDBOX_ID
        self.sandbox_domain = "aenv.internal"

    def get_host(self, port: int) -> str:
        return self.connection_config.get_host(self.sandbox_id, self.sandbox_domain, port)


def _handle(config: FakeConnectionConfig) -> SandboxHandle:
    return SandboxHandle(sandbox_id=SANDBOX_ID, provider_name="e2b", raw=FakeSandbox(config))


async def test_gateway_routing_repoints_the_port_away_from_envd() -> None:
    """E2b-Sandbox-Port defaults to envd, so every service port must override it."""
    provider = E2BProvider(connection={"sandbox_url": GATEWAY})
    handle = _handle(FakeConnectionConfig(sandbox_url=GATEWAY))

    resolved = await provider.endpoint(handle, 5000)

    assert resolved.endpoint == GATEWAY
    assert resolved.headers["E2b-Sandbox-Port"] == "5000"
    assert resolved.headers["E2b-Sandbox-Id"] == SANDBOX_ID
    assert resolved.headers["X-Access-Token"]


async def test_gateway_routing_gives_every_port_one_origin() -> None:
    provider = E2BProvider(connection={"sandbox_url": GATEWAY})
    handle = _handle(FakeConnectionConfig(sandbox_url=GATEWAY))

    resolved = {port: await provider.endpoint(handle, port) for port in (5000, 9222, 6901, 8080)}

    assert {r.endpoint for r in resolved.values()} == {GATEWAY}
    assert {port: r.headers["E2b-Sandbox-Port"] for port, r in resolved.items()} == {
        5000: "5000",
        9222: "9222",
        6901: "6901",
        8080: "8080",
    }


async def test_hostname_routing_needs_no_headers() -> None:
    """With wildcard DNS the caller dials the port's own name directly."""
    provider = E2BProvider(connection={})
    handle = _handle(FakeConnectionConfig())

    resolved = await provider.endpoint(handle, 5000)

    assert resolved.endpoint == f"https://5000-{SANDBOX_ID}.aenv.internal"
    assert resolved.headers == {}


async def test_auto_selects_gateway_from_the_sdk_env_var(monkeypatch) -> None:
    """A deployment configured only through E2B_SANDBOX_URL still routes by header."""
    monkeypatch.setenv("E2B_SANDBOX_URL", GATEWAY)
    provider = E2BProvider(connection={})

    resolved = await provider.endpoint(_handle(FakeConnectionConfig(sandbox_url=GATEWAY)), 5000)

    assert resolved.endpoint == GATEWAY
    assert resolved.headers["E2b-Sandbox-Port"] == "5000"


async def test_explicit_hostname_routing_overrides_a_configured_gateway() -> None:
    provider = E2BProvider(connection={"sandbox_url": GATEWAY, "port_routing": "hostname"})

    resolved = await provider.endpoint(_handle(FakeConnectionConfig(sandbox_url=GATEWAY)), 6901)

    assert resolved.endpoint == f"https://6901-{SANDBOX_ID}.aenv.internal"
    assert resolved.headers == {}


async def test_hostname_scheme_is_overridable_for_plain_http_dns() -> None:
    provider = E2BProvider(connection={"port_routing": "hostname", "port_scheme": "http"})

    resolved = await provider.endpoint(_handle(FakeConnectionConfig()), 5000)

    assert resolved.endpoint == f"http://5000-{SANDBOX_ID}.aenv.internal"


async def test_debug_mode_mirrors_the_sdk_local_shape() -> None:
    provider = E2BProvider(connection={})

    resolved = await provider.endpoint(_handle(FakeConnectionConfig(debug=True)), 5000)

    assert resolved.endpoint == "http://localhost:5000"


@pytest.mark.parametrize("port", [0, -1, 65536, True, "5000", 5000.0])
async def test_invalid_ports_are_rejected(port) -> None:
    provider = E2BProvider(connection={})
    with pytest.raises(ValueError):
        await provider.endpoint(_handle(FakeConnectionConfig()), port)


def test_invalid_port_routing_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="port_routing must be one of"):
        E2BProvider(connection={"port_routing": "dns"})


def test_invalid_port_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="port_scheme must be"):
        E2BProvider(connection={"port_scheme": "ftp"})


async def test_missing_sandbox_headers_names_the_sdk_requirement() -> None:
    """An older SDK without sandbox_headers must fail with the actionable cause."""

    class WithoutSandboxHeaders(FakeConnectionConfig):
        @property
        def sandbox_headers(self) -> dict[str, str]:
            raise AttributeError("sandbox_headers")

    provider = E2BProvider(connection={"sandbox_url": GATEWAY})
    with pytest.raises(RuntimeError, match="sandbox_headers"):
        await provider.endpoint(_handle(WithoutSandboxHeaders(sandbox_url=GATEWAY)), 5000)
