# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Backend/provider selection from config, and remote session bookkeeping.

The session tests use a fake provider: leaking a remote browser is the failure
mode that matters most here, and it must be provable without a cloud account.
"""

import pytest
from browser import (
    BrowserSessionError,
    BrowserSessionHandle,
    BrowserSessionSpec,
    LocalPlaywrightBackend,
    RemoteCDPBackend,
    create_backend,
    create_session_provider,
    list_backends,
    list_session_providers,
)


# ----- selection ---------------------------------------------------------- #
def test_shipped_backends_and_providers_are_discoverable():
    assert list_backends() == ["local_playwright", "remote_cdp"]
    providers = list_session_providers()
    assert "static_cdp" in providers
    # The in-tree example provider is selectable by name without the
    # environment importing its SDK.
    assert "lexmount" in providers


def test_create_backend_builds_the_named_backend():
    assert isinstance(create_backend({"local_playwright": {"headless": True}}), LocalPlaywrightBackend)
    remote = create_backend({"remote_cdp": {"session_provider": {"static_cdp": {"cdp_url": "http://x:1"}}}})
    assert isinstance(remote, RemoteCDPBackend)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"local_playwright": {}, "remote_cdp": {}},
        {"local_playwright": "headless"},
    ],
)
def test_create_backend_rejects_malformed_config(config):
    with pytest.raises((ValueError, TypeError)):
        create_backend(config)


def test_unknown_names_list_what_is_available():
    with pytest.raises(ValueError, match="local_playwright"):
        create_backend({"nope": {}})
    with pytest.raises(ValueError, match="static_cdp"):
        create_session_provider({"nope": {}})


def test_remote_cdp_without_a_provider_says_so():
    with pytest.raises(ValueError, match="session_provider"):
        create_backend({"remote_cdp": {}})


@pytest.mark.asyncio
async def test_static_cdp_without_an_endpoint_fails_loudly(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    provider = create_session_provider({"static_cdp": {}})
    with pytest.raises(BrowserSessionError, match="no endpoint"):
        await provider.acquire(BrowserSessionSpec())


@pytest.mark.asyncio
async def test_selecting_the_example_provider_needs_no_sdk_until_it_runs():
    """Choosing `lexmount` must not import its SDK; only acquiring needs it."""
    provider = create_session_provider({"lexmount": {"browser_mode": "normal"}})
    assert provider.name == "lexmount"
    try:
        import lexmount  # noqa: F401
    except ImportError:
        with pytest.raises(BrowserSessionError, match="SDK not installed"):
            await provider.acquire(BrowserSessionSpec())


# ----- remote session bookkeeping ----------------------------------------- #
class FakeProvider:
    """Records what the backend asked for, and hands out a chosen endpoint."""

    name = "fake"

    def __init__(self, cdp_url: str):
        self._cdp_url = cdp_url
        self.acquired: list = []
        self.released: list = []

    async def acquire(self, spec):
        self.acquired.append(spec)
        return BrowserSessionHandle(
            cdp_url=self._cdp_url, session_id=f"s{len(self.acquired)}", provider_name=self.name
        )

    async def release(self, handle):
        self.released.append(handle)


@pytest.mark.asyncio
async def test_session_is_released_once_however_often_close_is_called(cdp_endpoint):
    provider = FakeProvider(cdp_endpoint)
    backend = RemoteCDPBackend(provider, session_metadata={"rollout_session_id": "abc"})
    await backend.open("about:blank")
    assert len(provider.acquired) == 1
    assert provider.acquired[0].metadata == {"rollout_session_id": "abc"}

    await backend.close()
    await backend.close()
    assert len(provider.released) == 1
    assert provider.released[0].session_id == "s1"


@pytest.mark.asyncio
async def test_session_is_released_when_the_cdp_connect_fails():
    # Port 1 refuses connections: the session was acquired but is unusable, and
    # an unreleased session here is exactly how a run walks into its quota.
    provider = FakeProvider("http://127.0.0.1:1")
    backend = RemoteCDPBackend(provider, connect_timeout_s=5)
    with pytest.raises(Exception):
        await backend.open("about:blank")
    assert len(provider.acquired) == 1
    assert len(provider.released) == 1


@pytest.mark.asyncio
async def test_session_without_a_cdp_url_is_reported_and_released():
    provider = FakeProvider("")
    backend = RemoteCDPBackend(provider)
    with pytest.raises(BrowserSessionError, match="cdp_url"):
        await backend.open("about:blank")
    assert len(provider.released) == 1
