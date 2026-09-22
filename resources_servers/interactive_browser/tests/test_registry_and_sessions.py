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

import asyncio
import time
from types import SimpleNamespace
from typing import Any

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


class TestACreateThatLandsAfterTheRolloutGaveUp:
    """A session created past its rollout's deadline must not keep a quota slot.

    The SDK call runs in a thread and cannot be cancelled, so a create that is
    still polling when we stop waiting may well succeed afterwards. Nothing else
    knows that session exists -- the rollout that asked for it has already failed
    -- so the provider has to release it itself.
    """

    def _provider(self, create: Any, monkeypatch: Any) -> Any:
        from providers.lexmount import provider as provider_module

        # The SDK bound is the real one; shrink only our grace period so the
        # abandoned-create path runs in test time rather than in service time.
        monkeypatch.setattr(provider_module, "_CREATE_SLACK_S", 0.05)
        provider = provider_module.LexmountSessionProvider(create_timeout_s=0.0)
        client = SimpleNamespace(sessions=SimpleNamespace(create=create, delete=lambda session_id: None))
        provider._client = client
        return provider

    @pytest.mark.asyncio
    async def test_a_late_session_is_released(self, monkeypatch: Any) -> None:
        released: list[str] = []

        def slow_create(**kwargs: Any) -> Any:
            time.sleep(0.2)
            return SimpleNamespace(connect_url="ws://late", session_id="late-1", close=lambda: None)

        provider = self._provider(slow_create, monkeypatch)
        provider.release = lambda handle: released.append(handle.session_id)  # type: ignore[assignment]

        with pytest.raises(BrowserSessionError, match="did not complete"):
            await provider.acquire(BrowserSessionSpec())

        # The thread is still running at this point; the callback fires when it finishes.
        await asyncio.sleep(0.6)
        assert released == ["late-1"]

    @pytest.mark.asyncio
    async def test_a_create_that_fails_on_its_own_releases_nothing(self, monkeypatch: Any) -> None:
        """Nothing was allocated, so there is no quota slot to reclaim."""
        released: list[str] = []

        def failing_create(**kwargs: Any) -> Any:
            time.sleep(0.2)
            raise RuntimeError("provider refused")

        provider = self._provider(failing_create, monkeypatch)
        provider.release = lambda handle: released.append(handle.session_id)  # type: ignore[assignment]

        with pytest.raises(BrowserSessionError, match="did not complete"):
            await provider.acquire(BrowserSessionSpec())

        await asyncio.sleep(0.6)
        assert released == []
