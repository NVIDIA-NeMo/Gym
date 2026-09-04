# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import nemo_gym.web.browser_session as browser_session
from nemo_gym.web.browser_session import (
    BrowserSessionError,
    BrowserSessionHandle,
    BrowserSessionSpec,
    LocalProcessBrowserSessionProvider,
    create_browser_session_provider,
    list_browser_session_providers,
    register_browser_session_provider,
)


@pytest.mark.asyncio
async def test_local_provider_binds_the_lease_to_the_process_and_display(monkeypatch) -> None:
    monkeypatch.setenv("DISPLAY", ":73")
    provider = LocalProcessBrowserSessionProvider()

    handle = await provider.acquire(BrowserSessionSpec(metadata={"rollout_session_id": "rollout-a"}))

    assert handle.session_id == "rollout-a"
    assert handle.transport == "local_process"
    assert handle.endpoint == ":73"
    assert handle.owner_pid == os.getpid()
    await provider.heartbeat(handle)
    await provider.release(handle)

    handle.owner_pid = os.getpid() + 1
    with pytest.raises(BrowserSessionError, match="belongs to pid"):
        await provider.release(handle)


def test_provider_factory_supports_registered_agentenv_integrations() -> None:
    class ExampleAgentEnvProvider:
        name = "test_agentenv_provider"

        def __init__(self, *, pool: str) -> None:
            self.pool = pool

        async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
            return BrowserSessionHandle(
                session_id=spec.metadata.get("rollout_session_id"),
                provider_name=self.name,
                transport="agentenv",
            )

        async def release(self, handle: BrowserSessionHandle) -> None:
            del handle

    register_browser_session_provider(
        ExampleAgentEnvProvider.name,
        ExampleAgentEnvProvider,
        override=True,
    )

    provider = create_browser_session_provider({ExampleAgentEnvProvider.name: {"pool": "visual-web"}})

    assert provider.pool == "visual-web"
    assert ExampleAgentEnvProvider.name in list_browser_session_providers()


@pytest.mark.parametrize("config", [{}, {"local_process": {}, "other": {}}, {"local_process": "bad"}])
def test_provider_factory_rejects_ambiguous_or_invalid_configuration(config) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_browser_session_provider(config)


def test_provider_registration_rejects_empty_and_duplicate_names(monkeypatch) -> None:
    monkeypatch.setattr(browser_session, "_PROVIDER_REGISTRY", {"local_process": LocalProcessBrowserSessionProvider})

    with pytest.raises(ValueError, match="non-empty"):
        register_browser_session_provider("", LocalProcessBrowserSessionProvider)
    with pytest.raises(ValueError, match="already registered"):
        register_browser_session_provider("local_process", LocalProcessBrowserSessionProvider)


def test_entry_point_discovery_skips_registered_names_and_loads_external_provider(monkeypatch) -> None:
    class ExternalProvider:
        name = "external"

        async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
            del spec
            return BrowserSessionHandle(provider_name=self.name)

        async def release(self, handle: BrowserSessionHandle) -> None:
            del handle

    class EntryPoint:
        def __init__(self, name, provider_class) -> None:
            self.name = name
            self._provider_class = provider_class

        def load(self):
            return self._provider_class

    monkeypatch.setattr(
        browser_session,
        "entry_points",
        lambda **_kwargs: [
            EntryPoint("local_process", object),
            EntryPoint("external", ExternalProvider),
        ],
    )

    assert "external" in list_browser_session_providers()
    assert create_browser_session_provider({"external": {}}).name == "external"


def test_provider_factory_rejects_unknown_name_non_string_name_and_bad_implementation(monkeypatch) -> None:
    class InvalidProvider:
        pass

    monkeypatch.setattr(browser_session, "_PROVIDER_REGISTRY", {"invalid": InvalidProvider})
    monkeypatch.setattr(browser_session, "entry_points", lambda **_kwargs: [])

    with pytest.raises(ValueError, match="non-empty string"):
        create_browser_session_provider({1: {}})
    with pytest.raises(ValueError, match="unknown browser session provider"):
        create_browser_session_provider({"missing": {}})
    with pytest.raises(TypeError, match="does not implement"):
        create_browser_session_provider({"invalid": {}})
