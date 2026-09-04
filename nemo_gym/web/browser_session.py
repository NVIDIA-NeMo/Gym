# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser-supply leases shared by stateful web environments.

The benchmark owns tasks, actions, observations, and evaluation.  A browser
session provider owns only the lifecycle of the isolated runtime in which the
browser executes.  Keeping that boundary asynchronous lets a provider acquire
an AgentEnv, container, VM, or remote browser without blocking the resources
server event loop, while synchronous Playwright operations remain on their
session-affine worker thread.

Remote providers should issue expiring leases and renew them from
``heartbeat``.  If the Gym process dies, heartbeats stop and provider-side TTL
cleanup becomes the final backstop against orphaned training sessions.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Callable, Protocol, runtime_checkable


ENTRY_POINT_GROUP = "nemo_gym.browser_session_providers"


class BrowserSessionError(RuntimeError):
    """A provider could not supply or maintain a usable browser session."""


@dataclass(frozen=True, slots=True)
class BrowserSessionSpec:
    """Provider-neutral request for one rollout-scoped browser runtime."""

    metadata: dict[str, str] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
    lease_ttl_seconds: int = 900


@dataclass(slots=True)
class BrowserSessionHandle:
    """Opaque provider handle retained until the rollout is released.

    ``transport`` describes how the benchmark driver reaches the runtime.
    The built-in reference runtime uses ``local_process``.  Future providers
    may return ``agentenv`` or ``remote_cdp`` and pair that lease with a driver
    that implements the same ``WebBrowserDriver`` contract.

    The handle is deliberately transport-neutral. Providers may keep their
    SDK object in ``provider_state``; benchmark code must use only the declared
    transport, endpoint, and metadata instead of depending on that object.
    """

    session_id: str | None = None
    provider_name: str | None = None
    transport: str = "local_process"
    endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_pid: int | None = None
    provider_state: Any = None


@runtime_checkable
class BrowserSessionProvider(Protocol):
    """Acquire and release one isolated browser runtime per rollout.

    Implementations with synchronous SDKs must offload blocking calls with
    ``asyncio.to_thread``.  ``release`` must be idempotent.  Metered or remote
    providers must enforce ``BrowserSessionSpec.lease_ttl_seconds`` outside the
    Gym process so a process crash cannot leak the session indefinitely.
    """

    name: str

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle: ...

    async def release(self, handle: BrowserSessionHandle) -> None: ...


@runtime_checkable
class RenewableBrowserSessionProvider(BrowserSessionProvider, Protocol):
    """Optional provider extension for renewing an expiring external lease."""

    async def heartbeat(self, handle: BrowserSessionHandle) -> None: ...


class LocalProcessBrowserSessionProvider:
    """Represent the resources-server process and its DISPLAY as one lease.

    The provider itself does not launch Chromium; the visual-browser driver
    does that on its dedicated thread.  The handle records the owning process
    so a forked child cannot accidentally operate a parent's Playwright state.
    """

    name = "local_process"

    def __init__(self, display_env: str = "DISPLAY") -> None:
        self._display_env = display_env

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        display = os.environ.get(self._display_env, "")
        rollout_session_id = spec.metadata.get("rollout_session_id")
        return BrowserSessionHandle(
            session_id=rollout_session_id,
            provider_name=self.name,
            transport="local_process",
            endpoint=display or None,
            metadata={"display_env": self._display_env, "display": display},
            owner_pid=os.getpid(),
        )

    async def heartbeat(self, handle: BrowserSessionHandle) -> None:
        self._require_owner(handle)

    async def release(self, handle: BrowserSessionHandle) -> None:
        self._require_owner(handle)

    @staticmethod
    def _require_owner(handle: BrowserSessionHandle) -> None:
        if handle.owner_pid is not None and handle.owner_pid != os.getpid():
            raise BrowserSessionError(
                f"local browser lease belongs to pid={handle.owner_pid}, current pid={os.getpid()}"
            )


_PROVIDER_REGISTRY: dict[str, type[BrowserSessionProvider]] = {
    LocalProcessBrowserSessionProvider.name: LocalProcessBrowserSessionProvider,
}


def register_browser_session_provider(
    name: str,
    provider_class: type[BrowserSessionProvider],
    *,
    override: bool = False,
) -> None:
    """Register an in-process provider implementation by configuration name."""

    if not name:
        raise ValueError("browser session provider name must be non-empty")
    if not override and name in _PROVIDER_REGISTRY:
        raise ValueError(f"browser session provider {name!r} is already registered")
    _PROVIDER_REGISTRY[name] = provider_class


def _entry_point_providers() -> dict[str, type[BrowserSessionProvider]]:
    providers: dict[str, type[BrowserSessionProvider]] = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name in _PROVIDER_REGISTRY:
            continue
        providers[entry_point.name] = entry_point.load()
    return providers


def list_browser_session_providers() -> list[str]:
    return sorted({*_PROVIDER_REGISTRY, *_entry_point_providers()})


def create_browser_session_provider(config: Mapping[str, Any]) -> BrowserSessionProvider:
    """Instantiate a provider from a single-key ``{name: kwargs}`` mapping."""

    if not isinstance(config, Mapping) or len(config) != 1:
        raise ValueError("browser_session_provider must be a mapping with exactly one provider name")
    name, raw_kwargs = next(iter(config.items()))
    if not isinstance(name, str) or not name:
        raise ValueError("browser session provider name must be a non-empty string")
    kwargs = {} if raw_kwargs is None else raw_kwargs
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"browser session provider {name!r} config must be a mapping")
    provider_class = _PROVIDER_REGISTRY.get(name) or _entry_point_providers().get(name)
    if provider_class is None:
        available = ", ".join(list_browser_session_providers()) or "<none>"
        raise ValueError(f"unknown browser session provider {name!r}; available: {available}")
    provider = provider_class(**dict(kwargs))
    if not isinstance(provider, BrowserSessionProvider):
        raise TypeError(f"browser session provider {name!r} does not implement acquire/release")
    return provider


BrowserSessionProviderFactory = Callable[[Mapping[str, Any]], BrowserSessionProvider]


__all__ = [
    "BrowserSessionError",
    "BrowserSessionHandle",
    "BrowserSessionProvider",
    "BrowserSessionProviderFactory",
    "BrowserSessionSpec",
    "ENTRY_POINT_GROUP",
    "LocalProcessBrowserSessionProvider",
    "RenewableBrowserSessionProvider",
    "create_browser_session_provider",
    "list_browser_session_providers",
    "register_browser_session_provider",
]
