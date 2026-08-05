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
"""Remote backend: drive a browser that runs somewhere else, over CDP.

The browser may live in another container, on another host, or in a hosted
browser service. This module does not know which — it takes a CDP endpoint from
a `BrowserSessionProvider` and gives it back when the rollout ends. Everything
service-specific (auth, session create/delete, quotas) belongs in a provider.

The built-in `StaticCDPProvider` points at an endpoint that already exists
(`chromium --remote-debugging-port=9222`, a browser container, ...), so this
backend is fully usable — and testable in CI — without any third-party SDK.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .base import (
    BrowserSessionError,
    BrowserSessionHandle,
    BrowserSessionProvider,
    BrowserSessionSpec,
)
from .page import PlaywrightConnectedBackend


LOGGER = logging.getLogger(__name__)


class StaticCDPProvider:
    """Hands out one pre-existing CDP endpoint; releasing it is a no-op.

    Every rollout gets its own browser *context* on that endpoint, not its own
    browser process — fine for local development and CI, not for training
    concurrency against a single Chromium.
    """

    name = "static_cdp"

    def __init__(self, cdp_url: Optional[str] = None, env_var: str = "BROWSER_CDP_URL"):
        self._cdp_url = cdp_url or os.environ.get(env_var)
        self._env_var = env_var

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        if not self._cdp_url:
            raise BrowserSessionError(
                f"static_cdp provider has no endpoint: set `cdp_url` in the config or ${self._env_var}"
            )
        return BrowserSessionHandle(cdp_url=self._cdp_url, provider_name=self.name)

    async def release(self, handle: BrowserSessionHandle) -> None:
        return None


class RemoteCDPBackend(PlaywrightConnectedBackend):
    """Connects to a provider-supplied browser over CDP.

    One provider session per rollout. The provider decides what a session costs
    and how long it may live; this backend only guarantees that every acquired
    session is released exactly once, including when the CDP connect itself
    fails.
    """

    def __init__(
        self,
        session_provider: BrowserSessionProvider,
        *,
        connect_timeout_s: float = 60.0,
        provider_options: Optional[dict[str, Any]] = None,
        session_metadata: Optional[dict[str, str]] = None,
    ):
        super().__init__(session_metadata=session_metadata)
        self._provider = session_provider
        self._connect_timeout_s = connect_timeout_s
        self._provider_options = dict(provider_options or {})
        self._handle: Optional[BrowserSessionHandle] = None

    @property
    def session_handle(self) -> Optional[BrowserSessionHandle]:
        """The live session, or None before `open()` / after `close()`."""
        return self._handle

    async def _connect(self, playwright):
        spec = BrowserSessionSpec(
            metadata=dict(self.session_metadata),
            provider_options=dict(self._provider_options),
        )
        handle = await self._provider.acquire(spec)
        if not handle.cdp_url:
            # Release before raising: the provider may already hold a session
            # even though it failed to give us a usable endpoint.
            self._handle = handle
            raise BrowserSessionError(
                f"provider {getattr(self._provider, 'name', type(self._provider).__name__)!r} "
                "returned a session without a cdp_url"
            )
        self._handle = handle
        browser = await playwright.chromium.connect_over_cdp(handle.cdp_url, timeout=self._connect_timeout_s * 1000)
        try:
            # A remote browser usually ships with a default context; borrow it
            # rather than closing it on teardown.
            if browser.contexts:
                return browser, browser.contexts[0], False
            return browser, await browser.new_context(), True
        except Exception:
            await browser.close()
            raise

    async def _release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            await self._provider.release(handle)
        except Exception as exc:
            # Loud, not fatal: a session the provider still holds is a leaked
            # resource, and silence here is how a run walks into its quota.
            LOGGER.warning(
                "Provider %r failed to release session %s: %r",
                getattr(self._provider, "name", type(self._provider).__name__),
                handle.session_id or "?",
                exc,
            )
