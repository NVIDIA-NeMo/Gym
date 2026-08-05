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
"""Example remote provider: isolated cloud browsers from Lexmount.

Reference implementation of `BrowserSessionProvider` against a hosted browser
service — the browser runs off the training node and is reached over CDP. The
SDK is an optional dependency: nothing here is imported unless the config
selects `session_provider: {lexmount: ...}`.

Read `README.md` in this directory before running it at training concurrency.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Optional


try:  # package import (gym loads the resources server as a module)
    from ...browser.base import (
        BrowserSessionError,
        BrowserSessionHandle,
        BrowserSessionSpec,
    )
except ImportError:  # script/standalone import (python app.py, local tests)
    from browser.base import (  # type: ignore[no-redef]
        BrowserSessionError,
        BrowserSessionHandle,
        BrowserSessionSpec,
    )


LOGGER = logging.getLogger(__name__)

# How long teardown may spend on one blocking SDK call before we give up on it.
_RELEASE_TIMEOUT_S = 30.0


class LexmountSessionProvider:
    """One isolated Lexmount cloud browser per rollout.

    Credentials come from the environment, never from committed config:
    ``LEXMOUNT_API_KEY`` / ``LEXMOUNT_PROJECT_ID`` / ``LEXMOUNT_BASE_URL``.
    """

    name = "lexmount"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        browser_mode: str = "normal",
        create_timeout_s: float = 150.0,
    ):
        self._endpoint = endpoint  # optional LEXMOUNT_BASE_URL override
        self._browser_mode = browser_mode
        self._create_timeout_s = create_timeout_s
        self._client: Any = None

    def _sdk_client(self):
        if self._client is None:
            try:
                from lexmount import Lexmount
            except ImportError as e:
                raise BrowserSessionError(
                    "Lexmount SDK not installed. `pip install lexmount` and set "
                    "LEXMOUNT_API_KEY / LEXMOUNT_PROJECT_ID / LEXMOUNT_BASE_URL to use "
                    "session_provider `lexmount`."
                ) from e
            self._client = Lexmount(base_url=self._endpoint) if self._endpoint else Lexmount()
        return self._client

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        client = self._sdk_client()
        create_kwargs: dict[str, Any] = {"browser_mode": self._browser_mode, **spec.provider_options}
        # Feature-detect instead of retrying on TypeError: a TypeError raised
        # from *inside* create would make a retry allocate a second cloud
        # session and leak the first one.
        if "poll_timeout_sec" in inspect.signature(client.sessions.create).parameters:
            create_kwargs.setdefault("poll_timeout_sec", int(self._create_timeout_s))

        # The SDK is synchronous and creation polls until the session is active,
        # so calling it directly would block the event loop for this whole
        # server — stalling every other rollout's tool calls, not just this one.
        # The thread cannot be cancelled, so `poll_timeout_sec` above (not the
        # wait_for) is the real bound; wait_for only caps what *we* wait for.
        session = await asyncio.wait_for(
            asyncio.to_thread(client.sessions.create, **create_kwargs),
            timeout=self._create_timeout_s + 15,
        )
        cdp_url = getattr(session, "connect_url", None)
        session_id = getattr(session, "session_id", None) or getattr(session, "id", None)
        if not cdp_url:
            # Hand the session back through the normal path rather than
            # dropping it: it exists cloud-side even though it is unusable here.
            handle = BrowserSessionHandle(cdp_url="", session_id=session_id, provider_name=self.name, raw=session)
            await self.release(handle)
            raise BrowserSessionError("Lexmount session did not return a connect_url")
        return BrowserSessionHandle(
            cdp_url=cdp_url,
            session_id=session_id,
            provider_name=self.name,
            raw=session,
        )

    async def release(self, handle: BrowserSessionHandle) -> None:
        session = handle.raw
        session_id = handle.session_id
        # Both SDK calls are synchronous: run them off the event loop for the
        # same reason as create, and bound them so a hung service cannot pin
        # this rollout's teardown indefinitely.
        if session is not None:
            try:
                await asyncio.wait_for(asyncio.to_thread(session.close), timeout=_RELEASE_TIMEOUT_S)
            except Exception as exc:
                LOGGER.warning("Failed to close Lexmount session %s: %r", session_id or "?", exc)
        if session_id and self._client is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._client.sessions.delete, session_id=session_id),
                    timeout=_RELEASE_TIMEOUT_S,
                )
            except Exception as exc:
                # Not retried: the service may still hold this session, and the
                # run has no way to reclaim it.  Say so loudly.
                LOGGER.warning(
                    "Failed to delete Lexmount session %s (it may still be held service-side): %r",
                    session_id,
                    exc,
                )
        handle.raw = None
