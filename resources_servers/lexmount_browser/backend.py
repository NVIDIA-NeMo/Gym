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
"""Pluggable browser backends for the NeMo-Gym browser environment.

`BrowserBackend` is the thin contract the resources server depends on. The
`PlaywrightBackend` is a fully-working open-source reference (headless Chromium).
To use the **Lexmount** browser instead, implement the same async methods in
`LexmountBackend` and select it via config `backend: lexmount` — nothing else in
the environment changes.
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class Element:
    """One interactive element exposed to the policy, addressed by stable `id`."""

    id: int
    role: str          # "link" | "button" | "textbox" | ...
    name: str          # accessible name / visible text (truncated)


@dataclass
class Observation:
    """Compact, token-cheap view of the page handed to the policy each step."""

    url: str
    title: str
    elements: list[Element] = field(default_factory=list)
    # True when the backend stopped collecting at its element budget, so the
    # policy knows the element list is incomplete rather than exhaustive.
    truncated: bool = False

    def render(self, max_elements: int = 50) -> str:
        lines = [f"URL: {self.url}", f"TITLE: {self.title}", "ELEMENTS:"]
        for el in self.elements[:max_elements]:
            lines.append(f"  [{el.id}] {el.role}: {el.name}")
        if self.truncated or len(self.elements) > max_elements:
            lines.append(f"  ... (truncated at {max_elements} elements)")
        return "\n".join(lines)


class BrowserBackend(abc.ABC):
    """Per-episode browser contract. One instance == one isolated context/page.

    Implementations MUST be safe to instantiate many times concurrently (one per
    rollout) and MUST release all resources in `close()`.
    """

    @abc.abstractmethod
    async def open(self, initial_url: str) -> None: ...

    @abc.abstractmethod
    async def goto(self, url: str) -> None: ...

    @abc.abstractmethod
    async def click(self, element_id: int) -> None: ...

    @abc.abstractmethod
    async def type(self, element_id: int, text: str) -> None: ...

    @abc.abstractmethod
    async def observe(self, max_elements: int = 50) -> Observation:
        """Snapshot the page.  Implementations MUST stop collecting elements at
        `max_elements` (probing every interactive node costs a round-trip each)
        and set `Observation.truncated` when they do."""
        ...

    @abc.abstractmethod
    async def current_url(self) -> str: ...

    @abc.abstractmethod
    async def text(self) -> str:
        """Full visible page text (for dom_contains scoring)."""
        ...

    @abc.abstractmethod
    async def close(self) -> None: ...


_INTERACTIVE = (
    "a, button, input:not([type=hidden]), textarea, select, [role=button], [role=link]"
)


class PlaywrightBackend(BrowserBackend):
    """Reference backend over Playwright + headless Chromium.

    Each instance owns its own browser context (cookie/session isolation), so N
    rollouts do not interfere. Element ids are assigned by DOM order at each
    `observe()` and bound to element handles for `click`/`type`.
    """

    def __init__(self, headless: bool = True):
        self._headless = headless
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._handles: dict[int, object] = {}

    async def open(self, initial_url: str) -> None:
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        # One browser process can host many isolated contexts; here we keep it
        # simple and own a context per backend instance.
        self._browser = await self._pw.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        if initial_url:
            await self.goto(initial_url)

    async def goto(self, url: str) -> None:
        await self._page.goto(url, wait_until="domcontentloaded")

    async def observe(self, max_elements: int = 50) -> Observation:
        self._handles.clear()
        locator = self._page.locator(_INTERACTIVE)
        count = await locator.count()
        elements: list[Element] = []
        truncated = False
        for i in range(count):
            if len(elements) >= max_elements:
                # Each candidate below costs several CDP round-trips, so stop at
                # the budget instead of probing every node on a large page and
                # discarding the surplus at render time.
                truncated = True
                break
            try:
                # Bind an element handle rather than keeping the lazy locator:
                # `locator.nth(i)` is re-resolved against the *current* DOM at
                # action time, so a click after any DOM mutation could silently
                # land on a different element.  A detached handle raises instead,
                # which is an honest signal for both the policy and the trainer.
                node = await locator.nth(i).element_handle()
                if node is None or not await node.is_visible():
                    continue
                role = (await node.evaluate("e => e.tagName")).lower()
                name = (await node.inner_text()) or (await node.get_attribute("value")) or ""
                name = " ".join(name.split())[:80]
            except Exception:
                continue
            eid = len(elements)
            self._handles[eid] = node
            elements.append(Element(id=eid, role=role, name=name))
        return Observation(
            url=self._page.url,
            title=await self._page.title(),
            elements=elements,
            truncated=truncated,
        )

    async def click(self, element_id: int) -> None:
        node = self._require(element_id)
        await node.click(timeout=5000)

    async def type(self, element_id: int, text: str) -> None:
        node = self._require(element_id)
        await node.fill(text, timeout=5000)

    async def current_url(self) -> str:
        return self._page.url

    async def text(self) -> str:
        try:
            return await self._page.inner_text("body")
        except Exception:
            return ""

    async def close(self) -> None:
        for closer in (self._context, self._browser):
            try:
                if closer is not None:
                    await closer.close()
            except Exception as exc:
                # Never silent: an unreported close failure is a leaked browser
                # process that looks exactly like a clean teardown.
                LOGGER.warning("Failed to close %s: %r", type(closer).__name__, exc)
        if self._pw is not None:
            await self._pw.stop()

    def _require(self, element_id: int):
        if element_id not in self._handles:
            raise KeyError(
                f"element_id {element_id} not in last observation; call browser_observe first"
            )
        return self._handles[element_id]


class LexmountBackend(PlaywrightBackend):
    """**Experimental** backend: an isolated browser session in the Lexmount cloud.

    The browser runs off the training node; we connect to it over CDP and reuse
    PlaywrightBackend's page-driving logic (observe/click/type/goto). Only session
    setup/teardown differ. Credentials are read from the environment by the
    Lexmount SDK — ``LEXMOUNT_API_KEY`` / ``LEXMOUNT_PROJECT_ID`` /
    ``LEXMOUNT_BASE_URL`` — never from committed config. Select with
    ``backend: lexmount``.

    Known limits — read before running this at training concurrency:

    * **No client-side session cap.** One provider session is created per
      rollout with no admission control, so N concurrent rollouts bid for N
      provider sessions. Size the account quota above the rollout concurrency
      (with headroom for sessions still being torn down), or the run will
      exhaust the quota and every subsequent create will fail.
    * **No episode TTL.** A session is released when the rollout is scored or
      re-seeded (see ``app.py``). A rollout abandoned without either — trainer
      crash, client disconnect — leaks its cloud session until the provider's
      own timeout reclaims it.
    * **Close is best-effort.** A close/delete that fails is logged but not
      retried, and the provider may still hold the session afterwards.

    The reference ``playwright`` backend has none of these constraints (local
    processes, no quota) and remains the default.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        headless: bool = True,
        browser_mode: str = "normal",
        poll_timeout_sec: int = 150,
    ):
        super().__init__(headless=headless)
        self._endpoint = endpoint            # optional LEXMOUNT_BASE_URL override
        self._browser_mode = browser_mode
        self._poll_timeout_sec = poll_timeout_sec
        self._client = None
        self._session = None

    async def open(self, initial_url: str) -> None:
        try:
            from lexmount import Lexmount
        except ImportError as e:
            raise RuntimeError(
                "Lexmount SDK not installed. `pip install lexmount` and set "
                "LEXMOUNT_API_KEY / LEXMOUNT_PROJECT_ID / LEXMOUNT_BASE_URL to use "
                "`backend: lexmount`."
            ) from e
        from playwright.async_api import async_playwright

        # One isolated cloud browser session per rollout (browser runs off-node).
        self._client = Lexmount(base_url=self._endpoint) if self._endpoint else Lexmount()
        create_kwargs: dict = {"browser_mode": self._browser_mode}
        # Feature-detect instead of retrying on TypeError: a TypeError raised
        # from *inside* create would make a retry allocate a second provider
        # session and leak the first one.
        if "poll_timeout_sec" in inspect.signature(self._client.sessions.create).parameters:
            create_kwargs["poll_timeout_sec"] = self._poll_timeout_sec
        # The SDK is synchronous and creation polls until the session is active,
        # so calling it directly would block the event loop for this whole server
        # — stalling every other rollout's tool calls, not just this one.  The
        # thread cannot be cancelled, so `poll_timeout_sec` above (not the
        # wait_for) is the real bound; wait_for only caps what *we* wait for.
        self._session = await asyncio.wait_for(
            asyncio.to_thread(self._client.sessions.create, **create_kwargs),
            timeout=self._poll_timeout_sec + 15,
        )

        cdp_url = getattr(self._session, "connect_url", None)
        if not cdp_url:
            raise RuntimeError("Lexmount session did not return a connect_url")

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.connect_over_cdp(cdp_url)
        self._context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
        self._page = await self._context.new_page()
        if initial_url:
            await self.goto(initial_url)

    async def close(self) -> None:
        # Disconnect the CDP browser + stop playwright, then release the cloud session.
        try:
            await super().close()
        finally:
            session_id = getattr(self._session, "session_id", None) or getattr(self._session, "id", None)
            # Both SDK calls are synchronous: run them off the event loop for the
            # same reason as create, and bound them so a hung provider cannot
            # pin this rollout's teardown indefinitely.
            if self._session is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self._session.close), timeout=30.0
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to close Lexmount session %s: %r", session_id or "?", exc
                    )
            if self._client is not None and session_id:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self._client.sessions.delete, session_id=session_id
                        ),
                        timeout=30.0,
                    )
                except Exception as exc:
                    # Not retried: the provider may still hold this session, and
                    # the run has no way to reclaim it.  Say so loudly.
                    LOGGER.warning(
                        "Failed to delete Lexmount session %s (it may still be "
                        "held provider-side): %r",
                        session_id,
                        exc,
                    )


def make_backend(name: str, **kwargs) -> BrowserBackend:
    if name == "playwright":
        return PlaywrightBackend(headless=kwargs.get("headless", True))
    if name == "lexmount":
        return LexmountBackend(
            endpoint=kwargs.get("endpoint"),
            headless=kwargs.get("headless", True),
            browser_mode=kwargs.get("browser_mode", "normal"),
            poll_timeout_sec=kwargs.get("poll_timeout_sec", 150),
        )
    raise ValueError(f"unknown browser backend: {name!r}")
