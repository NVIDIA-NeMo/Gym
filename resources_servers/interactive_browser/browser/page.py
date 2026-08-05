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
"""Page driving and lifecycle shared by every Playwright-connected backend.

This is an implementation detail, not the abstraction boundary: a local
Chromium and a remote CDP browser differ only in how the browser is obtained,
so observe/click/type/goto and teardown live here once. A backend that does not
speak CDP at all (a vendor SDK with its own DOM API, say) implements
`BrowserBackend` directly and ignores this module.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import BrowserBackend, Element, Observation


LOGGER = logging.getLogger(__name__)

_INTERACTIVE = "a, button, input:not([type=hidden]), textarea, select, [role=button], [role=link]"


class PlaywrightPageDriver:
    """Drives one Playwright `Page` and owns the element ids handed to the policy.

    Element ids are assigned by DOM order at each `observe()` and bound to
    element handles for `click`/`type`.
    """

    def __init__(self, page):
        self._page = page
        self._handles: dict[int, object] = {}

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

    def _require(self, element_id: int):
        if element_id not in self._handles:
            raise KeyError(f"element_id {element_id} not in last observation; call browser_observe first")
        return self._handles[element_id]


class PlaywrightConnectedBackend(BrowserBackend):
    """Base for backends whose page is driven through Playwright.

    Subclasses supply the two ends of the lifecycle — `_connect()` obtains a
    browser + context, `_release()` gives back whatever `_connect()` took from
    the outside world — and inherit everything in between.
    """

    def __init__(self, *, session_metadata: Optional[dict[str, str]] = None):
        self.session_metadata: dict[str, str] = dict(session_metadata or {})
        self._pw = None
        self._browser = None
        self._context = None
        self._owns_context = False
        self._released = False
        self._driver: Optional[PlaywrightPageDriver] = None

    # ----- subclass hooks -------------------------------------------------- #
    async def _connect(self, playwright) -> tuple[object, object, bool]:
        """Return `(browser, context, owns_context)`.

        `owns_context` is False when the context predates us (a remote browser's
        default context), so teardown does not close something we were only
        borrowing. Implementations MUST release any external resource they
        acquired if they raise.
        """
        raise NotImplementedError

    async def _release(self) -> None:
        """Give back external resources (a provider session, ...). Called once,
        after the Playwright objects are closed, even if `_connect` failed."""

    # ----- lifecycle ------------------------------------------------------- #
    async def open(self, initial_url: str) -> None:
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        try:
            self._browser, self._context, self._owns_context = await self._connect(self._pw)
            page = await self._context.new_page()
        except Exception:
            # `open()` failing must not strand a Playwright process or a
            # provider session: unwind everything we managed to take.
            await self.close()
            raise
        self._driver = PlaywrightPageDriver(page)
        if initial_url:
            await self.goto(initial_url)

    async def close(self) -> None:
        try:
            closers = [self._browser]
            if self._owns_context:
                closers.insert(0, self._context)
            for closer in closers:
                try:
                    if closer is not None:
                        await closer.close()
                except Exception as exc:
                    # Never silent: an unreported close failure is a leaked
                    # browser that looks exactly like a clean teardown.
                    LOGGER.warning("Failed to close %s: %r", type(closer).__name__, exc)
            if self._pw is not None:
                try:
                    await self._pw.stop()
                except Exception as exc:
                    LOGGER.warning("Failed to stop playwright: %r", exc)
        finally:
            self._browser = self._context = self._pw = None
            self._driver = None
            # `open()` unwinds through close() on failure, and the server closes
            # again on verify/re-seed: release exactly once regardless.
            if not self._released:
                self._released = True
                await self._release()

    # ----- driving --------------------------------------------------------- #
    def _page(self) -> PlaywrightPageDriver:
        if self._driver is None:
            raise RuntimeError("backend is not open; call open() first")
        return self._driver

    async def goto(self, url: str) -> None:
        await self._page().goto(url)

    async def click(self, element_id: int) -> None:
        await self._page().click(element_id)

    async def type(self, element_id: int, text: str) -> None:
        await self._page().type(element_id, text)

    async def observe(self, max_elements: int = 50) -> Observation:
        return await self._page().observe(max_elements=max_elements)

    async def current_url(self) -> str:
        return await self._page().current_url()

    async def text(self) -> str:
        return await self._page().text()
