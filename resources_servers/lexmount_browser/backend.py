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
To use the **Lexmount** browser instead, implement the same five async methods in
`LexmountBackend` and select it via config `backend: lexmount` — nothing else in
the environment changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


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

    def render(self, max_elements: int = 50) -> str:
        lines = [f"URL: {self.url}", f"TITLE: {self.title}", "ELEMENTS:"]
        for el in self.elements[:max_elements]:
            lines.append(f"  [{el.id}] {el.role}: {el.name}")
        if len(self.elements) > max_elements:
            lines.append(f"  ... (+{len(self.elements) - max_elements} more)")
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
    async def observe(self) -> Observation: ...

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
    `observe()` and resolved back to handles for `click`/`type`.
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

    async def observe(self) -> Observation:
        self._handles.clear()
        locator = self._page.locator(_INTERACTIVE)
        count = await locator.count()
        elements: list[Element] = []
        for i in range(count):
            node = locator.nth(i)
            try:
                if not await node.is_visible():
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
            except Exception:
                pass
        if self._pw is not None:
            await self._pw.stop()

    def _require(self, element_id: int):
        if element_id not in self._handles:
            raise KeyError(
                f"element_id {element_id} not in last observation; call browser_observe first"
            )
        return self._handles[element_id]


class LexmountBackend(PlaywrightBackend):
    """Production backend: an isolated browser session in the Lexmount cloud.

    The browser runs off the training node; we connect to it over CDP and reuse
    PlaywrightBackend's page-driving logic (observe/click/type/goto). Only session
    setup/teardown differ. Credentials are read from the environment by the
    Lexmount SDK — ``LEXMOUNT_API_KEY`` / ``LEXMOUNT_PROJECT_ID`` /
    ``LEXMOUNT_BASE_URL`` — never from committed config. Select with
    ``backend: lexmount``.
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
        try:
            self._session = self._client.sessions.create(
                browser_mode=self._browser_mode, poll_timeout_sec=self._poll_timeout_sec
            )
        except TypeError:  # older SDK without poll_timeout_sec
            self._session = self._client.sessions.create(browser_mode=self._browser_mode)

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
            try:
                if self._session is not None:
                    self._session.close()
            except Exception:
                pass
            try:
                if self._client is not None and session_id:
                    self._client.sessions.delete(session_id=session_id)
            except Exception:
                pass


def make_backend(name: str, **kwargs) -> BrowserBackend:
    if name == "playwright":
        return PlaywrightBackend(headless=kwargs.get("headless", True))
    if name == "lexmount":
        return LexmountBackend(
            endpoint=kwargs.get("endpoint"),
            headless=kwargs.get("headless", True),
            browser_mode=kwargs.get("browser_mode", "normal"),
        )
    raise ValueError(f"unknown browser backend: {name!r}")
