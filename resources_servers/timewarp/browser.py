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
"""The browser a TimeWarp policy drives: one shared Chromium, one isolated context per rollout.

Observations are Playwright's AI-mode ARIA snapshot (``aria_snapshot(mode="ai")``): an
accessibility tree in which every element carries a ``[ref=eN]`` the policy passes back to
act on it. Refs resolve through Playwright's ``aria-ref=`` selector, the engine that resolves
AI-mode snapshot refs. That selector is not in the Python API reference, so the Playwright
pin in ``requirements.txt`` is narrow and ``tests/test_browser.py`` exercises it against a
real Chromium; an upgrade that changes it fails there instead of inside a rollout.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)

# Upstream TimeWarp runs BrowserGym with a 1280x720 viewport.
VIEWPORT = {"width": 1280, "height": 720}

_REF_PATTERN = re.compile(r"^[a-z0-9]+$")
# Playwright appends a multi-line call log to its errors; the first line says what happened.
_MAX_ERROR_CHARS = 300


# Several News themes link with target="_blank" and News v6 calls window.open; both load in the current tab.
_SAME_TAB_SCRIPT = """
document.addEventListener("click", (event) => {
  const link = event.target instanceof Element && event.target.closest("a[target], area[target]");
  if (link) link.target = "_self";
}, true);
document.addEventListener("submit", (event) => { event.target.target = "_self"; }, true);
window.open = (url) => { if (url) window.location.assign(url); return window; };
"""


class ToolInputError(ValueError):
    """The policy asked for something the current page cannot do; the message is shown to it."""


def origin_of(url: str) -> str:
    """``scheme://host[:port]`` of a URL, lowercased, or ``""`` when it has none."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}".lower()


def normalize_ref(ref: str) -> str:
    """Accept ``e12``, ``[ref=e12]`` or ``ref=e12``; reject anything that is not a ref."""
    cleaned = ref.strip().strip("[]").strip()
    if cleaned.startswith("ref="):
        cleaned = cleaned[len("ref=") :]
    if not _REF_PATTERN.match(cleaned):
        raise ToolInputError(f"'{ref}' is not an element reference; use a ref such as e12 from the latest observation")
    return cleaned


def split_into_parts(snapshot: str, max_chars: int) -> list[str]:
    """Split a snapshot on line boundaries into parts of at most ``max_chars`` characters.

    A single line longer than the budget is cut, so no part ever exceeds it.
    """
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in snapshot.splitlines():
        while len(line) > max_chars:
            if current:
                parts.append("\n".join(current))
                current, size = [], 0
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if current and size + len(line) + 1 > max_chars:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts or [""]


def describe_error(error: BaseException) -> str:
    message = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return message[:_MAX_ERROR_CHARS]


@dataclass(frozen=True)
class SessionLimits:
    max_observation_chars: int
    action_timeout_ms: int
    navigation_timeout_ms: int


class BrowserSession:
    """One rollout's isolated browser context, restricted to its UI version's sites.

    Top-level navigations to any other origin are refused, the in-browser counterpart of
    upstream TimeWarp scoring an episode 0 once a page leaves the TimeWarp sites. Subresources
    are not filtered: some themes load CSS and JavaScript from public CDNs, as they do upstream.
    Links and scripts that would open a new tab load in the current one, so an episode has a
    single tab whose history ``go_back`` can walk.
    """

    def __init__(self, context, *, allowed_origins: frozenset[str], limits: SessionLimits) -> None:
        self._context = context
        self._allowed_origins = allowed_origins
        self._limits = limits
        self._page = None
        self._blocked: list[str] = []
        context.on("page", self._on_new_page)

    @classmethod
    async def open(cls, browser, *, start_url: str, allowed_origins: frozenset[str], limits: SessionLimits):
        context = await browser.new_context(viewport=VIEWPORT)
        try:
            session = cls(context, allowed_origins=allowed_origins, limits=limits)
            await context.route("**/*", session._guard_navigation)
            await context.add_init_script(_SAME_TAB_SCRIPT)
            context.set_default_timeout(limits.action_timeout_ms)
            context.set_default_navigation_timeout(limits.navigation_timeout_ms)
            await (await context.new_page()).goto(start_url, wait_until="domcontentloaded")
        except BaseException:
            await context.close()
            raise
        return session

    async def close(self) -> None:
        await self._context.close()

    def is_alive(self) -> bool:
        """False once the browser process or this episode's tab is gone."""
        browser = self._context.browser
        return browser is not None and browser.is_connected() and self._page is not None and not self._page.is_closed()

    # ----- navigation guard ------------------------------------------------------------- #
    def _allowed(self, url: str) -> bool:
        return url.startswith(("about:", "data:")) or origin_of(url) in self._allowed_origins

    async def _guard_navigation(self, route, request) -> None:
        if self._is_top_level_navigation(request) and not self._allowed(request.url):
            self._blocked.append(request.url)
            # A 204 answer makes the browser stay on the current page; aborting would leave the
            # tab on Chrome's error page instead.
            await route.fulfill(status=204, body="")
        else:
            await route.continue_()

    @staticmethod
    def _is_top_level_navigation(request) -> bool:
        if not request.is_navigation_request():
            return False
        try:
            return request.frame.parent_frame is None
        except Exception:
            # Service-worker requests have no frame; they never navigate a tab.
            return False

    def _on_new_page(self, page) -> None:
        self._page = page
        page.on("close", self._on_page_closed)

    def _on_page_closed(self, page) -> None:
        if page is self._page and self._context.pages:
            self._page = self._context.pages[-1]

    # ----- observation ------------------------------------------------------------------ #
    async def observe(self, part: int = 1) -> str:
        page = self._page
        try:
            snapshot = await page.locator("body").aria_snapshot(mode="ai")
        except Exception as error:
            snapshot = f"(page content unavailable: {describe_error(error)})"
        chunks = split_into_parts(snapshot, self._limits.max_observation_chars)
        if not 1 <= part <= len(chunks):
            raise ToolInputError(f"part must be between 1 and {len(chunks)} for the current page")

        lines = [f"URL: {page.url}", f"Title: {await page.title()}"]
        if len(self._context.pages) > 1:
            lines.append(f"Tabs: {len(self._context.pages)} open; showing the most recently opened one.")
        while self._blocked:
            lines.append(f"Blocked navigation to {self._blocked.pop(0)}: only this task's websites are reachable.")
        if len(chunks) > 1:
            follow = f"; call observe with part={part + 1} to read further" if part < len(chunks) else ""
            lines.append(f"Page content (part {part} of {len(chunks)}{follow}):")
        else:
            lines.append("Page content:")
        lines.append(chunks[part - 1])
        return "\n".join(lines)

    # ----- actions (each returns the next observation) ---------------------------------- #
    async def goto(self, url: str) -> str:
        if not self._allowed(url):
            raise ToolInputError(f"cannot open {url}: only this task's websites are reachable")
        await self._page.goto(url, wait_until="domcontentloaded")
        return await self.observe()

    async def go_back(self) -> str:
        if await self._page.go_back(wait_until="domcontentloaded") is None:
            raise ToolInputError("there is no previous page in this tab")
        return await self.observe()

    async def go_forward(self) -> str:
        if await self._page.go_forward(wait_until="domcontentloaded") is None:
            raise ToolInputError("there is no next page in this tab")
        return await self.observe()

    async def click(self, ref: str) -> str:
        await (await self._element(ref)).click()
        return await self._observe_after_action()

    async def fill(self, ref: str, text: str) -> str:
        await (await self._element(ref)).fill(text)
        return await self.observe()

    async def press(self, ref: str, key: str) -> str:
        await (await self._element(ref)).press(key)
        return await self._observe_after_action()

    async def select_option(self, ref: str, option: str) -> str:
        await (await self._element(ref)).select_option(option)
        return await self._observe_after_action()

    async def _element(self, ref: str):
        locator = self._page.locator(f"aria-ref={normalize_ref(ref)}")
        if await locator.count() == 0:
            raise ToolInputError(
                f"no element with ref {ref} on the current page; refs change when the page changes, "
                "so call observe for fresh ones"
            )
        return locator

    async def _observe_after_action(self) -> str:
        # Observe the document a navigation started by the action loaded, not the one it replaced.
        try:
            await self._page.wait_for_load_state("domcontentloaded")
        except Exception as error:
            logger.debug("page did not settle after an action: %s", describe_error(error))
        return await self.observe()


class SharedBrowser:
    """One Chromium for the whole server, launched on first use and relaunched if it dies."""

    def __init__(self, *, headless: bool) -> None:
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._lock = asyncio.Lock()

    async def get(self):
        async with self._lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            # Imported here so the scorer, and its verifier fixture, load without Playwright installed.
            from playwright.async_api import async_playwright

            if self._playwright is None:
                self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
            return self._browser

    async def close(self) -> None:
        async with self._lock:
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
            self._browser = self._playwright = None
