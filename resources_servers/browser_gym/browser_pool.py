# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
BrowserPool manages Playwright browser instances with Semaphore-bounded concurrency.
Each browser session is identified by a unique env_id.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

logger = logging.getLogger(__name__)

PLAYWRIGHT_KEY_MAP = {
    "ctrl": "Control",
    "control": "Control",
    "control_l": "Control",
    "control_r": "Control",
    "cmd": "Meta",
    "command": "Meta",
    "meta": "Meta",
    "super_l": "Meta",
    "super_r": "Meta",
    "alt": "Alt",
    "alt_l": "Alt",
    "alt_r": "Alt",
    "option": "Alt",
    "shift": "Shift",
    "shift_l": "Shift",
    "shift_r": "Shift",
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "escape": "Escape",
    "esc": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "space": " ",
    "spacebar": " ",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "page_up": "PageUp",
    "page_down": "PageDown",
    "prior": "PageUp",
    "next": "PageDown",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
    "caps_lock": "CapsLock",
    "capslock": "CapsLock",
    "num_lock": "NumLock",
    "numlock": "NumLock",
    "scroll_lock": "ScrollLock",
    "scrolllock": "ScrollLock",
    "print": "PrintScreen",
    "printscreen": "PrintScreen",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}


def _normalize_key(key: str) -> str:
    """Normalize a key name (including X11/xdotool names) to Playwright format."""
    return PLAYWRIGHT_KEY_MAP.get(key.lower().strip(), key)


@dataclass
class BrowserSession:
    context: BrowserContext
    page: Page
    viewport_width: int
    viewport_height: int


class BrowserPool:
    def __init__(self, max_concurrent: int = 16, default_viewport_width: int = 1280, default_viewport_height: int = 720):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._sessions: dict[str, BrowserSession] = {}
        self._browser: Optional[Browser] = None
        self._playwright = None
        self._lock = asyncio.Lock()
        self._default_viewport_width = default_viewport_width
        self._default_viewport_height = default_viewport_height

    async def _ensure_browser(self) -> Browser:
        if self._browser is None or not self._browser.is_connected():
            async with self._lock:
                if self._browser is None or not self._browser.is_connected():
                    self._playwright = await async_playwright().start()
                    self._browser = await self._playwright.chromium.launch(headless=True)
        return self._browser

    async def create_session(
        self,
        env_id: str,
        start_url: str,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
    ) -> str:
        """Create a new browser session. Returns base64 screenshot of the initial page."""
        await self._semaphore.acquire()
        try:
            browser = await self._ensure_browser()
            vw = viewport_width or self._default_viewport_width
            vh = viewport_height or self._default_viewport_height

            context = await browser.new_context(viewport={"width": vw, "height": vh})
            page = await context.new_page()
            await page.goto(start_url, wait_until="domcontentloaded", timeout=30000)

            session = BrowserSession(context=context, page=page, viewport_width=vw, viewport_height=vh)
            self._sessions[env_id] = session

            screenshot_bytes = await page.screenshot(type="png", full_page=False)
            return base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception:
            self._semaphore.release()
            raise

    def get_session(self, env_id: str) -> BrowserSession:
        if env_id not in self._sessions:
            raise KeyError(f"No browser session found for env_id={env_id}")
        return self._sessions[env_id]

    async def take_screenshot(self, env_id: str) -> str:
        """Take a screenshot of the current page and return base64-encoded PNG."""
        session = self.get_session(env_id)
        screenshot_bytes = await session.page.screenshot(type="png", full_page=False)
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def get_current_url(self, env_id: str) -> str:
        session = self.get_session(env_id)
        return session.page.url

    async def dump_local_storage(self, env_id: str) -> str:
        """Dump localStorage as a JSON string."""
        session = self.get_session(env_id)
        local_storage = await session.page.evaluate("() => JSON.stringify(localStorage)")
        return local_storage

    async def close_session(self, env_id: str) -> bool:
        """Close a browser session. Idempotent -- returns False if session was already closed."""
        session = self._sessions.pop(env_id, None)
        if session is None:
            return False
        try:
            await session.context.close()
        except Exception as e:
            logger.warning(f"Error closing browser context for env_id={env_id}: {e}")
        finally:
            self._semaphore.release()
        return True

    async def shutdown(self):
        """Close all sessions and the browser."""
        env_ids = list(self._sessions.keys())
        for env_id in env_ids:
            await self.close_session(env_id)
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping playwright: {e}")
            self._playwright = None

    async def execute_action(self, env_id: str, action) -> tuple[str, str]:
        """Execute a BrowserAction on the page. Returns (screenshot_b64, current_url)."""
        session = self.get_session(env_id)
        page = session.page

        action_type = action.action_type

        if action_type == "click":
            x, y = action.coordinate or [0, 0]
            button = action.button or "left"
            await page.mouse.click(x, y, button=button)
            try:
                await page.wait_for_load_state(timeout=5000)
            except Exception:
                pass

        elif action_type == "double_click":
            x, y = action.coordinate or [0, 0]
            await page.mouse.dblclick(x, y)
            try:
                await page.wait_for_load_state(timeout=5000)
            except Exception:
                pass

        elif action_type == "triple_click":
            x, y = action.coordinate or [0, 0]
            await page.mouse.click(x, y, click_count=3)

        elif action_type == "right_click":
            x, y = action.coordinate or [0, 0]
            await page.mouse.click(x, y, button="right")

        elif action_type == "middle_click":
            x, y = action.coordinate or [0, 0]
            await page.mouse.click(x, y, button="middle")

        elif action_type == "type":
            if action.coordinate:
                x, y = action.coordinate
                await page.mouse.click(x, y)
                try:
                    await page.wait_for_load_state(timeout=5000)
                except Exception:
                    pass
            if action.clear_before_typing:
                import sys as _sys

                mod = "Meta" if _sys.platform == "darwin" else "Control"
                await page.keyboard.press(f"{mod}+a")
                await page.keyboard.press("Delete")
            text = action.text or ""
            try:
                for char in text:
                    await page.keyboard.type(char, delay=50)
            except Exception:
                try:
                    await page.fill("input, textarea", text)
                except Exception:
                    await page.evaluate(
                        """(text) => {
                            const el = document.activeElement;
                            if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
                                el.value = '';
                                el.value = text;
                                el.dispatchEvent(new Event('input', { bubbles: true }));
                                el.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                        }""",
                        text,
                    )
            try:
                await page.wait_for_load_state(timeout=5000)
            except Exception:
                pass
            if action.press_enter:
                await page.keyboard.press("Enter")
                try:
                    await page.wait_for_load_state(timeout=5000)
                except Exception:
                    pass

        elif action_type == "keypress":
            try:
                keys = action.keys
                if keys and len(keys) > 2:
                    unique = set(k.lower() for k in keys)
                    if len(unique) == 1:
                        normalized = _normalize_key(keys[0])
                        for i in range(len(keys)):
                            await page.keyboard.press(normalized)
                            if i < len(keys) - 1:
                                await asyncio.sleep(0.05)
                        keys = None
                if keys:
                    combo = "+".join(_normalize_key(k) for k in keys)
                    await page.keyboard.press(combo)
                elif action.key:
                    if "+" in action.key:
                        combo = "+".join(_normalize_key(k) for k in action.key.split("+"))
                        await page.keyboard.press(combo)
                    else:
                        await page.keyboard.press(_normalize_key(action.key))
            except Exception as e:
                logger.warning(f"Keypress failed for key={action.key!r} keys={action.keys!r}: {e}")

        elif action_type == "scroll":
            if action.coordinate:
                x, y = action.coordinate
            else:
                x, y = session.viewport_width // 2, session.viewport_height // 2

            await page.mouse.move(x, y)

            if action.scroll_x is not None or action.scroll_y is not None:
                dx = action.scroll_x or 0
                dy = action.scroll_y or 0
                await page.mouse.wheel(dx, dy)
            elif action.scroll_direction and action.scroll_amount:
                pixels = action.scroll_amount * 100
                direction_map = {"up": (0, -pixels), "down": (0, pixels), "left": (-pixels, 0), "right": (pixels, 0)}
                dx, dy = direction_map.get(action.scroll_direction, (0, 0))
                await page.mouse.wheel(dx, dy)

        elif action_type == "hover":
            x, y = action.coordinate or [0, 0]
            await page.mouse.move(x, y)
            try:
                await page.wait_for_load_state(timeout=5000)
            except Exception:
                pass

        elif action_type == "drag":
            if action.path and len(action.path) >= 2:
                def _pt(p):
                    if isinstance(p, dict):
                        return int(p["x"]), int(p["y"])
                    return int(p[0]), int(p[1])

                sx, sy = _pt(action.path[0])
                await page.mouse.move(sx, sy)
                await page.mouse.down()
                for point in action.path[1:]:
                    px, py = _pt(point)
                    await page.mouse.move(px, py)
                await page.mouse.up()
            elif action.start_coordinate and action.end_coordinate:
                sx, sy = action.start_coordinate
                ex, ey = action.end_coordinate
                await page.mouse.move(sx, sy)
                try:
                    await page.wait_for_load_state(timeout=5000)
                except Exception:
                    pass
                await page.mouse.down()
                await page.mouse.move(ex, ey)
                await page.mouse.up()
                try:
                    await page.wait_for_load_state(timeout=5000)
                except Exception:
                    pass

        elif action_type == "goto":
            url = action.url or ""
            if url and not url.startswith(("http://", "https://")):
                url = "https://" + url
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass
            except Exception:
                await page.goto(url, wait_until="networkidle", timeout=30000)

        elif action_type == "wait":
            duration_ms = action.duration or 1000
            await asyncio.sleep(duration_ms / 1000.0)

        elif action_type == "zoom":
            region = action.region
            if region and len(region) == 4:
                x1, y1, x2, y2 = region
                screenshot_bytes = await page.screenshot(type="png", full_page=False)
                from io import BytesIO
                from PIL import Image

                img = Image.open(BytesIO(screenshot_bytes))
                w, h = img.size
                x1c, y1c = max(0, min(x1, w)), max(0, min(y1, h))
                x2c, y2c = max(0, min(x2, w)), max(0, min(y2, h))
                cropped = img.crop((x1c, y1c, x2c, y2c))
                buf = BytesIO()
                cropped.save(buf, format="PNG")
                current_url = await self.get_current_url(env_id)
                return base64.b64encode(buf.getvalue()).decode("utf-8"), current_url

        elif action_type == "screenshot":
            pass  # just take screenshot below

        elif action_type == "new_tab":
            new_page = await session.context.new_page()
            await new_page.set_viewport_size(
                {"width": session.viewport_width, "height": session.viewport_height}
            )
            session.page = new_page
            if action.url:
                tab_url = action.url
                if not tab_url.startswith(("http://", "https://")):
                    tab_url = "https://" + tab_url
                try:
                    await new_page.goto(tab_url, wait_until="domcontentloaded", timeout=30000)
                    try:
                        await new_page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                except Exception:
                    await new_page.goto(tab_url, wait_until="networkidle", timeout=30000)

        elif action_type == "close_tab":
            pages = session.context.pages
            if len(pages) > 1:
                await page.close()
                session.page = session.context.pages[-1]

        elif action_type == "switch_tab":
            pages = session.context.pages
            idx = action.tab_index or 0
            if 0 <= idx < len(pages):
                session.page = pages[idx]
                await session.page.bring_to_front()

        elif action_type == "go_back":
            await page.go_back(wait_until="domcontentloaded", timeout=10000)

        elif action_type == "go_forward":
            await page.go_forward(wait_until="domcontentloaded", timeout=10000)

        else:
            logger.warning(f"Unknown action_type: {action_type}")

        await asyncio.sleep(0.3)

        screenshot_b64 = await self.take_screenshot(env_id)
        current_url = await self.get_current_url(env_id)
        return screenshot_b64, current_url
