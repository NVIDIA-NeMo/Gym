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
import sys
import time
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
    # Symbol keys models commonly emit
    "minus": "-",
    "dash": "-",
    "hyphen": "-",
    "endash": "-",
    "underscore": "_",
    "plus": "+",
    "equal": "=",
    "equals": "=",
    "period": ".",
    "dot": ".",
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "slash": "/",
    "forwardslash": "/",
    "backslash": "\\",
    "bracketleft": "[",
    "bracketright": "]",
    "braceleft": "{",
    "braceright": "}",
    "quote": "'",
    "doublequote": '"',
    "backtick": "`",
    "tilde": "~",
    "exclamation": "!",
    "at": "@",
    "hash": "#",
    "dollar": "$",
    "percent": "%",
    "caret": "^",
    "ampersand": "&",
    "asterisk": "*",
    "parenleft": "(",
    "parenright": ")",
    # Multi-word modifier aliases (models sometimes emit "Right Shift" etc.)
    "right shift": "Shift",
    "left shift": "Shift",
    "right ctrl": "Control",
    "left ctrl": "Control",
    "right alt": "Alt",
    "left alt": "Alt",
    "right control": "Control",
    "left control": "Control",
}

# Per-action timeout map (seconds) — values match the rl-gym-harness-ui
# base_playwright.py @with_timeout decorators exactly.
ACTION_TIMEOUTS: dict[str, float] = {
    "screenshot": 20.0,
    "click": 10.0,
    "double_click": 10.0,
    "triple_click": 10.0,
    "right_click": 10.0,
    "middle_click": 10.0,
    "type": 300.0,
    "keypress": 15.0,
    "scroll": 15.0,
    "hover": 10.0,
    "drag": 10.0,
    "goto": 90.0,
    "new_tab": 90.0,
    "close_tab": 10.0,
    "switch_tab": 10.0,
    "go_back": 10.0,
    "go_forward": 10.0,
    "zoom": 20.0,
    "wait": None,
}

CLOSE_SESSION_STEP_TIMEOUT = 60.0


def _normalize_key(key: str) -> str:
    """Normalize a key name (including X11/xdotool names) to Playwright format."""
    return PLAYWRIGHT_KEY_MAP.get(key.lower().strip(), key)


@dataclass
class BrowserSession:
    context: BrowserContext
    page: Page
    viewport_width: int
    viewport_height: int
    slot_index: int = 0
    last_accessed: float = field(default_factory=time.monotonic)


class _BrowserSlot:
    """A single Chromium process and its Playwright instance."""

    __slots__ = ("playwright", "browser", "session_count")

    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.session_count: int = 0

    def is_alive(self) -> bool:
        return self.browser is not None and self.browser.is_connected()

    async def ensure_launched(self):
        if not self.is_alive():
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            )
            self.session_count = 0

    async def close(self):
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                logger.warning("Error closing browser slot: %s", e)
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                logger.warning("Error stopping playwright slot: %s", e)
            self.playwright = None
        self.session_count = 0


class BrowserPool:
    def __init__(
        self,
        max_concurrent: int = 16,
        pool_size: int = 4,
        default_viewport_width: int = 1280,
        default_viewport_height: int = 720,
        session_ttl_seconds: float = 7200.0,
        reaper_interval_seconds: float = 300.0,
    ):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._sessions: dict[str, BrowserSession] = {}
        self._slots: list[_BrowserSlot] = [_BrowserSlot() for _ in range(max(1, pool_size))]
        self._lock = asyncio.Lock()
        self._default_viewport_width = default_viewport_width
        self._default_viewport_height = default_viewport_height
        self._session_ttl = session_ttl_seconds
        self._reaper_interval = reaper_interval_seconds
        self._reaper_task: Optional[asyncio.Task] = None

    async def _acquire_slot(self) -> tuple[Browser, int]:
        """Get or launch the least-loaded browser process. Must be called under self._lock."""
        best_idx = 0
        best_count = float("inf")
        for i, slot in enumerate(self._slots):
            if not slot.is_alive():
                await slot.ensure_launched()
                return slot.browser, i
            if slot.session_count < best_count:
                best_count = slot.session_count
                best_idx = i
        return self._slots[best_idx].browser, best_idx

    async def create_session(
        self,
        env_id: str,
        start_url: str,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
    ) -> str:
        """Create a new browser session. Returns base64 screenshot of the initial page."""
        await self._semaphore.acquire()
        slot_idx: Optional[int] = None
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None
        session_registered = False
        try:
            async with self._lock:
                browser, slot_idx = await self._acquire_slot()
                self._slots[slot_idx].session_count += 1
            vw = viewport_width or self._default_viewport_width
            vh = viewport_height or self._default_viewport_height

            context = await browser.new_context(viewport={"width": vw, "height": vh})
            page = await context.new_page()
            await page.goto(start_url, wait_until="domcontentloaded", timeout=30000)

            session = BrowserSession(
                context=context, page=page, viewport_width=vw, viewport_height=vh, slot_index=slot_idx
            )
            async with self._lock:
                self._sessions[env_id] = session
                session_registered = True

            screenshot_bytes = await page.screenshot(type="png", full_page=False)
            return base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception:
            if session_registered:
                async with self._lock:
                    self._sessions.pop(env_id, None)

            if page is not None:
                try:
                    await asyncio.wait_for(page.close(), timeout=CLOSE_SESSION_STEP_TIMEOUT)
                except Exception:
                    logger.warning("Failed to close page during create_session rollback for env_id=%s", env_id)

            if context is not None:
                try:
                    await asyncio.wait_for(context.close(), timeout=CLOSE_SESSION_STEP_TIMEOUT)
                except Exception:
                    logger.warning("Failed to close context during create_session rollback for env_id=%s", env_id)

            if slot_idx is not None:
                async with self._lock:
                    slot = self._slots[slot_idx]
                    slot.session_count = max(0, slot.session_count - 1)

            self._semaphore.release()
            raise

    def get_session(self, env_id: str) -> BrowserSession:
        if env_id not in self._sessions:
            raise KeyError(f"No browser session found for env_id={env_id}")
        session = self._sessions[env_id]
        session.last_accessed = time.monotonic()
        return session

    async def take_screenshot(self, env_id: str) -> str:
        """Take a screenshot of the current page and return base64-encoded PNG."""
        session = self.get_session(env_id)
        screenshot_bytes = await asyncio.wait_for(
            session.page.screenshot(type="png", full_page=False),
            timeout=ACTION_TIMEOUTS["screenshot"],
        )
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def get_current_url(self, env_id: str) -> str:
        session = self.get_session(env_id)
        return session.page.url

    async def dump_local_storage(self, env_id: str, timeout: float = 10.0) -> str:
        """Dump localStorage as a JSON string (with timeout to guard against stuck browsers)."""
        session = self.get_session(env_id)
        local_storage = await asyncio.wait_for(
            session.page.evaluate("() => JSON.stringify(localStorage)"),
            timeout=timeout,
        )
        return local_storage

    async def close_session(self, env_id: str) -> bool:
        """Close a browser session with timeout-protected escalation.

        Attempts page.close() then context.close(), each with a timeout.
        Always releases the semaphore regardless of cleanup success.
        """
        async with self._lock:
            session = self._sessions.pop(env_id, None)
        if session is None:
            return False

        try:
            try:
                await asyncio.wait_for(session.page.close(), timeout=CLOSE_SESSION_STEP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Page close timed out for env_id=%s (%.0fs)", env_id, CLOSE_SESSION_STEP_TIMEOUT)
            except Exception as e:
                logger.warning("Error closing page for env_id=%s: %s", env_id, e)

            try:
                await asyncio.wait_for(session.context.close(), timeout=CLOSE_SESSION_STEP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Context close timed out for env_id=%s (%.0fs)", env_id, CLOSE_SESSION_STEP_TIMEOUT)
            except Exception as e:
                logger.warning("Error closing context for env_id=%s: %s", env_id, e)
        finally:
            async with self._lock:
                slot = self._slots[session.slot_index]
                slot.session_count = max(0, slot.session_count - 1)
            self._semaphore.release()

        return True

    def start_reaper(self) -> None:
        """Start a background task that periodically closes stale sessions."""
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def stop_reaper(self) -> None:
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None

    async def _reaper_loop(self) -> None:
        """Periodically scan for sessions that exceeded their TTL and close them."""
        while True:
            await asyncio.sleep(self._reaper_interval)
            try:
                await self._reap_stale_sessions()
            except Exception:
                logger.exception("Session reaper encountered an error")

    async def _reap_stale_sessions(self) -> None:
        now = time.monotonic()
        stale_ids: list[str] = []
        async with self._lock:
            for env_id, session in self._sessions.items():
                if now - session.last_accessed >= self._session_ttl:
                    stale_ids.append(env_id)

        for env_id in stale_ids:
            logger.warning("Reaping stale session env_id=%s (idle >= %.0fs)", env_id, self._session_ttl)
            await self.close_session(env_id)

    async def shutdown(self):
        """Close all sessions, stop the reaper, and close all browser processes."""
        await self.stop_reaper()
        async with self._lock:
            env_ids = list(self._sessions.keys())
        for env_id in env_ids:
            await self.close_session(env_id)
        for slot in self._slots:
            await slot.close()

    # ──────────────────────────────────────────────────────────────
    # Action execution with per-action timeouts
    # ──────────────────────────────────────────────────────────────

    async def _do_action(self, session: BrowserSession, env_id: str, action) -> "Optional[str] | tuple[str, str]":
        """Execute the raw Playwright commands for a single BrowserAction.

        This is the inner helper called by execute_action, which wraps it
        in asyncio.wait_for with the appropriate per-action timeout.

        Returns:
            None — success, take a normal screenshot afterwards.
            str  — error message (action failed but page is still usable).
            tuple[str, str] — (screenshot_b64, current_url) for actions
                              that produce their own screenshot (e.g. zoom).
        """
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
                mod = "Meta" if sys.platform == "darwin" else "Control"
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
                    raw_key = action.key
                    if "+" in raw_key:
                        combo = "+".join(_normalize_key(k) for k in raw_key.split("+"))
                        await page.keyboard.press(combo)
                    elif " " in raw_key.strip():
                        for k in raw_key.split():
                            await page.keyboard.press(_normalize_key(k))
                            await asyncio.sleep(0.05)
                    else:
                        await page.keyboard.press(_normalize_key(raw_key))
            except Exception as e:
                error_msg = f"keypress failed: {e}"
                logger.warning("keypress failed for env_id=%s keys=%s key=%s: %s", env_id, action.keys, action.key, e)
                return error_msg

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
                await page.goto(url, wait_until="domcontentloaded", timeout=35000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass
            except Exception:
                await page.goto(url, wait_until="networkidle", timeout=35000)

        elif action_type == "wait":
            duration_ms = action.duration or 1000
            capped_ms = min(duration_ms, 30000)
            await asyncio.sleep(capped_ms / 1000.0)

        elif action_type == "zoom":
            region = action.region
            if region and len(region) == 4:
                x1, y1, x2, y2 = region
                screenshot_bytes = await page.screenshot(type="png", full_page=False)

                def _crop_screenshot():
                    from io import BytesIO

                    from PIL import Image

                    img = Image.open(BytesIO(screenshot_bytes))
                    w, h = img.size
                    x1c, y1c = max(0, min(x1, w)), max(0, min(y1, h))
                    x2c, y2c = max(0, min(x2, w)), max(0, min(y2, h))
                    cropped = img.crop((x1c, y1c, x2c, y2c))
                    buf = BytesIO()
                    cropped.save(buf, format="PNG")
                    return base64.b64encode(buf.getvalue()).decode("utf-8")

                cropped_b64 = await asyncio.to_thread(_crop_screenshot)
                current_url = await self.get_current_url(env_id)
                return cropped_b64, current_url

        elif action_type == "screenshot":
            pass

        elif action_type == "new_tab":
            new_page = await session.context.new_page()
            await new_page.set_viewport_size({"width": session.viewport_width, "height": session.viewport_height})
            session.page = new_page
            if action.url:
                tab_url = action.url
                if not tab_url.startswith(("http://", "https://")):
                    tab_url = "https://" + tab_url
                try:
                    await new_page.goto(tab_url, wait_until="domcontentloaded", timeout=35000)
                    try:
                        await new_page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                except Exception:
                    await new_page.goto(tab_url, wait_until="networkidle", timeout=35000)

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

        return None

    async def execute_action(self, env_id: str, action) -> tuple[str, str, Optional[str]]:
        """Execute a BrowserAction on the page with per-action timeout.

        Returns (screenshot_b64, current_url, error).
        error is None on success, or a message string if the action failed.

        If the action itself times out, we still attempt a screenshot so the
        agent can see the current page state and decide what to do next.
        Only raises if even the recovery screenshot fails.
        """
        session = self.get_session(env_id)
        action_type = action.action_type
        timeout = ACTION_TIMEOUTS.get(action_type, 30.0)
        action_error: Optional[str] = None
        result = None

        if timeout is not None:
            try:
                result = await asyncio.wait_for(
                    self._do_action(session, env_id, action),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Action '%s' timed out after %.0fs for env_id=%s — attempting recovery screenshot",
                    action_type,
                    timeout,
                    env_id,
                )
                action_error = f"Action '{action_type}' timed out after {timeout:.0f}s"
        else:
            result = await self._do_action(session, env_id, action)

        if isinstance(result, tuple):
            return result[0], result[1], None
        if isinstance(result, str):
            action_error = result

        await asyncio.sleep(0.3)

        try:
            screenshot_b64 = await asyncio.wait_for(
                self.take_screenshot(env_id),
                timeout=ACTION_TIMEOUTS["screenshot"],
            )
        except asyncio.TimeoutError:
            logger.error(
                "Post-action screenshot timed out for env_id=%s (action_error=%s)",
                env_id,
                action_error,
            )
            raise

        current_url = await self.get_current_url(env_id)
        return screenshot_b64, current_url, action_error
