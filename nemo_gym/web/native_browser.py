# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral headed Chromium/PyAutoGUI mechanics for native web sessions."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import Field, model_validator

from nemo_gym.web.actions import MAX_NATIVE_SCROLL_AMOUNT, parse_native_tool_calls
from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.models import (
    WebAction,
    WebArtifactRef,
    WebObservation,
    WebStepResult,
    WebTab,
    WebTask,
)
from nemo_gym.web.resource_config import WebResourcesServerConfig


LOG = logging.getLogger("nemo_gym.web.native_browser")


class NativeBrowserResourcesServerConfig(WebResourcesServerConfig):
    """Configuration shared by native-browser benchmark policies."""

    artifact_dir: str = "cache/native-browser/artifacts"
    headless: bool = False
    viewport_width: int = Field(default=1920, ge=640)
    viewport_height: int = Field(default=1080, ge=480)
    action_delay_seconds: float = Field(default=2.0, ge=0, le=30)
    default_timeout_ms: int = Field(default=45_000, ge=1_000, le=600_000)
    terminate_on_action_error: bool = True
    max_computer_actions: int = Field(default=20, ge=1, le=100)
    record_video: bool = False
    browser_channel: str | None = None
    task_image_root: str | None = None
    max_task_image_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    max_evidence_screenshots: int = Field(default=200, ge=1, le=500)

    @model_validator(mode="after")
    def validate_native_browser(self) -> "NativeBrowserResourcesServerConfig":
        if self.headless:
            raise ValueError("native visual actions require headed Chromium under Xvfb")
        if self.max_sessions != 1:
            raise ValueError("PyAutoGUI native runtimes require max_sessions=1 per isolated DISPLAY")
        return self


@dataclass(frozen=True, slots=True)
class NativeBrowserEvaluationContext:
    """Live process-local state exposed only to a colocated native evaluator."""

    page: Any
    browser_context: Any
    evidence: tuple[WebArtifactRef, ...]
    artifact_dir: Path | None = None


CHROME_ARGS = [
    "--window-position=0,0",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-breakpad",
    "--disable-component-extensions-with-background-pages",
    "--disable-dev-shm-usage",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
    "--disable-renderer-backgrounding",
    "--force-color-profile=srgb",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--mute-audio",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-quic",
    "--disable-http2",
]
SPECIAL_TEXT_KEYS = {"\n": "enter", "\t": "tab"}
SHIFT_TEXT_KEYS = {"<": ","}
# The pinned native runner settles the initial start URL on domcontentloaded but
# waits for `load` on every navigation the policy requests, so a tool-driven page
# transition is screenshotted after its subresources land.
RESET_WAIT_UNTIL = "domcontentloaded"
NAVIGATION_WAIT_UNTIL = "load"
# Transport-level faults that a benchmark policy may elect to retry. A
# Playwright timeout is deliberately absent: it is a slow page, not necessarily
# a dropped connection, and retrying it can multiply the wait.
RETRYABLE_NAVIGATION_ERRORS = (
    "net::ERR_EMPTY_RESPONSE",
    "net::ERR_PROXY_CONNECTION_FAILED",
    "net::ERR_TUNNEL_CONNECTION_FAILED",
    "net::ERR_CONNECTION_CLOSED",
    "net::ERR_CONNECTION_RESET",
    "net::ERR_TIMED_OUT",
)


def is_retryable_navigation_transport_error(exc: Exception) -> bool:
    """Return whether ``page.goto`` failed with a dropped transport."""

    message = str(exc)
    return "Page.goto:" in message and any(marker in message for marker in RETRYABLE_NAVIGATION_ERRORS)


def _url_origin(url: str) -> str:
    """Return a log-safe origin without URL paths, queries, or credentials."""

    parsed = urlparse(url)
    if not parsed.hostname:
        return "unknown"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme or 'unknown'}://{parsed.hostname}{port}"


def _stop_clipboard_owner(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        process.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=0.5)


def _paste_unicode(pyautogui: Any, text: str) -> None:
    xclip = shutil.which("xclip")
    if xclip is None:
        raise RuntimeError("xclip is required for Unicode browser text input")
    process = subprocess.Popen(
        [xclip, "-selection", "clipboard", "-in"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if process.stdin is None:
        raise RuntimeError("xclip stdin pipe was not created")
    try:
        process.stdin.write(text.encode("utf-8"))
        process.stdin.close()
        time.sleep(0.1)
        if process.poll() not in {None, 0}:
            raise RuntimeError(f"xclip exited before paste with code {process.returncode}")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)
    finally:
        _stop_clipboard_owner(process)


def _type_browser_text(pyautogui: Any, text: str) -> None:
    """Type text without losing Unicode or interpreting newlines as glyphs."""

    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        chunk = "".join(buffer)
        if chunk.isascii():
            pyautogui.write(chunk, interval=0.01)
        else:
            _paste_unicode(pyautogui, chunk)
        buffer.clear()

    for character in text:
        if character in SPECIAL_TEXT_KEYS:
            flush()
            pyautogui.press(SPECIAL_TEXT_KEYS[character])
        elif character in SHIFT_TEXT_KEYS:
            flush()
            pyautogui.hotkey("shift", SHIFT_TEXT_KEYS[character])
        else:
            buffer.append(character)
    flush()


class NativeBrowserDriver:
    """One thread-affine Playwright context with visible PyAutoGUI actions."""

    def __init__(
        self,
        config: NativeBrowserResourcesServerConfig,
        session_id: str,
        artifacts: WebArtifactStore,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.artifacts = artifacts
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._task: WebTask | None = None
        self._observation: WebObservation | None = None
        self._step = 0
        self._started_at = 0.0
        self._last_action = ""
        self._last_error = ""
        self._evidence: deque[WebArtifactRef] = deque(maxlen=config.max_evidence_screenshots)

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]:
        started = time.monotonic()
        LOG.info(
            "event=native_browser_reset_start session=%s benchmark=%s task=%s start_origin=%s "
            "display=%s viewport=%dx%d",
            self.session_id,
            task.benchmark.value,
            task.task_id,
            _url_origin(task.start_urls[0]) if task.start_urls else "none",
            os.environ.get("DISPLAY", "unset"),
            self.config.viewport_width,
            self.config.viewport_height,
        )
        self.close()
        if not os.environ.get("DISPLAY"):
            raise ValueError("DISPLAY is required; run the native resource server under Xvfb")

        from playwright.sync_api import sync_playwright

        self._configure_pyautogui()

        self._playwright = sync_playwright().start()
        launch: dict[str, Any] = {
            "headless": False,
            "args": CHROME_ARGS,
        }
        if self.config.browser_channel:
            launch["channel"] = self.config.browser_channel
        proxy = self._proxy_for_task(task)
        LOG.info(
            "event=native_browser_launch session=%s task=%s proxy_enabled=%s proxy_origin=%s "
            "access_helpers_enabled=%s browser_channel=%s",
            self.session_id,
            task.task_id,
            bool(proxy),
            _url_origin(proxy) if proxy else "none",
            self._access_helpers_enabled(),
            self.config.browser_channel or "bundled",
        )
        self._browser = self._playwright.chromium.launch(**launch)
        context_kwargs: dict[str, Any] = {
            "viewport": {"width": self.config.viewport_width, "height": self.config.viewport_height}
        }
        if proxy:
            context_kwargs["proxy"] = self._playwright_proxy(proxy)
        self._context = self._browser.new_context(**context_kwargs)
        # One context-wide deadline, as the reference runner sets, instead of a
        # per-navigation override. Every Playwright operation is then bounded.
        self._context.set_default_timeout(self.config.default_timeout_ms)
        self._context.set_default_navigation_timeout(self.config.default_timeout_ms)
        if proxy:
            self._configure_proxy_metadata(dict(context_kwargs["proxy"]))
        self._install_context_scripts(task)
        self._context.on("page", self._configure_page)
        self._page = self._context.new_page()
        self._task = task
        self._step = 0
        self._started_at = time.monotonic()
        self._last_action = ""
        self._last_error = ""
        self._evidence.clear()
        self._reset_task_state(task)
        start_urls = self._prepare_task(task)
        for index, start_url in enumerate(start_urls):
            if index > 0:
                self._page = self._context.new_page()
            self._goto_task_start(self._page, start_url)
            self._after_start_navigation(self._page, start_url)
        self._page.bring_to_front()
        time.sleep(self.config.action_delay_seconds)
        self._after_reset()
        self._observation = self._capture()
        LOG.info(
            "event=native_browser_reset_complete session=%s task=%s origin=%s tabs=%d elapsed_seconds=%.3f",
            self.session_id,
            task.task_id,
            _url_origin(self._observation.url),
            len(self._observation.tabs),
            time.monotonic() - started,
        )
        metadata = {
            "runtime_profile": "native_visual",
            "driver": "playwright_context_pyautogui_actions",
            "viewport": [self.config.viewport_width, self.config.viewport_height],
            "proxy_enabled": bool(proxy),
            "site_login_enabled": False,
        }
        metadata.update(self._reset_metadata())
        return self._observation, metadata

    def observe(self) -> WebObservation:
        if self._observation is None:
            raise RuntimeError("native browser has not been reset")
        return self._observation

    def step(self, action: WebAction) -> WebStepResult:
        if self._page is None:
            raise RuntimeError("native browser has not been reset")
        self._last_action = action.raw_model_output or action.name
        self._last_error = ""
        execution_ok = True
        started = time.monotonic()
        lifecycle_failure_step = self._step
        calls = action.arguments.get("calls", [])
        call_names = [str(call.get("name", "unknown")) for call in calls if isinstance(call, dict)]
        LOG.info(
            "event=native_browser_step_start session=%s task=%s step=%d action=%s calls=%s terminal=%s",
            self.session_id,
            self._task.task_id if self._task is not None else "unknown",
            self._step,
            action.name,
            ",".join(call_names) or "none",
            action.terminal,
        )
        try:
            validated_action = parse_native_tool_calls(
                [
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": call.get("name"),
                        "arguments": call.get("arguments"),
                    }
                    for call in calls
                    if isinstance(call, dict)
                ],
                max_computer_actions=self.config.max_computer_actions,
            )
            if validated_action.terminal != action.terminal:
                raise ValueError("native action terminal flag does not match its tool calls")
            calls = validated_action.arguments["calls"]
            for call in calls:
                call_started = time.monotonic()
                self._execute_call(call["name"], call.get("arguments") or {})
                LOG.info(
                    "event=native_browser_tool_complete session=%s task=%s step=%d tool=%s elapsed_seconds=%.3f",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    self._step,
                    call["name"],
                    time.monotonic() - call_started,
                )
                if call["name"] != "terminate":
                    self._after_tool_call(call["name"], failure_step=lifecycle_failure_step)
        except Exception as exc:  # A malformed/failed UI operation is policy-visible.
            execution_ok = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            LOG.exception(
                "event=native_browser_step_failed session=%s task=%s step=%d action=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                action.name,
            )
        self._step += 1
        if action.terminal:
            # The reference trajectory does not add a duplicate screenshot for
            # terminate; return the most recent observation unchanged.
            if self._observation is None:
                raise RuntimeError("terminal action has no prior observation")
        else:
            if not self._runtime_target_closed():
                time.sleep(self.config.action_delay_seconds)
            if not self._runtime_target_closed():
                try:
                    self._before_post_action_capture(failure_step=lifecycle_failure_step)
                except Exception as exc:
                    # Lifecycle hooks run after the tool-execution try block.
                    # They must still become a bounded task result rather than
                    # escaping as a resource-server 500 with no rollout row.
                    execution_ok = False
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    LOG.exception(
                        "event=native_browser_post_action_hook_failed session=%s task=%s step=%d",
                        self.session_id,
                        self._task.task_id if self._task is not None else "unknown",
                        self._step,
                    )
            if not self._runtime_target_closed():
                self._observation = self._capture()
        terminated = (
            action.terminal
            or self._runtime_should_terminate()
            or (not execution_ok and self.config.terminate_on_action_error)
        )
        LOG.info(
            "event=native_browser_step_complete session=%s task=%s step=%d execution_ok=%s "
            "terminated=%s origin=%s elapsed_seconds=%.3f",
            self.session_id,
            self._task.task_id if self._task is not None else "unknown",
            self._step,
            execution_ok,
            terminated,
            _url_origin(self._observation.url) if self._observation is not None else "none",
            time.monotonic() - started,
        )
        info = self._step_info(action)
        return WebStepResult(
            observation=self._observation,
            execution_ok=execution_ok,
            terminated=terminated,
            info=info,
        )

    def evaluation_context(self) -> NativeBrowserEvaluationContext:
        if self._task is None or self._page is None or self._context is None:
            raise RuntimeError("native browser has not been reset")
        return NativeBrowserEvaluationContext(
            page=self._page,
            browser_context=self._context,
            evidence=tuple(self._evidence),
            artifact_dir=self.artifacts.session_dir(self.session_id),
        )

    def close(self) -> None:
        had_runtime = any(owner is not None for owner in (self._context, self._browser, self._playwright))
        task_id = self._task.task_id if self._task is not None else "unknown"
        started = time.monotonic()
        for owner in (self._context, self._browser):
            if owner is not None:
                try:
                    owner.close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._playwright = self._browser = self._context = self._page = None
        self._task = None
        self._observation = None
        if had_runtime:
            LOG.info(
                "event=native_browser_close session=%s task=%s elapsed_seconds=%.3f",
                self.session_id,
                task_id,
                time.monotonic() - started,
            )

    def _proxy_for_task(self, task: WebTask) -> str:
        """Return a browser proxy for this task; local Arena sites use none."""

        del task
        return ""

    def _access_helpers_enabled(self) -> bool:
        return False

    def _configure_proxy_metadata(self, proxy_config: dict[str, str]) -> None:
        del proxy_config

    def _install_context_scripts(self, task: WebTask) -> None:
        del task

    def _reset_task_state(self, task: WebTask) -> None:
        del task

    def _prepare_task(self, task: WebTask) -> list[str]:
        return list(task.start_urls)

    def _after_start_navigation(self, page: Any, url: str) -> None:
        del page, url

    def _after_reset(self) -> None:
        return None

    def _reset_metadata(self) -> dict[str, Any]:
        return {}

    def _after_tool_call(self, name: str, *, failure_step: int) -> None:
        del name, failure_step

    def _before_post_action_capture(self, *, failure_step: int) -> None:
        del failure_step

    def _runtime_target_closed(self) -> bool:
        return False

    def _runtime_should_terminate(self) -> bool:
        return False

    def _step_info(self, action: WebAction) -> dict[str, Any]:
        if self._last_error:
            return {"action_error": self._last_error, "native_status": "error"}
        return {"native_status": "done" if action.terminal else "running"}

    @staticmethod
    def _playwright_proxy(proxy: str) -> dict[str, str]:
        parsed = urlparse(proxy if "://" in proxy else f"http://{proxy}")
        if not parsed.hostname:
            raise ValueError("WA_BROWSER_PROXY_SERVER is not a valid proxy URL")
        server = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            server += f":{parsed.port}"
        config = {"server": server}
        if parsed.username:
            config["username"] = unquote(parsed.username)
        if parsed.password:
            config["password"] = unquote(parsed.password)
        return config

    def _capture(self) -> WebObservation:
        from PIL import ImageGrab

        image = ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"))
        screenshot = self.artifacts.save_screenshot(self.session_id, self._step, image)
        if screenshot.artifact is not None:
            self._evidence.append(screenshot.artifact)
        pages = list(self._context.pages)
        active = pages.index(self._page) if self._page in pages else 0
        tabs = [
            WebTab(index=index, url=page.url, title=self._safe_title(page), active=index == active)
            for index, page in enumerate(pages)
        ]
        if screenshot.artifact is not None:
            LOG.info(
                "event=native_browser_screenshot session=%s task=%s step=%d origin=%s bytes=%d sha256=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                _url_origin(self._page.url if self._page is not None else ""),
                screenshot.artifact.size_bytes,
                screenshot.artifact.sha256[:12],
            )
        return WebObservation(
            goal=[{"type": "text", "text": self._task.intent if self._task else ""}],
            screenshot=screenshot,
            url=self._page.url if self._page is not None else "",
            tabs=tabs,
            active_tab_index=active,
            last_action=self._last_action,
            last_action_error=self._last_error,
            elapsed_time=max(0.0, time.monotonic() - self._started_at),
            metadata={"step": self._step, "runtime": "native_visual"},
        )

    @staticmethod
    def _safe_title(page: Any) -> str:
        try:
            return page.title()
        except Exception:
            return ""

    @staticmethod
    def _configure_page(page: Any) -> None:
        page.on("dialog", lambda dialog: dialog.accept())

    def _goto(self, page: Any, url: str, *, wait_until: str) -> Any:
        """Navigate using the retry policy supplied by a benchmark subclass.

        Only ``page.goto`` is retried. History navigation remains a single
        attempt because retrying ``go_back`` or ``go_forward`` can move twice.
        """

        retry_delays = self._navigation_retry_delays()
        attempts = len(retry_delays) + 1
        for attempt in range(1, attempts + 1):
            try:
                return page.goto(url, wait_until=wait_until)
            except Exception as exc:
                if attempt >= attempts or not self._should_retry_navigation(exc):
                    raise
                delay_seconds = retry_delays[attempt - 1]
                LOG.warning(
                    "event=native_browser_navigation_retry session=%s task=%s step=%d origin=%s "
                    "attempt=%d/%d error_type=%s sleep_seconds=%d",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    self._step,
                    _url_origin(url),
                    attempt,
                    attempts,
                    type(exc).__name__,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
        raise RuntimeError("navigation retry loop exited without a result")

    def _navigation_retry_delays(self) -> tuple[float, ...]:
        return ()

    def _should_retry_navigation(self, exc: Exception) -> bool:
        del exc
        return False

    def _goto_task_start(self, page: Any, url: str) -> Any:
        return self._goto(page, url, wait_until=RESET_WAIT_UNTIL)

    def _execute_call(self, name: str, arguments: dict[str, Any]) -> None:
        if name == "computer":
            for action in arguments["actions"]:
                self._execute_computer(action)
            return
        if name == "navigate":
            self._select_page(arguments.get("tab_id"))
            url = arguments["url"]
            if url == "back":
                self._page.go_back(wait_until=NAVIGATION_WAIT_UNTIL)
            elif url == "forward":
                self._page.go_forward(wait_until=NAVIGATION_WAIT_UNTIL)
            else:
                self._goto(self._page, url, wait_until=NAVIGATION_WAIT_UNTIL)
            self._page.bring_to_front()
            return
        if name == "tabs_create":
            self._page = self._context.new_page()
            url = arguments.get("url", "about:blank")
            if url != "about:blank":
                self._goto(self._page, url, wait_until=NAVIGATION_WAIT_UNTIL)
            self._page.bring_to_front()
            return
        if name == "tabs_focus":
            self._select_page(arguments["tab_id"])
            self._page.bring_to_front()
            return
        if name == "terminate":
            return
        raise ValueError(f"unsupported native tool: {name}")

    def _select_page(self, tab_id: int | None) -> None:
        if tab_id is None:
            return
        pages = list(self._context.pages)
        if not 0 <= tab_id < len(pages):
            raise ValueError(f"unknown tab_id: {tab_id}")
        self._page = pages[tab_id]

    def _execute_computer(self, spec: dict[str, Any]) -> None:
        import pyautogui

        name = spec["action"]
        coordinate = spec.get("coordinate")
        point = self._pixel(coordinate) if coordinate is not None else None
        if name in {"left_click", "middle_click", "right_click", "double_click", "triple_click"}:
            if point is None:
                raise ValueError(f"{name} requires coordinate")
            button = {"middle_click": "middle", "right_click": "right"}.get(name, "left")
            clicks = {"double_click": 2, "triple_click": 3}.get(name, 1)
            pyautogui.click(*point, clicks=clicks, button=button, interval=0.1)
        elif name == "mouse_move":
            pyautogui.moveTo(*point)
        elif name == "type":
            text = str(spec.get("text", ""))
            LOG.info(
                "event=native_browser_type session=%s task=%s step=%d characters=%d unicode=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                len(text),
                not text.isascii(),
            )
            _type_browser_text(pyautogui, text)
        elif name == "key_press":
            keys = [str(key).lower() for key in spec.get("keys") or []]
            if not keys:
                raise ValueError("key_press requires keys")
            normalized = [self._normalize_key(key) for key in keys]
            if len(normalized) == 1:
                pyautogui.press(normalized[0])
            else:
                pyautogui.hotkey(*normalized)
        elif name == "wait":
            time.sleep(float(spec.get("duration") or self.config.action_delay_seconds))
        elif name == "scroll":
            params = spec.get("scroll_parameters") or {}
            requested_amount = int(params.get("scroll_amount", 1))
            amount = max(0, min(requested_amount, MAX_NATIVE_SCROLL_AMOUNT))
            direction = params.get("scroll_direction", "down")
            if amount != requested_amount:
                LOG.warning(
                    "event=native_browser_scroll_clamped session=%s task=%s step=%d requested=%d applied=%d",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    self._step,
                    requested_amount,
                    amount,
                )
            if point is not None:
                pyautogui.moveTo(*point)
            if direction in {"up", "down"}:
                pyautogui.scroll(amount if direction == "up" else -amount)
            else:
                pyautogui.hscroll(amount if direction == "right" else -amount)
        elif name == "left_click_drag":
            start = self._pixel(spec.get("start_coordinate"))
            end = self._pixel(spec.get("coordinate"))
            pyautogui.moveTo(*start)
            pyautogui.dragTo(*end, duration=0.5, button="left")
        else:
            raise ValueError(f"unsupported computer action: {name}")
        time.sleep(0.3)

    def _pixel(self, coordinate: Any) -> tuple[int, int]:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError("coordinate must contain normalized x and y")
        x, y = float(coordinate[0]), float(coordinate[1])
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("coordinate values must be in [0, 1]")
        return (
            max(0, min(self.config.viewport_width - 1, round(x * self.config.viewport_width))),
            max(0, min(self.config.viewport_height - 1, round(y * self.config.viewport_height))),
        )

    @staticmethod
    def _normalize_key(key: str) -> str:
        aliases = {
            "cmd": "ctrl",
            "command": "ctrl",
            "control": "ctrl",
            "return": "enter",
            "escape": "esc",
            "option": "alt",
        }
        return aliases.get(key.lower(), key.lower())

    @staticmethod
    def _configure_pyautogui() -> None:
        os.environ.pop("WAYLAND_DISPLAY", None)
        import pyautogui

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
