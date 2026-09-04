# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.browser_session import BrowserSessionHandle
from nemo_gym.web.models import WebAction, WebObservation, WebTask
from nemo_gym.web.visual_browser import (
    NAVIGATION_WAIT_UNTIL,
    RESET_WAIT_UNTIL,
    VisualBrowserDriver,
    VisualBrowserDriverConfig,
    _paste_unicode,
    _stop_clipboard_owner,
    _type_browser_text,
    _url_origin,
    is_retryable_navigation_transport_error,
)


def _config(**updates: Any) -> VisualBrowserDriverConfig:
    return VisualBrowserDriverConfig.model_validate(
        {
            "name": "visual-browser",
            "host": "localhost",
            "port": 8010,
            "entrypoint": "app.py",
            "domain": "agent",
            "num_workers": 1,
            "headless": False,
            **updates,
        }
    )


def _task(**updates: Any) -> WebTask:
    return WebTask.model_validate(
        {
            "benchmark": "webvoyager",
            "task_id": "ArXiv--13",
            "intent": "Find a paper",
            "start_urls": ["https://one.example/start"],
            "runtime_profile": "visual_browser",
            "action_profile": "computer_use",
            **updates,
        }
    )


class _Page:
    def __init__(self, context: _Context | None = None, url: str = "about:blank") -> None:
        self.context = context
        self.url = url
        self.calls: list[tuple[Any, ...]] = []
        self.goto_results: list[Any] = []
        self.raise_title = False

    def on(self, event: str, callback: Any) -> None:
        self.calls.append(("on", event, callback))

    def title(self) -> str:
        if self.raise_title:
            raise RuntimeError("title unavailable")
        return f"Title for {self.url}"

    def goto(self, url: str, *, wait_until: str) -> str:
        self.calls.append(("goto", url, wait_until))
        if self.goto_results:
            result = self.goto_results.pop(0)
            if isinstance(result, Exception):
                raise result
        self.url = url
        return "response"

    def go_back(self, *, wait_until: str) -> None:
        self.calls.append(("back", wait_until))

    def go_forward(self, *, wait_until: str) -> None:
        self.calls.append(("forward", wait_until))

    def bring_to_front(self) -> None:
        self.calls.append(("front",))


class _Context:
    def __init__(self) -> None:
        self.pages: list[_Page] = []
        self.calls: list[tuple[Any, ...]] = []
        self.page_handler: Any = None

    def new_page(self) -> _Page:
        page = _Page(self)
        self.pages.append(page)
        return page

    def set_default_timeout(self, timeout: int) -> None:
        self.calls.append(("timeout", timeout))

    def set_default_navigation_timeout(self, timeout: int) -> None:
        self.calls.append(("navigation_timeout", timeout))

    def on(self, event: str, callback: Any) -> None:
        self.page_handler = callback
        self.calls.append(("on", event))

    def close(self) -> None:
        self.calls.append(("close",))


class _Browser:
    def __init__(self, context: _Context) -> None:
        self.context = context
        self.context_kwargs: dict[str, Any] = {}
        self.closed = False

    def new_context(self, **kwargs: Any) -> _Context:
        self.context_kwargs = kwargs
        return self.context

    def close(self) -> None:
        self.closed = True


class _Playwright:
    def __init__(self, browser: _Browser) -> None:
        self.browser = browser
        self.launch_kwargs: dict[str, Any] = {}
        self.stopped = False
        self.chromium = self

    def launch(self, **kwargs: Any) -> _Browser:
        self.launch_kwargs = kwargs
        return self.browser

    def stop(self) -> None:
        self.stopped = True


class _PyAutoGUI:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.FAILSAFE = True
        self.PAUSE = 1.0

    def __getattr__(self, name: str):
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, *args, kwargs))

        return record


def _install_pyautogui(monkeypatch: pytest.MonkeyPatch) -> _PyAutoGUI:
    module = _PyAutoGUI()
    monkeypatch.setitem(sys.modules, "pyautogui", module)
    return module


def _driver(tmp_path: Path, **config_updates: Any) -> VisualBrowserDriver:
    return VisualBrowserDriver(
        _config(**config_updates),
        "session-1",
        WebArtifactStore(tmp_path),
        BrowserSessionHandle(session_id="session-1", provider_name="local_process", owner_pid=os.getpid()),
    )


def test_config_transport_helpers_and_default_hooks(tmp_path: Path) -> None:
    assert _config().max_sessions == 1
    with pytest.raises(ValueError, match="headed Chromium"):
        _config(headless=True)
    with pytest.raises(ValueError, match="max_sessions=1"):
        _config(max_sessions=2)

    credential_url = "https://user:secret@example.test:8443/a?q=private"  # pragma: allowlist secret
    assert _url_origin(credential_url) == "https://example.test:8443"
    assert _url_origin("relative") == "unknown"
    assert is_retryable_navigation_transport_error(RuntimeError("Page.goto: net::ERR_CONNECTION_RESET"))
    assert not is_retryable_navigation_transport_error(RuntimeError("Page.goto: Timeout 45000ms exceeded"))
    assert not is_retryable_navigation_transport_error(RuntimeError("net::ERR_CONNECTION_RESET"))

    driver = _driver(tmp_path)
    task = _task()
    assert driver._proxy_for_task(task) == ""
    assert not driver._access_helpers_enabled()
    assert driver._prepare_task(task) == task.start_urls
    assert driver._reset_metadata() == {}
    assert driver._navigation_retry_delays() == ()
    assert not driver._should_retry_navigation(RuntimeError("no retry"))
    assert not driver._runtime_target_closed()
    assert not driver._runtime_should_terminate()
    driver._configure_proxy_metadata({})
    driver._install_context_scripts(task)
    driver._reset_task_state(task)
    driver._after_start_navigation(object(), "https://example.test")
    driver._after_reset()
    driver._after_tool_call("wait", failure_step=0)
    driver._before_post_action_capture(failure_step=0)


def test_reset_capture_evaluation_and_close(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _Context()
    browser = _Browser(context)
    playwright = _Playwright(browser)
    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: types.SimpleNamespace(start=lambda: playwright)
    package = types.ModuleType("playwright")
    package.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("PIL.ImageGrab.grab", lambda **_kwargs: b"png-payload")

    driver = _driver(tmp_path, browser_channel="chrome")
    driver._configure_pyautogui = lambda: None  # type: ignore[method-assign]
    driver._proxy_for_task = lambda _task: "http://user:pass@proxy.example:19407"  # type: ignore[method-assign]
    metadata_seen: list[dict[str, str]] = []
    driver._configure_proxy_metadata = metadata_seen.append  # type: ignore[method-assign]

    observation, metadata = driver.reset(_task(start_urls=["https://one.example", "https://two.example"]))

    assert playwright.launch_kwargs["headless"] is False
    assert playwright.launch_kwargs["channel"] == "chrome"
    assert browser.context_kwargs["proxy"] == {
        "server": "http://proxy.example:19407",
        "username": "user",
        "password": "pass",
    }
    assert metadata_seen == [browser.context_kwargs["proxy"]]
    assert [page.url for page in context.pages] == ["https://one.example", "https://two.example"]
    assert observation.url == "https://two.example"
    assert metadata["proxy_enabled"] is True
    assert driver.observe() == observation
    evaluation = driver.evaluation_context()
    assert evaluation.page is context.pages[-1]
    assert len(evaluation.evidence) == 1
    assert evaluation.artifact_dir == tmp_path / "session-1"

    browser.close = lambda: (_ for _ in ()).throw(RuntimeError("already closed"))  # type: ignore[method-assign]
    driver.close()
    assert playwright.stopped
    with pytest.raises(RuntimeError, match="not been reset"):
        driver.observe()
    with pytest.raises(RuntimeError, match="not been reset"):
        driver.evaluation_context()
    driver.close()


def test_reset_requires_display(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(ValueError, match="DISPLAY is required"):
        _driver(tmp_path).reset(_task())


def test_capture_handles_inactive_tab_and_title_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DISPLAY", ":88")
    monkeypatch.setattr("PIL.ImageGrab.grab", lambda **_kwargs: b"image")
    driver = _driver(tmp_path)
    context = _Context()
    first = context.new_page()
    first.url = "https://first.example"
    first.raise_title = True
    active = _Page(url="https://detached.example")
    driver._context = context
    driver._page = active
    driver._task = _task()
    driver._step = 4
    driver._started_at = 0.0

    observation = driver._capture()

    assert observation.active_tab_index == 0
    assert observation.tabs[0].title == ""
    assert observation.url == "https://detached.example"
    assert observation.screenshot is not None
    assert len(driver._evidence) == 1

    dialog = types.SimpleNamespace(accepted=False)
    dialog.accept = lambda: setattr(dialog, "accepted", True)
    VisualBrowserDriver._configure_page(first)
    callback = next(call[2] for call in first.calls if call[:2] == ("on", "dialog"))
    callback(dialog)
    assert dialog.accepted


def test_step_success_terminal_and_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", lambda _seconds: None)
    driver = _driver(tmp_path, action_delay_seconds=0)
    context = _Context()
    page = context.new_page()
    driver._context = context
    driver._page = page
    driver._task = _task()
    driver._observation = WebObservation(url="https://before.example")
    driver._capture = lambda: WebObservation(url=page.url)  # type: ignore[method-assign]

    navigate = WebAction(
        name="computer_use_tool_calls",
        script="",
        arguments={"calls": [{"id": "1", "name": "navigate", "arguments": {"url": "https://after.example"}}]},
    )
    result = driver.step(navigate)
    assert result.execution_ok and not result.terminated
    assert result.observation.url == "https://after.example"
    assert result.info == {"runtime_status": "running"}

    terminal = WebAction(
        name="computer_use_tool_calls",
        script="",
        terminal=True,
        arguments={
            "calls": [
                {
                    "id": "2",
                    "name": "terminate",
                    "arguments": {"status": "success", "answer": "done"},
                }
            ]
        },
    )
    prior = driver._observation
    result = driver.step(terminal)
    assert result.terminated and result.observation is prior
    assert result.info == {"runtime_status": "done"}

    driver._observation = None
    with pytest.raises(RuntimeError, match="no prior observation"):
        driver.step(terminal)

    driver._observation = WebObservation()
    mismatch = WebAction(
        name="computer_use_tool_calls",
        script="",
        terminal=True,
        arguments={"calls": [{"id": "3", "name": "navigate", "arguments": {"url": "https://x.example"}}]},
    )
    result = driver.step(mismatch)
    assert not result.execution_ok and result.terminated
    assert result.info["runtime_status"] == "error"
    assert "terminal flag" in result.info["action_error"]

    driver._page = None
    with pytest.raises(RuntimeError, match="not been reset"):
        driver.step(navigate)


def test_step_bounds_hook_failure_and_closed_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", lambda _seconds: None)

    class Driver(VisualBrowserDriver):
        closed = False

        def _before_post_action_capture(self, *, failure_step: int) -> None:
            raise RuntimeError(f"hook failed at {failure_step}")

        def _runtime_target_closed(self) -> bool:
            return self.closed

        def _runtime_should_terminate(self) -> bool:
            return self.closed

    driver = Driver(
        _config(action_delay_seconds=0),
        "session-1",
        WebArtifactStore(tmp_path),
        BrowserSessionHandle(
            session_id="session-1",
            provider_name="local_process",
            owner_pid=os.getpid(),
        ),
    )
    context = _Context()
    driver._context = context
    driver._page = context.new_page()
    driver._task = _task()
    driver._observation = WebObservation(url="https://before.example")
    driver._capture = lambda: WebObservation(url="https://captured.example")  # type: ignore[method-assign]
    action = WebAction(
        name="computer_use_tool_calls",
        script="",
        arguments={
            "calls": [{"id": "1", "name": "computer", "arguments": {"actions": [{"action": "wait", "duration": 0}]}}]
        },
    )
    driver._execute_call = lambda *_args: None  # type: ignore[method-assign]

    result = driver.step(action)
    assert not result.execution_ok and result.terminated
    assert "hook failed" in result.info["action_error"]

    driver.closed = True
    result = driver.step(action)
    assert result.terminated
    # A closed browser target cannot be captured safely. Preserve the last
    # known-good observation from the preceding bounded failure.
    assert result.observation.url == "https://captured.example"


def test_navigation_and_tool_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", sleeps.append)
    driver = _driver(tmp_path)
    context = _Context()
    first = context.new_page()
    second = context.new_page()
    driver._context = context
    driver._page = first

    first.goto_results = [RuntimeError("drop"), "ok"]
    driver._navigation_retry_delays = lambda: (0.25,)  # type: ignore[method-assign]
    driver._should_retry_navigation = lambda _exc: True  # type: ignore[method-assign]
    assert driver._goto(first, "https://retry.example", wait_until="load") == "response"
    assert sleeps == [0.25]

    first.goto_results = [RuntimeError("fatal")]
    driver._should_retry_navigation = lambda _exc: False  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="fatal"):
        driver._goto(first, "https://fatal.example", wait_until="load")

    executed: list[dict[str, Any]] = []
    driver._execute_computer = executed.append  # type: ignore[method-assign]
    driver._execute_call("computer", {"actions": [{"action": "wait", "duration": 1}]})
    assert executed == [{"action": "wait", "duration": 1}]

    driver._execute_call("navigate", {"url": "back", "tab_id": 1})
    assert driver._page is second
    driver._execute_call("navigate", {"url": "forward"})
    driver._execute_call("navigate", {"url": "https://new.example"})
    assert ("back", NAVIGATION_WAIT_UNTIL) in second.calls
    assert ("forward", NAVIGATION_WAIT_UNTIL) in second.calls

    before = len(context.pages)
    driver._execute_call("tabs_create", {})
    assert len(context.pages) == before + 1
    driver._execute_call("tabs_create", {"url": "https://tab.example"})
    assert driver._page.url == "https://tab.example"
    driver._execute_call("tabs_focus", {"tab_id": 0})
    assert driver._page is first
    driver._execute_call("terminate", {})
    driver._select_page(None)
    with pytest.raises(ValueError, match="unknown tab_id"):
        driver._select_page(99)
    with pytest.raises(ValueError, match="unsupported visual-browser tool"):
        driver._execute_call("unknown", {})

    first.goto_results = []
    driver._goto_task_start(first, "https://start.example")
    assert ("goto", "https://start.example", RESET_WAIT_UNTIL) in first.calls


@pytest.mark.parametrize(
    ("spec", "expected_method"),
    [
        ({"action": "left_click", "coordinate": [0.5, 0.5]}, "click"),
        ({"action": "middle_click", "coordinate": [0.5, 0.5]}, "click"),
        ({"action": "right_click", "coordinate": [0.5, 0.5]}, "click"),
        ({"action": "double_click", "coordinate": [0.5, 0.5]}, "click"),
        ({"action": "triple_click", "coordinate": [0.5, 0.5]}, "click"),
        ({"action": "mouse_move", "coordinate": [0.2, 0.3]}, "moveTo"),
        ({"action": "type", "text": "hello"}, "write"),
        ({"action": "key_press", "keys": ["return"]}, "press"),
        ({"action": "key_press", "keys": ["control", "a"]}, "hotkey"),
        ({"action": "wait", "duration": 0.1}, None),
        (
            {
                "action": "scroll",
                "coordinate": [0.1, 0.2],
                "scroll_parameters": {"scroll_direction": "up", "scroll_amount": 4},
            },
            "scroll",
        ),
        (
            {"action": "scroll", "scroll_parameters": {"scroll_direction": "right", "scroll_amount": 100000}},
            "hscroll",
        ),
        ({"action": "left_click_drag", "start_coordinate": [0, 0], "coordinate": [1, 1]}, "dragTo"),
    ],
)
def test_execute_computer_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    spec: dict[str, Any],
    expected_method: str | None,
) -> None:
    pyautogui = _install_pyautogui(monkeypatch)
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", lambda _seconds: None)
    driver = _driver(tmp_path, viewport_width=1000, viewport_height=500)

    driver._execute_computer(spec)

    if expected_method is not None:
        assert any(call[0] == expected_method for call in pyautogui.calls)


def test_reference_default_pointer_locations_and_zero_wait(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pyautogui = _install_pyautogui(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", sleeps.append)
    driver = _driver(tmp_path, viewport_width=1000, viewport_height=500, action_delay_seconds=2.0)

    driver._execute_computer(
        {"action": "scroll", "scroll_parameters": {"scroll_direction": "down", "scroll_amount": 3}}
    )
    driver._execute_computer({"action": "left_click_drag", "coordinate": [1.0, 1.0]})
    driver._execute_computer({"action": "wait", "duration": 0.0})

    assert ("moveTo", 500, 250, {}) in pyautogui.calls
    assert ("dragTo", 999, 499, {"duration": 0.5, "button": "left"}) in pyautogui.calls
    assert 0.0 in sleeps
    assert 2.0 not in sleeps


def test_computer_action_validation_and_coordinate_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_pyautogui(monkeypatch)
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", lambda _seconds: None)
    driver = _driver(tmp_path, viewport_width=1000, viewport_height=500)
    assert driver._pixel([0.5, 0.5]) == (500, 250)
    assert driver._normalize_key("Command") == "ctrl"
    assert driver._normalize_key("Escape") == "esc"
    assert driver._normalize_key("Option") == "alt"
    assert driver._normalize_key("X") == "x"
    for coordinate in (None, [1], [-0.1, 0], [0, 1.1]):
        with pytest.raises(ValueError, match="coordinate"):
            driver._pixel(coordinate)
    with pytest.raises(ValueError, match="requires keys"):
        driver._execute_computer({"action": "key_press", "keys": []})
    with pytest.raises(ValueError, match="unsupported computer action"):
        driver._execute_computer({"action": "unknown"})

    pyautogui = sys.modules["pyautogui"]
    VisualBrowserDriver._configure_pyautogui()
    assert pyautogui.FAILSAFE is False
    assert pyautogui.PAUSE == 0.0


def test_text_typing_preserves_special_and_unicode_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    pyautogui = _PyAutoGUI()
    pasted: list[str] = []
    monkeypatch.setattr("nemo_gym.web.visual_browser._paste_unicode", lambda _gui, text: pasted.append(text))

    _type_browser_text(pyautogui, "abc<\n雪\t")

    assert pasted == ["雪"]
    assert any(call[0] == "write" and call[1] == "abc" for call in pyautogui.calls)
    assert any(call[0] == "hotkey" and call[1:3] == ("shift", ",") for call in pyautogui.calls)
    assert [call[1] for call in pyautogui.calls if call[0] == "press"] == ["enter", "tab"]


class _ClipboardProcess:
    def __init__(self, *, poll: int | None = None, stdin: Any = object()) -> None:
        self.pid = 17
        self.returncode = poll
        self.stdin = stdin
        self.wait_calls = 0

    def wait(self, timeout: float) -> None:
        self.wait_calls += 1
        if self.wait_calls <= 2 and self.returncode is None:
            raise subprocess.TimeoutExpired("xclip", timeout)

    def poll(self) -> int | None:
        return self.returncode


def test_clipboard_process_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    _stop_clipboard_owner(None)
    finished = _ClipboardProcess(poll=0)
    _stop_clipboard_owner(finished)
    assert finished.wait_calls == 1

    signals: list[int] = []
    running = _ClipboardProcess()
    monkeypatch.setattr("nemo_gym.web.visual_browser.os.killpg", lambda _pid, sig: signals.append(sig))
    _stop_clipboard_owner(running)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert running.wait_calls == 3

    monkeypatch.setattr(
        "nemo_gym.web.visual_browser.os.killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    _stop_clipboard_owner(_ClipboardProcess())


def test_paste_unicode_success_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    pyautogui = _PyAutoGUI()
    monkeypatch.setattr("nemo_gym.web.visual_browser.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("nemo_gym.web.visual_browser.shutil.which", lambda _name: "/usr/bin/xclip")

    class Stdin:
        def __init__(self) -> None:
            self.payload = b""

        def write(self, payload: bytes) -> None:
            self.payload += payload

        def close(self) -> None:
            return None

    stdin = Stdin()
    process = _ClipboardProcess(poll=0, stdin=stdin)
    monkeypatch.setattr("nemo_gym.web.visual_browser.subprocess.Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr("nemo_gym.web.visual_browser._stop_clipboard_owner", lambda _process: None)
    _paste_unicode(pyautogui, "雪")
    assert stdin.payload == "雪".encode()
    assert any(call[0] == "hotkey" and call[1:3] == ("ctrl", "v") for call in pyautogui.calls)

    monkeypatch.setattr("nemo_gym.web.visual_browser.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="xclip is required"):
        _paste_unicode(pyautogui, "雪")

    monkeypatch.setattr("nemo_gym.web.visual_browser.shutil.which", lambda _name: "/usr/bin/xclip")
    monkeypatch.setattr(
        "nemo_gym.web.visual_browser.subprocess.Popen",
        lambda *_args, **_kwargs: _ClipboardProcess(poll=0, stdin=None),
    )
    with pytest.raises(RuntimeError, match="stdin pipe"):
        _paste_unicode(pyautogui, "雪")

    early = _ClipboardProcess(poll=2, stdin=Stdin())
    monkeypatch.setattr("nemo_gym.web.visual_browser.subprocess.Popen", lambda *_args, **_kwargs: early)
    with pytest.raises(RuntimeError, match="exited before paste"):
        _paste_unicode(pyautogui, "雪")
