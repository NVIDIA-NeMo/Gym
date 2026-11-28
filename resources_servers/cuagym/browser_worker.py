# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ray browser actor for CUA-Gym-Hub episodes.

Mirrors ``resources_servers.webarena.browser_worker`` (one persistent
Xvfb + Chromium per actor, one episode at a time) with hub-specific
``reset``/``evaluate``:

- ``reset`` mints a fresh episode sid, fills it into the row's inlined
  setup program (``__CUA_GYM_SID__`` placeholder) alongside the hub
  endpoints, runs the program in a subprocess (source over stdin — nothing
  is written to disk), then navigates the pooled browser to the mock app
  with that sid;
- ``evaluate`` fills the same placeholders into the row's inlined reward
  program, runs it, and parses the printed ``REWARD:`` line.

The class is a copy rather than a subclass because ``@ray.remote`` classes do
not subclass cleanly; ``step`` and the Xvfb/Chromium lifecycle started as copies
of the WebArena worker's (the cuagym knobs in schemas.py change parts of them).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import uuid
from typing import Any
from urllib.parse import urlparse

import psutil
import ray
import requests
from PIL import ImageGrab

from resources_servers.cuagym.egress import egress_allow_hosts, host_allowed, host_resolver_rules_arg
from resources_servers.cuagym.episode_code import (
    SID_PLACEHOLDER,
    app_session,
    app_url,
    parse_reward,
    run_code,
    substitute,
    tail,
)
from resources_servers.cuagym.hub_apps import SESSION_MODE_NONE, SESSION_MODE_SID, endpoint_overrides
from resources_servers.cuagym.internals_guard import (
    VIEW_SOURCE_DOCUMENT_JS,
    VIEW_SOURCE_REASON,
    blocked_message,
    blocked_navigation,
    computer_call_may_open_internals,
)
from resources_servers.cuagym.schemas import CuaGymResourcesServerConfig, task_info_from_row
from resources_servers.cuagym.session_sid import HubScope, hub_scope, sid_init_script
from resources_servers.webarena.actions import (
    decode_tool_arguments,
    execute_computer_actions,
    validate_tool_arguments,
)
from resources_servers.webarena.browser import (
    CHROME_ARGS,
    _install_dialog_handler,
    tab_context,
)
from resources_servers.webarena.browser_worker import _start_xvfb, _stop_process_group
from resources_servers.webarena.schemas import WebArenaTaskRow
from resources_servers.webarena.x11_typing import XKeymapTyper


logger = logging.getLogger(__name__)
_HUB_STATE_RESET_TIMEOUT_SECONDS = 10.0
# Mock-internals sweep (only after a computer call that could open a blocked URL): time for a
# navigation started by the call's last Enter to commit before the tabs are checked.
_SWEEP_SETTLE_SECONDS = 0.5
# _tab_in_front: every frame counts its keydown events (capture phase, installed on first use).
_KEYDOWN_COUNTER_JS = (
    "() => { if (window.__cuaGymKeydowns === undefined) { window.__cuaGymKeydowns = 0;"
    " window.addEventListener('keydown', () => { window.__cuaGymKeydowns += 1; }, true); }"
    " return window.__cuaGymKeydowns; }"
)

import time  # noqa: E402  (kept close to usage parity with the webarena worker)


def _best_effort_reset_hub_state(base_url: str, sid: str) -> None:
    """Delete one SID's remote current and initial state without blocking close."""
    try:
        response = requests.post(
            f"{base_url}/post",
            params={"sid": sid},
            json={"action": "reset"},
            timeout=_HUB_STATE_RESET_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning(
            "Best-effort CUA-Gym remote state reset failed for sid=%s",
            sid,
            exc_info=True,
        )


def _apply_terminate_properly_reward(
    reward: float,
    *,
    status: str | None,
    terminate_properly_reward: float,
) -> float:
    """Require valid termination and add its configured reward bonus."""
    if status in {"success", "failure"}:
        return reward + terminate_properly_reward
    return 0.0


def compute_start_url(base_url: str, sid: str, start_urls: list[str] | None, session_mode: str) -> str:
    """First page of an episode.

    Rows with empty ``start_urls`` open the app root; a non-empty first entry is
    used as-is when absolute (``__CUA_GYM_SID__`` substituted) or resolved against
    ``base_url`` when site-relative. Under the hub ``sid`` convention the session id
    is added as a query parameter, placed before any ``#fragment`` so an anchor such
    as ``#map=15/43.66/-70.25`` survives; apps declared ``sessions: none`` in the
    endpoints YAML get no sid at all.
    """
    row_start = next((u.strip() for u in (start_urls or []) if isinstance(u, str) and u.strip()), "")
    if row_start:
        url = row_start.replace(SID_PLACEHOLDER, sid)
        if not url.startswith(("http://", "https://")):
            url = base_url + (url if url.startswith("/") else "/" + url)
    else:
        url = f"{base_url}/"
    if session_mode == SESSION_MODE_NONE or f"sid={sid}" in url:
        return url
    head, sep, fragment = url.partition("#")
    head += ("&" if "?" in head else "?") + f"sid={sid}"
    return head + sep + fragment


def snapshot_urls(context: Any) -> list[str]:
    """URLs of every open tab, in tab order — the evidence exported as ``CUA_GYM_FINAL_URLS``.

    Blank tabs the agent never navigated carry no evidence and are dropped.
    """
    if context is None:
        return []
    urls: list[str] = []
    for page in list(context.pages):
        try:
            url = page.url
        except Exception:
            continue
        if isinstance(url, str) and url and url != "about:blank":
            urls.append(url)
    return urls


def reward_env(*, status: str | None, answer: str | None, final_urls: list[str]) -> dict[str, str]:
    """Variables handed to a row's reward program on top of run_code's allowlist.

    Reward programs are dataset-supplied and read exactly these: the agent's
    ``terminate(answer=...)`` text, its ``status``, and the open-tab URLs as a JSON list.
    """
    return {
        "CUA_GYM_AGENT_ANSWER": answer or "",
        "CUA_GYM_AGENT_STATUS": status or "",
        "CUA_GYM_FINAL_URLS": json.dumps(final_urls),
    }


def _browser_rss_snapshot() -> tuple[int, int, int]:
    """Return Chromium process count, summed RSS, and actor RSS in bytes."""
    actor = psutil.Process()
    chromium_count = 0
    chromium_rss = 0
    for process in actor.children(recursive=True):
        try:
            descriptor = " ".join((process.name(), *process.cmdline())).lower()
            if "chrome" not in descriptor and "chromium" not in descriptor:
                continue
            chromium_rss += process.memory_info().rss
            chromium_count += 1
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return chromium_count, chromium_rss, actor.memory_info().rss


async def _page_block_reason(page: Any, scope: HubScope) -> str | None:
    """Why the tab must leave its page (internals_guard.py), or None.

    The URL check, plus the document check for a hub page: a view-source tab reports the inner URL.
    """
    url = page.url
    reason = blocked_navigation(url, scope)
    if reason is not None or not scope.contains(url):
        return reason
    try:
        view_source = await asyncio.wait_for(page.evaluate(VIEW_SOURCE_DOCUMENT_JS), timeout=2.0)
    except Exception:
        return None
    return VIEW_SOURCE_REASON if view_source else None


async def _step_off(page: Any, scope: HubScope) -> None:
    """Take a tab off a blocked page: back in its history, else to about:blank."""
    try:
        await page.go_back(wait_until="domcontentloaded")
    except Exception:
        logger.debug("go_back from a blocked page failed", exc_info=True)
    if await _page_block_reason(page, scope) is not None:
        await page.goto("about:blank")


async def _open_episode(
    browser: Any,
    start_url: str,
    *,
    viewport_width: int,
    viewport_height: int,
    init_script: str | None = None,
    bring_to_front: bool = False,
) -> tuple[Any, Any]:
    """Create the episode context/page inside the worker's event loop.

    Playwright's async API (page.on, task scheduling) requires a running
    loop, so everything from new_context to the first goto happens here.
    Module-level on purpose: an ``async def`` method on the actor class would
    make Ray treat the whole actor as an async actor and run its methods
    inside Ray's own event loop, breaking the run_until_complete pattern.
    """
    context = await browser.new_context(viewport={"width": viewport_width, "height": viewport_height})
    context.set_default_timeout(45_000)
    context.set_default_navigation_timeout(45_000)
    if init_script:
        # Context-wide: runs in every document of every tab the agent opens later.
        await context.add_init_script(script=init_script)
    page = await context.new_page()
    _install_dialog_handler(page)
    await page.goto(start_url, wait_until="domcontentloaded")
    if bring_to_front:
        # bring_to_front: page_change. The real harness brings the start page to the front once; later computer
        # calls leave the keyboard focus where the agent put it.
        await page.bring_to_front()
    return context, page


@ray.remote(max_concurrency=1)
class CuaGymBrowserWorker:
    """One persistent Chromium/Xvfb process with at most one active episode."""

    def __init__(self, config: dict[str, Any]):
        self.config = CuaGymResourcesServerConfig.model_validate(config)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.xvfb_process: subprocess.Popen | None = None
        self.display: str | None = None
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None
        self.pyautogui: Any | None = None
        # Keymap-consistent typer for the `type` action (typing_backend: xkeymap); None = pyautogui.write.
        self.typer: XKeymapTyper | None = None
        self.task: WebArenaTaskRow | None = None
        self.sid: str | None = None
        self.reward_code: str | None = None
        self._session_mode: str = SESSION_MODE_SID
        # Open-tab URLs captured when the agent called terminate; None until then
        # (episodes that hit the step cap are snapshotted at evaluation time instead).
        self._final_urls: list[str] | None = None
        # Hosts Chromium may resolve (egress.py); None = open internet (egress_allowlist: false).
        self._egress_allow: list[str] | None = None
        # Hub origins whose internals are blocked this episode (internals_guard.py); None = guard off.
        self._internals_scope: HubScope | None = None
        self._blocked_in_history = False
        self._startup()

    def _startup(self) -> None:
        import os

        self.xvfb_process, self.display = _start_xvfb(
            self.config.viewport_width,
            self.config.viewport_height,
        )
        os.environ["DISPLAY"] = self.display
        os.environ.pop("WAYLAND_DISPLAY", None)
        try:
            import pyautogui
            from playwright.async_api import async_playwright

            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0
            self.pyautogui = pyautogui
            if self.config.typing_backend == "xkeymap":
                from Xlib.ext.xtest import fake_input

                unmapped = self.config.typing_unmapped_chars
                self.typer = XKeymapTyper(
                    pyautogui.platformModule._display,
                    fake_input,
                    unmapped=unmapped,
                    insert_text=self._insert_text if unmapped == "insert_text" else None,
                )
            self.playwright = self.loop.run_until_complete(async_playwright().start())
            args = [
                *CHROME_ARGS,
                f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
            ]
            if self.config.egress_allowlist:
                self._egress_allow = egress_allow_hosts(
                    hub_base_url=self.config.hub_base_url,
                    endpoints=endpoint_overrides()["endpoints"],
                    extra=self.config.egress_allow_extra_hosts,
                    exclude_apps=self.config.egress_exclude_endpoint_apps,
                )
                args.append(host_resolver_rules_arg(self._egress_allow))
                logger.info("CUA-Gym browser egress allowlist: %s", ", ".join(self._egress_allow))
            self.browser = self.loop.run_until_complete(
                self.playwright.chromium.launch(
                    headless=False,
                    args=args,
                )
            )
        except Exception:
            self.shutdown()
            raise

    def health(self) -> dict[str, Any]:
        return {
            "healthy": bool(
                self.xvfb_process
                and self.xvfb_process.poll() is None
                and self.browser is not None
                and self.browser.is_connected()
            ),
            "display": self.display,
            "active": self.context is not None,
        }

    def reset(self, task_payload: dict[str, Any]) -> dict[str, Any]:
        self.close_episode()
        task = WebArenaTaskRow.model_validate(task_payload)
        info = task_info_from_row(task)

        # Resolve the app's endpoint before running setup: an unknown app_dir must
        # fail before the program mutates hub state for an episode that could
        # never be opened. YAML override > row external_base_url > hub port math.
        base_url = app_url(
            info.app_dir,
            self.config.hub_base_url,
            self.config.hub_base_port,
            fallback=info.external_base_url,
        )
        session_mode = app_session(info.app_dir)

        sid = str(uuid.uuid4())
        # Honor a non-empty task start_urls[0] (absolute or site-relative); rows
        # with empty start_urls (all pre-existing manifests) open the app root.
        start_url = compute_start_url(base_url, sid, task.start_urls, session_mode)
        # Checked before setup mutates hub state: a start host the browser cannot
        # resolve is a deployment/config error, not something the policy did.
        if self._egress_blocks(start_url):
            raise RuntimeError(
                f"start URL of task {info.bundle_id} is outside the browser egress allowlist: {start_url} "
                "(add the host to $CUA_GYM_ENDPOINTS_YAML or egress_allow_extra_hosts)"
            )
        if info.initial_setup:
            setup_code = substitute(
                info.initial_setup,
                sid=sid,
                hub_base_url=self.config.hub_base_url,
                base_port=self.config.hub_base_port,
            )
            result = run_code(setup_code, timeout_seconds=self.config.setup_timeout_seconds)
            if result.returncode != 0:
                raise RuntimeError(
                    f"initial_setup failed for task {info.bundle_id} "
                    f"(rc={result.returncode}): {tail(result.stdout)} {tail(result.stderr)}"
                )

        self.task = task
        self.sid = sid
        self.reward_code = info.eval_reward_code
        self._session_mode = session_mode
        self._final_urls = None

        scope = None
        if self.config.propagate_sid_to_new_tabs or self.config.block_mock_internals:
            overrides = endpoint_overrides()
            scope = hub_scope(
                base_url=base_url,
                base_session_mode=session_mode,
                hub_base_url=self.config.hub_base_url,
                endpoints=overrides["endpoints"],
                sessions=overrides["sessions"],
            )
        init_script = sid_init_script(sid, scope) if self.config.propagate_sid_to_new_tabs else None
        self._internals_scope = scope if self.config.block_mock_internals else None
        self._blocked_in_history = False
        self.context, self.page = self.loop.run_until_complete(
            _open_episode(
                self.browser,
                start_url,
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
                init_script=init_script,
                bring_to_front=self.config.bring_to_front == "page_change",
            )
        )
        time.sleep(self.config.post_action_screenshot_delay_seconds)

        return {
            "screenshot": self._screenshot(),
            "tab_context": self.loop.run_until_complete(tab_context(self.page)),
            "task": self.task.model_dump(mode="json"),
        }

    def step(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        if self.context is None or self.page is None or self.task is None:
            raise RuntimeError("No CUA-Gym episode is active")
        name = str(tool_call["name"])
        call_id = str(tool_call["call_id"])
        done = False
        status = None
        answer = None
        function_output = None
        tool_error = None
        sweep_tabs = False
        before_pages = len(self.context.pages)
        try:
            arguments = validate_tool_arguments(
                name,
                decode_tool_arguments(tool_call.get("arguments")),
            )
            if name == "navigate":
                self._select_page(arguments.get("tab_id"))
                url = arguments["url"]
                if url == "back":
                    self.loop.run_until_complete(self.page.go_back(wait_until="domcontentloaded"))
                    function_output = self._leave_blocked_pages([self.page])
                elif url == "forward":
                    self.loop.run_until_complete(self.page.go_forward(wait_until="domcontentloaded"))
                    function_output = self._leave_blocked_pages([self.page])
                else:
                    function_output = self._guarded_goto(self.page, url)
                if self.config.bring_to_front == "page_change":
                    self.loop.run_until_complete(self.page.bring_to_front())
            elif name == "computer":
                if self.config.bring_to_front == "every_computer_call":
                    # Pre-2026-10-18. Page.bringToFront also gives the keyboard focus to the page content, so a
                    # click on the address bar or Ctrl+F in one call did not reach the text typed in the next.
                    self.loop.run_until_complete(self.page.bring_to_front())
                execute_computer_actions(
                    self.pyautogui,
                    arguments,
                    width=self.config.viewport_width,
                    height=self.config.viewport_height,
                    typer=self.typer,
                    normalize_keys=self.config.normalize_key_names,
                )
                sweep_tabs = self._internals_scope is not None and computer_call_may_open_internals(
                    arguments, blocked_page_in_history=self._blocked_in_history
                )
                if len(self.context.pages) > before_pages:
                    self.page = self.context.pages[-1]
            elif name == "tabs_create":
                self.page = self.loop.run_until_complete(self.context.new_page())
                function_output = self._guarded_goto(self.page, arguments.get("url") or "about:blank")
                if self.config.bring_to_front == "page_change":
                    self.loop.run_until_complete(self.page.bring_to_front())
            elif name == "tabs_focus":
                self._select_page(arguments["tab_id"])
                self.loop.run_until_complete(self.page.bring_to_front())
            elif name == "terminate":
                done = True
                status = arguments["status"]
                answer = arguments.get("answer")
                # Evidence for the reward program: the tabs as the agent left them.
                # Captured here because /finalize may run only after the agent
                # server round-trips, and /close never evaluates.
                self._final_urls = snapshot_urls(self.context)
        except Exception as exc:
            logger.info(
                "Recoverable CUA-Gym tool error for %s: %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
            function_output = f"Tool error: {type(exc).__name__}: {exc}"
            tool_error = f"{type(exc).__name__}: {exc}"

        result: dict[str, Any] = {
            "call_id": call_id,
            "function_output": function_output,
            "tool_error": tool_error,
            "done": done,
            "reward": 0.0,
            "termination_status": status,
            "answer": answer,
            "evaluator_details": {},
        }
        if not done:
            time.sleep(self.config.post_action_screenshot_delay_seconds)
            if sweep_tabs:
                # Only after a computer call that could have opened a blocked URL (typed URL, Ctrl+U, ...):
                # the check runs the event loop, which other steps leave idle until the screenshot.
                try:
                    blocked = self._leave_blocked_pages()
                except Exception as exc:
                    logger.warning("Mock-internals check failed", exc_info=True)
                    result["tool_error"] = f"mock-internals check failed: {type(exc).__name__}: {exc}"
                    blocked = None
                if blocked is not None:
                    result["function_output"] = blocked
                    # A tab just went back (or to about:blank): the page is committed, but the browser has not
                    # painted it yet (job 2305571: the screenshot still showed the view-source page).
                    time.sleep(self.config.post_action_screenshot_delay_seconds)
            result.update(
                screenshot=self._screenshot(),
                tab_context=self.loop.run_until_complete(tab_context(self.page)),
            )
        return result

    def evaluate(self, *, status: str | None, answer: str | None) -> dict[str, Any]:
        if self.reward_code is None or self.sid is None:
            raise RuntimeError("No CUA-Gym episode is active")
        # Prefer the snapshot taken at terminate; episodes that ended without a
        # terminate call (step cap) are snapshotted now, while the tabs are still open.
        snapshot = getattr(self, "_final_urls", None)
        final_urls = snapshot if snapshot is not None else snapshot_urls(getattr(self, "context", None))
        details: dict[str, Any] = {
            "sid": self.sid,
            "final_urls": final_urls,
            # The agent scores truncated rollouts as -terminate_properly_reward.
            "terminate_properly_reward": self.config.terminate_properly_reward,
        }
        if status not in {"success", "failure"}:
            details["message"] = "trajectory did not end with a valid terminate tool call; reward evaluation skipped"
            return {
                "reward": 0.0,
                "termination_status": status,
                "answer": answer,
                "evaluator_details": details,
            }
        reward = 0.0
        try:
            reward_code = substitute(
                self.reward_code,
                sid=self.sid,
                hub_base_url=self.config.hub_base_url,
                base_port=self.config.hub_base_port,
            )
            result = run_code(
                reward_code,
                timeout_seconds=self.config.reward_timeout_seconds,
                extra_env=reward_env(status=status, answer=answer, final_urls=final_urls),
            )
            parsed = parse_reward(result.stdout)
            details.update(
                returncode=result.returncode,
                stdout_tail=tail(result.stdout),
                stderr_tail=tail(result.stderr),
            )
            if parsed is None:
                details["message"] = "reward code produced no REWARD: line"
            else:
                reward = parsed
                details["message"] = f"reward code: score={parsed}"
        except subprocess.TimeoutExpired:
            details["message"] = f"reward code timed out after {self.config.reward_timeout_seconds}s"
        except Exception as exc:
            logger.exception("CUA-Gym reward execution failed")
            details["message"] = f"reward execution error: {type(exc).__name__}: {exc}"
        adjusted_reward = _apply_terminate_properly_reward(
            reward,
            status=status,
            terminate_properly_reward=self.config.terminate_properly_reward,
        )
        if self.config.terminate_properly_reward:
            details["message"] += (
                f"; terminate reward bonus applied: {reward} + "
                f"{self.config.terminate_properly_reward} -> {adjusted_reward}"
            )
        reward = adjusted_reward
        return {
            "reward": reward,
            "termination_status": status,
            "answer": answer,
            "evaluator_details": details,
        }

    def _egress_blocks(self, url: str) -> bool:
        """True when the egress rule makes Chromium fail to resolve the URL's host."""
        allow = getattr(self, "_egress_allow", None)
        if allow is None:
            return False
        try:
            host = urlparse(url).hostname
        except ValueError:
            return False
        return bool(host) and not host_allowed(host, allow)

    def _goto(self, page: Any, url: str) -> str | None:
        """Navigate ``page``; return a tool message when the egress rule blocked the host.

        A blocked host fails with net::ERR_NAME_NOT_RESOLVED and the tab shows Chrome's
        error page, as on the egress-restricted real eval host. The real harness feeds the
        error back to the model and the episode goes on, so it is reported as a recoverable
        function output, not as a tool_error (which ends and masks the rollout). Failures
        on allowed hosts keep the tool_error path.
        """
        try:
            self.loop.run_until_complete(page.goto(url, wait_until="domcontentloaded"))
        except Exception as exc:
            if not self._egress_blocks(url):
                raise
            logger.info("Egress-blocked navigation to %s: %s: %s", url, type(exc).__name__, exc)
            return f"Tool error: {type(exc).__name__}: {exc}"
        return None

    def _block_reason(self, url: str) -> str | None:
        scope = getattr(self, "_internals_scope", None)
        return None if scope is None else blocked_navigation(url, scope)

    def _guarded_goto(self, page: Any, url: str) -> str | None:
        """``_goto`` unless the URL is a mock internal (internals_guard.py): then nothing loads."""
        reason = self._block_reason(url)
        if reason is not None:
            logger.info("Blocked navigation to %s (%s)", url, reason)
            return blocked_message(url)
        return self._goto(page, url)

    def _leave_blocked_pages(self, pages: list[Any] | None = None) -> str | None:
        """Take tabs that sit on a blocked URL off it; return the tool message, or None.

        ``pages=None`` checks every tab after a short settle and a driver round trip, so that
        tabs opened and URLs committed by the computer call (no event loop ran meanwhile) are
        visible, including a navigation that the call's last Enter started.
        """
        scope = getattr(self, "_internals_scope", None)
        if scope is None or self.context is None:
            return None
        if pages is None:
            self.loop.run_until_complete(asyncio.sleep(_SWEEP_SETTLE_SECONDS))
            self.loop.run_until_complete(self.context.cookies())
            pages = list(self.context.pages)
        first_blocked: str | None = None
        for page in pages:
            reason = self.loop.run_until_complete(_page_block_reason(page, scope))
            if reason is None:
                continue
            url = page.url
            if reason == VIEW_SOURCE_REASON and not url.lower().startswith("view-source:"):
                url = f"view-source:{url}"  # the tab reports only the inner URL
            logger.info("Tab left blocked URL %s (%s)", url, reason)
            first_blocked = first_blocked or url
            self._blocked_in_history = True
            self.loop.run_until_complete(_step_off(page, scope))
        return None if first_blocked is None else blocked_message(first_blocked)

    def _insert_text(self, text: str) -> None:
        """typing_unmapped_chars: insert_text. CDP Input.insertText into the focused element of a page.

        With bring_to_front: page_change the tab in front can differ from self.page (a tab switch by key or click
        stays), so the target is _tab_in_front(); every_computer_call keeps self.page in front. insert_text never
        reaches the browser UI: when the address bar or the find bar has the keyboard focus, the characters go into
        the focused field of self.page. Never raises: when the insert fails, the characters are lost, as with
        pyautogui.
        """
        try:
            page = self._tab_in_front() if self.config.bring_to_front == "page_change" else self.page
            self.loop.run_until_complete(page.keyboard.insert_text(text))
        except Exception as exc:
            logger.warning(
                "type action: insert_text of %r failed, characters lost: %s: %s", text, type(exc).__name__, exc
            )

    def _tab_in_front(self) -> Any:
        """The tab whose page has the keyboard focus: the only tab, else the tab whose page sees a Shift tap.

        The real harness's method (nemotron_toolcall_agent._tab_in_front, WA_TYPING_FIX=1). Page state cannot tell
        the tab in front: Playwright keeps every tab visible and focused. A key event goes only to the page with the
        keyboard focus, so every frame of every tab counts keydown events, pyautogui taps Shift, and the tab whose
        count goes up is in front. When no tab sees the tap (the browser UI has the keyboard focus), self.page.
        """
        pages = list(self.context.pages)
        if len(pages) == 1:
            return pages[0]

        def keydowns(frame: Any) -> int | None:
            try:
                return self.loop.run_until_complete(asyncio.wait_for(frame.evaluate(_KEYDOWN_COUNTER_JS), timeout=1.0))
            except Exception:
                return None

        frames = [(page, frame) for page in pages for frame in page.frames]
        before = [keydowns(frame) for _, frame in frames]
        self.pyautogui.press("shift")
        hits: list[Any] = []
        for delay in (0.1, 0.15, 0.25):
            self.loop.run_until_complete(asyncio.sleep(delay))
            hits = [page for (page, frame), n in zip(frames, before) if n is not None and (keydowns(frame) or 0) > n]
            if hits:
                break
        if not hits or self.page in hits:
            return self.page
        return hits[-1]

    def _select_page(self, tab_id: Any) -> None:
        if tab_id is None:
            return
        index = int(tab_id)
        pages = list(self.context.pages)
        if index < 0 or index >= len(pages):
            raise ValueError(f"tab_id={index} is outside [0, {len(pages) - 1}]")
        self.page = pages[index]

    def _screenshot(self) -> bytes:
        image = ImageGrab.grab(xdisplay=self.display)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        # Opt-in rollout-visualization dump: set CUAGYM_SCREENSHOT_DUMP_DIR to
        # persist every observation as <dir>/<sid>/<seq>.png. Off by default.
        dump_dir = os.environ.get("CUAGYM_SCREENSHOT_DUMP_DIR")
        if dump_dir and self.sid:
            try:
                ep_dir = os.path.join(dump_dir, self.sid)
                os.makedirs(ep_dir, exist_ok=True)
                self._shot_seq = getattr(self, "_shot_seq", -1) + 1
                with open(os.path.join(ep_dir, f"{self._shot_seq:03d}.png"), "wb") as f:
                    f.write(data)
            except OSError:
                logging.getLogger(__name__).warning("screenshot dump failed", exc_info=True)
        return data

    def close_episode(self) -> bool:
        sid = self.sid
        base_url: str | None = None
        # Stateless apps (sessions: none) hold no per-sid state on the server, so
        # there is nothing to reset and the hub /post endpoint does not exist there.
        if sid is not None and self.task is not None and self._session_mode != SESSION_MODE_NONE:
            try:
                info = task_info_from_row(self.task)
                base_url = app_url(
                    info.app_dir,
                    self.config.hub_base_url,
                    self.config.hub_base_port,
                    fallback=info.external_base_url,
                )
            except ValueError:
                logger.warning(
                    "Could not resolve CUA-Gym endpoint for sid=%s; skipping remote state reset",
                    sid,
                    exc_info=True,
                )
        before_count, before_chromium_rss, before_actor_rss = _browser_rss_snapshot()
        ok = True
        if self.context is not None:
            try:
                self.loop.run_until_complete(self.context.close())
            except Exception:
                logger.warning("Failed to close CUA-Gym browser context", exc_info=True)
                ok = False
            finally:
                self.context = None
                self.page = None
        if sid is not None and base_url is not None:
            _best_effort_reset_hub_state(base_url, sid)
        self.task = None
        self.sid = None
        self.reward_code = None
        self._session_mode = SESSION_MODE_SID
        self._final_urls = None
        self._internals_scope = None
        self._blocked_in_history = False
        after_count, after_chromium_rss, after_actor_rss = _browser_rss_snapshot()
        mib = 1024**2
        logger.info(
            "CUA-Gym browser RSS after close_episode: "
            "sid=%s chromium_processes=%d->%d chromium_rss_mb=%.2f->%.2f "
            "chromium_delta_mb=%+.2f actor_rss_mb=%.2f->%.2f actor_delta_mb=%+.2f",
            sid,
            before_count,
            after_count,
            before_chromium_rss / mib,
            after_chromium_rss / mib,
            (after_chromium_rss - before_chromium_rss) / mib,
            before_actor_rss / mib,
            after_actor_rss / mib,
            (after_actor_rss - before_actor_rss) / mib,
        )
        return ok

    def shutdown(self) -> None:
        self.close_episode()
        if self.browser is not None:
            try:
                self.loop.run_until_complete(self.browser.close())
            except Exception:
                logger.debug("Failed to close Chromium", exc_info=True)
            self.browser = None
        if self.playwright is not None:
            try:
                self.loop.run_until_complete(self.playwright.stop())
            except Exception:
                logger.debug("Failed to stop Playwright", exc_info=True)
            self.playwright = None
        _stop_process_group(self.xvfb_process)
        self.xvfb_process = None
        if not self.loop.is_closed():
            self.loop.close()
