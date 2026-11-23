# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import fcntl
import io
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import ray
from PIL import ImageGrab

from resources_servers.webarena.actions import (
    decode_tool_arguments,
    execute_computer_actions,
    validate_tool_arguments,
)
from resources_servers.webarena.browser import setup_task_context, tab_context
from resources_servers.webarena.schemas import (
    WebArenaResourcesServerConfig,
    WebArenaTaskRow,
)


logger = logging.getLogger(__name__)
_DISPLAY_LOCK = Path("/tmp/nemo-gym-webarena-xvfb.lock")


def _display_available(number: int) -> bool:
    lock_path = Path(f"/tmp/.X{number}-lock")
    socket_path = Path(f"/tmp/.X11-unix/X{number}")
    if not lock_path.exists() and not socket_path.exists():
        return True
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            os.kill(pid, 0)
            return False
        except (ProcessLookupError, ValueError):
            lock_path.unlink(missing_ok=True)
            socket_path.unlink(missing_ok=True)
            return True
        except PermissionError:
            return False
    return False


def _start_xvfb(width: int, height: int, start_display: int = 99) -> tuple[subprocess.Popen, str]:
    _DISPLAY_LOCK.touch(exist_ok=True)
    with _DISPLAY_LOCK.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        display_number = next(
            (number for number in range(start_display, start_display + 1000) if _display_available(number)),
            None,
        )
        if display_number is None:
            raise RuntimeError("No free X11 display is available")
        display = f":{display_number}"
        process = subprocess.Popen(
            [
                "Xvfb",
                display,
                "-screen",
                "0",
                f"{width}x{height}x24",
                "-ac",
                "-nolisten",
                "tcp",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError(f"Xvfb {display} exited with code {process.returncode}")
            ready = subprocess.run(
                ["xdpyinfo", "-display", display],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if ready.returncode == 0:
                return process, display
            time.sleep(0.5)
        os.killpg(process.pid, signal.SIGKILL)
        raise RuntimeError(f"Xvfb {display} did not become ready")


def _stop_process_group(process: subprocess.Popen | None, timeout: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@ray.remote(max_concurrency=1)
class WebArenaBrowserWorker:
    """One persistent Chromium/Xvfb process with at most one active episode."""

    def __init__(self, config: dict[str, Any]):
        self.config = WebArenaResourcesServerConfig.model_validate(config)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.xvfb_process: subprocess.Popen | None = None
        self.display: str | None = None
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None
        self.pyautogui: Any | None = None
        self.task: WebArenaTaskRow | None = None
        self.collision_plan: dict[str, Any] | None = None
        self.before_snapshot: dict[str, Any] = {}
        self.evaluation_config: Any | None = None
        self._startup()

    def _startup(self) -> None:
        self.xvfb_process, self.display = _start_xvfb(
            self.config.viewport_width,
            self.config.viewport_height,
        )
        os.environ["DISPLAY"] = self.display
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ["WEBARENA_JUDGE_MODEL"] = self.config.judge_model
        os.environ["WEBARENA_JUDGE_BASE_URL"] = self.config.judge_base_url
        os.environ["WEBARENA_JUDGE_API_KEY"] = self.config.judge_api_key
        os.environ["WEBARENA_JUDGE_TIMEOUT"] = str(self.config.judge_timeout_seconds)
        try:
            import pyautogui
            from playwright.async_api import async_playwright

            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0
            self.pyautogui = pyautogui
            self.playwright = self.loop.run_until_complete(async_playwright().start())
            self.browser = self.loop.run_until_complete(
                self.playwright.chromium.launch(
                    headless=False,
                    args=[
                        *self._chrome_args(),
                        f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
                    ],
                )
            )
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _chrome_args() -> list[str]:
        from resources_servers.webarena.browser import CHROME_ARGS

        return list(CHROME_ARGS)

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
        self.task, self.context, self.page = self.loop.run_until_complete(
            setup_task_context(
                browser=self.browser,
                task=task,
                urls=self.config.site_urls,
                credentials=self.config.credentials,
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
            )
        )
        self._collect_before_snapshot()
        return {
            "screenshot": self._screenshot(),
            "tab_context": self.loop.run_until_complete(tab_context(self.page)),
            "task": self.task.model_dump(mode="json"),
        }

    def _collect_before_snapshot(self) -> None:
        self.before_snapshot = {}
        self.collision_plan = None
        try:
            from resources_servers.webarena.eval_collision import build_collision_plan
            from resources_servers.webarena.eval_snapshots import (
                collect_browser_snapshots,
                collect_snapshots,
                merge_snapshots,
            )
            from resources_servers.webarena.evaluation import (
                EvaluationConfig,
                HelperConfig,
                JudgeConfig,
            )

            task_dict = self.task.model_dump(mode="json")
            judge_config = (
                JudgeConfig(
                    base_url=self.config.judge_base_url,
                    api_key=self.config.judge_api_key,
                    model=self.config.judge_model,
                    timeout_seconds=self.config.judge_timeout_seconds,
                )
                if self.config.judge_api_key
                else None
            )
            shopping_admin_credentials = self.config.credentials.get(
                "shopping_admin",
                {},
            )
            self.evaluation_config = EvaluationConfig(
                judge_config=judge_config,
                helper_config=HelperConfig(
                    site_urls=self.config.site_urls,
                    shopping_admin_username=shopping_admin_credentials.get(
                        "username",
                        "admin",
                    ),
                    shopping_admin_password=shopping_admin_credentials.get(
                        "password",
                        "admin1234",
                    ),
                    timeout_seconds=self.config.judge_timeout_seconds,
                ),
            )
            self.collision_plan = build_collision_plan(task_dict)
            api_snapshot = collect_snapshots(
                self.collision_plan,
                config=self.evaluation_config,
            )
            browser_snapshot = self.loop.run_until_complete(
                collect_browser_snapshots(
                    self.page,
                    self.collision_plan,
                    config=self.evaluation_config,
                )
            )
            self.before_snapshot = merge_snapshots(api_snapshot, browser_snapshot)
        except Exception:
            logger.warning("Failed to collect WebArena pre-task snapshots", exc_info=True)

    def step(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        if self.context is None or self.page is None or self.task is None:
            raise RuntimeError("No WebArena episode is active")
        name = str(tool_call["name"])
        call_id = str(tool_call["call_id"])
        done = False
        reward = 0.0
        details: dict[str, Any] = {}
        status = None
        answer = None
        tool_error = None
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
                elif url == "forward":
                    self.loop.run_until_complete(self.page.go_forward(wait_until="domcontentloaded"))
                else:
                    self.loop.run_until_complete(self.page.goto(url, wait_until="domcontentloaded"))
            elif name == "computer":
                self.loop.run_until_complete(self.page.bring_to_front())
                execute_computer_actions(
                    self.pyautogui,
                    arguments,
                    width=self.config.viewport_width,
                    height=self.config.viewport_height,
                )
                if len(self.context.pages) > before_pages:
                    self.page = self.context.pages[-1]
            elif name == "tabs_create":
                self.page = self.loop.run_until_complete(self.context.new_page())
                self.loop.run_until_complete(
                    self.page.goto(
                        arguments.get("url") or "about:blank",
                        wait_until="domcontentloaded",
                    )
                )
            elif name == "tabs_focus":
                self._select_page(arguments["tab_id"])
                self.loop.run_until_complete(self.page.bring_to_front())
            elif name == "terminate":
                done = True
                status = arguments["status"]
                answer = arguments.get("answer")
            function_output = f"{name} completed successfully"
        except Exception as exc:
            logger.info(
                "Recoverable WebArena tool error for %s: %s: %s",
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
            "reward": reward,
            "termination_status": status,
            "answer": answer,
            "evaluator_details": details,
        }
        if not done:
            time.sleep(self.config.post_action_screenshot_delay_seconds)
            result.update(
                screenshot=self._screenshot(),
                tab_context=self.loop.run_until_complete(tab_context(self.page)),
            )
        return result

    def evaluate(self, *, status: str | None, answer: str | None) -> dict[str, Any]:
        if self.context is None or self.page is None or self.task is None:
            raise RuntimeError("No WebArena episode is active")
        reward, details = self._evaluate(status=status or "unknown", answer=answer)
        return {
            "reward": reward,
            "termination_status": status,
            "answer": answer,
            "evaluator_details": details,
        }

    def _select_page(self, tab_id: Any) -> None:
        if tab_id is None:
            return
        index = int(tab_id)
        pages = list(self.context.pages)
        if index < 0 or index >= len(pages):
            raise ValueError(f"tab_id={index} is outside [0, {len(pages) - 1}]")
        self.page = pages[index]

    def _evaluate(self, *, status: str, answer: str | None) -> tuple[float, dict[str, Any]]:
        try:
            from resources_servers.webarena.eval_snapshots import (
                build_snapshot_context,
                collect_browser_snapshots,
                collect_snapshots,
                merge_snapshots,
            )
            from resources_servers.webarena.evaluation import evaluate_classic_task

            after_api = collect_snapshots(
                self.collision_plan,
                config=self.evaluation_config,
            )
            after_browser = self.loop.run_until_complete(
                collect_browser_snapshots(
                    self.page,
                    self.collision_plan,
                    config=self.evaluation_config,
                )
            )
            after_snapshot = merge_snapshots(after_api, after_browser)
            eval_context = build_snapshot_context(
                self.before_snapshot,
                after_snapshot,
                self.collision_plan,
            )
            score, message = self.loop.run_until_complete(
                evaluate_classic_task(
                    self.task.model_dump(mode="json"),
                    {"status": status, "answer": answer},
                    self.page,
                    config=self.evaluation_config,
                    eval_context=eval_context,
                )
            )
            return float(score), {
                "message": message,
                "collision_plan": self.collision_plan or {},
            }
        except Exception as exc:
            logger.exception("Classic WebArena evaluation failed")
            return 0.0, {"message": f"Classic evaluation error: {type(exc).__name__}: {exc}"}

    def _screenshot(self) -> bytes:
        image = ImageGrab.grab(xdisplay=self.display)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def close_episode(self) -> bool:
        if self.context is not None:
            try:
                self.loop.run_until_complete(self.context.close())
            except Exception:
                logger.warning("Failed to close WebArena browser context", exc_info=True)
                return False
            finally:
                self.context = None
                self.page = None
                self.task = None
                self.collision_plan = None
                self.before_snapshot = {}
                self.evaluation_config = None
        return True

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
