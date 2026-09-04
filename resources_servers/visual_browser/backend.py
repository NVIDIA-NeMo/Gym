# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebVoyager-only policy for the shared headed visual-browser driver."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.browser_session import BrowserSessionHandle
from nemo_gym.web.models import (
    BROWSER_TARGET_CLOSED_STATUS,
    CAPTCHA_BUDGET_EXHAUSTED_STATUS,
    WebAction,
    WebBenchmark,
    WebTask,
)
from nemo_gym.web.visual_browser import (
    RESET_WAIT_UNTIL,
    VisualBrowserDriver,
    _url_origin,
    is_retryable_navigation_transport_error,
)
from resources_servers.visual_browser.captcha import (
    BROWSER_PROXY_CONFIG_ATTR,
    CAPTCHA_INTERCEPT_SCRIPT,
    captcha_solver_from_environment,
)
from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig


LOG = logging.getLogger("nemo_gym.resources_servers.visual_browser")

PROXY_START_URL_HOSTS = frozenset(
    {"www.allrecipes.com", "www.amazon.com", "dictionary.cambridge.org", "html.duckduckgo.com"}
)
PROXY_START_URL_PREFIXES = ("https://www.google.com/maps/",)
NAVIGATION_RETRY_DELAYS_S = (4.0, 4.0, 4.0, 8.0)
PRINT_INTERCEPT_SCRIPT = """(() => {
    if (window.__webvoyagerPrintHookInstalled) return;
    Object.defineProperty(window, "__webvoyagerPrintHookInstalled", {value: true});
    Object.defineProperty(window, "__webvoyagerPrintCalled", {value: false, writable: true});
    Object.defineProperty(window, "__webvoyagerPrintCalls", {value: [], writable: true});
    window.print = function() {
        window.__webvoyagerPrintCalled = true;
        window.__webvoyagerPrintCalls.push({url: window.location.href, timestamp: Date.now()});
    };
})();"""


class BrowserTargetClosedDuringCaptcha(RuntimeError):
    """The browser target disappeared while CAPTCHA handling inspected it."""


def _is_playwright_target_closed_error(exc: BaseException) -> bool:
    """Recognize the pinned Playwright target-closed type without a private import."""

    error_type = type(exc)
    return error_type.__name__ == "TargetClosedError" and error_type.__module__.startswith("playwright.")


class WebVoyagerDriver(VisualBrowserDriver):
    """Shared visual browser specialized for WebVoyager's public sites."""

    config: VisualBrowserResourcesServerConfig

    def __init__(
        self,
        config: VisualBrowserResourcesServerConfig,
        session_id: str,
        artifacts: WebArtifactStore,
        browser_lease: BrowserSessionHandle,
    ) -> None:
        super().__init__(config, session_id, artifacts, browser_lease)
        self._last_captcha_failure_step: int | None = None
        self._captcha_failures = 0
        self._captcha_budget_exhausted = False
        self._browser_target_closed = False
        self._captcha_solver = captcha_solver_from_environment()

    def _access_helpers_enabled(self) -> bool:
        return True

    def _navigation_retry_delays(self) -> tuple[float, ...]:
        return NAVIGATION_RETRY_DELAYS_S

    def _should_retry_navigation(self, exc: Exception) -> bool:
        return is_retryable_navigation_transport_error(exc)

    def _proxy_for_task(self, task: WebTask) -> str:
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("visual_browser only accepts WebVoyager tasks")
        if self.config.proxy_mode == "disabled":
            return ""
        proxy = self.config.browser_proxy()
        if self.config.proxy_mode == "always":
            return proxy
        for start_url in task.start_urls:
            parsed = urlparse(start_url)
            if parsed.scheme == "https" and parsed.netloc.lower() in PROXY_START_URL_HOSTS:
                return proxy
            if any(start_url.startswith(prefix) for prefix in PROXY_START_URL_PREFIXES):
                return proxy
        return ""

    def _configure_proxy_metadata(self, proxy_config: dict[str, str]) -> None:
        try:
            setattr(self._context, BROWSER_PROXY_CONFIG_ATTR, proxy_config)
        except Exception:
            LOG.warning(
                "event=visual_browser_proxy_metadata_failed session=%s task=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
            )

    def _install_context_scripts(self, task: WebTask) -> None:
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("visual_browser only accepts WebVoyager tasks")
        self._context.add_init_script(CAPTCHA_INTERCEPT_SCRIPT)
        self._context.add_init_script(PRINT_INTERCEPT_SCRIPT)

    def _reset_task_state(self, task: WebTask) -> None:
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("visual_browser only accepts WebVoyager tasks")
        self._last_captcha_failure_step = None
        self._captcha_failures = 0
        self._captcha_budget_exhausted = False
        self._browser_target_closed = False

    def _prepare_task(self, task: WebTask) -> list[str]:
        return list(task.start_urls)

    def _after_start_navigation(self, page: Any, url: str) -> None:
        del page, url

    def _after_reset(self) -> None:
        self._maybe_solve_captcha("initial")

    def _reset_metadata(self) -> dict[str, Any]:
        return {
            "captcha_enabled": bool(self.config.captcha_solver()),
            "site_login_enabled": False,
        }

    def _after_tool_call(self, name: str, *, failure_step: int) -> None:
        self._maybe_solve_captcha(f"after {name}", failure_step=failure_step)

    def _before_post_action_capture(self, *, failure_step: int) -> None:
        if not self._captcha_budget_exhausted and not self._browser_target_closed:
            self._maybe_solve_captcha("before post-action screenshot", failure_step=failure_step)

    def _runtime_target_closed(self) -> bool:
        return self._browser_target_closed

    def _runtime_should_terminate(self) -> bool:
        return self._captcha_budget_exhausted or self._browser_target_closed

    def _step_info(self, action: WebAction) -> dict[str, Any]:
        # CAPTCHA exhaustion is a site-access failure.  Target closure remains
        # distinguishable in metadata, but the agent treats it as a policy-
        # visible action failure because coordinate actions can close browser
        # chrome; this matches the maintained reference runner.
        if self._browser_target_closed:
            return {"action_error": self._last_error, "runtime_status": BROWSER_TARGET_CLOSED_STATUS}
        if self._captcha_budget_exhausted:
            return {"action_error": self._last_error, "runtime_status": CAPTCHA_BUDGET_EXHAUSTED_STATUS}
        return super()._step_info(action)

    def _goto_task_start(self, page: Any, url: str) -> Any:
        return self._goto(page, url, wait_until=RESET_WAIT_UNTIL)

    def _maybe_solve_captcha(self, phase: str, *, failure_step: int | None = None) -> bool:
        if self.config.require_captcha_solver and not self.config.captcha_solver():
            LOG.error(
                "event=captcha_precondition_failed session=%s task=%s phase=%s missing_env=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                phase,
                self.config.captcha_solver_env,
            )
            raise RuntimeError(f"captcha solver is required but {self.config.captcha_solver_env} is unset")
        try:
            solved = self._captcha_solver.maybe_solve(self._page, phase=phase)
        except Exception as exc:
            if _is_playwright_target_closed_error(exc):
                self._browser_target_closed = True
                LOG.error(
                    "event=captcha_browser_target_closed session=%s task=%s step=%d phase=%s "
                    "failure_budget_counted=false error_type=%s",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    self._step,
                    phase,
                    type(exc).__name__,
                )
                raise BrowserTargetClosedDuringCaptcha(
                    f"browser target closed during CAPTCHA handling at phase {phase!r}"
                ) from None
            LOG.warning(
                "event=captcha_solver_deferred session=%s task=%s step=%d phase=%s origin=%s error_type=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                phase,
                _url_origin(self._page.url if self._page is not None else ""),
                type(exc).__name__,
            )
            if failure_step is not None and self._last_captcha_failure_step != failure_step:
                self._last_captcha_failure_step = failure_step
                self._captcha_failures += 1
                try:
                    max_failures = int(os.environ.get("WA_MAX_CAPTCHA_FAILURES", "3"))
                except ValueError:
                    max_failures = 3
                    LOG.warning("event=captcha_failure_budget_invalid value_present=true fallback=%d", max_failures)
                max_failures = max(0, max_failures)
                LOG.warning(
                    "event=captcha_failure_counted session=%s task=%s step=%d failures=%d max_failures=%d",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    failure_step,
                    self._captcha_failures,
                    max_failures,
                )
                if self._captcha_failures > max_failures:
                    self._captcha_budget_exhausted = True
                    LOG.error(
                        "event=captcha_failure_budget_exhausted session=%s task=%s step=%d "
                        "failures=%d max_failures=%d",
                        self.session_id,
                        self._task.task_id if self._task is not None else "unknown",
                        failure_step,
                        self._captcha_failures,
                        max_failures,
                    )
                    raise RuntimeError(
                        f"Captcha solver failed more than {max_failures} times "
                        f"after VLM inference; aborting task at step {failure_step}"
                    ) from None
            return False
        if solved:
            LOG.info(
                "event=captcha_applied session=%s task=%s step=%d phase=%s origin=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                phase,
                _url_origin(self._page.url if self._page is not None else ""),
            )
        return solved


def visual_browser_backend_factory(
    config,
    session_id: str,
    artifacts: WebArtifactStore,
    browser_lease: BrowserSessionHandle,
):
    """Compose the dedicated browser with its evidence-only evaluator."""

    from nemo_gym.web.composed_backend import ComposedWebBackend
    from resources_servers.visual_browser.evaluators import WebVoyagerEvidenceEvaluator

    return ComposedWebBackend(
        WebVoyagerDriver(config, session_id, artifacts, browser_lease),
        WebVoyagerEvidenceEvaluator(),
    )
