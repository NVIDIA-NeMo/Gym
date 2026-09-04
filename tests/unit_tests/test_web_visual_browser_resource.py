# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.web.browser_session import BrowserSessionHandle
from nemo_gym.web.composed_backend import ComposedWebBackend
from nemo_gym.web.computer_use import nano_omni_tools
from nemo_gym.web.models import (
    BROWSER_TARGET_CLOSED_STATUS,
    CAPTCHA_BUDGET_EXHAUSTED_STATUS,
    WebAction,
    WebBenchmark,
    WebObservation,
    WebTask,
)
from nemo_gym.web.session import EvaluatorConfigurationError
from nemo_gym.web.visual_browser import VisualBrowserEvaluationContext
from resources_servers.visual_browser import backend
from resources_servers.visual_browser.backend import (
    BrowserTargetClosedDuringCaptcha,
    WebVoyagerDriver,
    visual_browser_backend_factory,
)
from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig
from resources_servers.visual_browser.evaluators import WebVoyagerEvidenceEvaluator
from resources_servers.visual_browser.session_manager import VisualBrowserSessionManager


COMPONENT_ROOT = Path(__file__).resolve().parents[2] / "resources_servers" / "visual_browser"


def _config() -> VisualBrowserResourcesServerConfig:
    return VisualBrowserResourcesServerConfig.model_validate(
        {
            "name": "visual-browser",
            "host": "localhost",
            "port": 8010,
            "entrypoint": "app.py",
            "domain": "agent",
            "num_workers": 1,
            "headless": False,
        }
    )


def _task(**updates) -> WebTask:
    return WebTask.model_validate(
        {
            "benchmark": "webvoyager",
            "task_id": "Allrecipes--0",
            "runtime_profile": "visual_browser",
            "action_profile": "computer_use",
            "verifier_profile": "webvoyager_gemini",
            **updates,
        }
    )


def test_session_manager_accepts_only_the_visual_webvoyager_contract() -> None:
    manager = VisualBrowserSessionManager(_config())

    manager._validate_task(_task())
    for updates, expected in (
        ({"benchmark": "webarena"}, "benchmark 'webarena' is disabled"),
        ({"verifier_profile": "webvoyager_llm_judge"}, "verifier_profile=webvoyager_gemini"),
    ):
        with pytest.raises(ValueError, match=expected):
            manager._validate_task(_task(**updates))

    for updates in (
        {"runtime_profile": "browsergym"},
        {"action_profile": "webvoyager_legacy"},
    ):
        with pytest.raises(ValidationError):
            _task(**updates)

    # The manager remains defensive when a task was constructed without normal
    # Pydantic validation by another in-process integration.
    with pytest.raises(ValueError, match="runtime_profile=visual_browser"):
        manager._validate_task(_task().model_copy(update={"runtime_profile": "other"}))
    with pytest.raises(ValueError, match="action_profile=computer_use"):
        manager._validate_task(_task().model_copy(update={"action_profile": "other"}))


def test_component_packages_only_its_own_resource_boundary() -> None:
    project = tomllib.loads((COMPONENT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "nemo-gym" in project["project"]["dependencies"]
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["resources_servers.visual_browser*"]


def test_domain_proxy_includes_google_search_duckduckgo_fallback(monkeypatch) -> None:
    monkeypatch.setenv("WA_BROWSER_PROXY_SERVER", "proxy.example.test:19407")
    driver = WebVoyagerDriver(
        _config(),
        "session-test",
        object(),
        BrowserSessionHandle(session_id="session-test", provider_name="local_process", owner_pid=os.getpid()),
    )

    assert (
        driver._proxy_for_task(_task(start_urls=["https://html.duckduckgo.com/html?q=weather"]))
        == "proxy.example.test:19407"
    )
    assert driver._proxy_for_task(_task(start_urls=["https://github.com/openai"])) == ""


def _driver(monkeypatch, **config_updates) -> WebVoyagerDriver:
    monkeypatch.delenv("CAPSOLVER_API_KEY", raising=False)
    monkeypatch.delenv("WA_CAPTCHA_SOLVER", raising=False)
    config = _config().model_copy(update=config_updates)
    return WebVoyagerDriver(
        config,
        "session-test",
        object(),
        BrowserSessionHandle(session_id="session-test", provider_name="local_process", owner_pid=os.getpid()),
    )


def test_visual_browser_policy_hooks_and_proxy_modes(monkeypatch) -> None:
    monkeypatch.setenv("WA_BROWSER_PROXY_SERVER", "proxy.example.test:19407")
    driver = _driver(monkeypatch)

    assert driver._access_helpers_enabled() is True
    assert driver._navigation_retry_delays() == backend.NAVIGATION_RETRY_DELAYS_S
    retryable = RuntimeError("Page.goto: net::ERR_CONNECTION_RESET")
    assert driver._should_retry_navigation(retryable) is True
    assert driver._should_retry_navigation(RuntimeError("page was slow")) is False
    assert driver._proxy_for_task(_task(start_urls=["https://www.amazon.com/item"]))
    assert driver._proxy_for_task(_task(start_urls=["https://www.google.com/maps/place/test"]))

    driver.config.proxy_mode = "disabled"
    assert driver._proxy_for_task(_task(start_urls=["https://www.amazon.com/item"])) == ""
    driver.config.proxy_mode = "always"
    assert driver._proxy_for_task(_task(start_urls=["https://github.com/openai"])) == "proxy.example.test:19407"
    with pytest.raises(ValueError, match="only accepts WebVoyager"):
        driver._proxy_for_task(_task().model_copy(update={"benchmark": "webarena"}))


class _Context:
    def __init__(self) -> None:
        self.scripts: list[str] = []

    def add_init_script(self, script: str) -> None:
        self.scripts.append(script)


class _Solver:
    def __init__(self, *, result: bool = False, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.phases: list[str] = []

    def maybe_solve(self, _page, *, phase: str) -> bool:
        self.phases.append(phase)
        if self.error is not None:
            raise self.error
        return self.result


def _action(*, terminal: bool = False) -> WebAction:
    return WebAction(name="computer_use", script="", arguments={"calls": []}, terminal=terminal)


def test_visual_browser_context_and_task_hooks(monkeypatch, caplog) -> None:
    driver = _driver(monkeypatch)
    context = _Context()
    driver._context = context
    task = _task(start_urls=["https://example.test", "https://example.org"])

    driver._install_context_scripts(task)
    assert context.scripts == [backend.CAPTCHA_INTERCEPT_SCRIPT, backend.PRINT_INTERCEPT_SCRIPT]
    driver._configure_proxy_metadata({"server": "http://proxy.example.test:19407"})
    assert getattr(context, backend.BROWSER_PROXY_CONFIG_ATTR)["server"].startswith("http://")
    driver._context = object()
    driver._task = task
    driver._configure_proxy_metadata({"server": "http://proxy.example.test:19407"})
    assert "visual_browser_proxy_metadata_failed" in caplog.text

    driver._last_captcha_failure_step = 4
    driver._captcha_failures = 2
    driver._captcha_budget_exhausted = True
    driver._browser_target_closed = True
    driver._reset_task_state(task)
    assert driver._prepare_task(task) == list(task.start_urls)
    assert driver._last_captcha_failure_step is None
    assert driver._captcha_failures == 0
    assert driver._runtime_should_terminate() is False
    driver._after_start_navigation(object(), task.start_urls[0])

    for hook in (driver._install_context_scripts, driver._reset_task_state):
        with pytest.raises(ValueError, match="only accepts WebVoyager"):
            hook(task.model_copy(update={"benchmark": "webarena"}))


def test_visual_browser_captcha_lifecycle_and_statuses(monkeypatch) -> None:
    driver = _driver(monkeypatch)
    driver._task = _task()
    driver._page = type("Page", (), {"url": "https://example.test/private?q=secret"})()
    driver._step = 7

    solver = _Solver(result=True)
    driver._captcha_solver = solver
    driver._after_reset()
    driver._after_tool_call("navigate", failure_step=0)
    driver._before_post_action_capture(failure_step=0)
    assert solver.phases == ["initial", "after navigate", "before post-action screenshot"]
    assert driver._reset_metadata() == {"captcha_enabled": False, "site_login_enabled": False}

    driver._last_error = "solver failed"
    driver._captcha_budget_exhausted = True
    assert driver._runtime_should_terminate() is True
    assert driver._step_info(_action())["runtime_status"] == CAPTCHA_BUDGET_EXHAUSTED_STATUS
    driver._captcha_budget_exhausted = False
    driver._browser_target_closed = True
    assert driver._runtime_target_closed() is True
    assert driver._step_info(_action())["runtime_status"] == BROWSER_TARGET_CLOSED_STATUS

    skipped = _Solver(result=True)
    driver._captcha_solver = skipped
    driver._before_post_action_capture(failure_step=1)
    assert skipped.phases == []


def test_visual_browser_captcha_failures_are_bounded_and_target_close_is_distinct(monkeypatch) -> None:
    driver = _driver(monkeypatch)
    driver._task = _task()
    driver._page = type("Page", (), {"url": "https://example.test"})()

    driver._captcha_solver = _Solver(error=RuntimeError("temporary provider failure"))
    monkeypatch.setenv("WA_MAX_CAPTCHA_FAILURES", "invalid")
    assert driver._maybe_solve_captcha("after navigate", failure_step=2) is False
    assert driver._captcha_failures == 1
    # A second lifecycle hook for the same model step must not spend the budget twice.
    assert driver._maybe_solve_captcha("before capture", failure_step=2) is False
    assert driver._captcha_failures == 1

    monkeypatch.setenv("WA_MAX_CAPTCHA_FAILURES", "0")
    with pytest.raises(RuntimeError, match="failed more than 0 times"):
        driver._maybe_solve_captcha("after navigate", failure_step=3)
    assert driver._captcha_budget_exhausted is True

    target_closed_type = type("TargetClosedError", (RuntimeError,), {"__module__": "playwright.sync_api"})
    target_driver = _driver(monkeypatch)
    target_driver._task = _task()
    target_driver._page = type("Page", (), {"url": "https://example.test"})()
    target_driver._captcha_solver = _Solver(error=target_closed_type("closed"))
    with pytest.raises(BrowserTargetClosedDuringCaptcha):
        target_driver._maybe_solve_captcha("after navigate", failure_step=0)
    assert target_driver._browser_target_closed is True
    assert target_driver._captcha_failures == 0


def test_visual_browser_required_solver_and_factory(monkeypatch) -> None:
    driver = _driver(monkeypatch, require_captcha_solver=True)
    driver._task = _task()
    driver._page = type("Page", (), {"url": "https://example.test"})()
    with pytest.raises(RuntimeError, match="captcha solver is required"):
        driver._maybe_solve_captcha("initial")

    lease = BrowserSessionHandle(provider_name="local_process", owner_pid=os.getpid())
    composed = visual_browser_backend_factory(_config(), "session-test", object(), lease)
    assert isinstance(composed, ComposedWebBackend)
    assert isinstance(composed.driver, WebVoyagerDriver)


def test_visual_browser_config_and_nano_tool_schema_are_environment_safe(monkeypatch) -> None:
    config = _config()
    monkeypatch.setenv("CAPSOLVER_API_KEY", "CAP-secret")
    assert config.captcha_api_key() == "CAP-secret"
    assert config.captcha_solver() == "builtin:capsolver"
    monkeypatch.setenv("WA_CAPTCHA_PROVIDER", "none")
    assert config.captcha_solver() == ""
    monkeypatch.setenv("WA_CAPTCHA_SOLVER", "approved.module:factory")
    assert config.captcha_solver() == "approved.module:factory"

    first = nano_omni_tools()
    first[0]["name"] = "mutated"
    assert nano_omni_tools()[0]["name"] == "navigate"


def test_webvoyager_evaluator_exposes_evidence_only_for_external_judging() -> None:
    evaluator = WebVoyagerEvidenceEvaluator()
    task = _task()
    observation = WebObservation()
    context = VisualBrowserEvaluationContext(page=object(), browser_context=object(), evidence=())

    evaluator.prepare(task=task, observation=observation, browser_context=context)
    result = evaluator.evaluate(
        task=task,
        observation=observation,
        final_answer="done",
        browser_context=context,
    )
    assert result.valid_sample is False
    assert result.failure_kind == "external_judge_required"
    assert result.metadata == {"final_answer": "done", "screenshots": 0}
    evaluator.close()

    with pytest.raises(EvaluatorConfigurationError, match="received benchmark"):
        evaluator.prepare(
            task=task.model_copy(update={"benchmark": WebBenchmark.WEBARENA}),
            observation=observation,
            browser_context=context,
        )


def test_visual_browser_requires_one_session_per_display(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_sessions=1 per isolated DISPLAY"):
        VisualBrowserResourcesServerConfig.model_validate(
            {
                "name": "visual-browser",
                "host": "localhost",
                "port": 8010,
                "entrypoint": "app.py",
                "domain": "agent",
                "num_workers": 1,
                "headless": False,
                "max_sessions": 2,
                "artifact_dir": str(tmp_path),
            }
        )
