# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import tomllib
from pathlib import Path

import pytest

from nemo_gym.web.browser_session import BrowserSessionHandle
from nemo_gym.web.models import WebObservation, WebTask
from nemo_gym.web.session import EvaluatorConfigurationError
from nemo_gym.web.visual_browser import VisualBrowserDriver, VisualBrowserEvaluationContext
from resources_servers.webarena_browser.app import WebArenaBrowserResourcesServer
from resources_servers.webarena_browser.backend import (
    LOCAL_SETUP_RETRY_DELAYS_S,
    WebArenaBrowserDriver,
    webarena_backend_factory,
)
from resources_servers.webarena_browser.config import WebArenaBrowserResourcesServerConfig
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.session_manager import WebArenaBrowserSessionManager


COMPONENT_ROOT = Path(__file__).resolve().parents[1]


def _config(**updates) -> WebArenaBrowserResourcesServerConfig:
    return WebArenaBrowserResourcesServerConfig.model_validate(
        {
            "name": "webarena",
            "host": "localhost",
            "port": 8010,
            "entrypoint": "app.py",
            "domain": "agent",
            "num_workers": 1,
            "headless": False,
            **updates,
        }
    )


def _webarena_task(**updates) -> WebTask:
    task = WebTask(
        benchmark="webarena",
        task_id="0",
        intent="Return the expected value",
        runtime_profile="visual_browser",
        action_profile="computer_use",
        verifier_profile="webarena_classic",
        original_metadata={
            "id": "webarena-0",
            "eval": {
                "eval_types": ["string_match"],
                "reference_answers": {"exact_match": "expected"},
            },
        },
    )
    return task.model_copy(update=updates)


def _evaluation_context() -> VisualBrowserEvaluationContext:
    return VisualBrowserEvaluationContext(page=object(), browser_context=object(), evidence=())


def _browser_lease() -> BrowserSessionHandle:
    return BrowserSessionHandle(
        session_id="session-test",
        provider_name="local_process",
        owner_pid=os.getpid(),
    )


def test_resource_rejects_non_webarena_benchmark() -> None:
    manager = WebArenaBrowserSessionManager(_config())
    with pytest.raises(ValueError, match="benchmark 'webvoyager' is disabled"):
        manager._validate_task(WebTask(benchmark="webvoyager", task_id="0"))


def test_resource_rejects_mixed_verifier_profile() -> None:
    manager = WebArenaBrowserSessionManager(_config())
    with pytest.raises(ValueError, match="verifier_profile=webarena_classic"):
        manager._validate_task(_webarena_task(verifier_profile="webvoyager_gemini"))


def test_resource_reuses_the_shared_visual_browser_contract() -> None:
    assert issubclass(WebArenaBrowserDriver, VisualBrowserDriver)
    assert issubclass(WebArenaBrowserResourcesServer, object)
    assert list(inspect.signature(webarena_backend_factory).parameters) == [
        "config",
        "session_id",
        "artifacts",
        "browser_lease",
    ]


def test_evaluator_fails_closed_without_installed_benchmark() -> None:
    with pytest.raises(EvaluatorConfigurationError, match="not installed"):
        WebArenaTaskEvaluator().prepare(
            task=_webarena_task(),
            observation=WebObservation(),
            browser_context=_evaluation_context(),
        )


def test_webarena_evaluator_scores_rule_only_task(monkeypatch) -> None:
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")  # pragma: allowlist secret
    evaluator = WebArenaTaskEvaluator(config=_config())
    task = _webarena_task()

    evaluator.prepare(task=task, observation=WebObservation(), browser_context=_evaluation_context())
    result = evaluator.evaluate(
        task=task,
        observation=WebObservation(),
        final_answer="expected",
        browser_context=_evaluation_context(),
    )

    assert result.reward == 1.0
    assert result.task_success
    assert result.valid_sample
    assert result.verifier_version == "webarena-reference-3b775dc"


def test_reference_judge_default_url_includes_single_v1_prefix(monkeypatch) -> None:
    from resources_servers.webarena_browser.reference_evaluation import classic_evaluation

    requested_urls = []

    class _Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == 120.0

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, url: str, **_kwargs) -> _Response:
            requested_urls.append(url)
            return _Response()

    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")  # pragma: allowlist secret
    monkeypatch.delenv("WEBARENA_JUDGE_BASE_URL", raising=False)
    monkeypatch.setattr(classic_evaluation.httpx, "Client", _Client)

    assert classic_evaluation._judge_chat([{"role": "user", "content": "judge this"}]) == "ok"
    assert requested_urls == ["https://inference-api.nvidia.com/v1/chat/completions"]


def test_webarena_evaluator_merges_api_and_browser_snapshots(monkeypatch) -> None:
    from resources_servers.webarena_browser import reference_evaluation

    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")  # pragma: allowlist secret
    api_snapshots = iter(
        [
            {"shopping_orders": [{"increment_id": "1"}]},
            {"shopping_orders": [{"increment_id": "1"}, {"increment_id": "2"}]},
        ]
    )
    browser_snapshots = iter(
        [
            {"program_html": [{"key": "shared", "value": "before"}]},
            {"program_html": [{"key": "shared", "value": "after"}]},
        ]
    )
    captured = {}
    monkeypatch.setattr(reference_evaluation, "collect_snapshots", lambda _plan: next(api_snapshots))
    monkeypatch.setattr(
        reference_evaluation,
        "collect_browser_snapshots_sync",
        lambda _page, _plan: next(browser_snapshots),
    )

    def build_context(plan, before, after):
        captured.update(plan=plan, before=before, after=after)
        return {"snapshots": {"before": before, "after": after}}

    monkeypatch.setattr(reference_evaluation, "build_snapshot_context", build_context)
    monkeypatch.setattr(reference_evaluation, "evaluate_classic_task_sync", lambda *_args, **_kwargs: (1.0, "ok"))
    collision_plan = {
        "snapshot_adapters": {
            "shopping_orders": {},
            "program_html": {"targets": []},
        },
        "target_overrides": {},
    }
    task = _webarena_task(task_kwargs={"collision_plan": collision_plan})
    evaluator = WebArenaTaskEvaluator(config=_config())

    evaluator.prepare(task=task, observation=WebObservation(), browser_context=_evaluation_context())
    evaluator.evaluate(
        task=task,
        observation=WebObservation(),
        final_answer="expected",
        browser_context=_evaluation_context(),
    )

    assert captured["before"] == {
        "shopping_orders": [{"increment_id": "1"}],
        "program_html": [{"key": "shared", "value": "before"}],
    }
    assert captured["after"] == {
        "shopping_orders": [{"increment_id": "1"}, {"increment_id": "2"}],
        "program_html": [{"key": "shared", "value": "after"}],
    }


def test_config_enforces_one_headed_session_per_display() -> None:
    with pytest.raises(ValueError, match="headed Chromium"):
        _config(headless=True)
    with pytest.raises(ValueError, match="max_sessions=1"):
        _config(max_sessions=2)


class _RecordingPage:
    def __init__(self) -> None:
        self.url = "https://example.test/"
        self.goto_calls = []
        self._errors = [RuntimeError("Timeout 45000ms exceeded"), RuntimeError("container still settling")]

    def goto(self, url: str, wait_until: str = "load"):
        self.goto_calls.append((url, wait_until))
        if self._errors:
            raise self._errors.pop(0)


def test_webarena_setup_retries_initial_local_navigation(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("resources_servers.webarena_browser.backend.time.sleep", sleeps.append)
    driver = WebArenaBrowserDriver(_config(), "session-test", object(), _browser_lease())
    driver._task = _webarena_task(task_id="17")
    page = _RecordingPage()

    driver._goto_task_start(page, "http://webarena.test/start")

    assert len(page.goto_calls) == 3
    assert sleeps == list(LOCAL_SETUP_RETRY_DELAYS_S)


def test_component_declares_isolated_runtime_dependencies() -> None:
    project = tomllib.loads((COMPONENT_ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert "nemo-gym" in dependencies
    assert "nemo-gym[dev]" not in dependencies
    assert "playwright==1.55.0" in dependencies
    assert project["tool"]["uv"]["sources"]["nemo-gym"] == {
        "path": "../..",
        "editable": True,
    }
