# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib
from pathlib import Path

import pytest

from nemo_gym.web.models import WebTask
from resources_servers.webvoyager_browser.backend import WebVoyagerBrowserDriver
from resources_servers.webvoyager_browser.config import WebVoyagerBrowserResourcesServerConfig
from resources_servers.webvoyager_browser.session_manager import WebVoyagerBrowserSessionManager


COMPONENT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> WebVoyagerBrowserResourcesServerConfig:
    return WebVoyagerBrowserResourcesServerConfig.model_validate(
        {
            "name": "webvoyager-browser",
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
            "runtime_profile": "native_visual",
            "action_profile": "native_toolcall",
            "verifier_profile": "native_webvoyager_gemini",
            **updates,
        }
    )


def test_session_manager_accepts_only_the_native_webvoyager_contract() -> None:
    manager = WebVoyagerBrowserSessionManager(_config())

    manager._validate_task(_task())
    for updates, expected in (
        ({"benchmark": "webarena"}, "benchmark 'webarena' is disabled"),
        ({"runtime_profile": "browsergym"}, "runtime_profile=native_visual"),
        ({"action_profile": "webvoyager_legacy"}, "action_profile=native_toolcall"),
        ({"verifier_profile": "webvoyager_llm_judge"}, "verifier_profile=native_webvoyager_gemini"),
    ):
        with pytest.raises(ValueError, match=expected):
            manager._validate_task(_task(**updates))


def test_component_packages_only_its_own_resource_boundary() -> None:
    project = tomllib.loads((COMPONENT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "nemo-gym" in project["project"]["dependencies"]
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["resources_servers.webvoyager_browser*"]


def test_domain_proxy_includes_google_search_duckduckgo_fallback(monkeypatch) -> None:
    monkeypatch.setenv("WA_BROWSER_PROXY_SERVER", "proxy.example.test:19407")
    driver = WebVoyagerBrowserDriver(_config(), "session-test", object())

    assert (
        driver._proxy_for_task(_task(start_urls=["https://html.duckduckgo.com/html?q=weather"]))
        == "proxy.example.test:19407"
    )
    assert driver._proxy_for_task(_task(start_urls=["https://github.com/openai"])) == ""
