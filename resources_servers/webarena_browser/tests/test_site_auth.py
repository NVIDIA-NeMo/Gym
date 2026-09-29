# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import MagicMock, Mock

import pytest

from nemo_gym.web.models import WebTask
from resources_servers.webarena_browser.site_auth import (
    _verify_shopping_login,
    configured_site_urls,
    login_sites,
    resolve_site_templates,
    resolve_start_urls,
)


def _task(**updates) -> WebTask:
    values = {
        "benchmark": "webarena",
        "task_id": "0",
        "runtime_profile": "visual_browser",
        "action_profile": "computer_use",
        "sites": ["gitlab", "reddit"],
        "start_urls": ["__GITLAB__/group/project", "__REDDIT__ |AND| __GITLAB_SSH__"],
    }
    values.update(updates)
    return WebTask.model_validate(values)


def test_site_urls_are_resolved_from_task_scoped_environment(monkeypatch) -> None:
    monkeypatch.setenv("WA_GITLAB", "http://sites.test:8023/")
    monkeypatch.setenv("WA_REDDIT", "http://sites.test:9999")
    task = _task()

    urls = configured_site_urls(task)

    assert urls == {
        "gitlab": "http://sites.test:8023",
        "reddit": "http://sites.test:9999",
    }
    assert resolve_start_urls(task, urls) == [
        "http://sites.test:8023/group/project",
        "http://sites.test:9999 |AND| sites.test:2222",
    ]


def test_missing_required_site_url_fails_before_browser_launch(monkeypatch) -> None:
    monkeypatch.delenv("WA_GITLAB", raising=False)
    monkeypatch.setenv("WA_REDDIT", "http://sites.test:9999")

    with pytest.raises(ValueError, match="WA_GITLAB"):
        configured_site_urls(_task())


def test_evaluator_placeholders_require_and_resolve_nested_site_urls(monkeypatch) -> None:
    monkeypatch.setenv("WA_GITLAB", "http://sites.test:8023")
    task = _task(
        sites=[],
        start_urls=[],
        original_metadata={
            "eval": {
                "program_html": [
                    {
                        "url": "__GITLAB__/group/project/-/issues/1",
                        "locator": "document.querySelector('body').innerText",
                    }
                ]
            }
        },
    )

    urls = configured_site_urls(task)
    resolved = resolve_site_templates(task.original_metadata, urls)

    assert urls == {"gitlab": "http://sites.test:8023"}
    assert resolved["eval"]["program_html"][0]["url"] == "http://sites.test:8023/group/project/-/issues/1"


@pytest.mark.parametrize("path", ["/customer/account/", "/customer/account/index/"])
def test_shopping_login_requires_a_visible_authenticated_dashboard(path) -> None:
    page = MagicMock(url="http://sites.test" + path)
    goto = Mock()

    _verify_shopping_login(page, base_url="http://sites.test", goto=goto)

    goto.assert_called_once_with(page, "http://sites.test/customer/account/")
    page.get_by_role.assert_called_once_with("heading", name="My Account", exact=True)
    page.get_by_role.return_value.wait_for.assert_called_once_with(state="visible", timeout=10000)


@pytest.mark.parametrize(
    "url",
    [
        "http://sites.test/customer/account/login/",
        "http://sites.test/",
        "http://other.test/customer/account/",
    ],
)
def test_shopping_login_rejects_a_login_redirect_or_unexpected_page(url) -> None:
    page = MagicMock(url=url)

    with pytest.raises(RuntimeError, match="authenticated account page"):
        _verify_shopping_login(page, base_url="http://sites.test", goto=Mock())

    page.get_by_role.assert_not_called()


def test_shopping_login_does_not_accept_account_url_without_dashboard() -> None:
    page = MagicMock(url="http://sites.test/customer/account/")
    page.get_by_role.return_value.wait_for.side_effect = TimeoutError("dashboard missing")

    with pytest.raises(TimeoutError, match="dashboard missing"):
        _verify_shopping_login(page, base_url="http://sites.test", goto=Mock())


@pytest.mark.parametrize("recover", [True, False])
def test_shopping_failed_auth_uses_bounded_setup_retries_and_closes_pages(monkeypatch, caplog, recover) -> None:
    monkeypatch.setattr("resources_servers.webarena_browser.site_auth.time.sleep", Mock())
    monkeypatch.setenv("WA_SHOPPING_USERNAME", "fixture-user")
    monkeypatch.setenv("WA_SHOPPING_PASSWORD", "fixture-password")
    caplog.set_level(logging.INFO, logger="nemo_gym.resources_servers.webarena_browser")
    pages = [MagicMock(url="http://sites.test/customer/account/login/") for _ in range(3)]
    if recover:
        pages[1].url = "http://sites.test/customer/account/"
    context = Mock()
    context.new_page.side_effect = pages
    task = _task(sites=["shopping"], start_urls=["__SHOPPING__/product.html"])
    original = task.model_dump()

    def run_login():
        login_sites(task, context=context, site_urls={"shopping": "http://sites.test"}, goto=Mock())

    if recover:
        run_login()
    else:
        with pytest.raises(RuntimeError, match="authenticated account page"):
            run_login()

    attempts = 2 if recover else 3
    assert context.new_page.call_count == attempts
    for page in pages[:attempts]:
        page.close.assert_called_once_with()
    for page in pages[attempts:]:
        page.close.assert_not_called()
    assert sum("webarena_site_login_complete" in r.message for r in caplog.records) == int(recover)
    assert sum("webarena_site_login_verified" in r.message for r in caplog.records) == int(recover)
    assert task.model_dump() == original
