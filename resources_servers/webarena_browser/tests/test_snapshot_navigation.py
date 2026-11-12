# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nemo_gym.web import EvaluatorInfrastructureError, snapshot_target_key
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots


@pytest.mark.parametrize(
    "requested,landed,expected",
    [
        ("http://site.test/page", "http://site.test/page/#section", True),
        ("http://site.test:80/page", "https://site.test:443/page", True),
        ("https://site.test/page", "http://site.test/page", False),
        ("http://site.test/page?a=1&b=2", "http://site.test/page?b=2&a=1", True),
        ("http://site.test/page?a=1", "http://site.test/page?a=2", False),
        ("http://site.test/page?flag=", "http://site.test/page", False),
        ("http://site.test/page", "http://other.test/page", False),
        ("http://site.test:8080/page", "http://site.test:8081/page", False),
        ("http://site.test/page", "http://site.test/login", False),
        ("http://site.test/page", "http://site.test/other", False),
        ("http://shop.test/store/wishlist/?limit=50", "http://shop.test/store/index.php/wishlist/?limit=50", True),
        ("http://other.test/wishlist", "http://other.test/index.php/wishlist", False),
        ("http://shop.test/store/other", "http://shop.test/store/index.phpother", False),
        ("http://reddit.test/f/news/123/old-title", "http://reddit.test/f/news/123/new-title", True),
        ("http://reddit.test/f/news/123/title", "http://reddit.test/f/news/124/title", False),
        ("http://reddit.test/f/news/123/title", "http://reddit.test/f/other/123/title", False),
        ("http://other.test/f/news/123/old", "http://other.test/f/news/123/new", False),
    ],
)
def test_snapshot_resource_identity(monkeypatch, requested, landed, expected):
    monkeypatch.setenv("WA_SHOPPING", "http://shop.test/store")
    monkeypatch.setenv("WA_REDDIT", "http://reddit.test")
    assert snapshots._snapshot_url_matches(requested, landed) is expected


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter", ["program_html", "page_image_query"])
@pytest.mark.parametrize("outcome", ["healthy", "unavailable", "login", "same-url-login", "no-response"])
@pytest.mark.asyncio
async def test_all_collectors_check_navigation_and_close_pages(monkeypatch, asynchronous, adapter, outcome):
    url = "http://snapshot.test/page"
    if adapter == "program_html":
        target = {"url": url, "locator": "document.body.innerText"}
        plan = {"snapshot_adapters": {adapter: {"targets": [{"key": snapshot_target_key(target), "target": target}]}}}
    else:
        query = {"eval_image_url": url, "eval_image_class": "img"}
        plan = {"snapshot_adapters": {adapter: {"queries": [{"key": snapshot_target_key(query), "query": query}]}}}
    method = AsyncMock if asynchronous else Mock

    def read(expression, *args):
        if expression == snapshots._SNAPSHOT_LOGIN_FORM:
            return outcome == "same-url-login"
        return "saved" if adapter == "program_html" else [url + "/image.png"]

    page = SimpleNamespace(
        url="http://snapshot.test/login" if outcome == "login" else url,
        evaluate=method(side_effect=read),
        close=method(),
    )
    response = None if outcome == "no-response" else SimpleNamespace(status=503 if outcome == "unavailable" else 200)
    if asynchronous:
        monkeypatch.setattr(snapshots, "goto", AsyncMock(return_value=response))
        root = SimpleNamespace(context=SimpleNamespace(new_page=AsyncMock(return_value=page)))
        captured = await snapshots.collect_browser_snapshots(root, plan)
        page.close.assert_awaited_once()
    else:
        page.goto = Mock(return_value=response)
        root = SimpleNamespace(context=SimpleNamespace(new_page=Mock(return_value=page)))
        captured = snapshots.collect_browser_snapshots_sync(root, plan)
        page.close.assert_called_once()
    record = captured[adapter][0]
    if outcome == "healthy":
        assert not record.get("error")
        assert record["navigation"] == {"status": 200, "url": url}
        assert record["value" if adapter == "program_html" else "image_urls"] == (
            "saved" if adapter == "program_html" else [url + "/image.png"]
        )
        snapshots.build_snapshot_context(plan, captured, captured)
    else:
        assert record["error"]
        if outcome == "same-url-login":
            page.evaluate.assert_called_once_with(snapshots._SNAPSHOT_LOGIN_FORM)
        else:
            page.evaluate.assert_not_called()
        with pytest.raises(EvaluatorInfrastructureError):
            snapshots.build_snapshot_context(plan, captured, captured)


@pytest.mark.parametrize(
    "phase,status,landed,valid",
    [
        ("before", 200, "/other", False),
        ("after", 200, "/grid", True),
        ("before", 404, "/page", True),
        ("after", 404, "/page", True),
        ("after", 410, "/page", True),
        ("before", 403, "/page", False),
        ("after", 401, "/page", False),
        ("after", 429, "/page", False),
        ("after", 503, "/page", False),
        ("after", 200, "/users/sign_in", False),
        ("after", 200, "/customer/account/login/", False),
        ("after", 200, "/index.php?page=login", False),
        ("before", 400, "/page", False),
        ("after", 400, "/page", True),
    ],
)
def test_navigation_validation_preserves_observed_task_outcomes(phase, status, landed, valid):
    url = "http://snapshot.test/page"
    record = {"url": url, "navigation": {"status": status, "url": "http://snapshot.test" + landed}}
    if valid:
        snapshots._validate_snapshot_navigation(record, phase)
    else:
        with pytest.raises(EvaluatorInfrastructureError):
            snapshots._validate_snapshot_navigation(record, phase)
