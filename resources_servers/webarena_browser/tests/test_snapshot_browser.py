# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.async_api import Route as AsyncRoute
from playwright.async_api import async_playwright
from playwright.sync_api import Browser, sync_playwright

from nemo_gym.web import EvaluatorInfrastructureError, snapshot_target_key
from resources_servers.webarena_browser.reference_evaluation import classic_evaluation as classic
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots


CASES = [
    (
        "status",
        "document.querySelector('.cover-status').lastChild.textContent",
        '<span class="cover-status">Cruising</span>',
        "Cruising",
    ),
    (
        "website",
        "document.querySelector('.profile-header [itemprop=\"url\"]').outerText",
        '<div class="profile-header"><a itemprop="url">egg.tart.com</a></div>',
        "egg.tart.com",
    ),
    (
        "wishlist",
        "document.querySelector('.products-grid.wishlist').outerText",
        '<ul class="products-grid wishlist"><li>new item</li></ul>',
        "new item",
    ),
    (
        "order_note",
        'document.querySelector("#order_history_block").querySelector(".note-list").firstElementChild'
        '.querySelector(".note-list-comment").outerText',
        '<div id="order_history_block"><ul class="note-list"><li>'
        '<span class="note-list-comment">new note</span></li></ul></div>',
        "new note",
    ),
    (
        "attribute",
        "document.querySelector('div.submission__vote').querySelector('form').getAttribute('class')",
        '<div class="submission__vote"><form class="voted"></form></div>',
        "voted",
    ),
    (
        "member_role",
        "func:gitlab_get_project_memeber_role(__page__, 'yjlou')",
        '<table><tr><td data-label="Account"><span class="gl-avatar-labeled-sublabel">@yjlou</span></td>'
        '<td class="col-max-role"><span>Guest</span></td></tr></table>',
        "Guest",
    ),
]


def _target(locator: str, expected: str) -> tuple[dict, dict]:
    target = {
        "url": "http://snapshot.test/page",
        "locator": locator,
        "required_contents": {"must_include": [expected]},
    }
    spec = {"key": snapshot_target_key(target), "target": target}
    return target, {"snapshot_adapters": {"program_html": {"targets": [spec]}}}


def _assert_transition(target: dict, plan: dict, before: dict, after: dict, expected: str) -> None:
    assert before["program_html"][0]["value"] == ""
    assert after["program_html"][0]["value"] == expected
    unchanged = snapshots.build_snapshot_context(plan, before, before)
    records = classic._program_html_snapshot_records(target, unchanged)
    assert classic._score_program_html_snapshot_records(target["required_contents"], records) == 0
    changed = snapshots.build_snapshot_context(plan, before, after)
    records = classic._program_html_snapshot_records(target, changed)
    assert classic._score_program_html_snapshot_records(target["required_contents"], records) == 1


@pytest.fixture
def chromium() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        executable = playwright.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Real-browser regression tests require the qualified Chromium bundle")
        browser = playwright.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        try:
            yield browser
        finally:
            browser.close()


@pytest.mark.parametrize("name,locator,html,expected", CASES, ids=[case[0] for case in CASES])
def test_sync_snapshot_absence_and_real_delta(
    chromium: Browser, name: str, locator: str, html: str, expected: str
) -> None:
    state = {"html": '<main><span class="cover-status"></span></main>'}
    context = chromium.new_context()
    try:
        context.route(
            "http://snapshot.test/**", lambda route: route.fulfill(content_type="text/html", body=state["html"])
        )
        page = context.new_page()
        target, plan = _target(locator, expected)
        before = snapshots.collect_browser_snapshots_sync(page, plan)
        state["html"] = html
        after = snapshots.collect_browser_snapshots_sync(page, plan)
        _assert_transition(target, plan, before, after, expected)
    finally:
        context.close()


def test_sync_snapshot_invalid_selector_stays_infrastructure_failure(chromium: Browser) -> None:
    context = chromium.new_context()
    try:
        context.route("http://snapshot.test/**", lambda route: route.fulfill(content_type="text/html", body="<main/>"))
        page = context.new_page()
        _, plan = _target("document.querySelector('[').outerText", "new")
        broken = snapshots.collect_browser_snapshots_sync(page, plan)
        with pytest.raises(EvaluatorInfrastructureError):
            snapshots.build_snapshot_context(plan, broken, broken)
    finally:
        context.close()


@pytest.mark.asyncio
async def test_async_snapshot_absence_real_delta_and_invalid_selector() -> None:
    async with async_playwright() as playwright:
        executable = playwright.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Real-browser regression tests require the qualified Chromium bundle")
        browser = await playwright.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        try:
            for _, locator, html, expected in CASES:
                state = {"html": '<main><span class="cover-status"></span></main>'}
                context = await browser.new_context()
                try:

                    async def serve(route: AsyncRoute) -> None:
                        await route.fulfill(content_type="text/html", body=state["html"])

                    await context.route("http://snapshot.test/**", serve)
                    page = await context.new_page()
                    target, plan = _target(locator, expected)
                    before = await snapshots.collect_browser_snapshots(page, plan)
                    state["html"] = html
                    after = await snapshots.collect_browser_snapshots(page, plan)
                    _assert_transition(target, plan, before, after, expected)
                finally:
                    await context.close()
            context = await browser.new_context()
            try:

                async def serve_empty(route: AsyncRoute) -> None:
                    await route.fulfill(content_type="text/html", body="<main/>")

                await context.route("http://snapshot.test/**", serve_empty)
                page = await context.new_page()
                _, plan = _target("document.querySelector('[').outerText", "new")
                broken = await snapshots.collect_browser_snapshots(page, plan)
                with pytest.raises(EvaluatorInfrastructureError):
                    snapshots.build_snapshot_context(plan, broken, broken)
            finally:
                await context.close()
        finally:
            await browser.close()
