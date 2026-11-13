# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from playwright.async_api import Route as AsyncRoute
from playwright.async_api import async_playwright
from playwright.sync_api import Browser, sync_playwright

from nemo_gym.web import (
    EvaluatorInfrastructureError,
    VisualBrowserEvaluationContext,
    WebObservation,
    build_collision_plan,
    snapshot_target_key,
)
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.reference_evaluation import classic_evaluation as classic
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.tests.test_contract import _config, _webarena_task


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


def _target(
    locator: str, expected: str, *, url: str = "http://snapshot.test/page", required_contents: dict | None = None
) -> tuple[dict, dict]:
    target = {
        "url": url,
        "locator": locator,
        "required_contents": required_contents if required_contents is not None else {"must_include": [expected]},
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


@pytest.mark.parametrize(
    "before_html,after_html,expected",
    [
        ("saved", "saved", 0),
        ("saved", "saved unrelated", 0),
        ("saved", "saved saved", 1),
        ("", "saved", 1),
        ("", "other", 0),
    ],
)
def test_public_collision_scoring_with_real_captures(chromium, monkeypatch, before_html, after_html, expected):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    target = {
        "url": "http://snapshot.test/page",
        "locator": "document.body.innerText",
        "required_contents": {"must_include": ["saved"]},
    }
    config = {"eval": {"eval_types": ["program_html"], "program_html": [target]}}
    plan = build_collision_plan(config, program_html_collision_keys={f"url:{target['url']}"})
    task = _webarena_task(original_metadata=config, task_kwargs={"collision_plan": plan})
    state = {"html": before_html}
    context = chromium.new_context()
    try:
        context.route(
            "http://snapshot.test/**", lambda route: route.fulfill(content_type="text/html", body=state["html"])
        )
        page = context.new_page()
        page.goto(target["url"])
        browser_context = VisualBrowserEvaluationContext(page=page, browser_context=context, evidence=())
        evaluator = WebArenaTaskEvaluator(config=_config())
        evaluator.prepare(task=task, observation=WebObservation(), browser_context=browser_context)
        state["html"] = after_html
        result = evaluator.evaluate(
            task=task, observation=WebObservation(), final_answer="done", browser_context=browser_context
        )
        assert result.valid_sample
        assert result.reward == expected
    finally:
        context.close()


@pytest.mark.parametrize("name,locator,html,expected", CASES, ids=[case[0] for case in CASES])
def test_sync_final_dom_absence_and_match_in_real_browser(chromium, name, locator, html, expected):
    context = chromium.new_context()
    try:
        page = context.new_page()
        page.set_content('<main><span class="cover-status"></span></main>')
        config = {
            "eval": {
                "eval_types": ["program_html"],
                "program_html": [
                    {"url": "last", "locator": locator, "required_contents": {"must_include": [expected]}}
                ],
            }
        }
        assert classic.evaluate_classic_task_sync(config, {}, page)[0] == 0
        page.set_content(html)
        assert classic.evaluate_classic_task_sync(config, {}, page)[0] == 1
    finally:
        context.close()


@pytest.mark.parametrize("locator", ["document.querySelector('[').outerText", "document.body.notAFunction()"])
def test_sync_invalid_final_dom_in_real_browser_is_inconclusive(chromium, locator):
    context = chromium.new_context()
    try:
        page = context.new_page()
        page.set_content("<main/>")
        config = {
            "eval": {
                "eval_types": ["program_html"],
                "program_html": [
                    {"url": "last", "locator": locator, "required_contents": {"must_include": ["saved"]}}
                ],
            }
        }
        with pytest.raises(EvaluatorInfrastructureError):
            classic.evaluate_classic_task_sync(config, {}, page)
    finally:
        context.close()


NAVIGATION_CASES = [
    ("before-503", (503, "/page", ""), (200, "/page", "saved\nunrelated"), None),
    ("before-login", (200, "/login", ""), (200, "/page", "saved\nunrelated"), None),
    (
        "before-login-document",
        (200, "/page", '<form id="login-form"><input type="password"></form>'),
        (200, "/page", "saved"),
        None,
    ),
    ("healthy-existing", (200, "/page", "saved"), (200, "/page", "saved\nunrelated"), 0),
    ("healthy-empty", (200, "/page", ""), (200, "/page", "saved"), 1),
    ("canonical-redirect", (200, "/page/", ""), (200, "/page/", "saved"), 1),
    ("before-not-found", (404, "/page", "Not found"), (200, "/page", "saved"), 1),
    ("after-not-found", (200, "/page", "saved"), (404, "/page", "Not found"), 0),
    ("after-deletion-redirect", (200, "/page", "saved"), (200, "/grid", "Deleted"), 0),
    ("after-503", (200, "/page", "saved"), (503, "/page", ""), None),
    ("after-login", (200, "/page", "saved"), (200, "/users/sign_in", ""), None),
    (
        "after-login-document",
        (200, "/page", "saved"),
        (200, "/page", '<form id="new_user"><input type="password"></form>'),
        None,
    ),
]
NAVIGATION_CASES = [
    (name, before, after, {"must_include": ["saved"]}, expected) for name, before, after, expected in NAVIGATION_CASES
]
_ERROR_TEXT = '<main id="content">saved</main>'
NAVIGATION_CASES += [
    (
        "before-404-login-form",
        (404, "/page", '<form id="login-form"><input type="password"></form>'),
        (200, "/page", "saved"),
        {"must_include": ["saved"]},
        None,
    ),
    (
        "after-404-login-form",
        (200, "/page", "saved"),
        (404, "/page", '<form id="new_user"><input type="password"></form>'),
        {"must_include": ["saved"]},
        None,
    ),
    ("after-404-matching-error-text", (200, "/page", ""), (404, "/page", _ERROR_TEXT), {"must_include": ["saved"]}, 0),
    ("after-410-matching-error-text", (200, "/page", ""), (410, "/page", _ERROR_TEXT), {"must_include": ["saved"]}, 0),
    (
        "after-redirect-matching-other-text",
        (200, "/page", ""),
        (200, "/grid", "saved"),
        {"must_include": ["saved"]},
        0,
    ),
    (
        "before-404-matching-error-text",
        (404, "/page", _ERROR_TEXT),
        (200, "/page", "saved"),
        {"must_include": ["saved"]},
        1,
    ),
    (
        "before-410-matching-error-text",
        (410, "/page", _ERROR_TEXT),
        (200, "/page", "saved"),
        {"must_include": ["saved"]},
        1,
    ),
    ("deletion-404-empty-criterion", (200, "/page", "old"), (404, "/page", _ERROR_TEXT), {"exact_match": ""}, 1),
    ("deletion-410-empty-criterion", (200, "/page", "old"), (410, "/page", _ERROR_TEXT), {"exact_match": ""}, 1),
    ("other-page-does-not-prove-empty-target", (200, "/page", "old"), (200, "/grid", "saved"), {"exact_match": ""}, 0),
    ("healthy-empty-criterion", (200, "/page", "old"), (200, "/page", ""), {"exact_match": ""}, 1),
    ("unchanged-empty-criterion", (200, "/page", ""), (200, "/page", ""), {"exact_match": ""}, 0),
    ("unchanged-absent-criterion", (404, "/page", _ERROR_TEXT), (404, "/page", _ERROR_TEXT), {"exact_match": ""}, 0),
    ("after-400-matching-error-text", (200, "/page", ""), (400, "/page", _ERROR_TEXT), {"must_include": ["saved"]}, 0),
]
NAVIGATION_CASES += [
    (
        f"absence-identity-{kind}-{status}",
        (200, "/page", "old"),
        (status, path, _ERROR_TEXT),
        {"exact_match": ""},
        expected,
    )
    for status in (404, 410)
    for kind, path, expected in (("unrelated", "/other-missing", 0), ("canonical", "/page/", 1))
]


@pytest.fixture
def navigation_site():
    """Serve redirects too: Playwright route handlers only intercept the first URL."""
    state = {"response": (200, "/page", "")}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, path, body = state["response"]
            redirect = self.path == "/page" and path != "/page"
            self.send_response(302 if redirect else status)
            if redirect:
                self.send_header("Location", path)
                payload = b""
            else:
                html = f'<main id="content">{body}</main>' if status == 200 else body or "Unavailable"
                payload = html.encode()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/page", state
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    "name,before,after,required,expected", NAVIGATION_CASES, ids=[case[0] for case in NAVIGATION_CASES]
)
def test_public_snapshot_navigation_with_real_browser(
    chromium, navigation_site, name, before, after, required, expected
):
    url, state = navigation_site
    target, plan = _target(
        "document.querySelector('#content').textContent", "saved", url=url, required_contents=required
    )
    config = {"eval": {"eval_types": ["program_html"], "program_html": [target]}}
    task = _webarena_task(original_metadata=config, task_kwargs={"collision_plan": plan})
    state["response"] = before
    context = chromium.new_context()
    evaluator = WebArenaTaskEvaluator(config=_config())
    try:
        page = context.new_page()
        browser_context = VisualBrowserEvaluationContext(page=page, browser_context=context, evidence=())

        def run():
            evaluator.prepare(task=task, observation=WebObservation(), browser_context=browser_context)
            state["response"] = after
            return evaluator.evaluate(
                task=task, observation=WebObservation(), final_answer="done", browser_context=browser_context
            )

        if expected is None:
            with pytest.raises(EvaluatorInfrastructureError):
                run()
        else:
            result = run()
            assert result.valid_sample and result.reward == expected
        assert context.pages == [page]
    finally:
        evaluator.close()
        context.close()


@pytest.mark.parametrize(
    "name,before,after,required,expected", NAVIGATION_CASES, ids=[case[0] for case in NAVIGATION_CASES]
)
@pytest.mark.asyncio
async def test_async_snapshot_navigation_with_real_browser(navigation_site, name, before, after, required, expected):
    async with async_playwright() as playwright:
        executable = playwright.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Real-browser regression tests require the qualified Chromium bundle")
        browser = await playwright.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        try:
            context = await browser.new_context()
            url, state = navigation_site
            target, plan = _target(
                "document.querySelector('#content').textContent", "saved", url=url, required_contents=required
            )
            config = {"eval": {"eval_types": ["program_html"], "program_html": [target]}}
            state["response"] = before
            page = await context.new_page()
            captured_before = await snapshots.collect_browser_snapshots(page, plan)
            state["response"] = after
            captured_after = await snapshots.collect_browser_snapshots(page, plan)
            if expected is None:
                with pytest.raises(EvaluatorInfrastructureError):
                    snapshots.build_snapshot_context(plan, captured_before, captured_after)
            else:
                evaluation_context = snapshots.build_snapshot_context(plan, captured_before, captured_after)
                score, _ = await classic.evaluate_classic_task(config, {}, page, eval_context=evaluation_context)
                assert score == expected
            assert context.pages == [page]
        finally:
            await browser.close()


IMAGE_DOCUMENT_CASES = [
    ("creation", (404, "/page"), (200, "/page"), 1),
    ("error-page", (200, "/page"), (404, "/page"), 0),
    ("other-page", (200, "/page"), (200, "/grid"), 0),
]


@pytest.mark.parametrize("name,before,after,expected", IMAGE_DOCUMENT_CASES)
def test_sync_image_document_ownership(chromium, navigation_site, name, before, after, expected):
    url, state = navigation_site
    query = {"eval_image_url": url, "eval_image_class": "#content"}
    plan = {
        "snapshot_adapters": {"page_image_query": {"queries": [{"key": snapshot_target_key(query), "query": query}]}}
    }
    image = '<main id="content"><img src="/artifact.png"></main>'
    context = chromium.new_context()
    try:
        page = context.new_page()
        state["response"] = (*before, image if name == "creation" else "")
        captured_before = snapshots.collect_browser_snapshots_sync(page, plan)
        state["response"] = (*after, image)
        captured_after = snapshots.collect_browser_snapshots_sync(page, plan)
        result = snapshots.build_snapshot_context(plan, captured_before, captured_after)
        records = result["deltas"]["page_image_query"]["added"] + result["deltas"]["page_image_query"]["changed"]
        assert sum(len(record["delta_image_urls"]) for record in records) == expected
    finally:
        context.close()


@pytest.mark.parametrize("name,before,after,expected", IMAGE_DOCUMENT_CASES)
@pytest.mark.asyncio
async def test_async_image_document_ownership(navigation_site, name, before, after, expected):
    async with async_playwright() as playwright:
        executable = playwright.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Real-browser regression tests require the qualified Chromium bundle")
        browser = await playwright.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        try:
            url, state = navigation_site
            query = {"eval_image_url": url, "eval_image_class": "#content"}
            plan = {
                "snapshot_adapters": {
                    "page_image_query": {"queries": [{"key": snapshot_target_key(query), "query": query}]}
                }
            }
            image = '<main id="content"><img src="/artifact.png"></main>'
            context = await browser.new_context()
            page = await context.new_page()
            state["response"] = (*before, image if name == "creation" else "")
            captured_before = await snapshots.collect_browser_snapshots(page, plan)
            state["response"] = (*after, image)
            captured_after = await snapshots.collect_browser_snapshots(page, plan)
            result = snapshots.build_snapshot_context(plan, captured_before, captured_after)
            records = result["deltas"]["page_image_query"]["added"] + result["deltas"]["page_image_query"]["changed"]
            assert sum(len(record["delta_image_urls"]) for record in records) == expected
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_async_final_dom_absence_match_and_invalid_scripts_in_real_browser():
    async with async_playwright() as playwright:
        executable = playwright.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Real-browser regression tests require the qualified Chromium bundle")
        browser = await playwright.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        try:
            page = await browser.new_page()
            for _, locator, html, expected in CASES:
                target = {"url": "last", "locator": locator, "required_contents": {"must_include": [expected]}}
                config = {"eval": {"eval_types": ["program_html"], "program_html": [target]}}
                await page.set_content('<main><span class="cover-status"></span></main>')
                assert (await classic.evaluate_classic_task(config, {}, page))[0] == 0
                await page.set_content(html)
                assert (await classic.evaluate_classic_task(config, {}, page))[0] == 1
            for locator in ["document.querySelector('[').outerText", "document.body.notAFunction()"]:
                target = {"url": "last", "locator": locator, "required_contents": {"must_include": ["saved"]}}
                config = {"eval": {"eval_types": ["program_html"], "program_html": [target]}}
                with pytest.raises(EvaluatorInfrastructureError):
                    await classic.evaluate_classic_task(config, {}, page)
        finally:
            await browser.close()
