# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Canonical task metadata: https://github.com/jayl940712/webarena_benchmarks/blob/6a2977939b157b0ab9de7799bb089c721f1ac115/visualwebarena.jsonl
# Revision: 6a2977939b157b0ab9de7799bb089c721f1ac115; license: Apache-2.0.
# Original rows are retained in fixtures/visualwebarena_review_tasks.json.

import copy
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlparse

import pytest
from PIL import Image
from playwright.async_api import async_playwright

from nemo_gym.web import VisualBrowserEvaluationContext, WebObservation, build_collision_plan, snapshot_target_key
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.reference_evaluation import visualwebarena_evaluation as vwa
from resources_servers.webarena_browser.tests.test_contract import _config, _visualwebarena_task
from resources_servers.webarena_browser.tests.test_snapshot_browser import chromium  # noqa: F401
from resources_servers.webarena_browser.tests.test_visualwebarena_evidence import ENTRYPOINTS, _evaluate, _page


TASKS = {
    row["runtime_task_id"]: row["source"]
    for row in json.loads((Path(__file__).parent / "fixtures/visualwebarena_review_tasks.json").read_text())
}
TRANSITIONS = [
    ("unchanged", ["red", "blue"], ["red", "blue"], 0),
    ("new-second-match", ["blue"], ["blue", "red"], 1),
    ("new-second-nonmatch", ["blue"], ["blue", "green"], 0),
    ("delete-only", ["red", "blue"], ["red"], 0),
    ("reorder-only", ["red", "blue"], ["blue", "red"], 0),
]
SELECTORS = [".product-image-photo", ".products-grid .wishlist .product-image-photo", ".card"]


@pytest.fixture(autouse=True)
def credentials_and_no_navigation_delay(monkeypatch):
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
    monkeypatch.setattr(vwa.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "task_id,text,expected",
    [
        (task_id, text, expected)
        for task_id, good in [("622", "aww"), ("623", "Newark nyc")]
        for text, expected in [
            (good, 1),
            ("", 0),
            *[
                (good + " " + forbidden, 0)
                for forbidden in TASKS[task_id]["eval"]["program_html"][0]["required_contents"]["must_exclude"]
            ],
        ]
    ],
)
@pytest.mark.asyncio
async def test_canonical_combined_predicates(monkeypatch, entrypoint, reverse, task_id, text, expected):
    assert TASKS[task_id]["id"] == "visualwebarena-" + str(int(task_id) + 2)
    target = copy.deepcopy(TASKS[task_id]["eval"]["program_html"][0])
    target["url"] = "last"
    if reverse:
        target["required_contents"] = dict(reversed(target["required_contents"].items()))
    from resources_servers.webarena_browser.reference_evaluation import classic_evaluation

    judge = Mock(side_effect=AssertionError("Canonical predicate controls must not call the judge"))
    monkeypatch.setattr(classic_evaluation, "_judge_chat", judge)
    assert (
        await _evaluate(entrypoint, target, _page(text, asynchronous=entrypoint == "async"), {}, {}, {}, monkeypatch)
        == expected
    )
    judge.assert_not_called()


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "before,after,expected",
    [("AskReddit", "AskReddit\naww", 0), ("AskReddit", "aww", 1), ("aww", "aww\nAskReddit", 0)],
)
@pytest.mark.asyncio
async def test_planned_exclusions_apply_to_the_whole_final_target(monkeypatch, entrypoint, before, after, expected):
    target = copy.deepcopy(TASKS["622"]["eval"]["program_html"][0])
    target.update(url="http://example.test/sidebar", locator="document.body.innerText")
    plan = build_collision_plan(
        {"eval": {"program_html": [target]}}, program_html_collision_keys={"url:" + target["url"]}
    )
    key = snapshot_target_key(target)
    monkeypatch.setattr(vwa, "_llm_must_include", lambda **kwargs: 0.0)
    assert (
        await _evaluate(
            entrypoint,
            target,
            _page(after, asynchronous=entrypoint == "async"),
            plan,
            {"program_html": [{"key": key, "value": before}]},
            {"program_html": [{"key": key, "value": after}]},
            monkeypatch,
        )
        == expected
    )


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("kind", ["must_include", "fuzzy_match"])
@pytest.mark.parametrize(
    "before,after,expected",
    [
        ("saved", "saved\nsuccessfully persisted", 1),
        ("saved", "saved unrelated", 0),
        ("saved\nother", "other\nsaved", 0),
        ("saved", "saved\nsaved", 1),
        ("&#115;aved", "saved unrelated", 0),
    ],
)
@pytest.mark.asyncio
async def test_added_semantic_matches_remain_scoreable(monkeypatch, entrypoint, kind, before, after, expected):
    def semantic_judge(**kwargs):
        return float(any(word in kwargs["pred"].split() for word in ("saved", "persisted")))

    monkeypatch.setattr(vwa, "_llm_must_include", semantic_judge)
    monkeypatch.setattr(vwa, "_llm_fuzzy_match", semantic_judge)
    target = {
        "url": "http://example.test/semantic",
        "locator": "document.body.innerText",
        "required_contents": {kind: ["saved"] if kind == "must_include" else "saved"},
    }
    plan = build_collision_plan(
        {"eval": {"program_html": [target]}}, program_html_collision_keys={"url:" + target["url"]}
    )
    key = snapshot_target_key(target)
    assert (
        await _evaluate(
            entrypoint,
            target,
            _page(after, asynchronous=entrypoint == "async"),
            plan,
            {"program_html": [{"key": key, "value": before}]},
            {"program_html": [{"key": key, "value": after}]},
            monkeypatch,
        )
        == expected
    )


@pytest.fixture
def image_site():
    state = {"colors": [], "selector": ""}
    images = {}
    for color in ("red", "blue", "green"):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        images[color] = buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            self.send_response(200)
            if path.endswith(".png"):
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(images[path.rsplit("/", 1)[-1].removesuffix(".png")])
                return
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            tags = [f'<img class="product-image-photo" src="/{color}.png">' for color in state["colors"]]
            if state["selector"] == ".card":
                body = "".join(f'<div class="card"><div class="card">{tag}</div></div>' for tag in tags)
            else:
                body = '<div class="products-grid wishlist">' + "".join(tags) + "</div>"
            self.wfile.write(("<html><body>" + body + "</body></html>").encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _image_task(base, selector):
    task_id = "108" if "wishlist" in selector else "286"
    query = copy.deepcopy(TASKS[task_id]["eval"]["page_image_query"][0])
    query.update(eval_image_url=base + "/target", eval_image_class=selector)
    config = {
        "intent": TASKS[task_id]["ques"],
        "eval": {"eval_types": ["page_image_query"], "page_image_query": [query]},
    }
    return task_id, query, config, build_collision_plan(config)


def _assert_complete_urls(capture, base, colors):
    assert capture["page_image_query"][0]["image_urls"] == [base + f"/{color}.png" for color in colors]


@pytest.fixture
def deterministic_vqa_boundary(monkeypatch):
    def judge(*, images, **kwargs):
        return float(any(image.getpixel((0, 0)) == (255, 0, 0) for image in images))

    monkeypatch.setattr(vwa, "_vlm_vqa_score", judge)


@pytest.mark.parametrize("entrypoint", ["sync", "public"])
@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("name,before,after,expected", TRANSITIONS, ids=[x[0] for x in TRANSITIONS])
def test_real_dom_image_capture_and_delta_scoring(
    chromium, image_site, deterministic_vqa_boundary, entrypoint, selector, name, before, after, expected
):
    base, state = image_site
    state.update(colors=before, selector=selector)
    task_id, query, config, plan = _image_task(base, selector)
    context = chromium.new_context()
    page = context.new_page()
    page.goto(base + "/target")
    browser = VisualBrowserEvaluationContext(page=page, browser_context=context, evidence=())
    task = _visualwebarena_task(task_id=task_id, original_metadata=config, task_kwargs={"collision_plan": plan})
    evaluator = WebArenaTaskEvaluator(config=_config())
    try:
        if entrypoint == "public":
            evaluator.prepare(task=task, observation=WebObservation(), browser_context=browser)
            capture_before = evaluator._active._before
        else:
            capture_before = snapshots.collect_browser_snapshots_sync(page, plan)
        _assert_complete_urls(capture_before, base, before)
        state["colors"] = after
        if entrypoint == "public":
            result = evaluator.evaluate(
                task=task, observation=WebObservation(), final_answer="done", browser_context=browser
            )
            assert result.valid_sample and result.reward == expected
        else:
            capture_after = snapshots.collect_browser_snapshots_sync(page, plan)
            _assert_complete_urls(capture_after, base, after)
            delta = snapshots.build_snapshot_context(plan, capture_before, capture_after)
            if name in {"delete-only", "reorder-only"}:
                assert delta["deltas"]["page_image_query"]["changed"][0]["delta_image_urls"] == []
            score, message = vwa.evaluate_visualwebarena_task_sync(config, {}, page, eval_context=delta)
            assert not message.startswith("VisualWebArena evaluation error:")
            assert score == expected
    finally:
        evaluator.close()
        context.close()


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("name,before,after,expected", TRANSITIONS, ids=[x[0] for x in TRANSITIONS])
@pytest.mark.asyncio
async def test_async_real_dom_image_capture_and_delta_scoring(
    image_site, deterministic_vqa_boundary, selector, name, before, after, expected
):
    base, state = image_site
    state.update(colors=before, selector=selector)
    task_id, query, config, plan = _image_task(base, selector)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=playwright.chromium.executable_path, headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(base + "/target")
            capture_before = await snapshots.collect_browser_snapshots(page, plan)
            _assert_complete_urls(capture_before, base, before)
            state["colors"] = after
            capture_after = await snapshots.collect_browser_snapshots(page, plan)
            _assert_complete_urls(capture_after, base, after)
            delta = snapshots.build_snapshot_context(plan, capture_before, capture_after)
            score, message = await vwa.evaluate_visualwebarena_task(config, {}, page, eval_context=delta)
            assert not message.startswith("VisualWebArena evaluation error:")
            assert score == expected
        finally:
            await browser.close()


def test_missing_image_delta_retains_legacy_capture_compatibility():
    query = {"eval_image_url": "http://example.test", "eval_image_class": ".image"}
    key = snapshot_target_key(query)
    context = {
        "collision_plan": {"snapshot_adapters": {"page_image_query": {"queries": [{"key": key}]}}},
        "deltas": {"page_image_query": {"added": [{"key": key, "image_urls": ["http://example.test/new.png"]}]}},
    }
    assert vwa._page_image_snapshot_urls(query, context) == ["http://example.test/new.png"]
    context["deltas"]["page_image_query"]["added"][0]["delta_image_urls"] = []
    assert vwa._page_image_snapshot_urls(query, context) == []
