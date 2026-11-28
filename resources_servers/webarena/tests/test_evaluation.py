# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from resources_servers.webarena.eval_collision import (
    build_collision_plan,
    build_collision_plans,
    extract_helper_calls,
    has_collision_mitigation,
    snapshot_target_key,
)
from resources_servers.webarena.eval_snapshots import (
    build_snapshot_context,
    collect_browser_snapshots,
    collect_snapshots,
    diff_records,
    merge_snapshots,
)
from resources_servers.webarena.evaluation import (
    EvaluationConfig,
    _eval_helper_expression,
    _parse_judge_json_label,
    evaluate_classic_task,
    reddit_get_post_url,
)


class FakeContext:
    def __init__(self, pages=None, new_page_factory=None):
        self.pages = list(pages or [])
        self.new_page_factory = new_page_factory

    async def new_page(self):
        page = self.new_page_factory() if self.new_page_factory else FakePage("about:blank")
        page.context = self
        self.pages.append(page)
        return page


class FakePage:
    def __init__(self, url: str, *, live_url: str | None = None, content: str = "", values=None):
        self.url = url
        self.live_url = live_url or url
        self._content = content
        self.values = values or {}
        self.closed = False
        self.context = FakeContext([self])
        self.visited = []

    async def evaluate(self, expression, *args):
        if expression == "window.location.href":
            return self.live_url
        if expression in self.values:
            value = self.values[expression]
            return value(*args) if callable(value) else value
        if "Account" in expression:
            return 0
        if "col-max-role" in expression:
            return self.values.get("member_role", "")
        return ""

    async def content(self):
        return self._content

    async def goto(self, url, **_kwargs):
        self.url = str(url)
        self.live_url = str(url)
        self.visited.append(str(url))

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True


def task(eval_types, **eval_fields):
    return {
        "id": "test",
        "intent": "Do the task",
        "eval": {"eval_types": eval_types, **eval_fields},
    }


@pytest.mark.asyncio
async def test_exact_and_must_include_match_without_judge():
    page = FakePage("https://example.test")
    exact = task(["string_match"], reference_answers={"exact_match": "'Quest Lumaflex™ Band'"})
    score, message = await evaluate_classic_task(exact, {"answer": "quest lumaflex™ band"}, page)
    assert score == 1
    assert message == "string_match: score=1.0"

    must = task(
        ["string_match"],
        reference_answers={"must_include": ["Quest", ["Sprite Ball", "Cruise Ball"]]},
    )
    score, _ = await evaluate_classic_task(must, {"answer": "Quest and CRUISE BALL"}, page)
    assert score == 1


@pytest.mark.asyncio
async def test_exact_and_must_include_use_injected_semantic_fallback():
    calls = []

    async def judge(messages):
        calls.append(messages)
        return 'prefix {"judgment":"correct","reasoning":"equivalent"} suffix'

    page = FakePage("https://example.test")
    exact = task(["string_match"], reference_answers={"exact_match": ["New York City", "NYC"]})
    assert (await evaluate_classic_task(exact, {"answer": "the big apple"}, page, judge=judge))[0] == 1

    must = task(["string_match"], reference_answers={"must_include": ["red", ["blue", "navy"]]})
    assert (await evaluate_classic_task(must, {"answer": "crimson and azure"}, page, judge=judge))[0] == 1
    assert len(calls) == 2
    assert "one of: blue or navy" in calls[1][1]["content"]


@pytest.mark.asyncio
async def test_fuzzy_and_unachievable_judgements():
    responses = iter(
        [
            '{"judgement":"correct","reasoning":"same meaning"}',
            '{"judgement":"correct","reasoning":"same meaning"}',
            '```json\n{"judgement":"same","reasoning":"aligned"}\n```',
        ]
    )

    async def judge(_messages):
        return next(responses)

    page = FakePage("https://example.test")
    fuzzy = task(["string_match"], reference_answers={"fuzzy_match": ["first", "second"]})
    assert (await evaluate_classic_task(fuzzy, {"answer": "both"}, page, judge=judge))[0] == 1

    impossible = task(
        ["string_match"],
        reference_answers={"fuzzy_match": "N/A"},
        string_note="No airport is within the requested distance",
    )
    assert (await evaluate_classic_task(impossible, {"answer": "There is no nearby airport"}, page, judge=judge))[
        0
    ] == 1

    called = False

    async def should_not_run(_messages):
        nonlocal called
        called = True

    assert (await evaluate_classic_task(fuzzy, {"answer": ""}, page, judge=should_not_run))[0] == 0
    assert not called


def test_judge_parser_and_reddit_url_parity():
    assert _parse_judge_json_label('{"judgement":"correct"}', {"correct"}) == "correct"
    duplicated = (
        '{"judgement":"incorrect","reasoning":"wrong number"}\n'
        "</think>\n"
        '{"judgement":"incorrect","reasoning":"wrong number"}'
    )
    assert _parse_judge_json_label(duplicated, {"incorrect"}) == "incorrect"
    assert reddit_get_post_url("https://reddit.test/f/python/42/title") == "https://reddit.test/f/python/42/"
    assert reddit_get_post_url("https://reddit.test/settings") == "https://reddit.test/settings"
    with pytest.raises(RuntimeError, match="Unexpected judge"):
        _parse_judge_json_label("not json", {"correct"})


@pytest.mark.asyncio
async def test_url_match_checks_all_open_tabs_and_live_urls():
    first = FakePage("https://gitlab.test/old")
    second = FakePage(
        "https://gitlab.test/stale",
        live_url="https://gitlab.test/a11y/project/-/issues/?state=opened&sort=created_date_desc",
    )
    context = FakeContext([first, second])
    first.context = second.context = context
    config = task(
        ["url_match"],
        reference_url="https://gitlab.test/a11y/project/-/issues/?sort=created_date&state=open |OR| "
        "https://gitlab.test/a11y/project/-/issues/?sort=created_date&state=opened",
        url_note="GOLD in PRED",
    )
    score, message = await evaluate_classic_task(config, {}, first)
    assert score == 1
    assert "candidates=" in message
    assert "stale" not in message


@pytest.mark.asyncio
async def test_all_six_classic_helper_expressions_are_injectable():
    page = FakePage("https://reddit.test/f/python/42/title")

    async def async_value(*_args):
        return "async"

    helpers = {
        "shopping_get_latest_order_url": lambda: "order",
        "shopping_admin_get_cart_price_rule": lambda name: f"rule:{name}",
        "shopping_get_sku_latest_review_author": lambda sku: f"author:{sku}",
        "shopping_get_sku_latest_review_rating": lambda sku: f"rating:{sku}",
        "reddit_get_post_url": reddit_get_post_url,
        "gitlab_get_project_memeber_role": async_value,
    }
    runtime = EvaluationConfig(helpers=helpers, navigation_wait_seconds=0)
    expressions = {
        "func:shopping_get_latest_order_url()": "order",
        "func:shopping_admin_get_cart_price_rule('Summer')": "rule:Summer",
        "func:shopping_get_sku_latest_review_author('SKU')": "author:SKU",
        "func:shopping_get_sku_latest_review_rating('SKU')": "rating:SKU",
        "func:reddit_get_post_url('__last_url__')": "https://reddit.test/f/python/42/",
        "func:gitlab_get_project_memeber_role(__page__, 'alice')": "async",
    }
    for expression, expected in expressions.items():
        assert await _eval_helper_expression(expression, page, runtime) == expected


@pytest.mark.asyncio
async def test_program_html_last_checks_all_tabs():
    first = FakePage("https://gitlab.test/first")
    second = FakePage("https://gitlab.test/second")
    context = FakeContext([first, second])
    first.context = second.context = context

    async def member_role(page, _account):
        return "Guest" if page is second else "Reporter"

    config = task(
        ["program_html"],
        program_html=[
            {
                "url": "last",
                "locator": "func:gitlab_get_project_memeber_role(__page__, 'alice')",
                "required_contents": {"must_include": ["Guest"]},
            }
        ],
    )
    score, _ = await evaluate_classic_task(
        config,
        {},
        first,
        helpers={"gitlab_get_project_memeber_role": member_role},
    )
    assert score == 1


@pytest.mark.asyncio
async def test_program_html_snapshot_and_review_deltas_precede_final_state():
    page = FakePage("https://shopping.test")
    target = {
        "url": "https://shopping.test/account",
        "locator": 'document.querySelector("#reviews").outerText',
        "required_contents": {"must_include": ["new review"]},
    }
    snapshot_context = {
        "deltas": {
            "program_html": {
                "added": [],
                "changed": [
                    {
                        "key": snapshot_target_key(target),
                        "before_value": "old review",
                        "value": "old review\nnew review",
                        "delta_value": "new review",
                    }
                ],
            }
        }
    }
    config = task(["program_html"], program_html=[target])
    runtime = EvaluationConfig(navigation_wait_seconds=0)
    assert (await evaluate_classic_task(config, {}, page, config=runtime, eval_context=snapshot_context))[0] == 1
    assert not page.visited

    review_target = {
        "url": "last",
        "locator": "func:shopping_get_sku_latest_review_author('SKU')",
        "required_contents": {"exact_match": "Emma Lopez"},
    }
    review_context = {"deltas": {"shopping_reviews": {"SKU": {"added": [{"nickname": "Emma Lopez"}], "changed": []}}}}
    review_config = task(["program_html"], program_html=[review_target])
    assert (
        await evaluate_classic_task(
            review_config,
            {},
            page,
            config=runtime,
            eval_context=review_context,
        )
    )[0] == 1


@pytest.mark.asyncio
async def test_program_html_order_delta_uses_each_new_order_url():
    page = FakePage(
        "https://shopping.test",
        values={'() => document.querySelector("#items").outerText': lambda: ""},
    )

    async def evaluate(expression, *_args):
        if expression == "window.location.href":
            return page.url
        if "document.querySelector" in expression:
            return "SKU-OK" if page.url.endswith("/2") else "wrong"
        return ""

    page.evaluate = evaluate
    target = {
        "url": "func:shopping_get_latest_order_url()",
        "locator": 'document.querySelector("#items").outerText',
        "required_contents": {"must_include": ["SKU-OK"]},
    }
    context = {
        "deltas": {
            "shopping_orders": {
                "added": [{"url": "https://shopping.test/orders/1"}, {"url": "https://shopping.test/orders/2"}],
                "changed": [],
            }
        }
    }
    score, _ = await evaluate_classic_task(
        task(["program_html"], program_html=[target]),
        {},
        page,
        config=EvaluationConfig(navigation_wait_seconds=0),
        eval_context=context,
    )
    assert score == 1
    assert page.visited == ["https://shopping.test/orders/1", "https://shopping.test/orders/2"]


def test_collision_plans_match_mutable_target_rules():
    helper_target = {
        "url": "last",
        "locator": "func:shopping_get_sku_latest_review_rating('SKU')",
        "required_contents": {"must_include": ["100"]},
    }
    plan = build_collision_plan(task(["program_html"], program_html=[helper_target]))
    assert plan["snapshot_adapters"]["shopping_reviews"]["skus"] == ["SKU"]
    assert has_collision_mitigation(plan)
    assert extract_helper_calls(helper_target["locator"]) == [
        {"name": "shopping_get_sku_latest_review_rating", "args": ["SKU"]}
    ]

    fixed_target = {
        "url": "https://gitlab.test/team/repo/-/blob/main/README.md",
        "locator": 'document.querySelector("article").outerText',
        "required_contents": {"must_include": ["text"]},
    }
    tasks = [
        {"id": "one", "eval": {"program_html": [fixed_target]}},
        {"id": "two", "eval": {"program_html": [fixed_target]}},
    ]
    plans = build_collision_plans(tasks)
    assert all(plan["snapshot_adapters"]["program_html"]["targets"] for plan in plans)


def test_snapshot_deltas_preserve_added_changed_removed_and_added_text():
    before = {
        "shopping_orders": [{"increment_id": "1", "status": "old"}],
        "shopping_reviews": {"SKU": [{"id": 1, "nickname": "Old"}]},
        "program_html": [{"key": "target", "value": "old"}],
        "page_image_query": [{"key": "image", "image_urls": ["old.png"]}],
    }
    after = {
        "shopping_orders": [
            {"increment_id": "1", "status": "new"},
            {"increment_id": "2", "status": "new"},
        ],
        "shopping_reviews": {"SKU": [{"id": 2, "nickname": "New"}]},
        "program_html": [{"key": "target", "value": "old\nnew"}],
        "page_image_query": [{"key": "image", "image_urls": ["old.png", "new.png"]}],
    }
    context = build_snapshot_context({"snapshot_adapters": {}}, before, after)
    worker_context = build_snapshot_context(before, after, {"snapshot_adapters": {}})
    assert worker_context["deltas"] == context["deltas"]
    assert context["deltas"]["shopping_orders"]["added"][0]["increment_id"] == "2"
    assert context["deltas"]["shopping_orders"]["changed"][0]["status"] == "new"
    assert context["deltas"]["shopping_reviews"]["SKU"]["removed"][0]["nickname"] == "Old"
    assert context["deltas"]["program_html"]["changed"][0]["delta_value"] == "new"
    assert context["deltas"]["page_image_query"]["changed"][0]["delta_image_urls"] == ["new.png"]
    assert merge_snapshots({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert diff_records([{"id": 1}], [], key_fields=("id",))["removed"][0]["id"] == 1


def test_snapshot_api_collection_supports_sync_browser_worker():
    snapshots = collect_snapshots(
        {"snapshot_adapters": {"shopping_orders": {}}},
        config=EvaluationConfig(),
        fetch_orders=lambda: [{"increment_id": "1"}],
    )
    assert snapshots == {"shopping_orders": [{"increment_id": "1"}]}


@pytest.mark.asyncio
async def test_snapshot_collectors_accept_fake_api_and_browser_dependencies():
    plan = {
        "snapshot_adapters": {
            "shopping_orders": {},
            "shopping_reviews": {"skus": ["SKU"]},
        }
    }

    async def orders():
        return [{"increment_id": "1"}]

    async def reviews(skus):
        return {skus[0]: [{"id": 1}]}

    snapshots = await collect_snapshots(
        plan,
        config=EvaluationConfig(),
        fetch_orders=orders,
        fetch_reviews=reviews,
    )
    assert snapshots == {
        "shopping_orders": [{"increment_id": "1"}],
        "shopping_reviews": {"SKU": [{"id": 1}]},
    }

    target = {
        "url": "https://example.test/result",
        "locator": 'document.querySelector("#result").outerText',
        "required_contents": {"must_include": ["done"]},
    }
    snapshot_page = FakePage(
        "about:blank",
        values={'() => document.querySelector("#result").outerText': "done"},
    )
    parent = FakePage("https://example.test")
    parent.context = FakeContext([parent], new_page_factory=lambda: snapshot_page)
    browser_plan = {
        "snapshot_adapters": {"program_html": {"targets": [{"key": snapshot_target_key(target), "target": target}]}}
    }
    browser = await collect_browser_snapshots(
        parent,
        browser_plan,
        config=EvaluationConfig(navigation_wait_seconds=0),
    )
    assert browser["program_html"][0]["value"] == "done"
    assert snapshot_page.closed


def test_example_dataset_contains_exact_requested_tasks():
    path = Path(__file__).parents[1] / "data" / "example.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["verifier_metadata"]["task"]["task_id"] for record in records] == [
        "webarena-0",
        "webarena-3",
        "webarena-8",
        "webarena-45",
        "webarena-585",
    ]
    assert all(record["responses_create_params"]["parallel_tool_calls"] is False for record in records)
    assert all(record["verifier_metadata"]["task"]["intent"] for record in records)
