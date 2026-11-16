# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from PIL import Image

from nemo_gym.web import (
    EvaluatorConfigurationError,
    EvaluatorInfrastructureError,
    VisualBrowserEvaluationContext,
    WebBenchmark,
    WebObservation,
    build_collision_plan,
    snapshot_target_key,
)
from resources_servers.webarena_browser import reference_evaluation as reference
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.reference_evaluation import visualwebarena_evaluation as classic
from resources_servers.webarena_browser.tests.test_contract import _config, _visualwebarena_task, _webarena_task
from resources_servers.webarena_browser.tests.test_snapshot_browser import chromium  # noqa: F401


ENTRYPOINTS = ["sync", "async", "public"]
URL = "https://example.test/shared"


def _config_for(target):
    return {"intent": "Synthetic test", "eval": {"eval_types": ["program_html"], "program_html": [target]}}


def _page(value="saved", *, asynchronous=False, failure=None):
    page = SimpleNamespace(url=URL, is_closed=Mock(return_value=False))

    def read(*args, **kwargs):
        if failure:
            raise failure
        # The account-index query precedes the member-role extraction.
        if "span.gl-avatar-labeled-sublabel" in args[0]:
            return 0
        return value

    method = AsyncMock if asynchronous else Mock
    page.evaluate = method(side_effect=read)
    page.content = method(side_effect=lambda: read("content"))
    page.goto = method(side_effect=lambda url, **kwargs: setattr(page, "url", url))
    page.context = SimpleNamespace(pages=[page])
    return page


@pytest.fixture(autouse=True)
def offline_judge(monkeypatch):
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
    monkeypatch.setattr(classic, "_llm_must_include", lambda **kwargs: 0.0)


@pytest.fixture
def no_navigation_delay(monkeypatch):
    async def goto(page, url, **kwargs):
        page.url = url

    monkeypatch.setattr(classic, "goto", goto)
    monkeypatch.setattr(classic.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr("time.sleep", lambda seconds: None)


async def _evaluate(entrypoint, target, page, plan, before, after, monkeypatch):
    config = _config_for(target)
    if entrypoint == "public":
        captures = iter([before, after])
        monkeypatch.setattr(reference, "collect_snapshots", lambda plan: next(captures))
        monkeypatch.setattr(reference, "collect_browser_snapshots_sync", lambda page, plan: {})
        task = _visualwebarena_task(original_metadata=config, task_kwargs={"collision_plan": plan})
        context = VisualBrowserEvaluationContext(page=page, browser_context=page.context, evidence=())
        evaluator = WebArenaTaskEvaluator(config=_config())
        evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
        result = evaluator.evaluate(
            task=task, observation=WebObservation(), final_answer="done", browser_context=context
        )
        assert result.valid_sample
        assert result.raw_score == result.reward
        assert result.task_success == (result.reward == 1)
        return result.reward
    context = snapshots.build_snapshot_context(plan, before, after)
    if entrypoint == "async":
        score, _ = await classic.evaluate_visualwebarena_task(config, {"answer": "done"}, page, eval_context=context)
    else:
        score, _ = classic.evaluate_visualwebarena_task_sync(config, {"answer": "done"}, page, eval_context=context)
    if _.startswith("VisualWebArena evaluation error:"):
        assert score == 0
        raise EvaluatorInfrastructureError(_)
    return score


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "before_value,after_value,expected",
    [
        ("saved", "saved", 0),
        ("saved", "saved unrelated", 0),
        ("&#115;aved", "saved unrelated", 0),
        ("saved", "saved\nunrelated", 0),
        ("saved", "saved\nsaved", 1),
        ("", "saved", 1),
        ("", "other", 0),
    ],
)
@pytest.mark.asyncio
async def test_planned_program_html_delta_is_authoritative(
    monkeypatch, no_navigation_delay, entrypoint, before_value, after_value, expected
):
    target = {"url": URL, "locator": "document.body.innerText", "required_contents": {"must_include": ["saved"]}}
    plan = build_collision_plan(_config_for(target), program_html_collision_keys={f"url:{target['url']}"})
    assert len(plan["snapshot_adapters"]["program_html"]["targets"]) == 1
    key = snapshot_target_key(target)
    before = {"program_html": [{"key": key, "value": before_value}]}
    after = {"program_html": [{"key": key, "value": after_value}]}
    page = _page(after_value, asynchronous=entrypoint == "async")
    assert await _evaluate(entrypoint, target, page, plan, before, after, monkeypatch) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("unrelated_adapter", [False, True])
@pytest.mark.asyncio
async def test_unplanned_target_retains_final_state_scoring(
    monkeypatch, no_navigation_delay, entrypoint, unrelated_adapter
):
    target = {"url": URL, "locator": "document.body.innerText", "required_contents": {"must_include": ["saved"]}}
    plan, before, after = {}, {}, {}
    if unrelated_adapter:
        other = {**target, "url": URL + "/other"}
        plan = build_collision_plan(_config_for(other), program_html_collision_keys={f"url:{other['url']}"})
        before = after = {"program_html": [{"key": snapshot_target_key(other), "value": "saved"}]}
    page = _page(asynchronous=entrypoint == "async")
    assert await _evaluate(entrypoint, target, page, plan, before, after, monkeypatch) == 1


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("case,expected", [("existing", 0), ("new-match", 1), ("new-miss", 0), ("unplanned", 1)])
@pytest.mark.asyncio
async def test_order_capture_never_falls_back_to_an_existing_order(
    monkeypatch, no_navigation_delay, entrypoint, case, expected
):
    target = {
        "url": "func:shopping_get_latest_order_url()",
        "locator": "document.body.innerText",
        "required_contents": {"must_include": ["saved"]},
    }
    old = {"increment_id": "1", "url": URL + "/order/1"}
    new = {"increment_id": "2", "url": URL + "/order/2"}
    plan = build_collision_plan(_config_for(target)) if case != "unplanned" else {}
    before = {"shopping_orders": [old]} if plan else {}
    after = {"shopping_orders": [old, new] if case.startswith("new-") else [old]} if plan else {}
    monkeypatch.setattr(classic, "shopping_get_latest_order_url", Mock(return_value=old["url"]))
    page = _page(asynchronous=entrypoint == "async")

    def content(*args, **kwargs):
        return "other" if case == "new-miss" and page.url == new["url"] else "saved"

    page.evaluate.side_effect = content
    assert await _evaluate(entrypoint, target, page, plan, before, after, monkeypatch) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("field,matching", [("author", "saved"), ("rating", "100"), ("text", "saved")])
@pytest.mark.parametrize(
    "case,expected", [("existing", 0), ("new-match", 1), ("new-miss", 0), ("missing-value", 0), ("other-sku", 1)]
)
@pytest.mark.asyncio
async def test_review_capture_is_authoritative_for_its_sku(
    monkeypatch, no_navigation_delay, entrypoint, field, matching, case, expected
):
    helper = "shopping_get_sku_latest_review_" + field
    target = {"url": "last", "locator": f"func:{helper}('sku')", "required_contents": {"exact_match": matching}}
    plan = build_collision_plan(_config_for(target))
    sku = "other" if case == "other-sku" else "sku"
    plan["snapshot_adapters"]["shopping_reviews"]["skus"] = [sku]
    old = {"id": "1", "nickname": "saved", "rating_percent": "100", "detail": "saved"}
    new = {"id": "2", "nickname": "other", "rating_percent": "20", "detail": "other"}
    if case == "new-match":
        new.update(nickname="saved", rating_percent="100", detail="saved")
    elif case == "missing-value":
        new = {"id": "2"}
    before = {"shopping_reviews": {sku: [old] if case in {"existing", "other-sku"} else []}}
    after = {"shopping_reviews": {sku: [old] if case in {"existing", "other-sku"} else [new]}}
    monkeypatch.setattr(classic, "_" + helper if field == "text" else helper, Mock(return_value=matching))
    page = _page(asynchronous=entrypoint == "async")
    assert await _evaluate(entrypoint, target, page, plan, before, after, monkeypatch) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "locator",
    [
        "document.body.innerText",
        "document.querySelector('.optional').outerText",
        "func:gitlab_get_project_memeber_role(__page__, 'member')",
        "",
    ],
)
@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (["error"], None),
        (["error", "error"], None),
        (["error", "miss"], None),
        (["miss", "error"], None),
        (["miss", "miss"], 0),
        (["error", "match"], 1),
        (["match", "error"], 1),
        (["absent"], 0),
    ],
)
@pytest.mark.asyncio
async def test_final_dom_evidence_preserves_candidate_errors(monkeypatch, entrypoint, locator, outcomes, expected):
    target = {"url": "last", "locator": locator, "required_contents": {"must_include": ["saved"]}}
    pages = [
        _page(
            "saved" if outcome == "match" else "" if outcome == "absent" else "other",
            asynchronous=entrypoint == "async",
            failure=ConnectionError("DOM transport disconnected") if outcome == "error" else None,
        )
        for outcome in outcomes
    ]
    page = pages[0]
    page.context.pages = pages
    if expected is None:
        with pytest.raises(EvaluatorInfrastructureError):
            await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch)
    else:
        assert await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch) == expected
    assert all(not candidate.is_closed() for candidate in pages)
    assert all(candidate.content.called if not locator else candidate.evaluate.called for candidate in pages)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("prep_action", [None, "invalid evaluator script"])
@pytest.mark.asyncio
async def test_invalid_final_dom_scripts_are_inconclusive(monkeypatch, entrypoint, prep_action):
    target = {
        "url": "last",
        "locator": "document.querySelector('[').outerText",
        "required_contents": {"must_include": ["saved"]},
    }
    page = _page(asynchronous=entrypoint == "async", failure=ValueError("invalid CSS selector"))
    if prep_action is not None:
        target["locator"] = "document.body.innerText"
        target["prep_actions"] = [prep_action]
        page.evaluate.side_effect = [ValueError("invalid evaluator script"), "saved"]
    with pytest.raises(EvaluatorInfrastructureError):
        await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "value,inequality,expected", [(0, ">= 0", 1), (0, "> 1", 0), (2, "> 1", 1), ([], ">= 0", 0), (None, ">= 0", 0)]
)
@pytest.mark.asyncio
async def test_numeric_zero_is_observed(monkeypatch, entrypoint, value, inequality, expected):
    target = {"url": "last", "locator": "lambda:() => 0", "required_contents": {"required_values": [inequality]}}
    page = _page(value, asynchronous=entrypoint == "async")
    assert await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "outcomes,expected",
    [
        ([1, 0], 1),
        ([0, 1], 1),
        ([0, 0], 0),
        ([1], 1),
        ([0], 0),
        (["error", 1], 1),
        ([1, "error"], 1),
        (["error", 0], None),
    ],
)
@pytest.mark.asyncio
async def test_dom_fuzzy_alternatives_are_or(monkeypatch, entrypoint, outcomes, expected):
    results = iter(outcomes)

    def judge(**kwargs):
        value = next(results)
        if value == "error":
            raise TimeoutError("Judge unavailable")
        return value

    monkeypatch.setattr(classic, "_llm_fuzzy_match", judge)
    target = {
        "url": "last",
        "locator": "document.body.innerText",
        "required_contents": {"fuzzy_match": " |OR| ".join(str(i) for i in range(len(outcomes)))},
    }
    page = _page(asynchronous=entrypoint == "async")
    if expected is None:
        with pytest.raises(EvaluatorInfrastructureError):
            await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch)
    else:
        assert await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch) == expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "answer,excluded,judge_score,expected",
    [
        ("blocked reason", "blocked", 1, 0),
        ("valid reason", "blocked", 1, 1),
        ("N/A", "blocked", 0, 1),
        ("", "blocked", 0, 0),
        ("wrong", "blocked", 0, 0),
    ],
)
def test_unachievable_answer_preserves_independent_predicates(
    monkeypatch, reverse, answer, excluded, judge_score, expected
):
    judge = Mock(return_value=judge_score)
    monkeypatch.setattr(classic, "_llm_ua_match", judge)
    predicates = [("must_exclude", [excluded]), ("fuzzy_match", "N/A")]
    if reverse:
        predicates.reverse()
    config = {"intent": "Synthetic test", "eval": {"reference_answers": dict(predicates)}}
    assert classic._string_match(config, answer) == expected
    if answer == "N/A":
        judge.assert_not_called()


@pytest.mark.parametrize("benchmark", [WebBenchmark.WEBARENA, WebBenchmark.VISUALWEBARENA])
@pytest.mark.parametrize(
    "predicate,needs_judge",
    [
        ("must_include", True),
        ("fuzzy_match", True),
        ("exact_match", False),
        ("must_exclude", False),
        ("required_values", False),
    ],
)
def test_dom_judge_is_checked_before_capture(monkeypatch, benchmark, predicate, needs_judge):
    monkeypatch.delenv("WEBARENA_JUDGE_API_KEY", raising=False)
    captures = Mock(return_value={})
    monkeypatch.setattr(reference, "collect_snapshots", captures)
    monkeypatch.setattr(reference, "collect_browser_snapshots_sync", Mock(return_value={}))
    value = (
        "saved"
        if predicate in {"exact_match", "fuzzy_match"}
        else [">= 0"]
        if predicate == "required_values"
        else ["saved"]
    )
    target = {"url": "last", "locator": "document.body.innerText", "required_contents": {predicate: value}}
    task_factory = _visualwebarena_task if benchmark == WebBenchmark.VISUALWEBARENA else _webarena_task
    task = task_factory(original_metadata=_config_for(target), task_kwargs={"collision_plan": {}})
    page = _page()
    context = VisualBrowserEvaluationContext(page=page, browser_context=page.context, evidence=())
    evaluator = WebArenaTaskEvaluator(config=_config())
    if benchmark == WebBenchmark.VISUALWEBARENA and needs_judge:
        with pytest.raises(EvaluatorConfigurationError, match="WEBARENA_JUDGE_API_KEY"):
            evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
        captures.assert_not_called()
    else:
        evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
        captures.assert_called_once()


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("failure", [True, False])
@pytest.mark.asyncio
async def test_exact_empty_distinguishes_disconnection(monkeypatch, entrypoint, failure):
    target = {"url": "last", "locator": "document.body.innerText", "required_contents": {"exact_match": ""}}
    page = _page("", asynchronous=entrypoint == "async", failure=ConnectionError("Disconnected") if failure else None)
    if failure:
        with pytest.raises(EvaluatorInfrastructureError):
            await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch)
    else:
        assert await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch) == 1


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (["error"], None),
        (["error", "miss"], None),
        (["miss", "error"], None),
        (["miss", "miss"], 0),
        (["error", "match"], 1),
        (["match", "error"], 1),
    ],
)
@pytest.mark.asyncio
async def test_judge_errors_across_tabs(monkeypatch, entrypoint, outcomes, expected):
    def judge(**kwargs):
        if kwargs["pred"] == "error":
            raise TimeoutError("Judge timed out")
        return float(kwargs["pred"] == "match")

    monkeypatch.setattr(classic, "_llm_fuzzy_match", judge)
    target = {"url": "last", "locator": "document.body.innerText", "required_contents": {"fuzzy_match": "accepted"}}
    pages = [_page(outcome, asynchronous=entrypoint == "async") for outcome in outcomes]
    pages[0].context.pages = pages
    if expected is None:
        with pytest.raises(EvaluatorInfrastructureError):
            await _evaluate(entrypoint, target, pages[0], {}, {}, {}, monkeypatch)
    else:
        assert await _evaluate(entrypoint, target, pages[0], {}, {}, {}, monkeypatch) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (["timeout"], None),
        (["browser"], None),
        (["503"], None),
        (["401"], None),
        (["timeout", "miss"], None),
        (["miss", "timeout"], None),
        (["timeout", "match"], 1),
        (["match", "timeout"], 1),
        (["miss"], 0),
        (["match"], 1),
        (["404"], 0),
        (["410"], 0),
        (["corrupt"], 0),
        ([], 0),
    ],
)
@pytest.mark.asyncio
async def test_image_evidence_errors_require_a_conclusive_match(
    monkeypatch, tmp_path, entrypoint, planned, outcomes, expected
):
    asynchronous = entrypoint == "async"
    red, blue = Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")
    ref = tmp_path / "reference.png"
    red.save(ref)

    def fetch(url, **kwargs):
        outcome = url.rsplit("/", 1)[-1].removesuffix(".png")
        request = httpx.Request("GET", url)
        if outcome == "timeout":
            raise httpx.ReadTimeout("Image timed out", request=request)
        if outcome in {"404", "410", "503", "401"}:
            return httpx.Response(int(outcome), request=request)
        content = io.BytesIO()
        (red if outcome == "match" else blue).save(content, format="PNG")
        return httpx.Response(
            200, content=b"not an image" if outcome == "corrupt" else content.getvalue(), request=request
        )

    monkeypatch.setattr(classic.httpx, "get", fetch)
    query = {"eval_image_url": "last", "eval_image_class": "", "eval_fuzzy_image_match": str(ref)}
    method = AsyncMock if asynchronous else Mock
    elements = []
    for outcome in outcomes:
        element = SimpleNamespace(
            get_attribute=method(
                side_effect=ConnectionError("Browser disconnected")
                if outcome == "browser"
                else lambda name, o=outcome: f"https://example.test/{o}.png" if name == "src" else None
            )
        )
        elements.append(element)
    page = _page(asynchronous=asynchronous)
    page.query_selector_all = method(return_value=elements)
    context = None
    if planned:
        if "browser" in outcomes:
            # Captured URLs have no live element read; test its transport error.
            outcomes = ["timeout" if item == "browser" else item for item in outcomes]
        key = snapshot_target_key(query)
        context = {
            "collision_plan": {"snapshot_adapters": {"page_image_query": {"queries": [{"key": key}]}}},
            "deltas": {
                "page_image_query": {
                    "added": [{"key": key, "image_urls": [f"https://example.test/{o}.png" for o in outcomes]}]
                }
            },
        }
    config = {"eval": {"eval_types": ["page_image_query"], "page_image_query": [query]}}
    if entrypoint == "public":
        task = _visualwebarena_task(original_metadata=config, task_kwargs={"collision_plan": {}})
        evaluator = WebArenaTaskEvaluator(config=_config(task_image_root=str(tmp_path)))
        browser_context = VisualBrowserEvaluationContext(page=page, browser_context=page.context, evidence=())
        monkeypatch.setattr(reference, "collect_snapshots", Mock(return_value={}))
        monkeypatch.setattr(reference, "collect_browser_snapshots_sync", Mock(return_value={}))
        # Snapshot deltas were exercised above; inject the already validated
        # captured context only at the native invocation for this adapter check.
        if planned:
            monkeypatch.setattr(reference, "build_snapshot_context", lambda *args: context)
        evaluator.prepare(task=task, observation=WebObservation(), browser_context=browser_context)
        if expected is None:
            with pytest.raises(EvaluatorInfrastructureError):
                evaluator.evaluate(
                    task=task, observation=WebObservation(), final_answer="done", browser_context=browser_context
                )
            return
        result = evaluator.evaluate(
            task=task, observation=WebObservation(), final_answer="done", browser_context=browser_context
        )
        assert result.valid_sample and result.reward == expected
        return
    if asynchronous:
        score, message = await classic.evaluate_visualwebarena_task(config, {}, page, eval_context=context)
    else:
        score, message = classic.evaluate_visualwebarena_task_sync(config, {}, page, eval_context=context)
    if expected is None:
        assert score == 0
        assert message.startswith("VisualWebArena evaluation error:")
    else:
        assert not message.startswith("VisualWebArena evaluation error:")
        assert score == expected


@pytest.mark.parametrize(
    "helper,argument,expected",
    [
        ("get_query_text", "'.absent'", ""),
        ("shopping_get_product_attributes", "'absent'", ""),
        ("shopping_get_product_price", "", 0),
        ("shopping_get_num_reviews", "", 0),
        ("shopping_get_rating_as_percentage", "", 0),
        ("shopping_get_order_product_name_list", "", ""),
        ("reddit_get_latest_comment_content_by_username", "'absent'", ""),
        ("gitlab_get_project_memeber_role", "'absent'", ""),
    ],
)
def test_dom_helpers_read_real_empty_pages(chromium, helper, argument, expected):
    page = chromium.new_page()
    try:
        page.set_content("<html><body>Empty task state</body></html>")
        expr = f"func:{helper}(__page__{', ' + argument if argument else ''})"
        assert classic._eval_helper_expression_sync(expr, page) == expected
    finally:
        page.close()


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.asyncio
async def test_dom_must_include_retains_visual_semantic_fallback(monkeypatch, entrypoint):
    judge = Mock(return_value=1)
    monkeypatch.setattr(classic, "_llm_must_include", judge)
    target = {"url": "last", "locator": "document.body.innerText", "required_contents": {"must_include": ["saved"]}}
    page = _page("successfully persisted", asynchronous=entrypoint == "async")
    assert await _evaluate(entrypoint, target, page, {}, {}, {}, monkeypatch) == 1
    judge.assert_called_once()
    assert judge.call_args.kwargs["pred"] == "successfully persisted"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["targets", "eval-types"])
@pytest.mark.asyncio
async def test_independent_dom_requirements_still_use_and(monkeypatch, asynchronous, kind):
    monkeypatch.setattr(classic, "_llm_fuzzy_match", lambda **kwargs: float(kwargs["reference"] == "saved"))
    target = {"url": "last", "locator": "document.body.innerText", "required_contents": {"fuzzy_match": "saved"}}
    config = _config_for(target)
    if kind == "targets":
        config["eval"]["program_html"].append({**target, "required_contents": {"fuzzy_match": "other"}})
    else:
        config["eval"]["eval_types"].append("string_match")
        config["eval"]["reference_answers"] = {"required_values": ["> 1"]}
    page = _page(asynchronous=asynchronous)
    if asynchronous:
        score, message = await classic.evaluate_visualwebarena_task(config, {"answer": "0"}, page)
    else:
        score, message = classic.evaluate_visualwebarena_task_sync(config, {"answer": "0"}, page)
    assert not message.startswith("VisualWebArena evaluation error:")
    assert score == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "helper,args",
    [
        ("get_query_text", "'.optional'"),
        ("shopping_get_order_product_name_list", ""),
        ("shopping_get_product_attributes", "'price'"),
        ("shopping_get_product_price", ""),
        ("shopping_get_num_reviews", ""),
        ("shopping_get_rating_as_percentage", ""),
        ("reddit_get_latest_comment_content_by_username", "'member'"),
        ("reddit_get_parent_comment_username_of_latest_comment_by_username", "'member'"),
        ("gitlab_get_project_memeber_role", "'member'"),
    ],
)
@pytest.mark.asyncio
async def test_dom_helpers_preserve_transport_errors(asynchronous, helper, args):
    page = _page(asynchronous=asynchronous, failure=ConnectionError("Browser transport disconnected"))
    expr = f"func:{helper}(__page__{', ' + args if args else ''})"
    with pytest.raises(RuntimeError, match="Browser transport disconnected"):
        if asynchronous:
            await classic._eval_helper_expression(expr, page)
        else:
            classic._eval_helper_expression_sync(expr, page)
