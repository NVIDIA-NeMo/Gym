# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nemo_gym.web import (
    EvaluatorInfrastructureError,
    VisualBrowserEvaluationContext,
    WebObservation,
    build_collision_plan,
    snapshot_target_key,
)
from resources_servers.webarena_browser import reference_evaluation as reference
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.reference_evaluation import classic_evaluation as classic
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.tests.test_contract import _config, _webarena_task


ENTRYPOINTS = ["sync", "async", "public"]
URL = "https://example.test/shared"


def _config_for(target):
    return {"eval": {"eval_types": ["program_html"], "program_html": [target]}}


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


@pytest.fixture
def no_navigation_delay(monkeypatch):
    async def goto(page, url, **kwargs):
        page.url = url

    monkeypatch.setattr(classic, "goto", goto)
    monkeypatch.setattr(classic, "resolve_after_navigation_sync", lambda page: None)
    monkeypatch.setattr(classic.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr("time.sleep", lambda seconds: None)


async def _evaluate(entrypoint, target, page, plan, before, after, monkeypatch):
    config = _config_for(target)
    if entrypoint == "public":
        captures = iter([before, after])
        monkeypatch.setattr(reference, "collect_snapshots", lambda plan: next(captures))
        monkeypatch.setattr(reference, "collect_browser_snapshots_sync", lambda page, plan: {})
        task = _webarena_task(original_metadata=config, task_kwargs={"collision_plan": plan})
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
        score, _ = await classic.evaluate_classic_task(config, {"answer": "done"}, page, eval_context=context)
    else:
        score, _ = classic.evaluate_classic_task_sync(config, {"answer": "done"}, page, eval_context=context)
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
@pytest.mark.parametrize("field,matching", [("author", "saved"), ("rating", "100")])
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
    old = {"id": "1", "nickname": "saved", "rating_percent": "100"}
    new = {"id": "2", "nickname": "other", "rating_percent": "20"}
    if case == "new-match":
        new.update(nickname="saved", rating_percent="100")
    elif case == "missing-value":
        new = {"id": "2"}
    before = {"shopping_reviews": {sku: [old] if case in {"existing", "other-sku"} else []}}
    after = {"shopping_reviews": {sku: [old] if case in {"existing", "other-sku"} else [new]}}
    monkeypatch.setattr(classic, helper, Mock(return_value=matching))
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
