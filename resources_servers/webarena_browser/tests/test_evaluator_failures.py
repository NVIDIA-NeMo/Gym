# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nemo_gym.web import (
    EvaluatorInfrastructureError,
    VisualBrowserEvaluationContext,
    WebObservation,
    WebTask,
    snapshot_target_key,
)
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator, _reference_task_config
from resources_servers.webarena_browser.reference_evaluation import classic_evaluation as classic
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.tests.test_contract import _config, _evaluation_context, _webarena_task


@pytest.mark.parametrize(
    "failure", [TimeoutError("judge timeout"), ConnectionError("judge disconnected"), ValueError("bad verdict")]
)
def test_public_evaluator_propagates_judge_failure(monkeypatch, failure) -> None:
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")  # pragma: allowlist secret

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(classic, "_judge_chat", fail)
    evaluator = WebArenaTaskEvaluator(config=_config())
    task = _webarena_task()
    evaluator.prepare(task=task, observation=WebObservation(), browser_context=_evaluation_context())
    with pytest.raises(EvaluatorInfrastructureError):
        evaluator.evaluate(
            task=task,
            observation=WebObservation(),
            final_answer="a different answer",
            browser_context=_evaluation_context(),
        )
    # An ordinary unanswered task is still a valid scored failure.
    result = evaluator.evaluate(
        task=task, observation=WebObservation(), final_answer="", browser_context=_evaluation_context()
    )
    assert result.valid_sample and result.reward == 0


@pytest.mark.asyncio
async def test_async_evaluator_propagates_judge_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise TimeoutError("judge timeout")

    monkeypatch.setattr(classic, "_judge_chat", fail)
    with pytest.raises(EvaluatorInfrastructureError):
        await classic.evaluate_classic_task(_webarena_task().original_metadata, {"answer": "different"}, object())


@pytest.mark.parametrize(
    "adapter,empty,nonempty",
    [
        ("shopping_orders", [], [{"increment_id": "1"}]),
        ("shopping_reviews", {"sku": []}, {"sku": [{"id": "1"}]}),
        ("program_html", [{"key": "target", "value": ""}], [{"key": "target", "value": "existing content"}]),
        (
            "page_image_query",
            [{"key": "target", "image_urls": []}],
            [{"key": "target", "image_urls": ["https://example.test/img"]}],
        ),
    ],
)
@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("failure", ["missing", "malformed", "error"])
def test_snapshot_context_rejects_invalid_required_capture(adapter, empty, nonempty, phase, failure) -> None:
    params = {"skus": ["sku"], "targets": [{"key": "target"}], "queries": [{"key": "target"}]}
    plan = {"snapshot_adapters": {adapter: params}}
    before, after = {adapter: copy.deepcopy(empty)}, {adapter: copy.deepcopy(nonempty)}
    broken = before if phase == "before" else after
    if failure == "missing":
        broken.clear()
    elif failure == "malformed":
        broken[adapter] = None
    elif adapter in ("program_html", "page_image_query"):
        broken[adapter][0]["error"] = True
    else:
        broken[adapter] = {"error": True}
    with pytest.raises(EvaluatorInfrastructureError):
        snapshots.build_snapshot_context(plan, before, after)
    # A genuine empty baseline remains valid and can produce a real delta.
    context = snapshots.build_snapshot_context(plan, {adapter: empty}, {adapter: nonempty})
    assert context["deltas"][adapter]


def test_api_capture_timeout_is_not_an_empty_baseline(monkeypatch) -> None:
    def fail():
        raise TimeoutError("orders unavailable")

    monkeypatch.setattr(snapshots, "fetch_shopping_orders", fail)
    with pytest.raises(EvaluatorInfrastructureError):
        snapshots.collect_snapshots({"snapshot_adapters": {"shopping_orders": {}}})


@pytest.mark.parametrize("payload", [{}, {"items": None}, {"items": {}}, {"items": [None]}])
def test_malformed_order_response_is_not_an_empty_capture(monkeypatch, payload) -> None:
    monkeypatch.setattr(snapshots, "_shopping_headers", lambda: {})
    monkeypatch.setattr(
        snapshots.httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload),
    )
    with pytest.raises(EvaluatorInfrastructureError):
        snapshots.collect_snapshots({"snapshot_adapters": {"shopping_orders": {}}})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_dom_capture_disconnect_cannot_create_positive_delta(monkeypatch, asynchronous) -> None:
    target = {
        "url": "https://example.test/page",
        "locator": "document.body.innerText",
        "required_contents": {"must_include": ["existing"]},
    }
    spec = {"key": snapshot_target_key(target), "target": target}
    plan = {"snapshot_adapters": {"program_html": {"targets": [spec]}}}
    if asynchronous:
        page = SimpleNamespace(
            url=target["url"], evaluate=AsyncMock(side_effect=ConnectionError("disconnected")), close=AsyncMock()
        )
        root = SimpleNamespace(context=SimpleNamespace(new_page=AsyncMock(return_value=page)))
        monkeypatch.setattr(snapshots, "goto", AsyncMock(return_value=SimpleNamespace(status=200)))
        before = await snapshots.collect_browser_snapshots(root, plan)
    else:

        def fail(*args, **kwargs):
            raise ConnectionError("disconnected")

        page = SimpleNamespace(
            url=target["url"],
            evaluate=Mock(side_effect=fail),
            goto=Mock(return_value=SimpleNamespace(status=200)),
            close=Mock(),
        )
        root = SimpleNamespace(context=SimpleNamespace(new_page=lambda: page))
        before = snapshots.collect_browser_snapshots_sync(root, plan)
        page.goto.assert_called_once()
        page.evaluate.assert_called_once()
    after = {"program_html": [{"key": spec["key"], "value": "existing"}]}
    with pytest.raises(EvaluatorInfrastructureError):
        snapshots.build_snapshot_context(plan, before, after)


def test_deployment_substitution_rekeys_prepared_collision_targets(monkeypatch) -> None:
    monkeypatch.setenv("WA_SHOPPING", "https://deployed.example")
    target = {
        "url": "__SHOPPING__/reviews",
        "locator": "document.body.innerText",
        "required_contents": {"must_include": ["new review"]},
    }
    original_key = snapshot_target_key(target)
    original_plan = {
        "snapshot_adapters": {"program_html": {"targets": [{"key": original_key, "target": target}]}},
        "site_locks": ["shopping"],
    }
    task = _webarena_task(
        original_metadata={"eval": {"eval_types": ["program_html"], "program_html": [target]}},
        task_kwargs={"collision_plan": original_plan},
    )
    config, plan = _reference_task_config(task, _config())
    resolved = config["eval"]["program_html"][0]
    spec = plan["snapshot_adapters"]["program_html"]["targets"][0]
    assert spec["key"] == snapshot_target_key(resolved)
    assert spec["key"] != original_key
    assert original_plan["snapshot_adapters"]["program_html"]["targets"][0]["key"] == original_key
    context = snapshots.build_snapshot_context(
        plan,
        {"program_html": [{"key": spec["key"], "value": ""}]},
        {"program_html": [{"key": spec["key"], "value": "new review"}]},
    )
    matched = classic._program_html_snapshot_records(resolved, context)
    assert classic._score_program_html_snapshot_records(resolved["required_contents"], matched) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (["error"], None),
        (["error", "error"], None),
        (["error", "miss"], None),
        (["miss", "error"], None),
        (["miss", "miss"], 0.0),
        (["error", "match"], 1.0),
        (["match", "error"], 1.0),
    ],
)
@pytest.mark.asyncio
async def test_last_cart_rule_candidate_service_failures(monkeypatch, asynchronous, outcomes, expected):
    import json
    from pathlib import Path

    task = json.loads((Path(__file__).parent / "fixtures/task699.json").read_text())["web_task"]
    config = task["original_metadata"]
    target = config["eval"]["program_html"][0]
    assert target["url"] == "last" and "shopping_admin_get_cart_price_rule" in target["locator"]
    monkeypatch.setenv("WA_SHOPPING_ADMIN", "https://admin.example.test")
    monkeypatch.setattr(classic, "_shopping_admin_get_auth_token", lambda: "test-token")
    remaining = iter(outcomes)

    def get(*args, **kwargs):
        outcome = next(remaining)
        if outcome == "error":
            raise TimeoutError("Magento GET timeout")
        name = kwargs["params"]["searchCriteria[filter_groups][0][filters][0][value]"]
        item = {"name": name, "customer_group_ids": [0, 1, 2, 3], "simple_action": "by_percent", "discount_amount": 10}
        # Preserve real HTTP/helper execution; vary the final predicate independently of canonical rule values.
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"items": [item]}, outcome=outcome)

    # The helper's serialized rule is checked by the real scorer. Use a distinctive discount for each outcome.
    def response_get(*args, **kwargs):
        response = get(*args, **kwargs)
        payload = response.json()
        payload["items"][0]["discount_amount"] = 42 if response.outcome == "match" else 17
        response.json = lambda: payload
        return response

    monkeypatch.setattr(classic.httpx, "get", response_get)
    target["required_contents"] = {"must_include": ['"discount_amount": "42"']}
    pages = [SimpleNamespace(url="https://admin.example.test/cart") for _ in outcomes]
    page = SimpleNamespace(context=SimpleNamespace(pages=pages))

    async def evaluate():
        if asynchronous:
            return await classic.evaluate_classic_task(config, {"answer": "done"}, page)
        public_task = WebTask.model_validate(task)
        evaluator = WebArenaTaskEvaluator(config=_config())
        context = VisualBrowserEvaluationContext(page=page, browser_context=page.context, evidence=())
        evaluator.prepare(task=public_task, observation=WebObservation(), browser_context=context)
        result = evaluator.evaluate(
            task=public_task, observation=WebObservation(), final_answer="done", browser_context=context
        )
        assert result.valid_sample
        return result.reward, ""

    if expected is None:
        with pytest.raises(EvaluatorInfrastructureError):
            await evaluate()
    else:
        result = await evaluate()
        assert result[0] == expected
