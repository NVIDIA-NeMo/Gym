# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nemo_gym.web import EvaluatorInfrastructureError, WebTask, adapt_visualwebarena_records, build_collision_plan
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.tests.test_visualwebarena_exact_evidence import ENTRYPOINTS, _score


URL = "http://snapshot.test/item?item=first&item=second&mode=edit"
REORDERED_KEYS = "http://snapshot.test/item?mode=edit&item=first&item=second"
REVERSED_VALUES = "http://snapshot.test/item?mode=edit&item=second&item=first"


def _task():
    target = {"url": URL, "locator": "document.body.innerText", "required_contents": {"exact_match": "saved"}}
    records = [
        {
            "id": f"query-order-{i}",
            "ques": "Save the first selected item",
            "web": [URL],
            "eval": {"eval_types": ["program_html"], "program_html": [target]},
        }
        for i in range(2)
    ]
    return WebTask.model_validate(adapt_visualwebarena_records(records)[0]["web_task"])


@pytest.mark.parametrize(
    "landed,expected",
    [
        (REORDERED_KEYS, True),
        (REVERSED_VALUES, False),
        ("http://snapshot.test/item?item=first&mode=edit&item=second", True),
        (URL, True),
        ("http://snapshot.test/item?item=first&mode=edit", False),
        ("http://snapshot.test/item?item=first&item=changed&mode=edit", False),
        ("http://snapshot.test/item?item=first&item=second&mode=", False),
        (URL + "&flag=", False),
    ],
)
def test_query_identity_preserves_each_keys_value_order(landed, expected):
    assert snapshots._snapshot_url_matches(URL, landed) is expected


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter", ["program_html", "page_image_query"])
@pytest.mark.parametrize("reversed_values", [False, True])
@pytest.mark.asyncio
async def test_collectors_and_capture_phases_preserve_query_identity(
    monkeypatch, asynchronous, adapter, reversed_values
):
    if adapter == "program_html":
        plan = _task().task_kwargs["collision_plan"]
        field = "value"
    else:
        plan = build_collision_plan(
            {"eval": {"page_image_query": [{"eval_image_url": URL, "eval_image_class": "img"}]}}
        )
        field = "image_urls"
    assert adapter in plan["snapshot_adapters"]
    method = AsyncMock if asynchronous else Mock
    pages = []
    for index, landed in enumerate([URL, REVERSED_VALUES if reversed_values else REORDERED_KEYS]):
        value = ("old", "saved")[index] if field == "value" else [f"http://snapshot.test/{index}.png"]

        def read(expression, *args, value=value):
            return False if expression == snapshots._SNAPSHOT_LOGIN_FORM else value

        pages.append(SimpleNamespace(url=landed, evaluate=method(side_effect=read), close=method()))
    response = SimpleNamespace(status=200)
    if asynchronous:
        monkeypatch.setattr(snapshots, "goto", AsyncMock(return_value=response))
        root = SimpleNamespace(context=SimpleNamespace(new_page=AsyncMock(side_effect=pages)))
        before = await snapshots.collect_browser_snapshots(root, plan)
        after = await snapshots.collect_browser_snapshots(root, plan)
        for page in pages:
            page.close.assert_awaited_once()
    else:
        for page in pages:
            page.goto = Mock(return_value=response)
        root = SimpleNamespace(context=SimpleNamespace(new_page=Mock(side_effect=pages)))
        before = snapshots.collect_browser_snapshots_sync(root, plan)
        after = snapshots.collect_browser_snapshots_sync(root, plan)
        for page in pages:
            page.close.assert_called_once()
    record = after[adapter][0]
    assert not record.get("error")
    assert record["target_document"] is (not reversed_values)
    snapshots.build_snapshot_context(plan, before, after)
    if reversed_values:
        assert record[field] == ("" if field == "value" else [])
        pages[1].evaluate.assert_called_once_with(snapshots._SNAPSHOT_LOGIN_FORM)
        with pytest.raises(EvaluatorInfrastructureError, match="baseline"):
            snapshots.build_snapshot_context(plan, after, after)
    else:
        assert record[field]
        snapshots.build_snapshot_context(plan, after, after)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("reversed_values", [False, True])
@pytest.mark.asyncio
async def test_retained_capture_cannot_credit_another_query_resource(monkeypatch, entrypoint, reversed_values):
    task = _task()

    def capture(plan, value, *, after):
        target = plan["snapshot_adapters"]["program_html"]["targets"][0]
        return {
            "program_html": [
                {
                    "key": target["key"],
                    "url": URL,
                    "value": value,
                    "navigation": {
                        "status": 200,
                        "url": (REVERSED_VALUES if reversed_values else REORDERED_KEYS) if after else URL,
                    },
                }
            ]
        }

    # Retained captures can contain the wrong page's actual text. The real
    # context producer must bind that evidence to its navigation identity.
    assert await _score(entrypoint, task, "old", "saved", monkeypatch, capture) == (0 if reversed_values else 1)
