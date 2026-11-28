# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.web import WebTask, adapt_visualwebarena_records
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.reference_evaluation import visualwebarena_evaluation as vwa
from resources_servers.webarena_browser.tests.test_visualwebarena_exact_evidence import ENTRYPOINTS, _score
from resources_servers.webarena_browser.tests.test_visualwebarena_semantic_freshness import _freshness_judge_response


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("kind", ["fuzzy_match", "must_include"])
@pytest.mark.parametrize(
    "before,after,expected",
    [
        pytest.param("not", "not\npersisted", 0, id="retained-multiline-negation"),
        pytest.param("not", "not persisted", 0, id="retained-inline-negation"),
        pytest.param("not", "persisted", 1, id="removed-negation"),
        pytest.param("persisted", "persisted\npersisted", 1, id="new-matching-copy"),
        pytest.param("persisted\nother", "other\npersisted", 0, id="moved-matching-content"),
    ],
)
@pytest.mark.asyncio
async def test_semantic_truth_keeps_complete_observed_context(monkeypatch, entrypoint, kind, before, after, expected):
    # These are configured synthetic tasks, not canonical population records.
    target = {
        "url": "https://example.test/semantic-context",
        "locator": "document.body.innerText",
        "required_contents": {kind: ["saved"] if kind == "must_include" else "saved"},
    }
    records = [
        {
            "id": f"semantic-context-{index}",
            "ques": "Observe a new saved confirmation",
            "web": [target["url"]],
            "eval": {"eval_types": ["program_html"], "program_html": [target]},
        }
        for index in range(2)
    ]
    task = WebTask.model_validate(adapt_visualwebarena_records(records)[0]["web_task"])
    plan = task.task_kwargs["collision_plan"]
    specs = plan["snapshot_adapters"]["program_html"]["targets"]
    assert len(specs) == 1
    assert specs[0]["target"] == target

    def capture(plan, value, *, after):
        return {
            "program_html": [{"key": plan["snapshot_adapters"]["program_html"]["targets"][0]["key"], "value": value}]
        }

    context = snapshots.build_snapshot_context(
        plan, capture(plan, before, after=False), capture(plan, after, after=True)
    )
    changed = context["deltas"]["program_html"]["changed"]
    assert len(changed) == 1
    assert changed[0]["value"] == after
    if after == "not\npersisted":
        assert changed[0]["delta_value"] == "persisted"

    predictions = []

    def judge(**kwargs):
        predictions.append(kwargs["pred"])
        words = kwargs["pred"].split()
        return float("not" not in words and "persisted" in words)

    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
    monkeypatch.setattr(vwa, "_llm_fuzzy_match", judge)
    monkeypatch.setattr(vwa, "_llm_must_include", judge)
    monkeypatch.setattr(vwa, "_judge_chat", _freshness_judge_response)
    assert await _score(entrypoint, task, before, after, monkeypatch, capture) == expected
    assert after in predictions
