# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Canonical source: https://github.com/jayl940712/webarena_benchmarks/blob/6a2977939b157b0ab9de7799bb089c721f1ac115/visualwebarena.jsonl
# Revision: 6a2977939b157b0ab9de7799bb089c721f1ac115; license: Apache-2.0.

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nemo_gym.web import (
    VisualBrowserEvaluationContext,
    WebObservation,
    WebTask,
    adapt_visualwebarena_records,
    snapshot_target_key,
)
from resources_servers.webarena_browser import reference_evaluation as reference
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.reference_evaluation import visualwebarena_evaluation as vwa
from resources_servers.webarena_browser.tests.test_contract import _config


TASKS = {
    x["runtime_task_id"]: x["source"]
    for x in yaml.safe_load((Path(__file__).parent / "fixtures/visualwebarena_exact_tasks.yaml").read_text())
}
ENTRYPOINTS = ["sync", "async", "public"]
EXPECTED = "It has 3 $%@"


def _capture(plan, comment, *, after):
    records = []
    for spec in plan["snapshot_adapters"]["program_html"]["targets"]:
        value = "liverblow" if after else "other-parent"
        if "parent_comment" not in spec["target"]["locator"]:
            value = comment
        records.append({"key": spec["key"], "value": value})
    return {"program_html": records}


async def _score(entrypoint, task, before, after, monkeypatch, capture):
    plan = task.task_kwargs["collision_plan"]
    page = SimpleNamespace(context=SimpleNamespace(pages=[]))
    if entrypoint == "public":
        phases = iter([(before, False), (after, True)])
        monkeypatch.setenv("WA_REDDIT", "https://reddit.test")
        monkeypatch.setattr(reference, "collect_snapshots", lambda plan: {})

        def collect(page, resolved_plan):
            value, changed = next(phases)
            return capture(resolved_plan, value, after=changed)

        monkeypatch.setattr(reference, "collect_browser_snapshots_sync", collect)
        evaluator = WebArenaTaskEvaluator(config=_config())
        context = VisualBrowserEvaluationContext(page=page, browser_context=page.context, evidence=())
        evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
        result = evaluator.evaluate(
            task=task, observation=WebObservation(), final_answer="done", browser_context=context
        )
        assert result.valid_sample
        return result.reward
    context = snapshots.build_snapshot_context(
        plan, capture(plan, before, after=False), capture(plan, after, after=True)
    )
    if entrypoint == "async":
        score, message = await vwa.evaluate_visualwebarena_task(task.original_metadata, {}, page, eval_context=context)
    else:
        score, message = vwa.evaluate_visualwebarena_task_sync(task.original_metadata, {}, page, eval_context=context)
    assert not message.startswith("VisualWebArena evaluation error:")
    return score


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "before,after,expected",
    [
        ("unrelated", EXPECTED, 1),
        ("unrelated", EXPECTED + " unrelated", 0),
        ("unrelated", "unrelated " + EXPECTED, 0),
        ("old", EXPECTED + " newly invalid", 0),
        (EXPECTED, EXPECTED, 0),
        (EXPECTED, EXPECTED + " suffix", 0),
        ("It has 4 $%@", EXPECTED, 1),
        ("unrelated " + EXPECTED, EXPECTED, 1),
        ("unrelated", EXPECTED + "\n" + EXPECTED, 0),
        ("unrelated", "  " + EXPECTED.lower() + "  ", 1),
    ],
)
@pytest.mark.asyncio
async def test_canonical_568_strictly_grades_the_observed_comment(monkeypatch, entrypoint, before, after, expected):
    # Normal preparation with an actual canonical peer plans both targets;
    # no collision keys or handcrafted snapshot plan are injected.
    prepared = adapt_visualwebarena_records([TASKS["568"], TASKS["569"]])
    task = WebTask.model_validate(prepared[0]["web_task"])
    assert task.original_metadata["id"] == "visualwebarena-568"
    specs = task.task_kwargs["collision_plan"]["snapshot_adapters"]["program_html"]["targets"]
    assert len(specs) == 2
    assert specs[1]["target"]["required_contents"] == {"exact_match": EXPECTED}
    assert await _score(entrypoint, task, before, after, monkeypatch, _capture) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "before,after,expected",
    [("old", "old 3", 0), ("3", "3 old", 0), ("old", "3", 1), ("3", "3.0", 0), ("3", "4", 0), ("2", "3", 1)],
)
@pytest.mark.asyncio
async def test_numeric_constraints_grade_the_observed_value(monkeypatch, entrypoint, before, after, expected):
    target = {
        "url": "https://example.test/number",
        "locator": "document.body.innerText",
        "required_contents": {"required_values": ["== 3"]},
    }
    plan = {
        "snapshot_adapters": {"program_html": {"targets": [{"key": snapshot_target_key(target), "target": target}]}}
    }
    task = WebTask(
        benchmark="visualwebarena",
        task_id="numeric-control",
        intent="Observe exactly three",
        original_metadata={"eval": {"eval_types": ["program_html"], "program_html": [target]}},
        task_kwargs={"collision_plan": plan},
    )

    def capture(plan, value, *, after):
        return {
            "program_html": [{"key": plan["snapshot_adapters"]["program_html"]["targets"][0]["key"], "value": value}]
        }

    assert await _score(entrypoint, task, before, after, monkeypatch, capture) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "before,after,expected",
    [
        ("not", "not persisted", 0),
        ("not", "persisted", 1),
        ("saved", "saved\npersisted", 1),
        ("saved", "saved unrelated", 0),
    ],
)
@pytest.mark.asyncio
async def test_semantic_freshness_cannot_replace_observed_text(monkeypatch, entrypoint, before, after, expected):
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")

    def judge(**kwargs):
        words = kwargs["pred"].split()
        return float("not" not in words and any(word in words for word in ("saved", "persisted")))

    monkeypatch.setattr(vwa, "_llm_fuzzy_match", judge)
    target = {
        "url": "https://example.test/semantic",
        "locator": "document.body.innerText",
        "required_contents": {"fuzzy_match": "saved"},
    }
    plan = {
        "snapshot_adapters": {"program_html": {"targets": [{"key": snapshot_target_key(target), "target": target}]}}
    }
    task = WebTask(
        benchmark="visualwebarena",
        task_id="semantic-control",
        intent="Observe saved content",
        original_metadata={"eval": {"eval_types": ["program_html"], "program_html": [target]}},
        task_kwargs={"collision_plan": plan},
    )

    def capture(plan, value, *, after):
        return {
            "program_html": [{"key": plan["snapshot_adapters"]["program_html"]["targets"][0]["key"], "value": value}]
        }

    assert await _score(entrypoint, task, before, after, monkeypatch, capture) == expected
