# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import html
import json
import re
from types import SimpleNamespace

import pytest

from nemo_gym.web import (
    EvaluatorInfrastructureError,
    VisualBrowserEvaluationContext,
    WebObservation,
    WebTask,
    adapt_visualwebarena_records,
)
from resources_servers.webarena_browser import reference_evaluation as reference
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.reference_evaluation import eval_snapshots as snapshots
from resources_servers.webarena_browser.reference_evaluation import visualwebarena_evaluation as vwa
from resources_servers.webarena_browser.tests.test_contract import _config
from resources_servers.webarena_browser.tests.test_visualwebarena_exact_evidence import ENTRYPOINTS, _score


def _saved_claims(text):
    """Small independent semantic world for the controlled judge boundary."""
    count = 0
    negated = False
    for word in re.findall(r"\w+", html.unescape(text).casefold()):
        if word in {"not", "never"}:
            negated = True
        elif word in {"saved", "persisted"}:
            count += not negated
            negated = False
        elif word not in {"yet", "successfully"}:
            negated = False
    return count


def _freshness_judge_response(messages):
    observation = json.loads(messages[-1]["content"].split("Observed target JSON:\n", 1)[1])
    correct = _saved_claims(observation["after"]) > _saved_claims(observation["before"])
    return json.dumps({"judgement": "correct" if correct else "incorrect", "reasoning": "Compare observed claims."})


def _task(kind):
    target = {
        "url": "https://example.test/semantic-freshness",
        "locator": "document.body.innerText",
        "required_contents": {kind: ["saved"] if kind == "must_include" else "saved"},
    }
    records = [
        {
            "id": f"semantic-freshness-{index}",
            "ques": "Observe a new saved confirmation",
            "web": [target["url"]],
            "eval": {"eval_types": ["program_html"], "program_html": [target]},
        }
        for index in range(2)
    ]
    return WebTask.model_validate(adapt_visualwebarena_records(records)[0]["web_task"])


def _capture(plan, value, *, after):
    spec = plan["snapshot_adapters"]["program_html"]["targets"][0]
    return {"program_html": [{"key": spec["key"], "value": value}]}


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("kind", ["must_include", "fuzzy_match"])
@pytest.mark.parametrize(
    "before,after,expected",
    [
        pytest.param("saved\nnot", "saved\nnot\npersisted", 0, id="retained-multiline-negation"),
        pytest.param("saved not", "saved not persisted", 0, id="retained-inline-negation"),
        pytest.param("saved\nnever", "saved\nnever\npersisted", 0, id="retained-never"),
        pytest.param("saved\nnot yet", "saved\nnot yet\npersisted", 0, id="retained-modified-negation"),
        pytest.param("saved", "saved\nnot persisted", 0, id="new-complete-negative"),
        pytest.param("saved\nnot persisted", "saved\npersisted", 1, id="inline-negation-removed"),
        pytest.param("saved\nnot\npersisted", "saved\npersisted", 1, id="negation-line-removed"),
        pytest.param("saved\nnot", "saved\npersisted", 1, id="negation-replaced"),
        pytest.param("saved", "saved\nsuccessfully persisted", 1, id="new-paraphrase"),
        pytest.param("saved", "saved\nsaved", 1, id="new-literal-copy"),
        pytest.param("saved\nother", "other\nsaved", 0, id="moved-literal"),
        pytest.param("saved\npersisted", "persisted\nsaved", 0, id="moved-semantic-copy"),
        pytest.param("saved\npersisted", "saved\npersisted\npersisted", 1, id="new-semantic-copy"),
        pytest.param("saved", "saved unrelated", 0, id="unrelated-append"),
        pytest.param("saved\nnot", "saved\nnot", 0, id="unchanged"),
        pytest.param(
            "saved\nnot\npersisted",
            "saved\nnot\npersisted\nsuccessfully persisted",
            1,
            id="new-positive-alongside-old-negative",
        ),
        pytest.param("saved\npersisted", "saved\npersisted\nnot persisted", 0, id="new-negative-after-paraphrase"),
        pytest.param("saved\nnot", "saved\nnot\nsuccessfully persisted", 0, id="completed-negative-paraphrase"),
        pytest.param("&#115;aved\nnot", "saved\nnot\npersisted", 0, id="entity-encoded-old-match"),
        pytest.param("saved\nother", "other\nsaved\nnote", 0, id="moved-match-with-new-noise"),
        pytest.param("saved", "Saved", 0, id="case-only-change"),
        pytest.param("saved\nother", "saved", 0, id="unrelated-removal"),
    ],
)
@pytest.mark.asyncio
async def test_semantic_freshness_preserves_mixed_context(monkeypatch, entrypoint, kind, before, after, expected):
    task = _task(kind)
    plan = task.task_kwargs["collision_plan"]
    assert len(plan["snapshot_adapters"]["program_html"]["targets"]) == 1
    context = snapshots.build_snapshot_context(
        plan, _capture(plan, before, after=False), _capture(plan, after, after=True)
    )
    changed = context["deltas"]["program_html"]["changed"]
    if before != after:
        assert len(changed) == 1 and changed[0]["value"] == after and changed[0]["before_value"] == before
    if before == "saved\nnot" and after == "saved\nnot\npersisted":
        assert changed[0]["delta_value"] == "persisted"
    # A target can contain an old affirmative and a newly negated claim. Its
    # truth boundary must not be used to infer freshness of that new claim.
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
    monkeypatch.setattr(vwa, "_llm_fuzzy_match", lambda **kw: float(_saved_claims(kw["pred"]) > 0))
    monkeypatch.setattr(vwa, "_llm_must_include", lambda **kw: float(_saved_claims(kw["pred"]) > 0))
    monkeypatch.setattr(vwa, "_judge_chat", _freshness_judge_response)
    assert await _score(entrypoint, task, before, after, monkeypatch, _capture) == expected


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("kind,expected", [("must_include", 1), ("fuzzy_match", 0)])
@pytest.mark.parametrize(
    "before,after",
    [("saved", "saved\nnot saved"), ("saved\nnot", "saved\nnot\nsaved")],
)
@pytest.mark.asyncio
async def test_literal_count_preserves_reference_semantics_while_fuzzy_checks_negation(
    monkeypatch, entrypoint, kind, expected, before, after
):
    task = _task(kind)
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
    monkeypatch.setattr(vwa, "_llm_fuzzy_match", lambda **kw: float(_saved_claims(kw["pred"]) > 0))
    monkeypatch.setattr(vwa, "_judge_chat", _freshness_judge_response)
    assert await _score(entrypoint, task, before, after, monkeypatch, _capture) == expected


@pytest.mark.parametrize("label,expected", [("correct", 1), ("incorrect", 0), ("partially correct", 0)])
def test_freshness_judge_receives_complete_observations_and_logs_them(monkeypatch, tmp_path, label, expected):
    before = "&#115;aved\nnot\n" + "retained context\n" * 1000
    after = before + "persisted"
    required = {"must_include": ["saved |OR| stored", "ready"], "fuzzy_match": "saved |OR| persisted"}
    observed = {"requirements": required, "before": html.unescape(before), "after": html.unescape(after)}
    requests = []
    response = json.dumps({"judgement": label, "reasoning": "Test label handling."})

    def judge(messages):
        requests.append(messages)
        assert json.loads(messages[-1]["content"].split("Observed target JSON:\n", 1)[1]) == observed
        return response

    monkeypatch.setattr(vwa, "_judge_chat", judge)
    log = tmp_path / "judge.jsonl"
    assert vwa._llm_program_html_fresh_match(required, before, after, log) == expected
    record = json.loads(log.read_text())
    assert record["judge_type"] == "visualwebarena_program_html_freshness"
    assert json.loads(record["prediction"]) == observed
    assert json.loads(record["reference"]) == required
    assert record["messages"] == requests[0] and record["response"] == response
    assert len(requests) == 1


@pytest.mark.parametrize("response", ["not JSON", "{}", '{"judgement":"unknown"}'])
def test_freshness_judge_malformed_result_is_inconclusive(monkeypatch, response):
    monkeypatch.setattr(vwa, "_judge_chat", lambda messages: response)
    with pytest.raises(RuntimeError):
        vwa._score_program_html_snapshot_records(
            {"must_include": ["saved"]},
            [{"before_value": "saved\nnot", "value": "saved\nnot\npersisted", "delta_value": "persisted"}],
            None,
        )


def test_freshness_judge_transport_error_is_inconclusive(monkeypatch):
    def unavailable(messages):
        raise RuntimeError("Synthetic judge unavailable")

    monkeypatch.setattr(vwa, "_judge_chat", unavailable)
    with pytest.raises(RuntimeError, match="Synthetic judge unavailable"):
        vwa._score_program_html_snapshot_records(
            {"must_include": ["saved"]}, [{"before_value": "saved", "value": "saved\npersisted"}], None
        )


@pytest.mark.parametrize("response", [None, "not JSON", "{}", '{"judgement":"unknown"}'])
def test_public_freshness_judge_failure_does_not_become_valid_task_failure(monkeypatch, response):
    task = _task("must_include")
    phases = iter([("saved\nnot", False), ("saved\nnot\npersisted", True)])
    page = SimpleNamespace(context=SimpleNamespace(pages=[]))

    def collect(page, plan):
        value, after = next(phases)
        return _capture(plan, value, after=after)

    def judge(messages):
        if response is None:
            raise RuntimeError("Synthetic judge unavailable")
        return response

    monkeypatch.setenv("WA_REDDIT", "https://reddit.test")
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
    monkeypatch.setattr(reference, "collect_snapshots", lambda plan: {})
    monkeypatch.setattr(reference, "collect_browser_snapshots_sync", collect)
    monkeypatch.setattr(vwa, "_judge_chat", judge)
    evaluator = WebArenaTaskEvaluator(config=_config())
    context = VisualBrowserEvaluationContext(page=page, browser_context=page.context, evidence=())
    evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
    try:
        with pytest.raises(EvaluatorInfrastructureError, match="VisualWebArena evaluation error:"):
            evaluator.evaluate(task=task, observation=WebObservation(), final_answer="done", browser_context=context)
    finally:
        evaluator.close()


def test_freshness_reuses_native_scores(monkeypatch):
    before, after = "saved\nnot", "saved\nnot\npersisted"
    predictions = []
    requests = []

    def native_judge(**kwargs):
        predictions.append(kwargs["pred"])
        return 1.0

    def freshness_judge(messages):
        requests.append(messages)
        return _freshness_judge_response(messages)

    monkeypatch.setattr(vwa, "_llm_fuzzy_match", native_judge)
    monkeypatch.setattr(vwa, "_judge_chat", freshness_judge)
    assert (
        vwa._score_program_html_snapshot_records(
            {"fuzzy_match": "saved"}, [{"before_value": before, "value": after, "delta_value": "persisted"}], None
        )
        == 0
    )
    assert predictions == [after, before]
    assert len(requests) == 1


@pytest.mark.parametrize(
    "required,before,after",
    [
        ({"must_include": ["saved"]}, "saved", "saved\nsaved"),
        ({"must_include": ["saved |OR| stored", "ready"]}, "saved\nready", "saved\nready\nstored\nready"),
        ({"fuzzy_match": "saved"}, "not persisted", "persisted"),
    ],
)
def test_literal_copies_and_improved_native_truth_need_no_freshness_call(monkeypatch, required, before, after):
    def unexpected_judge(messages):
        pytest.fail("Proven literal occurrences or improved native truth need no additional semantic call")

    monkeypatch.setattr(vwa, "_judge_chat", unexpected_judge)
    monkeypatch.setattr(vwa, "_llm_fuzzy_match", lambda **kw: float(_saved_claims(kw["pred"]) > 0))
    assert vwa._score_program_html_snapshot_records(required, [{"before_value": before, "value": after}], None) == 1
