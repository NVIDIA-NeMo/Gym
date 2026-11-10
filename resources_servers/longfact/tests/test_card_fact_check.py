# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.longfact.app import AtomicFacts, CardBatchMismatchError, LongFactConfig, LongFactSafeServer
from resources_servers.longfact.card_fact_check import (
    BROWSE_OFF_CLAIM_LISTING,
    BROWSE_OFF_FACT_CHECK,
    BatchSearchPlan,
    CardBatchVerdicts,
    build_fact_check_request,
)
from resources_servers.longfact.claim_prompts import build_claim_extraction_request


def _source(url):
    return SimpleNamespace(title="t", url=url, snippet=f"text from {url}")


def _requested(request, end_marker=None):
    """The claims a card request lists, as (claim_id, claim) pairs."""
    section = request.split("Claims:\n", 1)[1]
    if end_marker:
        section = section.split(end_marker, 1)[0]
    return [(item["claim_id"], item["claim"]) for item in json.loads(section)]


def _card_server(answers, *, plan_edits=(), verdict_edits=(), batch_size=10, max_attempts=3):
    """A server whose grader answers from ``answers``; each edit rewrites one attempt's items in turn."""
    searched = []
    constrained = []
    calls = {"plan": 0, "verdicts": 0}
    batches = {"plan": [], "verdicts": []}

    async def search(query):
        searched.append(query)
        return [_source(f"https://example.test/{query}")]

    async def generate_json(schema, request, *, constrain=None):
        if schema is BatchSearchPlan:
            constrained.append(constrain)
            requested = _requested(request)
            batches["plan"].append(tuple(claim for _, claim in requested))
            items = [{"claim_id": cid, "queries": ["q1", "q2", "q3", "q4"]} for cid, _ in requested]
            edit = plan_edits[calls["plan"]] if calls["plan"] < len(plan_edits) else None
            calls["plan"] += 1
            return BatchSearchPlan(claims=edit(items) if edit else items)
        if schema is CardBatchVerdicts:
            constrained.append(constrain)
            requested = _requested(request, "\n\nSearch results:")
            batches["verdicts"].append(tuple(claim for _, claim in requested))
            items = [
                {
                    "claim_id": cid,
                    "claim": claim,
                    "answer": answers[claim],
                    "reasoning": claim,
                    "supporting_evidence": [],
                }
                for cid, claim in requested
            ]
            edit = verdict_edits[calls["verdicts"]] if calls["verdicts"] < len(verdict_edits) else None
            calls["verdicts"] += 1
            return CardBatchVerdicts(verdicts=edit(items) if edit else items)
        return AtomicFacts(list(answers))

    server = SimpleNamespace(
        config=SimpleNamespace(
            max_queries=3,
            browse_off_instructions=True,
            fact_check_mode="gpt5_card_batch",
            fact_check_batch_size=batch_size,
            evaluator_max_attempts=max_attempts,
            max_claims=0,
            f1_max_claims=100,
            claim_verdict_mode="three_way",
            claim_extraction_prompt="claim_listing_v1",
        ),
        _search_provider=SimpleNamespace(search=search),
        _generate_json=generate_json,
        _claim_extraction_request=lambda *_args: "list claims",
        _f1_at_k=LongFactSafeServer._f1_at_k,
        _claim_id_problem=LongFactSafeServer._claim_id_problem,
    )
    server._generate_covering = lambda *args: LongFactSafeServer._generate_covering(server, *args)
    server._check_batch = lambda *args: LongFactSafeServer._check_batch(server, *args)
    server._check_batch_safely = lambda *args: LongFactSafeServer._check_batch_safely(server, *args)
    server.constrained = constrained
    server.calls = calls
    server.batches = batches
    return server, searched


def test_batch_labels_follow_the_card_answers_and_cap_queries():
    answers = {"claim a": "true", "claim b": "false", "claim c": "unsure"}
    server, searched = _card_server(answers)

    traces = asyncio.run(LongFactSafeServer._check_batch(server, "Q", "R", list(answers)))

    assert [trace["label"] for trace in traces] == ["SUPPORTED", "CONTRADICTED", "UNSURE"]
    assert [trace["claim_id"] for trace in traces] == ["C1", "C2", "C3"]
    assert [trace["judged_claim"] for trace in traces] == list(answers)
    assert all(trace["queries"] == ["q1", "q2", "q3"] for trace in traces)
    assert len(searched) == 9
    assert traces[1]["evidence"][0]["query"] == "q1"
    assert server.constrained == [True, True]


def test_reordered_verdicts_are_mapped_by_claim_id():
    answers = {"claim a": "true", "claim b": "false"}
    server, _ = _card_server(answers, verdict_edits=[lambda items: list(reversed(items))])

    traces = asyncio.run(LongFactSafeServer._check_batch(server, "Q", "R", list(answers)))

    assert [(trace["atomic_fact"], trace["label"]) for trace in traces] == [
        ("claim a", "SUPPORTED"),
        ("claim b", "CONTRADICTED"),
    ]
    assert [trace["explanation"] for trace in traces] == ["claim a", "claim b"]


def _duplicate_last(items):
    return items[:-2] + [items[-1], dict(items[-1], answer="true")]


def _drop_first(items):
    return items[1:]


def _rename_first(items):
    return [dict(items[0], claim_id="C99")] + items[1:]


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (_duplicate_last, "duplicated ['C2']; missing ['C1']"),
        (_drop_first, "missing ['C1']"),
        (_rename_first, "unknown ['C99']; missing ['C1']"),
    ],
)
def test_a_malformed_verdict_list_is_retried_then_accepted(edit, problem):
    answers = {"claim a": "true", "claim b": "false"}
    server, searched = _card_server(answers, verdict_edits=[edit])

    traces = asyncio.run(LongFactSafeServer._check_batch(server, "Q", "R", list(answers)))

    assert [trace["label"] for trace in traces] == ["SUPPORTED", "CONTRADICTED"]
    assert server.calls == {"plan": 1, "verdicts": 2}
    # The searches of the valid plan are kept, not repeated, when only the verdicts are retried.
    assert len(searched) == 6
    assert (
        LongFactSafeServer._claim_id_problem(
            [item["claim_id"] for item in edit([{"claim_id": "C1"}, {"claim_id": "C2"}])], {"C1": "a", "C2": "b"}
        )
        == problem
    )


def test_a_verdict_list_still_malformed_after_every_attempt_fails_verification():
    answers = {"claim a": "true", "claim b": "false"}
    server, searched = _card_server(answers, verdict_edits=[_duplicate_last] * 3, max_attempts=3)

    with pytest.raises(CardBatchMismatchError) as error:
        asyncio.run(LongFactSafeServer._check_batch_safely(server, "Q", "R", list(answers)))

    assert "fact-check did not cover the batch after 3 attempts: duplicated ['C2']" in str(error.value)
    assert server.calls == {"plan": 1, "verdicts": 3}
    assert len(searched) == 6


def test_a_search_plan_missing_a_claim_is_retried_before_searching():
    answers = {"claim a": "true", "claim b": "true"}
    server, searched = _card_server(answers, plan_edits=[_drop_first])

    traces = asyncio.run(LongFactSafeServer._check_batch(server, "Q", "R", list(answers)))

    assert [trace["label"] for trace in traces] == ["SUPPORTED", "SUPPORTED"]
    assert server.calls == {"plan": 2, "verdicts": 1}
    assert len(searched) == 6


def test_a_search_plan_malformed_after_every_attempt_fails_verification():
    server, searched = _card_server(
        {"claim a": "true", "claim b": "true"}, plan_edits=[_drop_first] * 2, max_attempts=2
    )

    with pytest.raises(CardBatchMismatchError) as error:
        asyncio.run(LongFactSafeServer._check_batch_safely(server, "Q", "R", ["claim a", "claim b"]))

    assert "search plan did not cover the batch after 2 attempts: missing ['C1']" in str(error.value)
    assert server.calls == {"plan": 2, "verdicts": 0}
    assert searched == []


def test_card_mode_checks_claims_in_separate_batches():
    answers = {f"claim {index}": ("false" if index < 3 else "true") for index in range(12)}
    server, searched = _card_server(answers, batch_size=10)

    evaluation = asyncio.run(LongFactSafeServer._evaluate(server, "Q", "R"))

    assert evaluation["num_extracted_claims"] == 12
    assert (evaluation["supported_claims"], evaluation["contradicted_claims"]) == (9, 3)
    assert evaluation["contradiction_rate"] == pytest.approx(3 / 12)
    assert evaluation["fact_check_mode"] == "gpt5_card_batch"
    assert len(searched) == 36
    expected_batches = {tuple(f"claim {index}" for index in range(10)), ("claim 10", "claim 11")}
    assert server.calls == {"plan": 2, "verdicts": 2}
    assert set(server.batches["plan"]) == expected_batches
    assert set(server.batches["verdicts"]) == expected_batches


def test_browse_off_sentences_follow_the_card():
    listing = build_claim_extraction_request("claim_listing_v1", "P", "R", browse_off=True)
    check = build_fact_check_request("P", "R", {"C1": "c"}, {"C1": []}, browse_off=True)
    plain = build_fact_check_request("P", "R", {"C1": "c"}, {"C1": []}, browse_off=False)

    assert BROWSE_OFF_CLAIM_LISTING in listing.split("### Formatting")[0]
    assert BROWSE_OFF_FACT_CHECK in check.split("### Connection of claims")[0]
    assert "did not have access to the web" not in plain
    with pytest.raises(ValueError, match="claim_listing_v1"):
        build_claim_extraction_request("safe_v1", "P", "R", browse_off=True)


def test_card_mode_requires_three_way_verdicts():
    config = LongFactConfig(
        host="",
        port=0,
        entrypoint="",
        name="longfact",
        evaluator_model_server={"type": "responses_api_models", "name": "judge"},
        evaluator_responses_create_params={"input": []},
        search_backend="tavily",
        tavily_api_key="unused",
        fact_check_mode="gpt5_card_batch",
        claim_extraction_prompt="claim_listing_v1",
        claim_verdict_mode="binary",
    )

    with pytest.raises(ValueError, match="three_way"):
        LongFactSafeServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    "options",
    [
        {"browse_off_instructions": True},
        {"fact_check_mode": "gpt5_card_batch"},
    ],
)
def test_question_aware_modes_reject_safe_extraction_at_startup(options):
    config = LongFactConfig(
        host="",
        port=0,
        entrypoint="",
        name="longfact",
        evaluator_model_server={"type": "responses_api_models", "name": "judge"},
        evaluator_responses_create_params={"input": []},
        search_backend="tavily",
        tavily_api_key="unused",
        claim_extraction_prompt="safe_v1",
        **options,
    )

    with pytest.raises(ValueError, match="claim_extraction_prompt=claim_listing_v1"):
        LongFactSafeServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_error_rates_are_reported_overall_and_per_subset():
    server = SimpleNamespace(config=SimpleNamespace(f1_max_claims=100))
    server._f1_at_k = LongFactSafeServer._f1_at_k
    server._error_rates = LongFactSafeServer._error_rates

    def record(subset, supported, contradicted, unsure):
        classified = supported + contradicted + unsure
        return {
            "info": {
                "subset": subset,
                "claim_verdict_mode": "three_way",
                "f1_at_k": 0.5,
                "precision": supported / classified,
                "decisive_precision": supported / (supported + contradicted),
                "contradiction_rate": contradicted / classified,
                "unsure_rate": unsure / classified,
                "supported_claims": supported,
                "not_supported_claims": contradicted + unsure,
                "contradicted_claims": contradicted,
                "unsure_claims": unsure,
            }
        }

    metrics = LongFactSafeServer.compute_metrics(
        server, [[record("objects", 8, 1, 1)], [record("concepts", 18, 2, 0)]]
    )

    assert metrics["longfact_safe/micro_error_rate"] == pytest.approx(3 / 30)
    assert metrics["longfact_safe/micro_decisive_error_rate"] == pytest.approx(3 / 29)
    assert metrics["longfact_safe/mean_error_rate"] == pytest.approx((0.1 + 0.1) / 2)
    assert metrics["longfact_safe/objects/micro_error_rate"] == pytest.approx(0.1)
    assert metrics["longfact_safe/objects/micro_decisive_error_rate"] == pytest.approx(1 / 9)
    assert metrics["longfact_safe/concepts/micro_error_rate"] == pytest.approx(0.1)
    assert metrics["longfact_safe/concepts/episodes_scored"] == 1
    assert metrics["longfact_safe/micro_unsure_rate"] == pytest.approx(1 / 30)
    assert metrics["longfact_safe/objects/micro_unsure_rate"] == pytest.approx(0.1)


def test_card_schemas_are_strict_object_roots():
    for schema in (CardBatchVerdicts.model_json_schema(), BatchSearchPlan.model_json_schema()):
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        for definition in schema.get("$defs", {}).values():
            assert definition["additionalProperties"] is False
            assert set(definition["required"]) == set(definition["properties"])


def test_a_bare_card_array_is_accepted_without_constrained_decoding():
    verdict = {"claim_id": "C1", "claim": "c", "answer": "false", "reasoning": "r", "supporting_evidence": []}

    assert [v.answer for v in CardBatchVerdicts.model_validate([verdict]).verdicts] == ["false"]
    with pytest.raises(ValueError):
        CardBatchVerdicts.model_validate({"verdicts": [verdict], "notes": []})
