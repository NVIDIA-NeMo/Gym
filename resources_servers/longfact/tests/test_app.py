# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from resources_servers.longfact.app import (
    BENCHMARK_VARIANT,
    AtomicFacts,
    BinaryClaimJudgment,
    LongFactSafeServer,
    ThreeWayClaimJudgment,
)
from resources_servers.longfact.evaluator import (
    EvaluatorIdentityError,
    EvaluatorParseError,
)


def test_f1_at_100_matches_safe_formula():
    assert LongFactSafeServer._f1_at_k(0, 4, 100) == 0
    assert LongFactSafeServer._f1_at_k(100, 0, 100) == 1
    assert LongFactSafeServer._f1_at_k(50, 50, 100) == pytest.approx(0.5)


def test_zero_judged_claims_have_perfect_precision_and_zero_f1():
    supported = 0
    not_supported = 0
    denominator = supported + not_supported
    precision = supported / denominator if denominator else 1.0

    assert precision == 1.0
    assert LongFactSafeServer._f1_at_k(supported, not_supported, 100) == 0.0


def test_claim_listing_prompt_includes_question_response_and_raw_json_contract():
    server = SimpleNamespace(config=SimpleNamespace(claim_extraction_prompt="claim_listing_v1"))

    request = LongFactSafeServer._claim_extraction_request(server, "Who is Ada?", "She was a mathematician.")

    assert "Prompt:\nWho is Ada?" in request
    assert "Response:\nShe was a mathematician." in request
    assert "single self-contained sentence" in request
    assert "without Markdown fences" in request


def test_legacy_claim_prompt_does_not_change_or_include_question():
    server = SimpleNamespace(config=SimpleNamespace(claim_extraction_prompt="legacy_v1"))

    request = LongFactSafeServer._claim_extraction_request(server, "SECRET QUESTION", "A factual response.")

    assert request.startswith("Break the RESPONSE")
    assert "A factual response." in request
    assert "SECRET QUESTION" not in request


def test_reset_rejects_missing_or_wrong_contract():
    server = SimpleNamespace(config=SimpleNamespace(benchmark_variant=BENCHMARK_VARIANT), session_state={})
    server._session_key = LongFactSafeServer._session_key
    server._info = lambda state: state
    with pytest.raises(ValueError, match="benchmark_variant"):
        import asyncio

        asyncio.run(LongFactSafeServer.reset(server, {"prompt": "Question", "source_provenance": {}}, "s"))


def test_compute_metrics_reports_safe_prompt_mean_and_micro_diagnostic():
    server = SimpleNamespace(config=SimpleNamespace(f1_max_claims=100))
    server._f1_at_k = LongFactSafeServer._f1_at_k
    metrics = LongFactSafeServer.compute_metrics(
        server,
        [
            [
                {
                    "info": {
                        "f1_at_k": 1.0,
                        "precision": 1.0,
                        "supported_claims": 100,
                        "not_supported_claims": 0,
                    }
                },
                {
                    "info": {
                        "f1_at_k": 0.0,
                        "precision": 0.0,
                        "supported_claims": 0,
                        "not_supported_claims": 1,
                        "tavily_cache_stats": {
                            "lookups": 20,
                            "completed_lookups": 20,
                            "persistent_hits": 7,
                            "effective_hit_rate": 0.6,
                        },
                    }
                },
            ]
        ],
    )
    assert metrics["longfact_safe/mean_f1_at_100"] == pytest.approx(0.5)
    assert metrics["longfact_safe/mean_precision"] == pytest.approx(0.5)
    # Pooled: precision 100/101, recall 100 / (2 x 100).
    assert metrics["longfact_safe/micro_f1_at_100"] == pytest.approx(200 / 301)
    assert metrics["longfact_safe/tavily_cache_lookups"] == 20
    assert metrics["longfact_safe/tavily_cache_persistent_hits"] == 7
    assert metrics["longfact_safe/tavily_cache_effective_hit_rate"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("mode", "label", "schema_type", "required_text", "forbidden_text"),
    [
        ("binary", "NOT_SUPPORTED", BinaryClaimJudgment, "NOT_SUPPORTED", "CONTRADICTED:"),
        ("three_way", "UNSURE", ThreeWayClaimJudgment, "CONTRADICTED:", "choose NOT_SUPPORTED"),
    ],
)
def test_claim_judge_mode_selects_matching_schema_and_prompt(mode, label, schema_type, required_text, forbidden_text):
    captured = {}

    async def generate_json(schema, prompt):
        captured.update(schema=schema, prompt=prompt)
        return schema(label=label, explanation="test")

    server = SimpleNamespace(
        config=SimpleNamespace(claim_verdict_mode=mode),
        _generate_json=generate_json,
    )
    judgment = asyncio.run(
        LongFactSafeServer._judge_fact(
            server,
            "The claim.",
            [{"url": "https://example.test", "text": "Evidence."}],
        )
    )

    assert captured["schema"] is schema_type
    assert required_text in captured["prompt"]
    assert forbidden_text not in captured["prompt"]
    assert judgment.label == label


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("UNSUPPORTED", "UNSURE"),
        ("NOT_SUPPORTED", "UNSURE"),
        (" not supported ", "UNSURE"),
        ("supported", "SUPPORTED"),
        (" contradicted ", "CONTRADICTED"),
        ("unsure", "UNSURE"),
    ],
)
def test_three_way_judgment_normalizes_common_label_variants(raw_label, expected):
    judgment = ThreeWayClaimJudgment(label=raw_label, explanation="No conclusive evidence.")

    assert judgment.label == expected
    assert judgment.explanation == "No conclusive evidence."


def test_prepare_fact_propagates_exhausted_json_retries():
    async def malformed_response(*_args, **_kwargs):
        raise EvaluatorParseError("Evaluator returned malformed JSON 6 times")

    server = SimpleNamespace(_generate_json=malformed_response)

    with pytest.raises(EvaluatorParseError):
        asyncio.run(LongFactSafeServer._prepare_fact(server, "Question?", "A response.", "A claim."))


@pytest.mark.parametrize("mode", ["binary", "three_way"])
def test_judge_without_a_verdict_raises_instead_of_unsure(mode):
    async def refused_response(*_args, **_kwargs):
        raise EvaluatorParseError("Evaluator returned malformed JSON 6 times")

    server = SimpleNamespace(config=SimpleNamespace(claim_verdict_mode=mode), _generate_json=refused_response)

    with pytest.raises(EvaluatorParseError):
        asyncio.run(LongFactSafeServer._judge_fact(server, "A sensitive quote.", []))


def test_unverdicted_claim_becomes_an_error_claim():
    async def extract(*_args, **_kwargs):
        return AtomicFacts(["A factual claim."])

    async def no_verdict(*_args, **_kwargs):
        raise EvaluatorParseError("Evaluator returned malformed JSON 6 times")

    server = SimpleNamespace(
        config=SimpleNamespace(
            max_claims=0, f1_max_claims=100, claim_verdict_mode="three_way", claim_extraction_prompt="legacy_v1"
        ),
        _generate_json=extract,
        _evaluate_fact=no_verdict,
        _claim_extraction_request=lambda *_args: "extract claims",
        _f1_at_k=LongFactSafeServer._f1_at_k,
    )

    evaluation = asyncio.run(LongFactSafeServer._evaluate(server, "Question", "Response"))

    assert evaluation["error_claims"] == 1
    assert evaluation["traces"][0]["label"] == "ERROR"


@pytest.mark.parametrize("constrain", [True, False])
def test_generate_json_uses_configured_retry_budget_and_decoding(constrain):
    attempts = 0
    seen = []

    async def eventually_valid(*_args, **kwargs):
        nonlocal attempts
        attempts += 1
        seen.append(kwargs["constrain_json_schema"])
        if attempts < 6:
            raise ValueError("malformed JSON")
        return ThreeWayClaimJudgment(label="SUPPORTED", explanation="valid")

    server = SimpleNamespace(
        config=SimpleNamespace(evaluator_max_attempts=6, evaluator_constrain_json_schema=constrain),
        _evaluator=SimpleNamespace(generate_json=eventually_valid),
    )
    result = asyncio.run(LongFactSafeServer._generate_json(server, ThreeWayClaimJudgment, "Judge this claim."))

    assert attempts == 6
    assert result.label == "SUPPORTED"
    assert seen == [constrain] * 6


def test_evaluator_identity_mismatch_is_not_downgraded_to_claim_error():
    async def extract(*_args, **_kwargs):
        return AtomicFacts(["A factual claim."])

    async def mismatched_identity(*_args, **_kwargs):
        raise EvaluatorIdentityError("judge model changed")

    server = SimpleNamespace(
        config=SimpleNamespace(max_claims=0, f1_max_claims=100, claim_verdict_mode="three_way"),
        _generate_json=extract,
        _evaluate_fact=mismatched_identity,
        _claim_extraction_request=lambda *_args: "extract claims",
    )

    with pytest.raises(EvaluatorIdentityError, match="judge model changed"):
        asyncio.run(LongFactSafeServer._evaluate(server, "Question", "Response"))


def test_longfact_json_parser_salvages_unfenced_qwen_commentary():
    from resources_servers.longfact.evaluator import JsonEvaluator

    parsed = JsonEvaluator._parse_json(
        ThreeWayClaimJudgment,
        'Here is the verdict: {"label":"SUPPORTED","explanation":"Evidence agrees."} Thanks.',
        salvage_embedded_json=True,
    )

    assert parsed.label == "SUPPORTED"
    assert parsed.explanation == "Evidence agrees."


def test_compute_metrics_reports_three_way_diagnostics():
    server = SimpleNamespace(config=SimpleNamespace(f1_max_claims=100))
    server._f1_at_k = LongFactSafeServer._f1_at_k
    metrics = LongFactSafeServer.compute_metrics(
        server,
        [
            [
                {
                    "info": {
                        "claim_verdict_mode": "three_way",
                        "f1_at_k": 0.02,
                        "precision": 0.5,
                        "decisive_precision": 2 / 3,
                        "contradiction_rate": 0.25,
                        "unsure_rate": 0.25,
                        "supported_claims": 2,
                        "not_supported_claims": 2,
                        "contradicted_claims": 1,
                        "unsure_claims": 1,
                    }
                },
                {
                    "info": {
                        "claim_verdict_mode": "three_way",
                        "f1_at_k": 0.0,
                        "precision": 0.0,
                        "decisive_precision": 1.0,
                        "contradiction_rate": 0.0,
                        "unsure_rate": 1.0,
                        "supported_claims": 0,
                        "not_supported_claims": 2,
                        "contradicted_claims": 0,
                        "unsure_claims": 2,
                    }
                },
            ]
        ],
    )

    assert metrics["longfact_safe/contradicted_claims"] == 1
    assert metrics["longfact_safe/unsure_claims"] == 3
    assert metrics["longfact_safe/mean_decisive_precision"] == pytest.approx(5 / 6)
    assert metrics["longfact_safe/mean_contradiction_rate"] == pytest.approx(0.125)
    assert metrics["longfact_safe/mean_unsure_rate"] == pytest.approx(0.625)
    assert metrics["longfact_safe/micro_supported_rate"] == pytest.approx(1 / 3)
    assert metrics["longfact_safe/micro_decisive_precision"] == pytest.approx(2 / 3)


def _step_server(evaluate):
    state = {"prompt": "Question", "task_id": "t", "subset": "objects", "topic": "x", "source_provenance": {}}
    server = SimpleNamespace(session_state={"s": state}, _evaluate=evaluate)
    server._session_key = LongFactSafeServer._session_key
    server._info = lambda state: {"f1_at_k": state.get("f1_at_k")}
    return server


def _run_step(server):
    action = SimpleNamespace(output=[])
    return asyncio.run(LongFactSafeServer.step(server, action, {}, "s"))


def test_step_fails_the_rollout_when_search_fails(monkeypatch):
    from resources_servers.longfact import app
    from resources_servers.longfact.tavily_provider import TavilyProviderError

    monkeypatch.setattr(app, "extract_text", lambda _action: "Response")

    async def search_down(*_args):
        raise TavilyProviderError("quota exhausted")

    with pytest.raises(TavilyProviderError, match="quota exhausted"):
        _run_step(_step_server(search_down))


def test_step_fails_the_rollout_with_unchecked_claims(monkeypatch):
    from resources_servers.longfact import app

    monkeypatch.setattr(app, "extract_text", lambda _action: "Response")

    async def one_claim_errored(*_args):
        return {
            "f1_at_k": 0.02,
            "num_extracted_claims": 2,
            "error_claims": 1,
            "traces": [{"label": "SUPPORTED"}, {"label": "ERROR", "fact_error": "TimeoutError()"}],
        }

    with pytest.raises(app.LongFactClaimCheckError, match=r"1 of 2 claim\(s\).*TimeoutError\(\)"):
        _run_step(_step_server(one_claim_errored))


def test_step_scores_a_fully_checked_episode(monkeypatch):
    from resources_servers.longfact import app

    monkeypatch.setattr(app, "extract_text", lambda _action: "Response")

    async def all_checked(*_args):
        return {"f1_at_k": 0.5, "num_extracted_claims": 1, "error_claims": 0, "traces": [{"label": "SUPPORTED"}]}

    _obs, reward, terminated, _truncated, info = _run_step(_step_server(all_checked))

    assert (reward, terminated, info["f1_at_k"]) == (0.5, True, 0.5)


def test_compute_metrics_names_f1_by_k():
    server = SimpleNamespace(config=SimpleNamespace(f1_max_claims=64))
    server._f1_at_k = LongFactSafeServer._f1_at_k
    record = {"f1_at_k": 1.0, "precision": 1.0, "supported_claims": 64, "not_supported_claims": 0}

    metrics = LongFactSafeServer.compute_metrics(server, [[{"info": record}]])

    assert metrics["longfact_safe/mean_f1_at_64"] == pytest.approx(1.0)
    assert metrics["longfact_safe/micro_f1_at_64"] == pytest.approx(1.0)
    assert "longfact_safe/mean_f1_at_100" not in metrics


def test_micro_f1_does_not_grow_with_repeated_identical_episodes():
    server = SimpleNamespace(config=SimpleNamespace(f1_max_claims=100))
    server._f1_at_k = LongFactSafeServer._f1_at_k
    record = {"f1_at_k": 0.4, "precision": 1.0, "supported_claims": 25, "not_supported_claims": 0}

    one = LongFactSafeServer.compute_metrics(server, [[{"info": record}]])
    two = LongFactSafeServer.compute_metrics(server, [[{"info": record}], [{"info": dict(record)}]])

    assert one["longfact_safe/micro_f1_at_100"] == pytest.approx(0.4)
    assert two["longfact_safe/micro_f1_at_100"] == pytest.approx(0.4)


def _http_server():
    from unittest.mock import MagicMock

    from nemo_gym.server_utils import ServerClient
    from resources_servers.longfact.app import LongFactConfig

    config = LongFactConfig(
        host="",
        port=0,
        entrypoint="",
        name="longfact",
        evaluator_model_server={"type": "responses_api_models", "name": "judge"},
        evaluator_responses_create_params={"input": []},
        search_backend="tavily",
        tavily_api_key="unused",
    )
    return LongFactSafeServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_failed_step_session_is_released_by_close():
    import httpx

    server = _http_server()

    async def judge_down(*_args):
        raise EvaluatorParseError("judge down")

    server._evaluate = judge_down
    row = {
        "responses_create_params": {"input": []},
        "id": "t",
        "benchmark_variant": BENCHMARK_VARIANT,
        "prompt": "Tell me about the Eiffel Tower.",
        "source_provenance": {"path": "x"},
    }
    response = {
        "id": "r",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }

    async def main():
        transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            reset = await client.post("/reset", json=row)
            step = await client.post("/step", json={"responses_create_params": {"input": []}, "response": response})
            open_sessions = len(server.session_state)
            close = await client.post("/close", json={})
        return reset, step, open_sessions, close

    reset, step, open_sessions, close = asyncio.run(main())

    assert reset.json()["info"]["supports_explicit_close"] is True
    assert step.status_code == 500
    assert open_sessions == 1
    assert close.status_code == 200
    assert server.session_state == {}
