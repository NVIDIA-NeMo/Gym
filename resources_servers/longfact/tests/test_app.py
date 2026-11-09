# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
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


@pytest.mark.parametrize("launch_mode", ["ordinary_import", "worker_import", "main_launch"])
def test_entrypoint_import_constructs_only_the_app_requested_by_the_launcher(tmp_path: Path, launch_mode: str):
    cache_dir = tmp_path / "tavily-replay"
    cache_dir.mkdir(mode=0o700)
    global_config = {
        "dry_run": False,
        "head_server": {"host": "127.0.0.1", "port": 1},
        "worker_longfact": {
            "resources_servers": {
                "longfact": {
                    "host": "127.0.0.1",
                    "port": 0,
                    "entrypoint": "app.py",
                    "num_workers": 2 if launch_mode == "worker_import" else 1,
                    "uvicorn_reuse_port": False,
                    "evaluator_model_server": {"type": "responses_api_models", "name": "offline_judge"},
                    "evaluator_responses_create_params": {"input": []},
                    "search_backend": "tavily",
                    "tavily_cache_mode": "replay",
                    "tavily_cache_dir": str(cache_dir),
                }
            }
        },
    }
    env = os.environ | {
        "LONGFACT_ENTRYPOINT_TEST_MODE": launch_mode,
        "IS_NEMO_GYM_FASTAPI_WORKER": "1" if launch_mode == "worker_import" else "0",
        "IS_NEMO_GYM_FASTAPI_ENTRYPOINT": "resources_servers/longfact/app.py",
        "NEMO_GYM_FASTAPI_NUM_WORKERS": "2" if launch_mode == "worker_import" else "1",
        "NEMO_GYM_CONFIG_PATH": "worker_longfact",
        "NEMO_GYM_CONFIG_DICT": "invalid: [" if launch_mode == "ordinary_import" else json.dumps(global_config),
        "NEMO_GYM_OTEL_ENABLED": "0",
    }
    env.pop("NEMO_GYM_SESSION_SOCKET_DIR", None)
    program = dedent("""
        import importlib
        import os
        import runpy

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        mode = os.environ["LONGFACT_ENTRYPOINT_TEST_MODE"]
        if mode == "ordinary_import":
            module = importlib.import_module("resources_servers.longfact.app")
            assert not hasattr(module, "app"), "ordinary library import exported a running app"
        else:
            if mode == "main_launch":
                import uvicorn

                launches = []
                uvicorn.run = lambda **kwargs: launches.append(kwargs)
                runpy.run_module("resources_servers.longfact.app", run_name="__main__", alter_sys=True)
                assert len(launches) == 1, "the main entrypoint must launch exactly once"
                assert "workers" not in launches[0], "this fixture uses the one-worker socket-launch path"
                app = launches[0]["app"]
            else:
                module = importlib.import_module("resources_servers.longfact.app")
                app = module.app
            assert isinstance(app, FastAPI), "the entrypoint must construct the real configured app"
            paths = {route.path for route in app.routes}
            assert {"/verify", "/seed_session", "/close_session", "/reset", "/step"} <= paths
            with TestClient(app) as client:
                reply = client.get("/health")
                assert reply.status_code == 200
                assert reply.json() == {"status": "ok"}
        """)

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_f1_at_100_matches_safe_formula():
    assert LongFactSafeServer._f1_at_k(0, 4, 100) == 0
    assert LongFactSafeServer._f1_at_k(100, 0, 100) == 1
    assert LongFactSafeServer._f1_at_k(50, 50, 100) == pytest.approx(0.5)


def test_zero_judged_claims_have_perfect_precision_and_zero_f1():
    async def extract(*_args, **_kwargs):
        return AtomicFacts([])

    server = SimpleNamespace(
        config=SimpleNamespace(
            max_claims=0,
            f1_max_claims=100,
            claim_extraction_prompt="claim_listing_v1",
            claim_verdict_mode="three_way",
            fact_check_mode="per_claim",
        ),
        _generate_json=extract,
        _claim_extraction_request=lambda *_args: "extract claims",
        _f1_at_k=LongFactSafeServer._f1_at_k,
    )

    evaluation = asyncio.run(LongFactSafeServer._evaluate(server, "Question", "Response"))

    assert evaluation["num_extracted_claims"] == 0
    assert evaluation["precision"] == 1.0
    assert evaluation["f1_at_k"] == 0.0
    assert evaluation["traces"] == []


def test_onboarding_fixture_exercises_stateless_native_f1_at_100_scoring():
    from nemo_gym.verification import exercise_verifier_fixture, load_verifier_fixture
    from resources_servers.longfact.app import VERIFIER_FIXTURE
    from resources_servers.longfact.verifier_fixture import invoke_fixture

    async def run():
        results = await exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
        )
        case = next(case for case in load_verifier_fixture(VERIFIER_FIXTURE.cases_path) if case.kind == "full_reward")
        server = VERIFIER_FIXTURE.server_factory()
        request = VERIFIER_FIXTURE.request_model.model_validate(case.request)
        scored = await invoke_fixture(server, request)
        return results, server, scored

    results, server, scored = asyncio.run(run())

    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed"}
    assert scored["reward"] == 1.0
    assert scored["info"]["safe_variant"]["f1_max_claims"] == 100
    assert scored["info"]["num_extracted_claims"] == scored["info"]["supported_claims"] == 100
    assert server.session_state == {}


def test_claim_listing_prompt_includes_question_response_and_raw_json_contract():
    server = SimpleNamespace(
        config=SimpleNamespace(claim_extraction_prompt="claim_listing_v1", browse_off_instructions=False)
    )

    request = LongFactSafeServer._claim_extraction_request(server, "Who is Ada?", "She was a mathematician.")

    assert "Prompt:\nWho is Ada?" in request
    assert "Response:\nShe was a mathematician." in request
    assert "single self-contained sentence" in request
    assert "without Markdown fences" in request


def test_legacy_claim_prompt_does_not_change_or_include_question():
    server = SimpleNamespace(
        config=SimpleNamespace(claim_extraction_prompt="legacy_v1", browse_off_instructions=False)
    )

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
    server._error_rates = LongFactSafeServer._error_rates
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


def test_unverdicted_claim_propagates_its_typed_judge_failure():
    async def extract(*_args, **_kwargs):
        return AtomicFacts(["A factual claim."])

    async def no_verdict(*_args, **_kwargs):
        raise EvaluatorParseError("Evaluator returned malformed JSON 6 times")

    server = SimpleNamespace(
        config=SimpleNamespace(
            max_claims=0,
            f1_max_claims=100,
            claim_verdict_mode="three_way",
            claim_extraction_prompt="legacy_v1",
            fact_check_mode="per_claim",
        ),
        _generate_json=extract,
        _evaluate_fact=no_verdict,
        _claim_extraction_request=lambda *_args: "extract claims",
        _f1_at_k=LongFactSafeServer._f1_at_k,
    )

    with pytest.raises(EvaluatorParseError, match="malformed JSON 6 times"):
        asyncio.run(LongFactSafeServer._evaluate(server, "Question", "Response"))


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
        config=SimpleNamespace(
            max_claims=0, f1_max_claims=100, claim_verdict_mode="three_way", fact_check_mode="per_claim"
        ),
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
    server._error_rates = LongFactSafeServer._error_rates
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
    server._require_checked_claims = LongFactSafeServer._require_checked_claims
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
    server._error_rates = LongFactSafeServer._error_rates
    record = {"f1_at_k": 1.0, "precision": 1.0, "supported_claims": 64, "not_supported_claims": 0}

    metrics = LongFactSafeServer.compute_metrics(server, [[{"info": record}]])

    assert metrics["longfact_safe/mean_f1_at_64"] == pytest.approx(1.0)
    assert metrics["longfact_safe/micro_f1_at_64"] == pytest.approx(1.0)
    assert "longfact_safe/mean_f1_at_100" not in metrics


def test_micro_f1_does_not_grow_with_repeated_identical_episodes():
    server = SimpleNamespace(config=SimpleNamespace(f1_max_claims=100))
    server._f1_at_k = LongFactSafeServer._f1_at_k
    server._error_rates = LongFactSafeServer._error_rates
    record = {"f1_at_k": 0.4, "precision": 1.0, "supported_claims": 25, "not_supported_claims": 0}

    one = LongFactSafeServer.compute_metrics(server, [[{"info": record}]])
    two = LongFactSafeServer.compute_metrics(server, [[{"info": record}], [{"info": dict(record)}]])

    assert one["longfact_safe/micro_f1_at_100"] == pytest.approx(0.4)
    assert two["longfact_safe/micro_f1_at_100"] == pytest.approx(0.4)


def _http_server():
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
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


@pytest.mark.parametrize(("label", "reward", "supported"), [("SUPPORTED", 1.0, 100), ("CONTRADICTED", 0.0, 0)])
def test_native_verify_binds_the_original_answer_reward_and_task_metadata(label, reward, supported):
    from nemo_gym.trajectory import grading_result_binding
    from resources_servers.longfact.verifier_fixture import _response, _server_factory

    async def run():
        import httpx

        server = _server_factory()
        server.server_client.label = label
        server.server_client.global_config_dict = {"observability_enabled": True}
        body = {
            "responses_create_params": {"input": [{"role": "user", "content": "Question"}]},
            "response": _response("The saved policy answer.").model_dump(mode="json"),
            "id": "source-task",
            "task_id": "source-task",
            "benchmark_variant": BENCHMARK_VARIANT,
            "prompt": "Question",
            "subset": "objects",
            "topic": "history",
            "source_provenance": {"source_path": "published-source.jsonl", "row": 7},
        }
        transport = httpx.ASGITransport(app=server.setup_webserver())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/ng-rollout/longfact-native-test/verify", json=body)
        return server, body, response

    server, body, response = asyncio.run(run())

    assert response.status_code == 200
    row = response.json()
    assert row["response"] == body["response"]
    assert row["reward"] == reward
    assert row["info"]["task_id"] == "source-task"
    assert row["info"]["source_provenance"] == body["source_provenance"]
    assert row["info"]["num_extracted_claims"] == 100
    assert row["info"]["supported_claims"] == supported
    assert row["info"]["safe_variant"]["f1_max_claims"] == 100
    assert not row["mask_sample"]
    (invocation,) = row["ng_verifier_invocations"]
    assert invocation["status"] == "completed"
    (operation,) = invocation["grading_operations"]
    assert (operation["kind"], operation["outcome"]) == ("llm_judge", "completed")
    assert len(operation["dependencies"]) == 201
    assert operation["result_binding_sha256"] == grading_result_binding(row)
    assert grading_result_binding(row | {"reward": 0.5}) != operation["result_binding_sha256"]
    changed_response = body["response"] | {"output": []}
    assert grading_result_binding(row | {"response": changed_response}) != operation["result_binding_sha256"]
    assert all(
        dependency["model_ref"] == {"type": "responses_api_models", "name": "offline_judge"}
        for dependency in operation["dependencies"]
    )
    assert server.session_state == {}


class _CardJudgeTransport:
    """Script the external judge replies while the real scorer owns concurrency and validation."""

    global_config_dict = {"observability_enabled": True}

    def __init__(self, fault=None):
        self.fault = fault
        self.counts = {}
        self.received_ids = []
        self.invocation_ids = []
        self.second_plan_received = asyncio.Event()
        self.completed_plans = []
        self.sibling_started = asyncio.Event()
        self.sibling_finished = asyncio.Event()

    async def post(self, *, json, **kwargs):
        import json as jsonlib

        from resources_servers.longfact.verifier_fixture import _JudgeResponse, _response

        schema = json.text["format"]["name"]
        prompt = json.input[-1].content
        self.counts[schema] = self.counts.get(schema, 0) + 1
        count = self.counts[schema]
        if schema == "AtomicFacts":
            payload = ["First factual claim.", "Second factual claim."]
            if self.fault == "exhausted_malformed" or (self.fault == "malformed_extraction" and count == 1):
                text = '{"claims": []'
            elif self.fault == "filtered_extraction":
                payload.append("   ")
                text = jsonlib.dumps(payload)
            else:
                text = jsonlib.dumps(payload)
        else:
            claims = jsonlib.loads(prompt.rsplit("Claims:\n", 1)[1].split("\n\nSearch results:", 1)[0])
            if schema == "BatchSearchPlan":
                # Reverse batch completion so the scorer must bind concurrent accepted replies correctly.
                if claims[0]["claim"].startswith("First"):
                    await self.second_plan_received.wait()
                else:
                    self.second_plan_received.set()
                self.completed_plans.append(claims[0]["claim"])
                payload = {"claims": [{"claim_id": item["claim_id"], "queries": [item["claim"]]} for item in claims]}
                if self.fault == "invalid_id" and count == 1:
                    payload["claims"][0]["claim_id"] = "UNKNOWN"
            elif schema == "CardBatchVerdicts":
                if self.fault == "failed_batch":
                    if claims[0]["claim"].startswith("First"):
                        self.sibling_started.set()
                        try:
                            await asyncio.Event().wait()
                        finally:
                            self.sibling_finished.set()
                    await self.sibling_started.wait()
                    response = _response('{"verdicts": []')
                    response.id = f"malformed-verdict-{count}"
                    self.received_ids.append(response.id)
                    return _JudgeResponse(response.model_dump_json().encode())
                payload = {
                    "verdicts": [
                        {
                            "claim_id": item["claim_id"],
                            "claim": item["claim"],
                            "answer": "true",
                            "reasoning": "Scripted external verdict.",
                            "supporting_evidence": [],
                        }
                        for item in claims
                    ]
                }
            else:
                raise AssertionError(f"Unexpected schema {schema}")
            text = jsonlib.dumps(payload)
        response = _response(text)
        response.id = f"external-{schema}-{count}"
        self.received_ids.append(response.id)
        self.invocation_ids.append(kwargs["headers"]["x-ng-verifier-invocation-id"])
        return _JudgeResponse(response.model_dump_json().encode())


def _card_http_verification(fault=None):
    from resources_servers.longfact.evaluator import JsonEvaluator
    from resources_servers.longfact.verifier_fixture import _response, _server_factory

    async def run():
        import httpx

        server = _server_factory()
        transport = _CardJudgeTransport(fault)
        server.server_client = transport
        server.config.fact_check_mode = "gpt5_card_batch"
        server.config.fact_check_batch_size = 1
        server.config.evaluator_max_attempts = 2
        server._evaluator = JsonEvaluator(
            server_client=transport,
            model_server=server.config.evaluator_model_server,
            create_params=server.config.evaluator_responses_create_params,
            max_concurrency=16,
            max_attempts=1,
            timeout_seconds=1,
        )
        body = {
            "responses_create_params": {"input": [{"role": "user", "content": "Question"}]},
            "response": _response("The saved policy answer.").model_dump(mode="json"),
            "id": "card-task",
            "benchmark_variant": BENCHMARK_VARIANT,
            "prompt": "Question",
            "source_provenance": {"source_path": "published-source.jsonl"},
        }
        http_transport = httpx.ASGITransport(app=server.setup_webserver())
        async with httpx.AsyncClient(transport=http_transport, base_url="http://test") as client:
            response = await client.post("/ng-rollout/longfact-card-test/verify", json=body)
            transport.sibling_finished_at_route_return = transport.sibling_finished.is_set()
        assert response.status_code == 200
        return body, response.json(), transport

    return asyncio.run(run())


def test_concurrent_card_batches_bind_every_accepted_judge_dependency():
    from nemo_gym.trajectory import grading_result_binding

    body, row, transport = _card_http_verification()

    assert row["reward"] == pytest.approx(4 / 102)
    assert row["response"] == body["response"]
    assert [trace["atomic_fact"] for trace in row["info"]["traces"]] == [
        "First factual claim.",
        "Second factual claim.",
    ]
    (invocation,) = row["ng_verifier_invocations"]
    (operation,) = invocation["grading_operations"]
    assert operation["outcome"] == "completed"
    assert operation["result_binding_sha256"] == grading_result_binding(row)
    assert [dependency["response_id"] for dependency in operation["dependencies"]] == transport.received_ids
    assert len(operation["dependencies"]) == len(set(transport.received_ids)) == 5
    assert set(transport.invocation_ids) == {invocation["invocation_id"]}
    assert transport.completed_plans == ["Second factual claim.", "First factual claim."]


@pytest.mark.parametrize("fault", ["invalid_id", "malformed_extraction", "filtered_extraction"])
def test_rejected_or_filtered_judge_reply_keeps_retry_grading_unknown(fault):
    body, row, transport = _card_http_verification(fault)

    assert row["reward"] == pytest.approx(4 / 102)
    assert row["response"] == body["response"]
    assert row["info"]["supported_claims"] == 2
    assert not row["mask_sample"]
    (invocation,) = row["ng_verifier_invocations"]
    (operation,) = invocation["grading_operations"]
    assert operation["outcome"] == "unknown"
    assert len(operation["dependencies"]) == len(transport.received_ids)
    assert [dependency["response_id"] for dependency in operation["dependencies"]] == transport.received_ids
    if fault == "invalid_id":
        assert transport.counts["BatchSearchPlan"] == 3
    elif fault == "malformed_extraction":
        assert transport.counts["AtomicFacts"] == 2


def test_native_unparseable_judge_masks_the_row_and_preserves_the_saved_answer():
    body, row, transport = _card_http_verification("exhausted_malformed")

    assert row["response"] == body["response"]
    assert row["mask_sample"] is True
    assert row["failure_kind"] == row["_ng_failure_class"] == "judge_failed"
    assert transport.counts == {"AtomicFacts": 2}
    (invocation,) = row["ng_verifier_invocations"]
    assert invocation["status"] == "failed"
    (operation,) = invocation["grading_operations"]
    assert operation["outcome"] == "failed"
    assert len(operation["dependencies"]) == 2


def test_failed_card_batch_joins_cancelled_judge_siblings_before_verify_returns():
    body, row, transport = _card_http_verification("failed_batch")

    assert row["response"] == body["response"]
    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"
    assert transport.sibling_started.is_set()
    assert transport.sibling_finished_at_route_return is True
    (invocation,) = row["ng_verifier_invocations"]
    assert invocation["grading_operations"][0]["outcome"] == "failed"


def test_native_pairing_sends_one_unchanged_policy_prompt_without_tool_grants():
    import httpx
    from omegaconf import OmegaConf
    from pydantic import BaseModel, Field

    from environment_servers.single_agent_turn_legacy.app import (
        SingleAgentTurnLegacyEnvironmentServer,
        SingleAgentTurnLegacyEnvironmentServerConfig,
    )
    from nemo_gym.client import ServerClient
    from nemo_gym.core import current_rollout_id
    from resources_servers.longfact.verifier_fixture import _response, _server_factory
    from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig

    class Reply:
        def __init__(self, payload, *, status=200, cookies=None, headers=None):
            self.payload = payload
            self.status = status
            self.ok = status < 400
            self.cookies = cookies or {}
            self.headers = headers or {}

        async def read(self):
            return self.payload

    class RoutingClient(ServerClient):
        apps: dict[str, object] = Field(default_factory=dict)
        calls: list[tuple[str, str, dict]] = Field(default_factory=list)
        policy_requests: list[dict] = Field(default_factory=list)

        def _resolve_base_url(self, server_name):
            return f"http://{server_name}"

        async def post(self, *, server_name, url_path, json, **kwargs):
            payload = json.model_dump(mode="json", exclude_unset=True) if isinstance(json, BaseModel) else json
            self.calls.append((server_name, url_path, payload))
            if server_name == "policy_model":
                self.policy_requests.append(payload)
                return Reply(_response("The saved policy answer.").model_dump_json().encode())
            rollout_id = current_rollout_id()
            if rollout_id and server_name == "longfact" and not url_path.startswith("/ng-rollout/"):
                url_path = f"/ng-rollout/{rollout_id}{url_path}"
            transport = httpx.ASGITransport(app=self.apps[server_name])
            async with httpx.AsyncClient(transport=transport, base_url=f"http://{server_name}") as client:
                response = await client.post(
                    url_path, json=payload, cookies=kwargs.get("cookies"), headers=kwargs.get("headers")
                )
            assert response.status_code == 200, response.text
            return Reply(
                response.content,
                status=response.status_code,
                cookies={key: SimpleNamespace(value=value) for key, value in response.cookies.items()},
                headers=response.headers,
            )

    async def run():
        client = RoutingClient(
            head_server_config={"host": "head", "port": 1},
            global_config_dict=OmegaConf.create(
                {
                    "observability_enabled": True,
                    "longfact": {"resources_servers": {"longfact": {"entrypoint": "app.py"}}},
                    "agent": {
                        "responses_api_agents": {
                            "simple_agent": {
                                "entrypoint": "app.py",
                                "model_server": {"type": "responses_api_models", "name": "policy_model"},
                            }
                        }
                    },
                }
            ),
        )
        resources = _server_factory()
        resources.server_client.label = "SUPPORTED"
        resources.server_client = client
        agent = SimpleAgent(
            config=SimpleAgentConfig(
                name="agent",
                host="agent",
                port=8001,
                entrypoint="app.py",
                max_steps=1,
                execute_tools=False,
                model_server={"type": "responses_api_models", "name": "policy_model"},
                resources_server={"type": "resources_servers", "name": "longfact"},
            ),
            server_client=client,
        )
        client.apps = {"longfact": resources.setup_webserver(), "agent": agent.setup_webserver()}
        environment = SingleAgentTurnLegacyEnvironmentServer(
            config=SingleAgentTurnLegacyEnvironmentServerConfig(
                name="environment",
                host="environment",
                port=8002,
                entrypoint="app.py",
                resources_server={"type": "resources_servers", "name": "longfact"},
                agent_server={"type": "responses_api_agents", "name": "agent"},
                resources_tool_transports=[],
                tool_servers=[],
                default_episode_timeout_seconds=None,
            ),
            server_client=client,
        )
        params = {
            "input": [{"role": "user", "content": "Tell me about the Eiffel Tower."}],
            "temperature": 0.25,
            "max_output_tokens": 1234,
        }
        row = {
            "responses_create_params": params,
            "id": "published-task",
            "task_id": "published-task",
            "_ng_rollout_id": "longfact-policy-test",
            "benchmark_variant": BENCHMARK_VARIANT,
            "prompt": params["input"][0]["content"],
            "source_provenance": {"source_path": "published-source.jsonl", "row": 7},
        }
        result = await environment.run_legacy(row)
        return client, params, result

    client, params, result = asyncio.run(run())

    assert result["reward"] == 1.0
    assert len(client.policy_requests) == 1
    policy_request = client.policy_requests[0]
    assert policy_request["input"] == params["input"]
    assert policy_request["temperature"] == params["temperature"]
    assert policy_request["max_output_tokens"] == params["max_output_tokens"]
    assert policy_request.get("tools", []) == []
    agent_seeds = [body for name, path, body in client.calls if name == "agent" and path == "/v1/agent_sessions"]
    assert len(agent_seeds) == 1
    assert agent_seeds[0]["tool_accesses"] == []
    assert [path for name, path, _body in client.calls if name == "longfact"] == [
        "/seed_session",
        "/verify",
        "/close_session",
    ]
    assert result["info"]["source_provenance"] == {"source_path": "published-source.jsonl", "row": 7}
    (invocation,) = result["ng_verifier_invocations"]
    assert invocation["grading_operations"][0]["outcome"] == "completed"
