# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import ClientResponseError
from pydantic import BaseModel, Field

import resources_servers.longfact.evaluator as evaluator_module
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.longfact.evaluator import (
    EvaluatorIdentityError,
    EvaluatorResponseError,
    JsonEvaluator,
)


class SearchDecision(BaseModel):
    reasoning: str
    continue_searching: bool = True
    search_query: str | None = None
    relevant_urls: list[str] = Field(default_factory=list)


def test_free_form_json_parser_accepts_markdown_fence_and_canonical_planner_fields():
    parsed = JsonEvaluator._parse_json(
        SearchDecision,
        (
            "Here is the result:\n```json\n"
            '{"reasoning":"Need the primary authority",'
            '"continue_searching":true,'
            '"search_query":"Nix v Williams",'
            '"relevant_urls":[]}'
            "\n```"
        ),
    )

    assert parsed == SearchDecision(
        reasoning="Need the primary authority",
        continue_searching=True,
        search_query="Nix v Williams",
        relevant_urls=[],
    )


def test_free_form_json_parser_repairs_invalid_latex_escape():
    parsed = JsonEvaluator._parse_json(
        SearchDecision,
        (
            r'{"reasoning":"Find the equation",'
            r'"continue_searching":true,'
            r'"search_query":"pressure \alpha relation",'
            r'"relevant_urls":[]}'
        ),
    )

    assert parsed.search_query == r"pressure \alpha relation"


@pytest.mark.parametrize(
    "command",
    [
        r"\frac{1}{2}",
        r"\upsilon",
        r"\underline{x}",
        r"\nu",
        r"\theta",
        r"\beta",
        r"\to",
        r"\textbf{x}",
        r"\tan",
        r"\neg",
        r"\ne",
        r"\notin",
        r"\rm",
        r"\forall",
        r"\bf",
        r"\bullet",
    ],
)
def test_free_form_json_parser_preserves_latex_control_words(command):
    parsed = JsonEvaluator._parse_json(
        SearchDecision,
        '{"reasoning":"Find the equation","search_query":"' + command + '"}',
    )

    assert parsed.search_query == command


def test_escape_repair_preserves_valid_json_escapes_and_literal_backslashes():
    payload = r'{"reasoning":"Quoted \"text\" and slash \/","search_query":"line\nnext, tab\ttext, Greek \u03b1, literal \\frac{1}{2}, \\upsilon"}'

    parsed = JsonEvaluator._parse_json(SearchDecision, payload)

    assert parsed.search_query == "line\nnext, tab\ttext, Greek α, literal \\frac{1}{2}, \\upsilon"
    assert parsed.reasoning == 'Quoted "text" and slash /'


def test_free_form_json_parser_rejects_unfenced_commentary():
    with pytest.raises(ValueError, match="parseable JSON"):
        JsonEvaluator._parse_json(
            SearchDecision,
            (
                "Here is the result: "
                '{"reasoning":"Search","continue_searching":true,'
                '"search_query":"authority","relevant_urls":[]}'
            ),
        )


@pytest.mark.parametrize(
    "response",
    [
        '["The notation [] means an empty array.',
        '["The notation {} means an empty object.',
        '"The notation [] means an empty array.',
        'Here are the claims: "The notation [] means an empty array.',
        'Here are the claims: ["The notation [] means an empty array.',
        '{"claims": ["The notation [] means an empty array.',
        '["The notation ```[]``` denotes an empty array.',
        '"The notation ```[]``` denotes an empty array.',
        'Here are the claims: ["The notation ```[]``` denotes an empty array.',
        'Here are the claims: "The notation ```[]``` denotes an empty array.',
    ],
)
def test_embedded_json_recovery_rejects_a_malformed_outer_payload(response):
    with pytest.raises(ValueError, match="parseable JSON"):
        JsonEvaluator._load_json(response, salvage_embedded_json=True)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("[]", []),
        ('["The notation [] means an empty array."]', ["The notation [] means an empty array."]),
        ('Claims: ["The notation [] means an empty array."] Done.', ["The notation [] means an empty array."]),
        (
            '["The Markdown notation ```[]``` denotes an empty array."]',
            ["The Markdown notation ```[]``` denotes an empty array."],
        ),
        (
            '```json\n["The Markdown notation ```[]``` denotes an empty array."]\n```',
            ["The Markdown notation ```[]``` denotes an empty array."],
        ),
        (
            '```\n["The Markdown notation ```[]``` denotes an empty array."]\n```',
            ["The Markdown notation ```[]``` denotes an empty array."],
        ),
        ('```json\n["A complete claim."]', ["A complete claim."]),
        (
            'Here is the verdict: {"label":"SUPPORTED","explanation":"Evidence agrees."} Thanks.',
            {"label": "SUPPORTED", "explanation": "Evidence agrees."},
        ),
    ],
)
def test_embedded_json_recovery_preserves_complete_outer_payloads(response, expected):
    assert JsonEvaluator._load_json(response, salvage_embedded_json=True) == expected


def test_judgment_parser_mode_does_not_repair_invalid_escapes():
    with pytest.raises(ValueError, match="parseable JSON"):
        JsonEvaluator._parse_json(
            SearchDecision,
            (
                r'{"reasoning":"Find \alpha","continue_searching":true,'
                r'"search_query":"authority","relevant_urls":[]}'
            ),
            sanitize_json=False,
        )


def test_free_form_json_parser_defaults_missing_continue_searching_to_true():
    parsed = JsonEvaluator._parse_json(
        SearchDecision,
        '{"reasoning":"No decision","search_query":"query","relevant_urls":[]}',
    )

    assert parsed.continue_searching is True


class _CreateParams(NeMoGymResponseCreateParamsNonStreaming):
    def __init__(self, *, temperature=None, top_p=None):
        super().__init__(
            input=[],
            text={"format": {"type": "json_schema", "name": "inherited_constraint", "schema": {}}},
            temperature=temperature,
            top_p=top_p,
        )


def _wire_response(text, model="judge-snapshot"):
    return {
        "id": "fixture-response",
        "created_at": 0,
        "model": model,
        "object": "response",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "fixture-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


_DECISION = '{"reasoning":"Enough evidence","continue_searching":false,"search_query":null,"relevant_urls":[]}'


class _Response:
    ok = True

    def __init__(self, payload):
        self.payload = payload

    async def read(self):
        return json.dumps(self.payload).encode()


class _CapturingClient:
    def __init__(self, responses=None):
        self.responses = [_wire_response(_DECISION)] if responses is None else list(responses)
        self.requests = []
        self.wire_requests = []

    async def post(self, **kwargs):
        self.requests.append(kwargs)
        self.wire_requests.append(kwargs["json"].model_dump(exclude_unset=True))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Response(response)


def _evaluator(client, *, attempts=1, identity=None, temperature=None, top_p=None):
    return JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(temperature=temperature, top_p=top_p),
        max_concurrency=1,
        max_attempts=attempts,
        timeout_seconds=1,
        expected_model_identity=identity,
    )


@pytest.mark.parametrize(("temperature", "top_p"), [(None, None), (0.1, 0.8), (0.7, 0.8)])
def test_json_transport_preserves_prompt_and_omits_disabled_guided_decoding(temperature, top_p):
    client = _CapturingClient()

    parsed = asyncio.run(
        _evaluator(client, temperature=temperature, top_p=top_p).generate_json(
            SearchDecision, "Exact LongFact judge prompt"
        )
    )

    params = client.requests[0]["json"]
    assert parsed.reasoning == "Enough evidence"
    assert parsed.continue_searching is False
    assert len(params.input) == 1
    assert params.input[0].role == "user"
    assert params.input[0].content == "Exact LongFact judge prompt"
    assert params.text is None
    assert "text" not in client.wire_requests[0]
    assert "reasoning" not in client.wire_requests[0]
    if temperature is None:
        assert "temperature" not in client.wire_requests[0]
        assert "top_p" not in client.wire_requests[0]
    else:
        assert client.wire_requests[0]["temperature"] == temperature
        assert client.wire_requests[0]["top_p"] == top_p
    assert client.requests[0]["_max_num_tries"] == 1
    assert client.requests[0]["timeout"] == 1


def test_generate_json_does_not_resample_a_malformed_completed_response():
    client = _CapturingClient([_wire_response("not json")])

    with pytest.raises(ValueError, match="parseable JSON"):
        asyncio.run(_evaluator(client, attempts=3).generate_json(SearchDecision, "Judge this claim"))

    assert len(client.requests) == 1


@pytest.mark.parametrize(
    ("status", "reason"), [("incomplete", "content_filter"), ("incomplete", "max_output_tokens"), (None, None)]
)
def test_unusable_judge_response_cannot_supply_a_parseable_grade(status, reason):
    from resources_servers.longfact.app import AtomicFacts, LongFactSafeServer

    response = _wire_response('["A complete-looking partial claim."]')
    response.update(status=status, incomplete_details={"reason": reason} if reason else None)
    client = _CapturingClient([response])
    server = SimpleNamespace(
        config=SimpleNamespace(evaluator_max_attempts=3, evaluator_constrain_json_schema=False),
        _evaluator=_evaluator(client, attempts=3),
    )

    with pytest.raises(EvaluatorResponseError, match=f"fixture-response.*{status}.*{reason}"):
        asyncio.run(LongFactSafeServer._generate_json(server, AtomicFacts, "Extract the claims."))

    assert len(client.requests) == 1


def test_malformed_claim_with_backticks_exhausts_extraction_instead_of_scoring_zero_claims():
    from resources_servers.longfact.app import AtomicFacts, EvaluatorParseError, LongFactSafeServer

    malformed = '["The notation ```[]``` denotes an empty array.'
    client = _CapturingClient([_wire_response(malformed), _wire_response(malformed)])
    server = SimpleNamespace(
        config=SimpleNamespace(evaluator_max_attempts=2, evaluator_constrain_json_schema=False),
        _evaluator=_evaluator(client),
    )

    with pytest.raises(EvaluatorParseError, match="malformed JSON 2 times"):
        asyncio.run(LongFactSafeServer._generate_json(server, AtomicFacts, "Extract the claims."))

    assert len(client.requests) == 2


@pytest.mark.parametrize("model", ["judge-snapshot", "other-judge"])
def test_pinned_model_identity_is_checked_without_retrying(model):
    client = _CapturingClient([_wire_response(_DECISION, model=model)])
    evaluator = _evaluator(client, attempts=3, identity="judge-snapshot")

    if model == "judge-snapshot":
        parsed = asyncio.run(evaluator.generate_json(SearchDecision, "Judge this claim"))
        assert parsed.reasoning == "Enough evidence"
    else:
        with pytest.raises(EvaluatorIdentityError, match="model identity mismatch"):
            asyncio.run(evaluator.generate_json(SearchDecision, "Judge this claim"))

    assert len(client.requests) == 1
    assert evaluator.response_model_validation == "exact"


def test_unknown_model_identity_is_explicitly_not_enforced():
    client = _CapturingClient([_wire_response(_DECISION, model="any-judge")])
    evaluator = _evaluator(client, identity="unknown")

    parsed = asyncio.run(evaluator.generate_json(SearchDecision, "Judge this claim"))

    assert parsed.reasoning == "Enough evidence"
    assert evaluator.response_model_validation == "not_enforced"


def test_json_request_retries_transport_failures_with_jitter_and_backoff(monkeypatch):
    sleeps = []
    jitter_values = iter((0.1, 0.25, 0.1))

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(evaluator_module.random, "uniform", lambda _low, _high: next(jitter_values))
    client = _CapturingClient([RuntimeError("transient transport failure"), _wire_response(_DECISION)])

    parsed = asyncio.run(_evaluator(client, attempts=2).generate_json(SearchDecision, "Judge this claim"))

    assert parsed.reasoning == "Enough evidence"
    assert len(client.requests) == 2
    assert sleeps == [0.1, 1.25, 0.1]


def _http_error(status):
    return ClientResponseError(
        SimpleNamespace(real_url="https://judge.invalid/v1/responses"), (), status=status, message="fixture error"
    )


def test_http_400_is_not_retried(monkeypatch):
    sleeps = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(evaluator_module.random, "uniform", lambda _low, _high: 0.1)
    client = _CapturingClient([_http_error(400)])

    with pytest.raises(EvaluatorResponseError, match="rejected with HTTP 400"):
        asyncio.run(_evaluator(client, attempts=3).generate_json(SearchDecision, "Judge this claim"))

    assert len(client.requests) == 1
    assert sleeps == [0.1]


def test_http_429_remains_retryable(monkeypatch):
    sleeps = []
    jitter_values = iter((0.1, 0.25, 0.1))

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(evaluator_module.random, "uniform", lambda _low, _high: next(jitter_values))
    client = _CapturingClient([_http_error(429), _wire_response(_DECISION)])

    parsed = asyncio.run(_evaluator(client, attempts=2).generate_json(SearchDecision, "Judge this claim"))

    assert parsed.reasoning == "Enough evidence"
    assert len(client.requests) == 2
    assert sleeps == [0.1, 1.25, 0.1]


def _retrieval_failure_server(mode, error_type):
    from resources_servers.longfact.verifier_fixture import _JudgeResponse, _response, _server_factory

    class JudgeTransport:
        global_config_dict = {"observability_enabled": True}

        def __init__(self):
            self.requests = []

        async def post(self, *, json: NeMoGymResponseCreateParamsNonStreaming, **kwargs):
            from json import dumps

            schema = json.text["format"]["name"]
            payloads = {
                "AtomicFacts": ["A factual claim."],
                "PreparedFact": {"self_contained_fact": "A factual claim.", "relevant": True, "queries": ["evidence"]},
                "ThreeWayClaimJudgment": {"label": "SUPPORTED", "explanation": "Evidence agrees."},
                "BatchSearchPlan": {"claims": [{"claim_id": "C1", "queries": ["evidence"]}]},
                "CardBatchVerdicts": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "claim": "A factual claim.",
                            "answer": "true",
                            "reasoning": "Evidence agrees.",
                            "supporting_evidence": [],
                        }
                    ]
                },
            }
            self.requests.append({"schema": schema, **kwargs})
            response = _response(dumps(payloads[schema]))
            response.id = f"recovery-judge-{len(self.requests)}"
            return _JudgeResponse(response.model_dump_json().encode())

    class SearchBoundary:
        available = False
        cache_stats = {}

        async def search(self, query):
            if not self.available:
                raise error_type("temporary retrieval outage")
            return [
                SimpleNamespace(
                    title="Evidence", url="https://example.test/evidence", snippet="A factual claim.", search_score=1.0
                )
            ]

    server = _server_factory()
    judge = JudgeTransport()
    server.server_client = judge
    server.config.fact_check_mode = mode
    if mode == "gpt5_card_batch":
        server.config.claim_extraction_prompt = "claim_listing_v1"
    server._evaluator = JsonEvaluator(
        server_client=judge,
        model_server=server.config.evaluator_model_server,
        create_params=server.config.evaluator_responses_create_params,
        max_concurrency=1,
        max_attempts=1,
        timeout_seconds=1,
    )
    provider = SearchBoundary()
    server._search_provider = provider
    body = {
        "responses_create_params": {"input": [{"role": "user", "content": "Question"}]},
        "response": _response("The original saved policy answer.").model_dump(mode="json"),
        "id": "source-task",
        "task_id": "source-task",
        "benchmark_variant": "longfact-safe-v1",
        "prompt": "Question",
        "source_provenance": {"path": "published-source.jsonl", "row_index": 7},
    }
    return server, provider, judge, body


@pytest.mark.parametrize("mode", ["per_claim", "gpt5_card_batch"])
@pytest.mark.parametrize("backend", ["tavily", "tantivy", "milvus"])
def test_known_retrieval_failure_recovers_the_saved_answer_without_policy_generation(
    tmp_path, monkeypatch, mode, backend
):
    import httpx
    from omegaconf import OmegaConf

    from nemo_gym.eval import rollout_reverification as recovery
    from resources_servers.longfact.milvus_provider import MilvusProviderError
    from resources_servers.longfact.tantivy_provider import TantivyProviderError
    from resources_servers.longfact.tavily_provider import TavilyProviderError

    errors = {"tavily": TavilyProviderError, "tantivy": TantivyProviderError, "milvus": MilvusProviderError}

    async def run():
        server, provider, judge, body = _retrieval_failure_server(mode, errors[backend])
        server.config.search_backend = backend
        transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            failed_response = await client.post("/ng-rollout/retrieval-failure/verify", json=body)
            assert failed_response.status_code == 200
            failed = failed_response.json()
            assert failed["mask_sample"] is True
            assert failed["failure_kind"] == failed["_ng_failure_class"] == "judge_failed"
            assert failed["response"] == body["response"]
            assert failed["source_provenance"] == body["source_provenance"]
            source = {key: value for key, value in body.items() if key != "response"}
            source |= {"_ng_task_index": 0, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}}
            settled_input = source | {"_ng_task_index": 1, "id": "settled-task", "task_id": "settled-task"}
            success = settled_input | {"response": body["response"] | {"id": "settled-response"}, "reward": 0.5}
            failed |= {"_ng_task_index": 0, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}}
            files = {
                "inputs.jsonl": [source, settled_input],
                "rollouts.jsonl": [success],
                "rollouts_failures.jsonl": [failed],
            }
            for name, rows in files.items():
                (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            original = {name: (tmp_path / name).read_bytes() for name in files}

            class RecoveryTransport:
                global_config_dict = OmegaConf.create(
                    {
                        "agent": {
                            "responses_api_agents": {
                                "simple_agent": {"resources_server": {"type": "resources_servers", "name": "longfact"}}
                            }
                        }
                    }
                )

                def __init__(self):
                    self.dispatched = []

                async def post(self, *, server_name, url_path, json, **kwargs):
                    assert (server_name, url_path) == ("longfact", "/verify")
                    self.dispatched.append(json)
                    response = await client.post("/ng-rollout/retrieval-recovery/verify", json=json)

                    class Reply:
                        ok = response.status_code < 400
                        status = response.status_code
                        headers = response.headers

                        async def read(self):
                            return response.content

                    return Reply()

            recovered_transport = RecoveryTransport()
            monkeypatch.setattr(recovery, "setup_server_client", lambda: recovered_transport)
            monkeypatch.setattr(recovery, "get_exporters", list)
            provider.available = True
            results = await recovery.RolloutReverificationHelper().run_from_config(
                recovery.RolloutReverificationConfig(
                    materialized_inputs_jsonl_fpath=str(tmp_path / "inputs.jsonl"),
                    rollouts_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
                    output_jsonl_fpath=str(tmp_path / "recovered.jsonl"),
                    judge_failed_only=True,
                    disable_aggregation=True,
                    resume_from_cache=False,
                )
            )
        return results, recovered_transport, body, success, original, judge

    results, transport, body, success, original, judge = asyncio.run(run())
    assert len(transport.dispatched) == 1
    assert transport.dispatched[0]["response"] == body["response"]
    assert transport.dispatched[0]["prompt"] == body["prompt"]
    assert transport.dispatched[0]["source_provenance"] == body["source_provenance"]
    rows = {row["_ng_task_index"]: row for row in results}
    assert rows[1] == success
    assert rows[0]["response"] == body["response"]
    assert rows[0]["reward"] == pytest.approx(2 / 101)
    assert not rows[0]["mask_sample"]
    assert "_ng_failure_class" not in rows[0]
    assert "ng_trajectory" not in rows[0]
    (invocation,) = rows[0]["ng_verifier_invocations"]
    assert invocation["status"] == "completed"
    (operation,) = invocation["grading_operations"]
    assert operation["outcome"] == "completed"
    assert all(request["server_name"] == "offline_judge" for request in judge.requests)
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())


@pytest.mark.parametrize("mode", ["per_claim", "gpt5_card_batch"])
def test_retrieval_programming_error_remains_a_verifier_error(mode):
    async def run():
        import httpx

        server, _provider, _judge, body = _retrieval_failure_server(mode, ValueError)
        transport = httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/ng-rollout/retrieval-programming-error/verify", json=body)

    response = asyncio.run(run())
    assert response.status_code == 500
    row = response.json()
    assert "_ng_failure_class" not in row
    (invocation,) = row["ng_verifier_invocations"]
    assert invocation["failure_kind"] == "verifier_error"
