# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import ClientResponseError
from pydantic import BaseModel, Field

import resources_servers.longfact.evaluator as evaluator_module
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
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
    def __init__(self):
        super().__init__(
            input=[],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "inherited_constraint",
                    "schema": {},
                }
            },
            reasoning=None,
        )


class _CapturingClient:
    def __init__(self):
        self.requests = []
        self.wire_requests = []

    async def post(self, **kwargs):
        self.requests.append(kwargs)
        self.wire_requests.append(kwargs["json"].model_dump(exclude_unset=True))
        return object()


class _FakeResponseModel:
    @classmethod
    def model_validate(cls, payload):
        return payload


def test_generate_json_does_not_append_a_schema_instruction_by_default(monkeypatch):
    async def fake_raise_for_status(response):
        return None

    async def fake_get_response_json(response):
        return {
            "reasoning": {"effort": "minimal"},
            "text": (
                '{"reasoning":"Enough evidence","continue_searching":false,'
                '"search_query":null,"relevant_urls":["[1.1]"]}'
            ),
        }

    monkeypatch.setattr(evaluator_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(evaluator_module, "get_response_json", fake_get_response_json)
    monkeypatch.setattr(evaluator_module, "NeMoGymResponse", _FakeResponseModel)
    monkeypatch.setattr(evaluator_module, "extract_text", lambda response: response["text"])

    client = _CapturingClient()
    evaluator = JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=1,
        timeout_seconds=1,
    )

    parsed = asyncio.run(
        evaluator.generate_json(
            SearchDecision,
            "Exact upstream system prompt",
            "Exact upstream user prompt",
            reasoning_effort="minimal",
        )
    )

    params = client.requests[0]["json"]
    assert parsed.continue_searching is False
    assert params.input[0].content == "Exact upstream system prompt"
    assert "OUTPUT FORMAT" not in params.input[0].content
    assert params.input[1].content == "Exact upstream user prompt"
    assert params.text is None
    assert "text" not in client.wire_requests[0]
    assert params.reasoning == {"effort": "minimal"}
    assert client.requests[0]["_max_num_tries"] == 1
    assert client.requests[0]["timeout"] == 1


def test_generate_text_omits_inherited_text_constraint_from_wire(monkeypatch):
    async def fake_raise_for_status(response):
        return None

    async def fake_get_response_json(response):
        return {
            "reasoning": {"effort": "minimal"},
            "text": "follow-up question?",
        }

    monkeypatch.setattr(evaluator_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(evaluator_module, "get_response_json", fake_get_response_json)
    monkeypatch.setattr(evaluator_module, "NeMoGymResponse", _FakeResponseModel)
    monkeypatch.setattr(evaluator_module, "extract_text", lambda response: response["text"])

    client = _CapturingClient()
    evaluator = JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=1,
        timeout_seconds=1,
    )

    result = asyncio.run(
        evaluator.generate_text(
            "Exact upstream follow-up prompt",
            reasoning_effort="minimal",
        )
    )

    params = client.requests[0]["json"]
    assert result == "follow-up question?"
    assert params.text is None
    assert "text" not in client.wire_requests[0]
    assert params.reasoning == {"effort": "minimal"}


def test_generate_json_does_not_resample_a_malformed_completed_response(monkeypatch):
    async def fake_raise_for_status(response):
        return None

    async def fake_get_response_json(response):
        return {"reasoning": {"effort": "minimal"}, "text": "not json"}

    monkeypatch.setattr(evaluator_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(evaluator_module, "get_response_json", fake_get_response_json)
    monkeypatch.setattr(evaluator_module, "NeMoGymResponse", _FakeResponseModel)
    monkeypatch.setattr(evaluator_module, "extract_text", lambda response: response["text"])

    client = _CapturingClient()
    evaluator = JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=3,
        timeout_seconds=1,
    )

    with pytest.raises(ValueError, match="parseable JSON"):
        asyncio.run(
            evaluator.generate_json(
                SearchDecision,
                None,
                "Exact upstream planner prompt",
                reasoning_effort="minimal",
            )
        )

    assert len(client.requests) == 1


def test_pinned_model_identity_is_checked_without_retrying(monkeypatch):
    async def fake_raise_for_status(response):
        return None

    async def fake_get_response_json(response):
        return {
            "model": "gpt-5-mini",
            "text": (
                '{"reasoning":"Enough evidence","continue_searching":false,"search_query":null,"relevant_urls":[]}'
            ),
        }

    class FakeResponseModel:
        @classmethod
        def model_validate(cls, payload):
            return SimpleNamespace(**payload)

    monkeypatch.setattr(evaluator_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(evaluator_module, "get_response_json", fake_get_response_json)
    monkeypatch.setattr(evaluator_module, "NeMoGymResponse", FakeResponseModel)
    monkeypatch.setattr(evaluator_module, "extract_text", lambda response: response.text)

    client = _CapturingClient()
    evaluator = JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=3,
        timeout_seconds=1,
        expected_model_identity="gpt-5-mini-2025-08-07",
    )

    with pytest.raises(EvaluatorIdentityError, match="model identity mismatch"):
        asyncio.run(
            evaluator.generate_json(
                SearchDecision,
                None,
                "Exact upstream planner prompt",
                reasoning_effort="minimal",
            )
        )

    assert len(client.requests) == 1
    assert evaluator.response_model_validation == "exact"


def test_unknown_model_identity_is_explicitly_not_enforced():
    evaluator = JsonEvaluator(
        server_client=_CapturingClient(),
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=1,
        timeout_seconds=1,
        expected_model_identity="unknown",
    )

    evaluator._validate_model_identity(SimpleNamespace(model="any-model"))

    assert evaluator.response_model_validation == "not_enforced"
    assert evaluator.response_reasoning_validation == "exact"


def test_response_reasoning_identity_is_checked():
    with pytest.raises(EvaluatorIdentityError, match="reasoning identity mismatch"):
        JsonEvaluator._validate_reasoning_identity(
            SimpleNamespace(reasoning=SimpleNamespace(effort="medium")),
            "minimal",
        )


def test_request_retry_matches_upstream_jitter_and_backoff(monkeypatch):
    sleeps = []
    jitter_values = iter((0.1, 0.25, 0.1))

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def fake_raise_for_status(response):
        return None

    async def fake_get_response_json(response):
        return {
            "reasoning": {"effort": "minimal"},
            "text": "follow-up question?",
        }

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        async def post(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return object()

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        evaluator_module.random,
        "uniform",
        lambda _low, _high: next(jitter_values),
    )
    monkeypatch.setattr(evaluator_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(evaluator_module, "get_response_json", fake_get_response_json)
    monkeypatch.setattr(evaluator_module, "NeMoGymResponse", _FakeResponseModel)
    monkeypatch.setattr(evaluator_module, "extract_text", lambda response: response["text"])
    client = FlakyClient()
    evaluator = JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=2,
        timeout_seconds=1,
    )

    result = asyncio.run(evaluator.generate_text("Exact upstream follow-up prompt", reasoning_effort="minimal"))

    assert result == "follow-up question?"
    assert client.calls == 2
    assert sleeps == [0.1, 1.25, 0.1]


@pytest.mark.parametrize("method", ["json", "text"])
def test_http_400_is_not_retried_like_upstream_bad_request(monkeypatch, method):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def fake_raise_for_status(response):
        del response
        raise ClientResponseError(
            SimpleNamespace(real_url="http://judge/v1/responses"),
            (),
            status=400,
            message="bad request",
        )

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(evaluator_module.random, "uniform", lambda _low, _high: 0.1)
    monkeypatch.setattr(evaluator_module, "raise_for_status", fake_raise_for_status)

    client = _CapturingClient()
    evaluator = JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=3,
        timeout_seconds=1,
    )

    with pytest.raises(EvaluatorResponseError, match="rejected with HTTP 400"):
        if method == "json":
            asyncio.run(
                evaluator.generate_json(
                    SearchDecision,
                    None,
                    "Exact upstream planner prompt",
                    reasoning_effort="minimal",
                )
            )
        else:
            asyncio.run(
                evaluator.generate_text(
                    "Exact upstream follow-up prompt",
                    reasoning_effort="minimal",
                )
            )

    assert len(client.requests) == 1
    assert sleeps == [0.1]


def test_http_429_remains_retryable_like_upstream_rate_limit(monkeypatch):
    sleeps = []
    jitter_values = iter((0.1, 0.25, 0.1))
    status_checks = 0

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def fake_raise_for_status(response):
        nonlocal status_checks
        del response
        status_checks += 1
        if status_checks == 1:
            raise ClientResponseError(
                SimpleNamespace(real_url="http://judge/v1/responses"),
                (),
                status=429,
                message="rate limited",
            )

    async def fake_get_response_json(response):
        del response
        return {
            "reasoning": {"effort": "minimal"},
            "text": "follow-up question?",
        }

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        evaluator_module.random,
        "uniform",
        lambda _low, _high: next(jitter_values),
    )
    monkeypatch.setattr(evaluator_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(evaluator_module, "get_response_json", fake_get_response_json)
    monkeypatch.setattr(evaluator_module, "NeMoGymResponse", _FakeResponseModel)
    monkeypatch.setattr(evaluator_module, "extract_text", lambda response: response["text"])

    client = _CapturingClient()
    evaluator = JsonEvaluator(
        server_client=client,
        model_server=SimpleNamespace(name="judge"),
        create_params=_CreateParams(),
        max_concurrency=1,
        max_attempts=2,
        timeout_seconds=1,
    )

    result = asyncio.run(
        evaluator.generate_text(
            "Exact upstream follow-up prompt",
            reasoning_effort="minimal",
        )
    )

    assert result == "follow-up question?"
    assert len(client.requests) == 2
    assert sleeps == [0.1, 1.25, 0.1]
