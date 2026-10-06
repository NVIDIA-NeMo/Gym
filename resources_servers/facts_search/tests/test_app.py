# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.facts_search.app import (
    GRADER_TEMPLATE_SHA256,
    FACTSSearchChatCompletion,
    FACTSSearchConfig,
    FACTSSearchResourcesServer,
    FACTSSearchVerifyRequest,
    extract_final_answer,
    format_brave_results,
    load_grader_template,
    parse_grade,
    strict_grade,
)


def _policy_response(text: str, *, metadata=None, incomplete=False) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=[
            {
                "type": "message",
                "id": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        metadata=metadata,
        incomplete_details={"reason": "max_output_tokens"} if incomplete else None,
    )


class _HTTPResponse:
    status = 200
    ok = True
    cookies = {}

    def __init__(self, payload):
        self.payload = payload

    async def read(self):
        return json.dumps(self.payload).encode()


def _config(**overrides):
    fields = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "facts_search",
        "brave_api_key": "test-key",
        "judge_model_server": {"type": "responses_api_models", "name": "facts_search_judge"},
    }
    fields.update(overrides)
    return FACTSSearchConfig(**fields)


def _server(judge_text="A"):
    client = MagicMock(spec=ServerClient)

    async def post(*, server_name, url_path, json=None, **kwargs):
        assert server_name == "facts_search_judge"
        assert url_path == "/v1/chat/completions"
        assert "Question: q?" in json.messages[0]["content"]
        return _HTTPResponse(
            {
                "id": "judge-id",
                "object": "chat.completion",
                "created": 0,
                "model": "google/gemini-3.5-flash",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": judge_text}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        )

    client.post = AsyncMock(side_effect=post)
    return FACTSSearchResourcesServer(config=_config(), server_client=client), client


def _verify_request(text="Final Answer: Paris", **kwargs):
    return FACTSSearchVerifyRequest(
        id="row",
        problem="q?",
        gold_answer="Paris",
        responses_create_params={"input": [{"role": "user", "content": "q?"}]},
        response=_policy_response(
            text,
            metadata={
                "facts_search_hops": "3",
                "facts_search_queries": "5",
                "facts_search_forced_final": "true",
            },
            **kwargs,
        ),
    )


def test_grader_prompt_is_byte_pinned_to_public_notebook_v3():
    prompt = load_grader_template()
    assert hashlib.sha256(prompt.encode()).hexdigest() == GRADER_TEMPLATE_SHA256
    assert prompt.endswith('Just return the letters "A", "B", or "C", with no text around it.')


def test_openrouter_provisioned_service_tier_is_a_valid_judge_response():
    completion = FACTSSearchChatCompletion.model_validate(
        {
            "id": "judge",
            "object": "chat.completion",
            "created": 0,
            "model": "google/gemini-3.5-flash",
            "service_tier": "provisioned",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "A"}, "finish_reason": "stop"}],
        }
    )
    assert completion.service_tier == "provisioned"


def test_grader_prompt_drift_and_custom_prompt_path_are_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr("resources_servers.facts_search.app.GRADER_TEMPLATE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="prompt drifted"):
        load_grader_template()
    custom = tmp_path / "grader.txt"
    custom.write_text("Question: {question}; Gold: {target}; Prediction: {predicted_answer}")
    server = FACTSSearchResourcesServer(
        config=_config(grader_template_path=str(custom)), server_client=MagicMock(spec=ServerClient)
    )
    assert server._grader_template.startswith("Question:")


@pytest.mark.parametrize(
    "text,public,strict",
    [
        ("A", "A", "A"),
        (" B ", "B", "B"),
        ("C: NOT_ATTEMPTED", "C", "C"),
        ("AB", "A", "UNKNOWN"),
        ("x", "C", "UNKNOWN"),
    ],
)
def test_public_and_strict_grade_parsers(text, public, strict):
    assert parse_grade(text) == public
    assert strict_grade(text) == strict


def test_brave_result_format_matches_public_notebook_and_preserves_all_snippets():
    payload = {
        "web": {
            "results": [
                {"title": "T", "url": "U", "description": "D", "extra_snippets": ["S1", "S2"]},
                {"title": "T2", "url": "U2"},
            ]
        }
    }
    assert format_brave_results(payload) == (
        "Title: T\nURL: U\nDescription: D\nExtra Snippets:\n  - S1\n  - S2\n\n"
        "Title: T2\nURL: U2\nDescription: \nExtra Snippets:\n\n"
    )
    assert format_brave_results({}) == "No results returned."


def test_extract_final_answer_uses_last_message_and_public_marker():
    assert extract_final_answer(_policy_response("reasoning\nFinal Answer: Paris")) == "Paris"
    empty = _policy_response("x").model_copy(update={"output": []})
    assert extract_final_answer(empty) == ""


def test_setup_webserver_registers_the_canonical_tool_route():
    server, _ = _server()
    paths = {route.path for route in server.setup_webserver().routes}
    assert "/brave_search" in paths and "/verify" in paths


class _BraveResponse:
    ok = True
    cookies = {}

    def __init__(self, status, payload=None):
        self.status = status
        self.payload = payload or {}

    async def json(self):
        return self.payload


async def test_brave_search_success_and_request_shape(monkeypatch):
    server, _ = _server()
    fake_request = AsyncMock(
        return_value=_BraveResponse(
            200, {"web": {"results": [{"title": "T", "url": "U", "description": "D", "extra_snippets": []}]}}
        )
    )
    monkeypatch.setattr("resources_servers.facts_search.app.request", fake_request)
    result = await server.brave_search({"query": "  query  "})
    assert result["result"].startswith("Title: T")
    call = fake_request.await_args.kwargs
    assert call["method"] == "GET" and call["params"] == {"q": "  query  ", "count": 5}
    assert call["headers"]["X-Subscription-Token"] == "test-key"


async def test_brave_search_retries_rate_limit_and_rejects_missing_inputs(monkeypatch):
    server, _ = _server()
    fake_request = AsyncMock(side_effect=[_BraveResponse(429), _BraveResponse(200, {})])
    fake_sleep = AsyncMock()
    monkeypatch.setattr("resources_servers.facts_search.app.request", fake_request)
    monkeypatch.setattr("resources_servers.facts_search.app.asyncio.sleep", fake_sleep)
    assert await server.brave_search({"query": "x"}) == {"result": "No results returned."}
    fake_sleep.assert_awaited_once_with(2)
    assert await server.brave_search({"query": ""}) == {"result": "Error: query must be a non-empty string."}
    no_key = FACTSSearchResourcesServer(config=_config(brave_api_key=""), server_client=MagicMock(spec=ServerClient))
    with pytest.raises(RuntimeError, match="no substitute"):
        await no_key.brave_search({"query": "x"})


async def test_brave_search_surfaces_terminal_http_failure(monkeypatch):
    server, _ = _server()
    responses = [_BraveResponse(503), _BraveResponse(503), _BraveResponse(503)]
    monkeypatch.setattr("resources_servers.facts_search.app.request", AsyncMock(side_effect=responses))
    monkeypatch.setattr("resources_servers.facts_search.app.asyncio.sleep", AsyncMock())
    failure = RuntimeError("brave unavailable")
    fake_raise = AsyncMock(side_effect=failure)
    monkeypatch.setattr("resources_servers.facts_search.app.raise_for_status", fake_raise)
    with pytest.raises(RuntimeError, match="brave unavailable"):
        await server.brave_search({"query": "x"})
    fake_raise.assert_awaited_once_with(responses[-1])


async def test_verify_records_current_judge_receipt_and_search_accounting():
    server, client = _server("A")
    result = await server.verify(_verify_request())
    assert result.reward == 1.0 and result.grade_letter == "A" and result.is_correct == 1.0
    assert result.final_answer == "Paris"
    assert result.search_hops == 3 and result.search_queries == 5 and result.forced_final is True
    assert result.judge_valid is True
    assert result.judge_receipt["judge_model"] == "google/gemini-3.5-flash"
    assert client.post.await_count == 1


def test_judge_request_overrides_and_degenerate_metrics():
    server = FACTSSearchResourcesServer(
        config=_config(
            judge_responses_create_params={"input": [], "temperature": 0.0, "top_p": 1.0, "max_output_tokens": 17}
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    params = server._judge_params("prompt")
    assert params.temperature == 0.0 and params.top_p == 1.0 and params.max_tokens == 17
    assert server._bootstrap_f1(["A"])[0] != server._bootstrap_f1(["A"])[0]
    assert server.compute_metrics([])["num_rollouts"] == 0
    assert server.get_key_metrics({"f1": 1.0, "ignored": 2}) == {"f1": 1.0}


async def test_invalid_judge_reply_uses_public_c_fallback_but_is_flagged():
    server, _ = _server("I cannot decide")
    result = await server.verify(_verify_request())
    assert result.reward == 0.0 and result.grade_letter == "C"
    assert result.is_not_attempted == 1.0 and result.is_unknown == 1.0
    assert result.judge_valid is False and result.failure_reason


def test_public_metrics_include_f1_efficiency_and_deterministic_ci():
    server, _ = _server()
    rows = [
        {"grade_letter": "A", "judge_valid": True, "search_hops": 2, "search_queries": 3},
        {"grade_letter": "A", "judge_valid": True, "search_hops": 4, "search_queries": 4},
        {"grade_letter": "B", "judge_valid": True, "search_hops": 3, "search_queries": 5},
        {"grade_letter": "C", "judge_valid": True, "search_hops": 7, "search_queries": 9, "forced_final": True},
        {"grade_letter": "C", "judge_valid": False, "search_hops": 0, "search_queries": 0},
    ]
    metrics = server.compute_metrics([[row] for row in rows])
    assert metrics["accuracy"] == pytest.approx(0.4)
    assert metrics["attempted_accuracy"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["hedging_rate"] == pytest.approx(0.4)
    assert metrics["judge_valid_rate"] == 0.8
    assert metrics["average_searches"] == pytest.approx(16 / 5)
    assert metrics["f1_ci95_low"] <= metrics["f1"] <= metrics["f1_ci95_high"]
