# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_reverification import InputRolloutPair, _build_verify_payload
from nemo_gym.server_utils import ServerClient
from nemo_gym.web.judge_evidence import compact_webvoyager_judge_evidence
from nemo_gym.web.models import WebBenchmark, WebTask
from resources_servers.webvoyager_judge.app import (
    WebVoyagerJudgeResourcesServer,
    parse_gemini_verdict,
)
from resources_servers.webvoyager_judge.config import WebVoyagerJudgeConfig
from resources_servers.webvoyager_judge.models import (
    MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS,
    WebVoyagerJudgeRequest,
    WebVoyagerStandardVerifyRequest,
)


def _model_response(text: str) -> dict:
    return {
        "id": "judge-response",
        "created_at": 0.0,
        "model": "judge",
        "object": "response",
        "output": [
            {
                "id": "judge-message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class _FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    @property
    def content(self):
        payload = self.payload

        class _Body:
            async def read(self):
                return json.dumps(payload).encode()

        return _Body()

    async def read(self):
        return json.dumps(self.payload).encode()


def _server(**overrides):
    config_values = {
        "name": "webvoyager_judge",
        "host": "localhost",
        "port": 8002,
        "entrypoint": "app.py",
        "domain": "agent",
        "judge_model_server": ModelServerRef(type="responses_api_models", name="judge-model"),
        "judge_responses_create_params": {"input": "placeholder", "temperature": 0},
    }
    config_values.update(overrides)
    config = WebVoyagerJudgeConfig(
        **config_values,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return WebVoyagerJudgeResourcesServer(config=config, server_client=client)


def _request(answer="answer", *, screenshot_count=1, page_url_count=0):
    return WebVoyagerJudgeRequest(
        task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="site--0",
            intent="Find the requested fact",
            start_urls=["https://example.test"],
        ),
        final_answer=answer,
        screenshots=[f"data:image/png;base64,image-{index}" for index in range(screenshot_count)],
        page_urls=[f"https://example.test/step-{index}" for index in range(page_url_count)],
    )


def _standard_request() -> WebVoyagerStandardVerifyRequest:
    evidence_request = _request()
    return WebVoyagerStandardVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Solve"},
            "response": _model_response("final"),
            "web_task": evidence_request.task.model_dump(mode="json"),
            "final_answer": evidence_request.final_answer,
            "screenshots": evidence_request.screenshots,
            "page_urls": evidence_request.page_urls,
        }
    )


@pytest.mark.asyncio
async def test_judge_exposes_only_standard_stateless_verify_route():
    server = _server()
    paths = {route.path for route in server.setup_webserver().routes}

    assert "/verify" in paths
    assert "/verify_webvoyager" not in paths
    assert await server.get_reverify_mode() == ReverifyMode.STATELESS


@pytest.mark.asyncio
async def test_standard_route_classifies_judge_transport_failure_for_sidecar():
    server = _server()
    server.server_client.post = AsyncMock(side_effect=TimeoutError("judge timed out"))
    verify_endpoint = next(route.endpoint for route in server.setup_webserver().routes if route.path == "/verify")

    response = await verify_endpoint(body=_standard_request())
    payload = json.loads(response.body)

    assert payload["reward"] == 0.0
    assert payload["_ng_failure_class"] == "judge_failed"
    assert "TimeoutError" in payload["_ng_failure_judge_error"]
    assert payload["response"] == _standard_request().response.model_dump(mode="json")


def test_gemini_verdict_requires_json_success_or_failure():
    assert parse_gemini_verdict('{"thought":"ok","verdict":"SUCCESS"}') == (
        True,
        {"thought": "ok", "verdict": "SUCCESS"},
    )
    assert parse_gemini_verdict("SUCCESS") is None
    assert parse_gemini_verdict('{"verdict":"NOT SUCCESS"}') is None


def test_request_contract_accepts_a_full_100_step_trajectory():
    request = _request(screenshot_count=101, page_url_count=101)

    assert len(request.screenshots) == 101
    assert len(request.page_urls) == 101
    assert (
        WebVoyagerStandardVerifyRequest.model_json_schema()["properties"]["screenshots"]["maxItems"]
        == MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS
    )


def test_request_contract_rejects_evidence_above_the_configured_ceiling():
    with pytest.raises(ValidationError, match="List should have at most 200 items"):
        _request(screenshot_count=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS + 1)


def test_standard_request_recovers_evidence_from_saved_rollout_response():
    response = _model_response("final")
    response["webvoyager_judge_evidence"] = {
        "final_answer": "42",
        "screenshots": ["data:image/png;base64,evidence"],
        "page_urls": ["https://example.test/result"],
    }
    request = WebVoyagerStandardVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Solve"},
            "response": response,
            "web_task": _request().task.model_dump(mode="json"),
        }
    )

    assert request.final_answer == "42"
    assert request.screenshots == ["data:image/png;base64,evidence"]
    assert request.page_urls == ["https://example.test/result"]


@pytest.mark.asyncio
async def test_generic_reverify_reconstructs_compact_evidence_without_browser_replay():
    trajectory_image = "data:image/png;base64,trajectory"
    terminal_image = "data:image/png;base64,terminal"
    response = _model_response("final")
    response["output"].insert(
        0,
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": trajectory_image, "detail": "high"}],
        },
    )
    response["webvoyager_judge_evidence"] = compact_webvoyager_judge_evidence(
        response=response,
        final_answer="42",
        screenshots=[trajectory_image, terminal_image],
        page_urls=["https://example.test/start", "https://example.test/result"],
    )
    task = _request().task
    payload = _build_verify_payload(
        InputRolloutPair(
            input={
                "responses_create_params": {"input": "Solve"},
                "web_task": task.model_dump(mode="json"),
            },
            rollout={"response": response},
        )
    )
    request = WebVoyagerStandardVerifyRequest.model_validate(payload)
    assert request.final_answer == "42"
    assert request.screenshots == [trajectory_image, terminal_image]
    assert request.page_urls == ["https://example.test/start", "https://example.test/result"]

    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"retained evidence proves completion","verdict":"SUCCESS"}')
        )
    )
    result = await server.verify(request)

    assert result.reward == 1.0
    assert result.task_success is True
    server.server_client.post.assert_awaited_once()
    judge_call = server.server_client.post.await_args
    assert judge_call.kwargs["server_name"] == "judge-model"
    assert judge_call.kwargs["url_path"] == "/v1/responses"
    assert result.response.model_dump(mode="json") == request.response.model_dump(mode="json")


@pytest.mark.asyncio
async def test_standard_verify_response_does_not_echo_top_level_evidence():
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshot proves completion","verdict":"SUCCESS"}')
        )
    )

    result = await server.verify(_standard_request())
    dumped = result.model_dump(mode="json")

    assert result.reward == 1.0
    assert "screenshots" not in dumped
    assert "page_urls" not in dumped
    assert "final_answer" not in dumped


@pytest.mark.asyncio
async def test_judge_uses_a_full_100_step_trajectory():
    server = _server(max_screenshots=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS)
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(_model_response('{"thought":"enough evidence","verdict":"SUCCESS"}'))
    )

    response = await server._judge_evidence(_request(screenshot_count=101, page_url_count=101))

    assert response.result.valid_sample is True
    assert response.result.metadata["screenshots_used"] == 101
    params = server.server_client.post.await_args.kwargs["json"]
    assert len(params.input[0].content) == 203


@pytest.mark.asyncio
async def test_empty_answer_still_uses_visual_trajectory_evidence():
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshots do not prove completion","verdict":"FAILURE"}')
        )
    )

    response = await server._judge_evidence(_request(answer=""))

    assert response.result.valid_sample is True
    assert response.result.reward == 0.0
    assert response.result.failure_kind is None
    server.server_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_success_returns_binary_reward():
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshot proves completion","verdict":"SUCCESS"}')
        )
    )

    response = await server._judge_evidence(_request())

    assert response.result.valid_sample is True
    assert response.result.reward == 1.0
    assert response.result.task_success is True


@pytest.mark.asyncio
async def test_received_unparseable_verdict_is_a_valid_zero_reward():
    server = _server()
    server.server_client.post = AsyncMock(return_value=_FakeHttpResponse(_model_response("unclear")))

    response = await server._judge_evidence(_request())

    assert response.result.valid_sample is True
    assert response.result.reward == 0.0
    assert response.result.failure_kind == "judge_unparseable"


@pytest.mark.asyncio
async def test_judge_logs_lifecycle_without_screenshot_payload(caplog):
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(
            _model_response('{"thought":"screenshot proves completion","verdict":"SUCCESS"}')
        )
    )

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.webvoyager_judge"):
        response = await server._judge_evidence(_request())

    assert response.result.task_success is True
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=webvoyager_judge_start" in messages
    assert "event=webvoyager_judge_model_start" in messages
    assert "event=webvoyager_judge_model_complete" in messages
    assert "event=webvoyager_judge_complete" in messages
    assert "origins=none" in messages
    assert "data:image/png;base64,image-0" not in messages
    assert "The screenshot proves completion" not in messages
