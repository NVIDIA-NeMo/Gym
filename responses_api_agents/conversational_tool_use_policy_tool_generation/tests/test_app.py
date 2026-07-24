# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import Request, Response
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.conversational_tool_use_policy_tool_generation.app import (
    PolicyToolGenerationAgent,
    PolicyToolGenerationAgentConfig,
)
from responses_api_agents.conversational_tool_use_policy_tool_generation.models import (
    PolicyToolGenerationRunRequest,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def chat_completion(text: str, completion_id: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"content": text, "refusal": None, "role": "assistant"},
            }
        ],
        "created": 0,
        "model": "test-chat-model",
        "object": "chat.completion",
    }


def responses_payload() -> dict[str, Any]:
    return {
        "id": "response-bridge",
        "created_at": 0.0,
        "model": "test-responses-model",
        "object": "response",
        "output": [
            {
                "id": "message-bridge",
                "content": [{"annotations": [], "text": "bridged", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class FakeHTTPResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict[str, Any], cookies: dict[str, str] | None = None) -> None:
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class QueueClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []
        self.global_config_dict = {"observability_enabled": False}

    async def post(self, **kwargs: Any) -> FakeHTTPResponse:
        self.calls.append(kwargs)
        return FakeHTTPResponse(self.payloads.pop(0))


def agent(client: QueueClient, *, max_retries: int = 0) -> PolicyToolGenerationAgent:
    config = PolicyToolGenerationAgentConfig(
        host="0.0.0.0",
        port=8000,
        entrypoint="app.py",
        name="conversational_tool_use_policy_tool_generation",
        policy_model_server=ModelServerRef(
            type="responses_api_models",
            name="policy_generation_model",
        ),
        judge_model_server=ModelServerRef(
            type="responses_api_models",
            name="policy_tool_judge_model",
        ),
        max_retries=max_retries,
    )
    return PolicyToolGenerationAgent.model_construct(config=config, server_client=client)


def request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 1),
            "scheme": "http",
        }
    )


def run_body() -> PolicyToolGenerationRunRequest:
    return PolicyToolGenerationRunRequest(
        responses_create_params={"input": []},
        profile="general",
        domain={"name": "Order Support", "applications": [{"raw": True}]},
    )


@pytest.mark.asyncio
async def test_internal_run_uses_message_only_chat_calls_and_converts_final_completion() -> None:
    final_tool = '{"name":"lookup","doc":"Lookup","params":null,"returns":null}'
    payloads = [
        chat_completion("<policy>draft</policy>", "policy-v1"),
        chat_completion(f"<tools>{final_tool}</tools>", "tools-v1"),
        chat_completion("<policy>final</policy>", "policy-refine"),
        chat_completion(f"<tools>{final_tool}</tools>", "tools-refine"),
        *[chat_completion("<judgment>true</judgment>", f"cohesion-{index}") for index in range(3)],
        *[chat_completion("<judgment>0</judgment>", f"golden-{index}") for index in range(4)],
    ]
    client = QueueClient(payloads)
    result = await agent(client).run(request("/run"), run_body())

    assert result.reward == 1.0
    assert result.result.accepted is True
    assert result.response.output[0].content[0].text == f"<tools>{final_tool}</tools>"
    assert result.generation_trace.attempts[0].calls[3].response["id"] == "tools-refine"
    assert len(client.calls) == 11
    assert [call["server_name"] for call in client.calls] == ["policy_generation_model"] * 4 + [
        "policy_tool_judge_model"
    ] * 7
    assert all(call["url_path"] == "/v1/chat/completions" for call in client.calls)
    assert all(set(call["json"]) == {"messages"} for call in client.calls)
    assert all(
        call["json"]["messages"][0]["role"] == "user" and isinstance(call["json"]["messages"][0]["content"], str)
        for call in client.calls
    )


@pytest.mark.asyncio
async def test_public_responses_endpoint_transparently_forwards_body() -> None:
    client = QueueClient([responses_payload()])
    server = agent(client)
    body = NeMoGymResponseCreateParamsNonStreaming(
        model="caller-model",
        input=[{"role": "user", "content": "hello"}],
        temperature=0.7,
        top_p=0.8,
        max_output_tokens=91,
    )
    result = await server.responses(request("/v1/responses"), Response(), body)

    assert result.id == "response-bridge"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["server_name"] == "policy_generation_model"
    assert call["url_path"] == "/v1/responses"
    assert call["json"] is body


def test_config_example_and_routes() -> None:
    raw_config = OmegaConf.load(PACKAGE_DIR / "configs" / "conversational_tool_use_policy_tool_generation.yaml")
    inner = OmegaConf.to_container(
        raw_config["conversational_tool_use_policy_tool_generation"]["responses_api_agents"][
            "conversational_tool_use_policy_tool_generation"
        ],
        resolve=True,
    )
    config = PolicyToolGenerationAgentConfig.model_validate(
        inner
        | {
            "host": "0.0.0.0",
            "port": 8000,
            "name": "conversational_tool_use_policy_tool_generation",
        }
    )
    assert config.max_retries == 20
    assert config.policy_model_server.name == "policy_generation_model"
    assert config.judge_model_server.name == "policy_tool_judge_model"

    example = json.loads((PACKAGE_DIR / "data" / "example.jsonl").read_text(encoding="utf-8"))
    parsed = PolicyToolGenerationRunRequest.model_validate(example)
    assert parsed.profile == "general"
    assert parsed.domain.applications[1] == {"any_raw_shape": ["is", "accepted"]}

    routes = {route.path for route in agent(QueueClient([])).setup_webserver().routes}
    assert {"/run", "/v1/responses", "/aggregate_metrics"}.issubset(routes)
