# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.conversational_tool_use_domain_generation.app import (
    FOLLOWUP_INSTRUCTION,
    DomainGenerationAgent,
    DomainGenerationAgentConfig,
    DomainGenerationRunRequest,
)
from responses_api_agents.conversational_tool_use_domain_generation.assets import (
    DOMAIN_PROMPT_PATH,
    archive_prompt_paths,
    load_domain_prompt,
)


INITIAL_PROMPT = "Generate domains."


class FakeHttpResponse:
    ok = True
    status = 200
    cookies: dict = {}

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def chat_completion(text: str, *, suffix: str) -> dict:
    return {
        "id": f"chatcmpl-{suffix}",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"content": text, "refusal": None, "role": "assistant"},
            }
        ],
        "created": 0,
        "model": "domain-model",
        "object": "chat.completion",
    }


def responses_payload() -> dict:
    return {
        "id": "response-bridge",
        "created_at": 0,
        "model": "domain-model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def make_agent(*responses: dict, observability: bool = True) -> DomainGenerationAgent:
    config = DomainGenerationAgentConfig(
        host="",
        port=0,
        entrypoint="app.py",
        name="domain_generation_agent",
        model_server=ModelServerRef(type="responses_api_models", name="domain_model"),
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": observability}
    server_client.post = AsyncMock(side_effect=[FakeHttpResponse(response) for response in responses])
    return DomainGenerationAgent(config=config, server_client=server_client)


def run_request() -> DomainGenerationRunRequest:
    return DomainGenerationRunRequest(
        responses_create_params={"input": [{"role": "user", "content": INITIAL_PROMPT}]},
        **{TASK_INDEX_KEY_NAME: 7, ROLLOUT_INDEX_KEY_NAME: 0},
    )


@pytest.mark.asyncio
async def test_responses_is_one_call_bridge_preserving_caller_parameters() -> None:
    agent = make_agent(responses_payload())
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "Generate."}],
        model="caller-model",
        temperature=0.25,
        top_p=0.75,
        max_output_tokens=123,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/ng-rollout/task-1/v1/responses",
            "headers": [],
            "query_string": b"",
            "path_params": {"rollout_id": "task-1"},
        }
    )

    response = await agent.responses(request, body)

    assert response.id == "response-bridge"
    call = agent.server_client.post.await_args
    assert call.kwargs == {
        "server_name": "domain_model",
        "url_path": "/ng-rollout/task-1/v1/responses",
        "json": body,
    }


@pytest.mark.asyncio
async def test_run_makes_exactly_two_message_only_chat_completions() -> None:
    first_candidates = [
        {
            "name": "Retail",
            "applications": [{"function": "Track an order"}],
            "unvalidated_extra": {"preserved": True},
        }
    ]
    second_candidates = [{"name": "Airlines", "applications": []}]
    agent = make_agent(
        chat_completion(json.dumps(first_candidates), suffix="initial"),
        chat_completion(f"```json\n{json.dumps(second_candidates)}\n```", suffix="followup"),
    )

    result = await agent.run(run_request())

    assert result.reward == 1.0
    assert result.result.candidates == [*first_candidates, *second_candidates]
    assert result.generation_trace.protocol_version == "domain-generation/v1"
    assert result.generation_trace.request_index == 7
    assert [phase.phase for phase in result.generation_trace.phases] == ["initial", "followup"]
    assert [phase.parse_error for phase in result.generation_trace.phases] == [None, None]

    calls = agent.server_client.post.await_args_list
    assert len(calls) == 2
    expected_path = "/ng-rollout/7-0/v1/chat/completions"
    assert [call.kwargs["url_path"] for call in calls] == [expected_path, expected_path]
    assert [call.kwargs["server_name"] for call in calls] == ["domain_model", "domain_model"]

    first_request = calls[0].kwargs["json"]
    second_request = calls[1].kwargs["json"]
    assert first_request.model_dump(exclude_unset=True) == {"messages": [{"role": "user", "content": INITIAL_PROMPT}]}
    expected_followup = INITIAL_PROMPT + "\n\nPreviously brainstormed domains: ['Retail'].\n" + FOLLOWUP_INSTRUCTION
    assert second_request.model_dump(exclude_unset=True) == {
        "messages": [{"role": "user", "content": expected_followup}]
    }


@pytest.mark.asyncio
async def test_parse_failure_returns_empty_batch_and_still_runs_followup() -> None:
    followup_candidates = [{"name": "Telecom", "arbitrary": ["raw", "object"]}]
    agent = make_agent(
        chat_completion("not json", suffix="initial"),
        chat_completion(json.dumps(followup_candidates), suffix="followup"),
        observability=False,
    )

    result = await agent.run(run_request())

    assert result.reward == 1.0
    assert result.result.candidates == followup_candidates
    assert result.generation_trace.phases[0].parsed_value == []
    assert result.generation_trace.phases[0].parse_error
    calls = agent.server_client.post.await_args_list
    assert [call.kwargs["url_path"] for call in calls] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    followup_request = calls[1].kwargs["json"].model_dump(exclude_unset=True)
    assert "Previously brainstormed domains: []" in followup_request["messages"][0]["content"]


def test_agent_routes_are_registered() -> None:
    agent = make_agent()
    routes = {route.path for route in agent.setup_webserver().routes}
    assert {"/run", "/v1/responses", "/aggregate_metrics"}.issubset(routes)


def test_prompt_assets_are_complete_and_whitespace_free() -> None:
    expected_hashes = {
        "domain_generation.txt": "f90c8b57ed564fb8c918b4d2c2d9dc4da537285fe8bcc56500db168a54200211",  # pragma: allowlist secret
        "domains_sample_1.txt": "f0786c302797ef8543a062b41fe3da37094e9863d5ff8adabcd1fca861fe3688",  # pragma: allowlist secret
        "domains_sample_sectors.txt": "0f7d851ffa1502fc24b82758d86731ac52bd5633023a5978f01928cd7211b5bd",  # pragma: allowlist secret
    }
    prompt_paths = (DOMAIN_PROMPT_PATH, *archive_prompt_paths())

    assert {path.name for path in prompt_paths} == set(expected_hashes)
    assert all(not any(character.isspace() for character in path.name) for path in prompt_paths)
    for path in prompt_paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hashes[path.name]
    assert load_domain_prompt() == DOMAIN_PROMPT_PATH.read_text(encoding="utf-8").strip()


def test_run_request_requires_one_string_user_message() -> None:
    agent = make_agent()
    body = DomainGenerationRunRequest(responses_create_params={"input": []})
    with pytest.raises(ValueError, match="exactly one input message"):
        # Validation is synchronous and occurs before the first await in run().
        agent.run(body).send(None)
