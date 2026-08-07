# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.conversational_tool_use.domain_generation.app import (
    DomainGenerationAgent,
    DomainGenerationAgentConfig,
    DomainGenerationRunRequest,
)
from responses_api_agents.conversational_tool_use.domain_generation.assets import (
    PROMPT_FILENAMES,
    load_domain_prompt,
    load_followup_instruction,
)


INITIAL_PROMPT = "Generate domains."
PACKAGE_DIR = Path(__file__).resolve().parents[1]


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


def make_agent(
    *responses: dict,
    observability: bool = True,
    followup_count: int = 1,
) -> DomainGenerationAgent:
    config = DomainGenerationAgentConfig(
        host="",
        port=0,
        entrypoint="app.py",
        name="domain_generation_agent",
        model_server=ModelServerRef(type="responses_api_models", name="domain_model"),
        followup_count=followup_count,
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
            "name": "Home Services",
            "applications": [{"function": "Schedule a visit"}],
            "unvalidated_extra": {"preserved": True},
        }
    ]
    second_candidates = [{"name": "Event Support", "applications": []}]
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
    expected_followup = (
        INITIAL_PROMPT + "\n\nPreviously brainstormed domains: ['Home Services'].\n" + load_followup_instruction()
    )
    assert second_request.model_dump(exclude_unset=True) == {
        "messages": [{"role": "user", "content": expected_followup}]
    }


@pytest.mark.asyncio
async def test_followup_count_controls_rounds_and_uses_all_prior_names() -> None:
    batches = [
        [{"name": "Home Services"}],
        [{"description": "kept without a name"}, {"name": "Event Support"}],
        [{"name": "Property Management"}],
    ]
    agent = make_agent(
        *(chat_completion(json.dumps(batch), suffix=str(index)) for index, batch in enumerate(batches)),
        followup_count=2,
    )

    result = await agent.run(run_request())

    assert result.result.candidates == [candidate for batch in batches for candidate in batch]
    assert result.generation_trace.protocol_version == "domain-generation/v2"
    assert result.generation_trace.followup_count == 2
    assert [phase.phase for phase in result.generation_trace.phases] == [
        "initial",
        "followup",
        "followup",
    ]
    calls = agent.server_client.post.await_args_list
    assert len(calls) == 3
    assert "Previously brainstormed domains: ['Home Services']" in calls[1].kwargs["json"].messages[0]["content"]
    assert (
        "Previously brainstormed domains: ['Home Services', 'Event Support']"
        in calls[2].kwargs["json"].messages[0]["content"]
    )
    assert result.response.output[0].content[0].text == json.dumps(batches[-1])


@pytest.mark.asyncio
async def test_zero_followups_returns_the_initial_completion() -> None:
    candidates = [{"name": "Home Services"}]
    agent = make_agent(
        chat_completion(json.dumps(candidates), suffix="initial"),
        followup_count=0,
    )

    result = await agent.run(run_request())

    assert result.result.candidates == candidates
    assert result.generation_trace.followup_count == 0
    assert result.generation_trace.protocol_version == "domain-generation/v2"
    assert [phase.phase for phase in result.generation_trace.phases] == ["initial"]
    assert len(agent.server_client.post.await_args_list) == 1
    assert result.response.output[0].content[0].text == json.dumps(candidates)


@pytest.mark.asyncio
async def test_parse_failure_returns_empty_batch_and_still_runs_followup() -> None:
    followup_candidates = [{"name": "Property Management", "arbitrary": ["raw", "object"]}]
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


def test_checked_in_config_exposes_default_followup_count() -> None:
    raw_config = OmegaConf.load(PACKAGE_DIR / "configs" / "conversational_tool_use_domain_generation.yaml")
    inner = OmegaConf.to_container(
        raw_config["conversational_tool_use_domain_generation"]["responses_api_agents"][
            "conversational_tool_use/domain_generation"
        ],
        resolve=True,
    )
    parsed = DomainGenerationAgentConfig.model_validate(
        inner
        | {
            "host": "0.0.0.0",
            "port": 8000,
            "name": "conversational_tool_use_domain_generation",
        }
    )

    assert parsed.followup_count == 1
    with pytest.raises(ValueError):
        DomainGenerationAgentConfig.model_validate(parsed.model_dump() | {"followup_count": -1})
    with pytest.raises(ValueError):
        DomainGenerationAgentConfig.model_validate(parsed.model_dump() | {"followup_cout": 2})


def test_prompt_assets_are_complete_and_whitespace_free() -> None:
    assert PROMPT_FILENAMES == ("domain_followup.txt", "domain_generation.txt")
    assert all(not any(character.isspace() for character in filename) for filename in PROMPT_FILENAMES)
    assert load_domain_prompt() == "Generate domains."
    assert load_followup_instruction() == "Do not repeat domains."


def test_run_request_requires_one_string_user_message() -> None:
    agent = make_agent()
    body = DomainGenerationRunRequest(responses_create_params={"input": []})
    with pytest.raises(ValueError, match="exactly one input message"):
        # Validation is synchronous and occurs before the first await in run().
        agent.run(body).send(None)
