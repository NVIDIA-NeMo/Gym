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

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.conversational_tool_use.scenario_generation import app
from responses_api_agents.conversational_tool_use.scenario_generation.app import (
    ConversationalToolUseScenarioGenerationAgent,
    ScenarioGenerationAgentConfig,
    ScenarioGenerationRunRequest,
)
from responses_api_agents.conversational_tool_use.scenario_generation.assets import (
    SCHEMA_PATH,
    SYSTEM_PROMPT_PATH,
    USER_PROMPT_PATH,
    ScenarioAssets,
    load_assets,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def scenario(
    reason: str,
    *,
    persona: str = "A customer",
    include_unknown_info: bool = True,
) -> dict[str, Any]:
    value = {
        "customer_persona": persona,
        "reason_for_contact": reason,
        "customer_details": "Order O-123",
        "task_instructions": "Ask for help.",
    }
    if include_unknown_info:
        value["unknown_info"] = None
    return value


def chat_completion(
    response_id: str,
    scenarios: list[dict[str, Any]],
    *,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": json.dumps({"scenarios": scenarios}),
                    "refusal": None,
                    "role": "assistant",
                },
            }
        ],
        "created": 1,
        "model": "scenario-model",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 3,
            "prompt_tokens": 5,
            "total_tokens": 8,
        },
    }


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.ok = True

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class BlockingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def post(self, **kwargs: Any) -> FakeHTTPResponse:
        request_index = len(self.calls)
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == 20:
            self.all_started.set()
        await self.release.wait()
        self.active -= 1
        return FakeHTTPResponse(
            chat_completion(
                f"response-{request_index}",
                [scenario(f"Reason {request_index}")],
            )
        )


class RecordingClient:
    def __init__(self, response_payload: dict[str, Any]) -> None:
        self.response_payload = response_payload
        self.calls: list[dict[str, Any]] = []

    async def post(self, **kwargs: Any) -> FakeHTTPResponse:
        self.calls.append(kwargs)
        return FakeHTTPResponse(self.response_payload)


def agent(client: Any) -> ConversationalToolUseScenarioGenerationAgent:
    config = ScenarioGenerationAgentConfig(
        host="0.0.0.0",
        port=8000,
        entrypoint="app.py",
        name="conversational_tool_use_scenario_generation",
        model_server=ModelServerRef(
            type="responses_api_models",
            name="scenario_generation_model",
        ),
    )
    return ConversationalToolUseScenarioGenerationAgent.model_construct(
        config=config,
        server_client=client,
    )


def run_request() -> ScenarioGenerationRunRequest:
    return ScenarioGenerationRunRequest(
        id="domain-1",
        profile="general",
        domain_name="order support",
        policy="Authenticate before changing an order.\n",
        tools=[],
        responses_create_params={"input": []},
    )


def test_prompt_and_schema_bytes() -> None:
    expected = {
        SYSTEM_PROMPT_PATH: (
            673,
            "9b799cf3bd1110e9d637ca4db718f68d626c90943ecf5bcbf2ad808a511f3d90",  # pragma: allowlist secret
        ),
        USER_PROMPT_PATH: (
            243,
            "cad5e03e2ad1365c919da4520af339a0616ec0954a29e415779244df13b73cbd",  # pragma: allowlist secret
        ),
        SCHEMA_PATH: (
            1877,
            "b0a4d8385fbda3b77d8d9626bc5998f1d5e62a2f75a225c3f6198a663aa5991e",  # pragma: allowlist secret
        ),
    }
    for path, (expected_length, expected_hash) in expected.items():
        raw_bytes = path.read_bytes()
        assert len(raw_bytes) == expected_length
        assert hashlib.sha256(raw_bytes).hexdigest() == expected_hash

    assets = load_assets()
    assert assets.system_prompt == SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    assert assets.user_prompt == USER_PROMPT_PATH.read_text(encoding="utf-8")
    assert assets.schema == SCHEMA_PATH.read_text(encoding="utf-8")


def test_config_and_example_data_contract() -> None:
    raw_config = OmegaConf.load(PACKAGE_DIR / "configs" / "conversational_tool_use_scenario_generation.yaml")
    assert raw_config["scenario_generation_model"]["_copy"] == "policy_model"
    inner = OmegaConf.to_container(
        raw_config["conversational_tool_use_scenario_generation"]["responses_api_agents"][
            "conversational_tool_use/scenario_generation"
        ],
        resolve=True,
    )
    parsed_config = ScenarioGenerationAgentConfig.model_validate(
        inner
        | {
            "host": "0.0.0.0",
            "port": 8000,
            "name": "conversational_tool_use_scenario_generation",
        }
    )
    assert parsed_config.entrypoint == "app.py"
    assert parsed_config.model_server.name == "scenario_generation_model"

    example = json.loads((PACKAGE_DIR / "data" / "example.jsonl").read_text(encoding="utf-8"))
    assert set(example) == {
        "id",
        "profile",
        "domain_name",
        "policy",
        "tools",
        "responses_create_params",
        "agent_ref",
    }
    assert example["profile"] == "general"
    parsed_request = ScenarioGenerationRunRequest.model_validate(example)
    assert parsed_request.domain_name == "order support"
    assert parsed_request.responses_create_params.input == []


@pytest.mark.asyncio
async def test_responses_is_one_call_bridge_preserving_caller_parameters() -> None:
    response_payload = {
        "id": "resp-bridge",
        "created_at": 1,
        "model": "caller-model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 0.25,
        "top_p": 0.75,
    }
    client = RecordingClient(response_payload)
    server = agent(client)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "Generate."}],
        model="caller-model",
        temperature=0.25,
        top_p=0.75,
        max_output_tokens=123,
        metadata={"request": "kept"},
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

    response = await server.responses(request, body)

    assert response.id == "resp-bridge"
    assert len(client.calls) == 1
    assert client.calls[0] == {
        "server_name": "scenario_generation_model",
        "url_path": "/ng-rollout/task-1/v1/responses",
        "json": body,
    }
    assert client.calls[0]["json"].model_dump(exclude_unset=True) == body.model_dump(exclude_unset=True)


@pytest.mark.asyncio
async def test_run_launches_exactly_twenty_message_only_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_values = iter([0.05, 0.5] * 10)
    monkeypatch.setattr(app, "random", lambda: next(scope_values))
    client = BlockingClient()
    server = agent(client)

    run_task = asyncio.create_task(server.run(run_request()))
    await asyncio.wait_for(client.all_started.wait(), timeout=2)

    assert client.max_active == 20
    assert len(client.calls) == 20
    assert all(call["server_name"] == "scenario_generation_model" for call in client.calls)
    assert all(call["url_path"] == "/v1/chat/completions" for call in client.calls)
    assert all(set(call["json"]) == {"messages"} for call in client.calls)
    assert all(len(call["json"]["messages"]) == 2 for call in client.calls)
    assert sum("does not cover" in call["json"]["messages"][0]["content"] for call in client.calls) == 10
    assert all(
        "Please create 80 different customer scenarios" in call["json"]["messages"][1]["content"]
        for call in client.calls
    )

    client.release.set()
    result = await run_task
    assert result.reward == 1.0
    assert result.generation_trace.successful_call_count == 20
    assert result.generation_trace.failed_call_count == 0
    assert len(result.result.scenarios) == 20
    assert set(result.result.model_dump()) == {"domain_name", "scenarios"}


@pytest.mark.asyncio
async def test_failures_are_isolated_and_completion_order_controls_dedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "random", lambda: 0.5)
    server = agent(object())
    completion_order = [3, 2, *range(4, 20), 1, 0]
    delays = {
        request_index: completion_index / 1_000 for completion_index, request_index in enumerate(completion_order)
    }

    async def generate_one(
        *,
        body: ScenarioGenerationRunRequest,
        request_index: int,
        outside_policy_scope: bool,
        assets: ScenarioAssets,
    ) -> app._CallOutcome:
        await asyncio.sleep(delays[request_index])
        messages = [{"role": "user", "content": assets.schema}]
        if request_index == 2:
            return app._CallOutcome(
                trace=app.ScenarioCallTrace(
                    request_index=request_index,
                    completion_index=-1,
                    outside_policy_scope=outside_policy_scope,
                    messages=messages,
                    status="failed",
                    error_type="ValueError",
                    error_message="malformed",
                ),
                chat_completion=None,
            )
        duplicate = app.CustomerScenario.model_validate(scenario("duplicate", persona=f"winner-{request_index}"))
        duplicate.representative_domain = body.domain_name
        duplicate.outside_policy_scope = outside_policy_scope
        if request_index == 1:
            duplicate.customer_persona = "SAME PERSONA"
            duplicate.reason_for_contact = "DUPLICATE"
        elif request_index == 3:
            duplicate.customer_persona = "same persona"
        completion = app.NeMoGymChatCompletion.model_validate(
            chat_completion(
                f"response-{request_index}",
                [duplicate.model_dump()],
                finish_reason="length" if request_index == 0 else "stop",
            )
        )
        return app._CallOutcome(
            trace=app.ScenarioCallTrace(
                request_index=request_index,
                completion_index=-1,
                outside_policy_scope=outside_policy_scope,
                messages=messages,
                status="success",
                scenarios=[duplicate],
                raw_chat_completion=completion.model_dump(mode="json"),
            ),
            chat_completion=completion,
        )

    monkeypatch.setattr(server, "_generate_one", generate_one)
    result = await server.run(run_request())

    assert result.reward == 1.0
    assert result.generation_trace.failed_call_count == 1
    assert [call.request_index for call in result.generation_trace.calls][:2] == [3, 2]
    same_persona = [item for item in result.result.scenarios if item.customer_persona == "same persona"]
    assert len(same_persona) == 1
    assert same_persona[0].reason_for_contact == "duplicate"
    assert result.generation_trace.calls[-1].raw_chat_completion["id"] == "response-0"
    response_payload = result.response.model_dump(mode="json")
    assert response_payload["output"][0]["content"][0]["text"] == json.dumps(
        {"scenarios": [result.result.scenarios[-1].model_dump(mode="json")]}
    )
    assert response_payload["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response_payload["usage"]["input_tokens"] == 5
    assert response_payload["usage"]["output_tokens"] == 3
    assert response_payload["usage"]["total_tokens"] == 8


@pytest.mark.asyncio
async def test_omitted_unknown_info_is_dropped_but_explicit_null_is_retained() -> None:
    completion = chat_completion(
        "response",
        [
            scenario("explicit null"),
            scenario("omitted", include_unknown_info=False),
        ],
    )
    client = RecordingClient(completion)
    outcome = await agent(client)._generate_one(
        body=run_request(),
        request_index=0,
        outside_policy_scope=True,
        assets=load_assets(),
    )

    assert outcome.trace.status == "success"
    assert outcome.trace.parsed_scenario_count == 2
    assert outcome.trace.omitted_unknown_info_count == 1
    assert len(outcome.trace.scenarios) == 1
    assert "unknown_info" in outcome.trace.scenarios[0].model_fields_set
    assert outcome.trace.scenarios[0].unknown_info is None
    assert outcome.trace.scenarios[0].representative_domain == "order support"
    assert outcome.trace.scenarios[0].outside_policy_scope is True


@pytest.mark.asyncio
async def test_parse_failure_retains_the_raw_completion() -> None:
    completion = chat_completion("response-malformed", [])
    completion["choices"][0]["message"]["content"] = "not json"
    outcome = await agent(RecordingClient(completion))._generate_one(
        body=run_request(),
        request_index=0,
        outside_policy_scope=False,
        assets=load_assets(),
    )

    assert outcome.trace.status == "failed"
    assert outcome.trace.error_type == "ValidationError"
    assert outcome.trace.raw_chat_completion["id"] == "response-malformed"


@pytest.mark.asyncio
async def test_all_failed_is_normal_empty_reward_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "random", lambda: 0.5)
    server = agent(object())

    async def fail_one(**kwargs: Any) -> app._CallOutcome:
        return app._CallOutcome(
            trace=app.ScenarioCallTrace(
                request_index=kwargs["request_index"],
                completion_index=-1,
                outside_policy_scope=kwargs["outside_policy_scope"],
                messages=[],
                status="failed",
                error_type="RuntimeError",
                error_message="provider failed",
            ),
            chat_completion=None,
        )

    monkeypatch.setattr(server, "_generate_one", fail_one)
    result = await server.run(run_request())

    assert result.reward == 1.0
    assert result.generation_trace.failed_call_count == 20
    assert result.result.scenarios == []
    assert result.response.id == "conversational_tool_use_scenario_generation_empty"
    assert result.response.output == []


@pytest.mark.asyncio
async def test_fatal_coordinator_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_random() -> float:
        raise RuntimeError("coordinator failed")

    monkeypatch.setattr(app, "random", fail_random)
    with pytest.raises(RuntimeError, match="coordinator failed"):
        await agent(object()).run(run_request())
