# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.processors.nemo_sim_processor import (
    NeMoSimProcessor,
    NeMoSimProcessorConfig,
    NeMoSimRunRequest,
    _ConversationBridge,
    _GymModelFacade,
)
from nemo_gym.server_utils import ServerClient


def _model_response(response_id: str, text: str) -> dict:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": f"{response_id}-message",
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


def _processor() -> NeMoSimProcessor:
    config = NeMoSimProcessorConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="nemo-sim-processor",
        user_agent=AgentServerRef(type="responses_api_agents", name="user-agent"),
        assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant-agent"),
        judge_model=ModelServerRef(type="responses_api_models", name="judge-model"),
        summary_model=ModelServerRef(type="responses_api_models", name="summary-model"),
        api_response_model=ModelServerRef(type="responses_api_models", name="api-response-model"),
        max_turns=2,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return NeMoSimProcessor(config=config, server_client=client)


def _request() -> NeMoSimRunRequest:
    return NeMoSimRunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "scenario": {
                "persona": {"first_name": "Morgan", "age": 42},
                "probe_type": "general_open_ended",
                "theme": {"type": "recommendation", "description": "Plan dinner."},
                "locale": "en_US",
            },
        }
    )


@pytest.mark.asyncio
async def test_conversation_loop_calls_participant_agents_and_support_models(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor()
    posts: list[dict] = []

    async def post(**kwargs):
        posts.append(kwargs)
        response = MagicMock(status=200, ok=True, cookies={})
        response.content.read = AsyncMock(return_value=b"")
        response.read = AsyncMock(
            return_value=json.dumps(_model_response(f"response-{len(posts)}", f"output-{len(posts)}"))
        )
        return response

    processor.server_client.post = post

    def run_interaction_protocol(bridge, body):
        del body
        models = {
            alias: _GymModelFacade(alias, bridge)
            for alias in ("user_model", "assistant_model", "judge_model", "summary_model")
        }
        models["user_model"].completion(
            [{"role": "system", "content": "Act as the user."}],
            max_tokens=77,
        )
        models["judge_model"].completion([{"role": "user", "content": "Judge the user turn."}])
        models["assistant_model"].completion([{"role": "user", "content": "Help me."}])
        models["summary_model"].completion([{"role": "user", "content": "Should the episode stop?"}])
        return {
            "conversation_status": True,
            "conversation_messages": "[]",
            "simulation_outcome": "{}",
        }

    monkeypatch.setattr(processor, "_run_nemo_sim", run_interaction_protocol)
    result = await processor.run(SimpleNamespace(cookies={"session": "shared"}), _request())

    assert [post["server_name"] for post in posts] == [
        "user-agent",
        "judge-model",
        "assistant-agent",
        "summary-model",
    ]
    assert posts[0]["json"]["max_output_tokens"] == 77
    assert [(call.alias, call.executor) for call in result.invocations] == [
        ("user_model", "agent"),
        ("judge_model", "model"),
        ("assistant_model", "agent"),
        ("summary_model", "model"),
    ]
    assert result.response.output[0].content[0].text == "output-3"
    assert result.episode_interaction_protocol == "nemo_sim.ConversationLoop"


def test_rejects_unknown_response_parameter_alias() -> None:
    with pytest.raises(ValueError, match="unknown aliases"):
        NeMoSimRunRequest.model_validate(
            {
                **_request().model_dump(mode="json"),
                "model_responses_create_params": {"not_a_nemo_sim_alias": {"input": []}},
            }
        )


@pytest.mark.asyncio
async def test_preserves_failure_before_first_assistant_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor()

    def fail_user_gate(bridge, body):
        del bridge, body
        return {
            "conversation_status": False,
            "conversation_messages": "[]",
            "simulation_outcome": '{"status":"failed"}',
        }

    monkeypatch.setattr(processor, "_run_nemo_sim", fail_user_gate)
    result = await processor.run(SimpleNamespace(cookies={}), _request())

    assert result.nemo_sim_result["conversation_status"] is False
    assert result.response.output == []
    assert result.invocations == []


@pytest.mark.asyncio
async def test_rejects_sync_bridge_call_on_processor_event_loop() -> None:
    processor = _processor()
    bridge = _ConversationBridge(
        processor=processor,
        body=_request(),
        event_loop=asyncio.get_running_loop(),
        cookies={},
    )

    with pytest.raises(RuntimeError, match="must run outside"):
        bridge.complete_from_worker("user_model", [], max_tokens=None)
