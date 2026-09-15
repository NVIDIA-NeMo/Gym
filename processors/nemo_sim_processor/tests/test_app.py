# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.processors import EpisodeId, EpisodeRequest, TaskIdentity
from nemo_gym.server_utils import ServerClient
from processors.nemo_sim_processor.app import (
    NeMoSimProcessor,
    NeMoSimProcessorConfig,
    _ConversationBridge,
    _GymModelFacade,
)
from processors.nemo_sim_processor.contracts import NeMoSimTaskData


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


def _request() -> EpisodeRequest:
    return EpisodeRequest(
        episode_id=EpisodeId(rollout_id="0-0"),
        task=TaskIdentity(task_source="nemo_sim", task_id="0"),
        responses_create_params={"input": []},
        task_data={
            "scenario": {
                "persona": {"first_name": "Morgan"},
                "theme": {"type": "recommendation", "description": "Plan dinner."},
            }
        },
    )


@pytest.mark.asyncio
async def test_returns_ordered_agent_turns_and_focal_response(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def run_protocol(bridge, scenario):
        del scenario
        for alias in ("user_model", "judge_model", "assistant_model", "summary_model"):
            _GymModelFacade(alias, bridge).completion([{"role": "user", "content": alias}])
        return {"conversation_status": True, "conversation_messages": [], "simulation_outcome": {}}

    monkeypatch.setattr(processor, "_run_nemo_sim", run_protocol)
    result = await processor.run(SimpleNamespace(cookies={"session": "shared"}), _request())

    assert result.verification.reward == 1
    assert result.response.output_text == "output-3"
    turns = result.verification.verifier_data["agent_turns"]
    assert [(turn["sequence"], turn["participant"]) for turn in turns] == [(0, "user"), (1, "assistant")]
    assert turns[1]["request"]["input"][0]["content"] == "assistant_model"


@pytest.mark.asyncio
async def test_missing_assistant_is_typed_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor()
    monkeypatch.setattr(
        processor,
        "_run_nemo_sim",
        lambda bridge, scenario: {"conversation_status": False, "conversation_messages": []},
    )
    result = await processor.run(SimpleNamespace(cookies={}), _request())
    assert result.failure.kind == "agent"
    assert result.response is None


def test_rejects_unknown_response_parameter_alias() -> None:
    with pytest.raises(ValueError, match="unknown aliases"):
        NeMoSimTaskData.model_validate(
            {
                **_request().task_data,
                "model_responses_create_params": {"unknown": {"input": []}},
            }
        )


@pytest.mark.asyncio
async def test_rejects_sync_bridge_call_on_processor_event_loop() -> None:
    processor = _processor()
    request = _request()
    bridge = _ConversationBridge(
        processor,
        request,
        NeMoSimTaskData.model_validate(request.task_data),
        asyncio.get_running_loop(),
        {},
    )
    with pytest.raises(RuntimeError, match="must run outside"):
        bridge.complete_from_worker("user_model", [], max_tokens=None)
