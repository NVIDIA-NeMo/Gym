# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic demonstration of the NeMoSimProcessor bridge."""

from __future__ import annotations

import asyncio
import json as json_module
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.processors.nemo_sim_processor import (
    NeMoSimProcessor,
    NeMoSimProcessorConfig,
    NeMoSimRunRequest,
)


_RESPONSES = {
    "user_model": [
        "I need a vegetarian dinner for two that costs no more than $20. What would you suggest?",
        "Could you include a simple shopping list and approximate prices?",
    ],
    "assistant_model": [
        "Make chickpea tomato pasta: pasta, chickpeas, canned tomatoes, onion, and spinach should total about $12.",
        "Shopping list: pasta $2, chickpeas $2, tomatoes $3, onion $1, spinach $3, and seasoning $1.",
    ],
    "judge_model": ["<explanation>Role and task are correct.</explanation><rating>success</rating>"],
    "summary_model": ["no"],
    "api_response_model": ['{"result":"unused in this non-tool probe"}'],
}


class _FakeResponse:
    ok = True
    cookies: dict[str, str] = {}

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def read(self) -> bytes:
        return json_module.dumps(self._payload).encode()


class _ScriptedAgentClient:
    """Stand in for ServerClient while preserving the real async call boundary."""

    def __init__(self, server_to_alias: dict[str, str]) -> None:
        self.server_to_alias = server_to_alias
        self.call_counts: defaultdict[str, int] = defaultdict(int)

    async def post(
        self,
        *,
        server_name: str,
        url_path: str,
        json: dict[str, Any],
        cookies: dict[str, Any],
    ) -> _FakeResponse:
        del url_path, cookies
        alias = self.server_to_alias[server_name]
        call_index = self.call_counts[alias]
        self.call_counts[alias] += 1
        scripted = _RESPONSES[alias]
        text = scripted[min(call_index, len(scripted) - 1)]
        print(
            json_module.dumps(
                {
                    "event": "gym_agent_turn",
                    "alias": alias,
                    "call_index": call_index,
                    "input_roles": [item["role"] for item in json["input"]],
                    "output": text,
                },
                indent=2,
            )
        )
        return _FakeResponse(
            {
                "id": f"resp_{uuid4().hex}",
                "created_at": 0,
                "model": server_name,
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid4().hex}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "tools": [],
            }
        )


async def main() -> None:
    user_agent = AgentServerRef(type="responses_api_agents", name="demo-user-agent")
    assistant_agent = AgentServerRef(type="responses_api_agents", name="demo-assistant-agent")
    judge_model = ModelServerRef(type="responses_api_models", name="demo-judge-model")
    summary_model = ModelServerRef(type="responses_api_models", name="demo-summary-model")
    api_response_model = ModelServerRef(type="responses_api_models", name="demo-api-response-model")
    config = NeMoSimProcessorConfig(
        name="nemo-sim-processor",
        host="0.0.0.0",
        port=0,
        entrypoint="processors.nemo_sim_processor.app",
        user_agent=user_agent,
        assistant_agent=assistant_agent,
        judge_model=judge_model,
        summary_model=summary_model,
        api_response_model=api_response_model,
        max_turns=2,
        skip_verification=True,
    )
    client = _ScriptedAgentClient(
        {
            user_agent.name: "user_model",
            assistant_agent.name: "assistant_model",
            judge_model.name: "judge_model",
            summary_model.name: "summary_model",
            api_response_model.name: "api_response_model",
        }
    )
    processor = NeMoSimProcessor.model_construct(config=config, server_client=client)
    base_params = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": []})
    body = NeMoSimRunRequest.model_validate(
        {
            "responses_create_params": base_params,
            "scenario": {
                "locale": "en_US",
                "probe_type": "general_open_ended",
                "theme": {
                    "type": "recommendation",
                    "description": "Plan a vegetarian dinner for two that costs no more than $20.",
                },
                "persona": {
                    "first_name": "Morgan",
                    "last_name": "Lee",
                    "age": 42,
                    "occupation": "building inspector",
                    "agreeableness": {"t_score": 63},
                    "neuroticism": {"t_score": 51},
                },
            },
            "simulation_config": {
                "context_compression": False,
                "enforce_user_language": False,
                "verbosity": 0,
            },
        }
    )
    result = await processor.run(SimpleNamespace(cookies={}), body)
    print(
        json_module.dumps(
            {
                "event": "episode_complete",
                "episode_interaction_protocol": result.episode_interaction_protocol,
                "conversation_status": result.nemo_sim_result["conversation_status"],
                "conversation_messages": result.nemo_sim_result["conversation_messages"],
                "simulation_outcome": result.nemo_sim_result["simulation_outcome"],
                "invocations": [
                    {"alias": invocation.alias, "executor": invocation.executor} for invocation in result.invocations
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
