# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
import verifiers.v1 as vf

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import POLICY_MODEL_NAME_KEY_NAME
from nemo_gym.server_utils import ServerClient
from responses_api_agents.verifiers_agent.app import (
    VerifiersAgent,
    VerifiersAgentConfig,
    VerifiersAgentRunRequest,
)
from responses_api_agents.verifiers_agent.example_taskset import ExampleData


def make_agent() -> VerifiersAgent:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {POLICY_MODEL_NAME_KEY_NAME: "policy"}
    return VerifiersAgent(
        config=VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="verifiers_agent",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
            verifiers={
                "taskset": {"id": "example_taskset"},
                "agent": {
                    "harness": {"id": "null"},
                    "max_turns": 1,
                },
            },
        ),
        server_client=server_client,
    )


class TestApp:
    async def test_loads_v1_taskset_and_harness(self) -> None:
        agent = make_agent()
        task = agent._tasks[0]
        trace = vf.Trace(
            task=vf.TraceTask(type="ExampleTask", data=task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(model="policy")),
            nodes=[
                vf.MessageNode(
                    message=vf.AssistantMessage(content="desserts"),
                    sampled=True,
                )
            ],
        )
        await task.score(trace)

        assert task.data.answer == "desserts"
        assert agent._env.config.agent.harness.id == "null"
        assert trace.reward == 1.0
        assert not any(
            getattr(route, "path", "").endswith("/v1/responses") for route in agent.setup_webserver().routes
        )

    @pytest.mark.parametrize("endpoint", ["/responses", "/chat/completions"])
    async def test_runs_and_scores_v1_trace(self, monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
        agent = make_agent()
        task = ExampleData(idx=0, prompt="question", answer="desserts")
        function_call = {
            "id": "call_1",
            "call_id": "call_1",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
            "status": "completed",
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
        }
        answer = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "desserts",
                    "annotations": [],
                }
            ],
            "prompt_token_ids": [1, 2, 3],
            "generation_token_ids": [4],
            "generation_log_probs": [-0.2],
        }
        trace = vf.Trace(
            task=vf.TraceTask(type="ExampleTask", data=task),
            agent=vf.AgentInfo(config=vf.AgentConfig(model="policy")),
            tools=[
                vf.Tool(
                    name="lookup",
                    description="Look up the answer",
                    parameters={"type": "object", "properties": {}},
                )
            ],
            nodes=[
                vf.MessageNode(message=vf.UserMessage(content="question")),
                vf.MessageNode(
                    parent=0,
                    message=vf.AssistantMessage(
                        tool_calls=[
                            vf.ToolCall(
                                id="call_1",
                                name="lookup",
                                arguments="{}",
                            )
                        ],
                        provider_state=[function_call] if endpoint == "/responses" else None,
                    ),
                    sampled=True,
                ),
                vf.MessageNode(
                    parent=1,
                    message=vf.ToolMessage(
                        tool_call_id="call_1",
                        name="lookup",
                        content="desserts",
                    ),
                ),
                vf.MessageNode(
                    parent=2,
                    message=vf.AssistantMessage(
                        content="desserts",
                        provider_state=[answer] if endpoint == "/responses" else None,
                    ),
                    sampled=True,
                ),
            ],
            rewards={"exact_match": vf.Reward(score=1.0)},
            metrics={"used_tool": 1.0},
            calls=[
                vf.ModelCall(node=1, model="policy", endpoint=endpoint),
                vf.ModelCall(node=3, model="policy", endpoint=endpoint),
            ],
            is_completed=True,
            ok=True,
        )
        agent._env = MagicMock()
        slot = MagicMock()
        agent._env.slots.return_value = [slot]
        agent._env.run_slot = AsyncMock(return_value=vf.Episode.of(trace))
        monkeypatch.setattr(
            VerifiersAgent,
            "resolve_model_base_url",
            lambda self, name, rollout_id=None: "http://model/v1",
        )

        request = VerifiersAgentRunRequest(
            task_idx=0,
            responses_create_params={
                "input": "stale prompt",
                "metadata": {"extra_body": '{"seed":7}'},
            },
        )
        result = await agent.responses(request)

        assert [item.type for item in result.output] == [
            "function_call",
            "function_call_output",
            "message",
        ]
        assert result.output[1].output == "desserts"
        if endpoint == "/responses":
            assert result.output[0].prompt_token_ids == [1]
            assert result.output[2].generation_token_ids == [4]
        assert result.reward == 1.0
        assert result.metrics == {"used_tool": 1.0}
        assert agent._env.run_slot.call_args.args[0] is slot
        context = agent._env.run_slot.call_args.args[1]
        assert context.sampling.seed == 7
        assert context.client.base_url == "http://model/v1"
        assert context.client.api_key_var == "NEMO_GYM_API_KEY"
        assert context.model == "policy"
        assert result.model == "policy"
        assert result.tools[0].name == "lookup"
        assert request.responses_create_params.input[0].content == "question"
        assert request.responses_create_params.tools[0]["name"] == "lookup"

    async def test_rejects_failed_v1_episode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agent()
        agent._env = MagicMock()
        agent._env.slots.return_value = [MagicMock()]
        agent._env.run_slot = AsyncMock(
            return_value=vf.Episode(errors=[vf.Error(type="RuntimeError", message="rollout failed")])
        )
        monkeypatch.setattr(
            VerifiersAgent,
            "resolve_model_base_url",
            lambda self, name, rollout_id=None: "http://model/v1",
        )

        with pytest.raises(RuntimeError, match="rollout failed"):
            await agent.responses(VerifiersAgentRunRequest(task_idx=0, responses_create_params={"input": "question"}))
