# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import socket
from unittest.mock import Mock

import litellm
import pytest
from minisweagent.exceptions import Submitted
from nemo_fabric_adapter_contract import models as contract

from responses_api_agents.nemo_fabric_agent.app import _fabric_output_items
from responses_api_agents.nemo_fabric_agent.mini_swe_adapter import MiniSweAgentRuntime


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network is forbidden")))


@pytest.mark.parametrize("base_url", ["https://model-gateway.example/v1", "http://127.0.0.1:9000/rollout/v1"])
@pytest.mark.parametrize("cost_tracking", ["default", "ignore_errors"])
@pytest.mark.parametrize("known_cost", [None, 0.25])
def test_catalog_model_unknown_cost_keeps_route_and_complete_trace(
    monkeypatch: pytest.MonkeyPatch, base_url: str, cost_tracking: str, known_cost: float | None
) -> None:
    model = "azure/anthropic/claude-opus-5-5"
    commands = iter(["printf mock-output", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"])
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        assert kwargs["custom_llm_provider"] == "openai"
        assert kwargs["model"] == model
        assert kwargs["api_base"] == base_url
        return litellm.ModelResponse(
            model=model,
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "mock reasoning",
                        "tool_calls": [
                            {
                                "id": f"call-{len(calls)}",
                                "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": next(commands)})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    cost = (
        Mock(side_effect=ValueError("unknown catalog model")) if known_cost is None else Mock(return_value=known_cost)
    )
    monkeypatch.setattr(litellm.cost_calculator, "completion_cost", cost)

    async def run():
        runtime = MiniSweAgentRuntime()
        await runtime.start(
            {
                "config": contract.AgentConfig.from_mapping(
                    {
                        "harness": {"settings": {}},
                        "models": {
                            "default": {
                                "provider": "openai-compatible",
                                "model": model,
                                "base_url": base_url,
                                "settings": {"cost_tracking": cost_tracking},
                            }
                        },
                        "runtime": {"max_turns": 3},
                    }
                )
            }
        )

        def execute(action):
            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in action["command"]:
                raise Submitted(
                    {
                        "role": "exit",
                        "content": "mock-final",
                        "extra": {"exit_status": "Submitted", "submission": "mock-final"},
                    }
                )
            return {"output": "mock-output", "returncode": 0, "exception_info": ""}

        monkeypatch.setattr(runtime._environment, "execute", execute)
        return await runtime.invoke(contract.AgentRunRequest(input="mock task"), None)

    result = asyncio.run(run())
    if known_cost is None and cost_tracking == "default":
        assert result.status == contract.AgentRunStatus.FAILED
        assert "unknown catalog model" in result.error.message
        assert result.usage.cost_usd is None
        return
    assert result.status == contract.AgentRunStatus.SUCCEEDED
    assert result.output["output"] == "mock-final"
    assert result.usage.cost_usd == (None if known_cost is None else 2 * known_cost)
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 6
    assert result.usage.total_tokens == 10
    messages = result.output["messages"]
    assert len([m for m in messages if m["role"] == "assistant"]) == 2
    assert all(m["extra"]["cost"] == known_cost for m in messages if m["role"] == "assistant")
    assert all(m["extra"]["response"]["usage"]["total_tokens"] == 5 for m in messages if m["role"] == "assistant")
    items = _fabric_output_items(result.output, "mock-final")
    assert [i.type for i in items].count("function_call") == 2
    assert [i.type for i in items].count("function_call_output") == 2
    assert items[-1].content[0].text == "mock-final"


@pytest.mark.parametrize("provider", ["openai", "openai-compatible", "nvidia"])
def test_litellm_provider_resolution_preserves_catalog_model(provider: str) -> None:
    model = "azure/anthropic/catalog-model"
    base_url = "https://model-gateway.example/v1"
    runtime = MiniSweAgentRuntime()
    asyncio.run(
        runtime.start(
            {
                "config": contract.AgentConfig.from_mapping(
                    {"models": {"default": {"provider": provider, "model": model, "base_url": base_url}}}
                )
            }
        )
    )
    resolved, selected, _, endpoint = litellm.get_llm_provider(
        model=runtime._model.config.model_name,
        custom_llm_provider=runtime._model.config.model_kwargs["custom_llm_provider"],
        api_base=runtime._model.config.model_kwargs["api_base"],
    )
    assert (resolved, selected, endpoint) == (model, "openai", base_url)


@pytest.mark.parametrize("cost_tracking", ["default", "ignore_errors"])
@pytest.mark.parametrize("error_type", [litellm.AuthenticationError, litellm.NotFoundError])
def test_model_authentication_errors_are_failed_with_trace(
    monkeypatch: pytest.MonkeyPatch, cost_tracking: str, error_type: type[Exception]
) -> None:
    completion = Mock(
        side_effect=error_type(message="mock provider failed", llm_provider="openai", model="catalog-model")
    )
    monkeypatch.setattr(litellm, "completion", completion)

    async def run():
        runtime = MiniSweAgentRuntime()
        await runtime.start(
            {
                "config": contract.AgentConfig.from_mapping(
                    {
                        "models": {
                            "default": {
                                "provider": "openai-compatible",
                                "model": "catalog-model",
                                "settings": {"cost_tracking": cost_tracking},
                            }
                        },
                    }
                )
            }
        )
        return await runtime.invoke(contract.AgentRunRequest(input="mock task"), None)

    result = asyncio.run(run())
    assert result.status == contract.AgentRunStatus.FAILED
    assert result.error.code == error_type.__name__
    assert "mock provider failed" in result.error.message
    assert completion.call_count == 1
    assert result.output["messages"][-1]["role"] == "exit"
    assert result.usage.cost_usd is None


def test_deepagents_preserves_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_fabric_adapters.deepagents.adapter import build_chat_model

    monkeypatch.setenv("NEMO_GYM_FABRIC_TEST_KEY", "local")
    model, name, url = build_chat_model(
        contract.AgentModelConfig(
            provider="openai-compatible",
            model="azure/anthropic/catalog-model",
            base_url="https://model-gateway.example/v1",
            api_key_env="NEMO_GYM_FABRIC_TEST_KEY",
        )
    )
    assert model.model_name == name == "azure/anthropic/catalog-model"
    assert model.openai_api_base == url == "https://model-gateway.example/v1"
