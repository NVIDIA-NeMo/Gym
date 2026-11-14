# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use actual Gym response schemas at the OpenCode-specific adapter boundary."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


sys.path.insert(0, str(BENCHMARK_ROOT))
from responses_api_models.vllm_model import saas_opencode as module


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_real_gym_schema_preserves_reasoning_and_rejects_unexecuted_action(monkeypatch, malformed):
    async def upstream(self, request, body):
        return NeMoGymChatCompletion.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "super",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "<think>Next<tool_call><function=saas_browser_interact></function></tool_call></think>"
                            if malformed
                            else "<think>Checked UI</think>Saved document",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    monkeypatch.setattr("nemo_gym.config.global_config.get_global_config_dict", dict)
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="saas_opencode.py",
        name="policy_model",
        base_url="http://localhost:9999/v1",
        api_key="EMPTY",
        model="super",
        return_token_id_information=False,
        uses_reasoning_parser=True,
    )
    adapter = module.SaasOpenCodeModel(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": [{"role": "user", "content": "Complete task"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "saas_browser_interact",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    if malformed:
        with pytest.raises(HTTPException) as error:
            await adapter.chat_completions(None, body)
        assert error.value.status_code == 422
        assert "after 3 provider attempts" in error.value.detail
        with pytest.raises(HTTPException) as retry:
            await adapter.chat_completions(None, body)
        assert retry.value.status_code == 422
    else:
        response = await adapter.chat_completions(None, body)
        msg = response.model_dump()["choices"][0]["message"]
        assert msg["content"] == "Saved document"
        assert msg["reasoning_content"] == "Checked UI"


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("frequency_penalty,client_penalty", [(0.0, 0.3), (0.3, 0.0)])
async def test_effective_outbound_super_sampling_overrides_client_defaults(
    monkeypatch, thinking, frequency_penalty, client_penalty
):
    from starlette.requests import Request

    monkeypatch.setattr("nemo_gym.config.global_config.get_global_config_dict", dict)
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="saas_opencode.py",
        name="policy_model",
        base_url="http://127.0.0.1:9/v1",
        api_key="EMPTY",
        model="super",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        sampling_overrides={
            "temperature": 0.6,
            "top_p": 0.95,
            "frequency_penalty": frequency_penalty,
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    client = MagicMock()
    client.create_chat_completion = AsyncMock(
        return_value={
            "id": "response",
            "object": "chat.completion",
            "created": 1,
            "model": "super",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Saved"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
    )
    monkeypatch.setattr(VLLMModel, "_resolve_client", lambda self, request: client)
    adapter = module.SaasOpenCodeModel(config=config, server_client=MagicMock(spec=ServerClient))
    request = Request({"type": "http", "headers": [], "session": {}})
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": [{"role": "user", "content": "Use the app"}],
            "temperature": 1.0,
            "top_p": 1.0,
            "frequency_penalty": client_penalty,
            "max_tokens": 16384,
        }
    )
    await adapter.chat_completions(request, body)
    sent = client.create_chat_completion.call_args.kwargs
    assert sent["temperature"] == 0.6 and sent["top_p"] == 0.95
    assert sent["frequency_penalty"] == frequency_penalty
    assert sent["max_tokens"] == 16384
    assert sent["chat_template_kwargs"]["enable_thinking"] is thinking


@pytest.mark.asyncio
async def test_oversized_batches_have_bounded_retries_and_retained_provider_evidence(monkeypatch, tmp_path):
    calls = []

    async def upstream(self, request, body):
        calls.append(body)
        return NeMoGymChatCompletion.model_validate(
            {
                "id": "oversized-response",
                "object": "chat.completion",
                "created": 1,
                "model": "super",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{i}",
                                    "type": "function",
                                    "function": {
                                        "name": "saas_browser_observe",
                                        "arguments": "{}",
                                    },
                                }
                                for i in range(33)
                            ],
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    monkeypatch.setattr(
        "nemo_gym.config.global_config.get_global_config_dict",
        lambda: {"model_call_capture_dir": str(tmp_path)},
    )
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="saas_opencode.py",
        name="policy_model",
        base_url="http://127.0.0.1:9/v1",
        api_key="EMPTY",
        model="super",
        return_token_id_information=False,
        uses_reasoning_parser=True,
    )
    adapter = module.SaasOpenCodeModel(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": [{"role": "user", "content": "Use tools"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "saas_browser_observe",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    for status in [422, 422]:
        with pytest.raises(HTTPException) as error:
            await adapter.chat_completions(None, body)
        assert error.value.status_code == status
    assert len(calls) == 3
    records = [
        json.loads(line) for path in tmp_path.glob("saas_protocol_*.jsonl") for line in path.read_text().splitlines()
    ]
    assert len(records) == 3
    assert [record["provider_attempt"] for record in records] == [1, 2, 3]
    assert all(len(r["provider_response"]["choices"][0]["message"]["tool_calls"]) == 33 for r in records)
