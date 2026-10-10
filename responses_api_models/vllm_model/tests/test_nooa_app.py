# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, mark

import nemo_gym.server_utils
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_models.vllm_model.app import VLLMModelConfig
from responses_api_models.vllm_model.nooa_app import NOOAVLLMModel


class TestNOOAGateway:
    def _setup_server(self, monkeypatch: MonkeyPatch) -> NOOAVLLMModel:
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", lambda: {})
        return NOOAVLLMModel(
            config=VLLMModelConfig(
                host="localhost",
                port=8081,
                name="policy_model",
                entrypoint="nooa_app.py",
                base_url="http://localhost:8000/v1",
                api_key="dummy",
                model="test",
                return_token_id_information=False,
                uses_reasoning_parser=False,
            ),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

    @mark.parametrize(
        "base_urls, gateway_only",
        [
            (["https://inference-api.nvidia.com/v1"], True),
            (["https://inference-api.nvidia.com/v1/"], True),
            (["https://inference-api.nvidia.com/v1", "https://inference-api.nvidia.com/v1/"], True),
            (["http://localhost:8000/v1"], False),
            (["https://inference-api.nvidia.com/v1", "http://localhost:8000/v1"], False),
            (["https://inference-api.nvidia.com/v1/other"], False),
            ([], False),
        ],
    )
    def test_nvidia_gateway_normalizes_only_equivalent_reasoning_aliases(
        self, monkeypatch: MonkeyPatch, base_urls: list[str], gateway_only: bool
    ) -> None:
        server = self._setup_server(monkeypatch)
        server.config.base_url = base_urls
        aliases = [
            {"reasoning_content": "thought", "reasoning": "thought"},
            {"reasoning_content": "thought", "reasoning": "different thought"},
            {"reasoning_content": "thought"},
            {"reasoning": "thought"},
            {"reasoning_content": "", "reasoning": ""},
            {"reasoning_content": None, "reasoning": None},
            {},
        ]
        messages = [{"role": "assistant", "content": "answer", **fields} for fields in aliases]
        original_messages = deepcopy(messages)
        body = {"prompt_cache_key": "rollout-cache", "messages": messages}

        result = server._apply_sampling_overrides(body)

        assert ("prompt_cache_key" in result) == (not gateway_only)
        expected_messages = deepcopy(original_messages)
        if gateway_only:
            for index in (0, 4, 5):
                expected_messages[index].pop("reasoning")
        assert result["messages"] == expected_messages
        assert messages == original_messages

    def test_nvidia_gateway_preserves_shared_sampling_overrides(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server(monkeypatch)
        server.config.base_url = ["https://inference-api.nvidia.com/v1"]
        overrides = {
            "prompt_cache_key": "configured-cache",
            "temperature": 0.7,
            "messages": [{"role": "assistant", "reasoning_content": "thought", "reasoning": "thought"}],
        }
        server.config.sampling_overrides = deepcopy(overrides)

        result = server._apply_sampling_overrides({"temperature": 0.2})

        assert "prompt_cache_key" not in result
        assert result["temperature"] == 0.7
        assert result["messages"] == [{"role": "assistant", "reasoning_content": "thought"}]
        assert server.config.sampling_overrides == overrides

    @mark.parametrize("gateway", [False, True])
    async def test_nvidia_gateway_chat_forwarding_preserves_request_history(
        self, monkeypatch: MonkeyPatch, gateway: bool
    ) -> None:
        server = self._setup_server(monkeypatch)
        server.config.base_url = ["https://inference-api.nvidia.com/v1" if gateway else "http://localhost:8000/v1"]
        server.config.uses_reasoning_parser = True
        client = MagicMock(spec=NeMoGymAsyncOpenAI)
        client.create_chat_completion = AsyncMock(return_value=server._create_empty_chat_completion().model_dump())
        server._clients = [client]
        body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
            {
                "prompt_cache_key": "rollout-cache",
                "messages": [
                    {"role": "assistant", "content": "<think>thought</think>answer"},
                    {"role": "user", "content": "continue"},
                ],
            }
        )
        original_body = body.model_dump()
        request = MagicMock(session={SESSION_ID_KEY: "gateway-test"}, headers={})

        await server.chat_completions(request, body)

        outbound = client.create_chat_completion.call_args.kwargs
        expected_assistant = {"role": "assistant", "content": "answer", "reasoning_content": "thought"}
        if not gateway:
            expected_assistant["reasoning"] = "thought"
        assert outbound["messages"] == [expected_assistant, {"role": "user", "content": "continue"}]
        assert ("prompt_cache_key" in outbound) == (not gateway)
        assert body.model_dump() == original_body
