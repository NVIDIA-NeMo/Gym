# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from responses_api_agents.spatialclaw_agent.native_client import GymChatClient, create_native_client


async def test_native_transport_preserves_extra_body_and_usage(monkeypatch) -> None:
    send = AsyncMock(
        return_value={
            "id": "native-response",
            "object": "chat.completion",
            "created": 0,
            "model": "policy",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "B"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
    )
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", send)
    client = GymChatClient(base_url="http://policy/v1/", api_key="gym")
    messages = [{"role": "user", "content": "Where is the cup?"}]

    response = await client.chat.completions.create(
        model="policy", messages=messages, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    send.assert_awaited_once_with(model="policy", messages=messages, chat_template_kwargs={"enable_thinking": False})
    assert response.choices[0].message.content == "B"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 3
    assert client._client.base_url == "http://policy/v1"
    await client.close()


async def test_native_transport_rejects_streaming() -> None:
    client = GymChatClient(base_url="http://policy/v1", api_key="gym")
    with pytest.raises(ValueError, match="non-streaming"):
        await client.chat.completions.create(model="policy", messages=[], stream=True)


@pytest.mark.skipif(not os.environ.get("SPATIALCLAW_ROOT"), reason="requires the original SpatialClaw checkout")
def test_original_client_survives_kernel_serialization(monkeypatch) -> None:
    import cloudpickle

    monkeypatch.syspath_prepend(str(Path(os.environ["SPATIALCLAW_ROOT"]).resolve()))
    from spatial_agent.config import SpatialAgentConfig
    from spatial_agent.llm.client import LLMClient

    config = SpatialAgentConfig(llm_base_url="http://policy/v1", llm_model="policy", llm_api_key="gym")
    client = create_native_client(config)
    assert isinstance(client, LLMClient)
    assert isinstance(client._get_client("http://policy/v1"), GymChatClient)

    restored = cloudpickle.loads(cloudpickle.dumps(client))
    assert restored._client_pool == {}
    assert isinstance(restored._get_client("http://policy/v1"), GymChatClient)
