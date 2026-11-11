# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import io
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from PIL import Image

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalAgent, OpenCodeMultimodalAgentConfig
from responses_api_agents.opencode_multimodal_agent.rewards import final_answer


def image_url():
    output = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def body(**updates):
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Count red objects"},
                        {"type": "input_image", "image_url": image_url(), "detail": "auto"},
                    ],
                }
            ],
            "temperature": 0.6,
            "top_p": 0.95,
        }
        | updates
    )


def config(**updates):
    return OpenCodeMultimodalAgentConfig.model_validate(
        {
            "name": "visual",
            "host": "127.0.0.1",
            "port": 8080,
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
            "model": "vision",
            "context_window": 49152,
            "sandbox_provider": "sandbox",
            "sandbox_config": {"image": "test", "workdir": "/app"},
        }
        | updates
    )


def agent(monkeypatch=None, **updates):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return OpenCodeMultimodalAgent(config=config(**updates), server_client=client)


async def test_unseeded_episode_cannot_use_local_training_or_eval(monkeypatch):
    with pytest.raises(HTTPException) as error:
        await agent()._create_episode(body())
    assert error.value.status_code == 409


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>\\boxed{oven}</think>stove", "stove"),
        ("<think>Answer: oven</think>Final answer: stove", "Final answer: stove"),
        ("<thinking>wrong\nanswer</thinking>42", "42"),
        ("<think>\\boxed{42}", ""),
        ("wrong</think>42", "42"),
        ("42", "42"),
    ],
)
def test_final_answer_excludes_inline_reasoning(text, expected):
    output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
    assert final_answer(output, strip_reasoning=True) == expected
    assert output[0]["content"][0]["text"] == text
