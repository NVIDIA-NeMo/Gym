# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable agent-test helpers also used by paired NeMo-RL compatibility tests."""

import json
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.core import NeMoGymResponse


def answer(index, *, output=None, incomplete=None):
    return NeMoGymResponse(
        id=f"response-{index}",
        created_at=0,
        error=None,
        incomplete_details=incomplete,
        instructions=None,
        metadata=None,
        model="test",
        object="response",
        parallel_tool_calls=True,
        temperature=None,
        tool_choice="auto",
        tools=[],
        top_p=None,
        output=output
        if output is not None
        else [
            {
                "id": f"message-{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": f"answer {index}", "annotations": []}],
            }
        ],
    )


def http_response(payload, *, content=None, cookies=None, status=200):
    response = MagicMock(ok=status < 400, status=status, cookies=cookies or {})
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.content.read = AsyncMock(return_value=(content if content is not None else json.dumps(payload)).encode())
    return response


def observation(image, text="observation"):
    return {
        "role": "user",
        "type": "message",
        "content": [
            {"type": "input_image", "image_url": f"https://example.invalid/{image}.png", "detail": "auto"},
            {"type": "input_text", "text": text},
        ],
    }


def reasoning(index):
    return {
        "id": f"reason-{index}",
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": f"private {index}"}],
    }
