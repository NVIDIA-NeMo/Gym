# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline judge transport for Gym's verifier-fixture contract."""

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.vials.app import VialsVerifier, VialsVerifyRequest, VialsVerifyResponse


def judge_http_reply(text: str, *, tokens: int = 10) -> MagicMock:
    """Build a model-server HTTP response without calling a model provider."""
    payload = {
        "id": "fixture-judge",
        "created_at": 0,
        "model": "gpt-5-mini",
        "object": "response",
        "output": [
            {
                "id": "fixture-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        "usage": {
            "input_tokens": tokens,
            "output_tokens": 5,
            "total_tokens": tokens + 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    reply = MagicMock(ok=True, status=200)
    reply.read = AsyncMock(return_value=json.dumps(payload).encode())
    return reply


def create_fixture_verifier() -> "VialsVerifier":
    from resources_servers.vials.app import VialsVerifier

    return VialsVerifier(MagicMock(spec=ServerClient))


async def invoke_fixture(server: "VialsVerifier", body: "VialsVerifyRequest") -> "VialsVerifyResponse":
    async def post(**kwargs: object) -> MagicMock:
        params = kwargs["json"].model_dump(mode="json")
        prompt = params["input"][1]["content"]
        candidate = prompt.split("Candidate final answer:\n", 1)[1].split("\n\n", 1)[0]
        verdicts = {"twelve": True, "eleven": False}
        if candidate not in verdicts:
            raise AssertionError(f"Unexpected fixture candidate: {candidate!r}")
        return judge_http_reply(json.dumps({"equivalent": verdicts[candidate], "reasoning": "Fixed offline verdict."}))

    server._judge.server_client.post = AsyncMock(side_effect=post)
    return await server.verify(body)
