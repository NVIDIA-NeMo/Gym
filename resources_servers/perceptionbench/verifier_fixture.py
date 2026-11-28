# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier fixture for perceptionbench (``gym env test perceptionbench``)."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion


if TYPE_CHECKING:
    from resources_servers.perceptionbench.app import (
        PerceptionbenchResourcesServer,
        PerceptionbenchVerifyRequest,
        PerceptionbenchVerifyResponse,
    )


def fixture_judge_reply(text: str, *, reply_id: str = "fixture_judge") -> MagicMock:
    """Mock HTTP reply returning a single chat completion with ``text``."""
    completion = NeMoGymChatCompletion.model_validate(
        {
            "id": reply_id,
            "object": "chat.completion",
            "created": 0,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
        }
    )
    reply = MagicMock(status=200, ok=True)
    reply.read = AsyncMock(return_value=completion.model_dump_json().encode())
    return reply


def create_fixture_server() -> "PerceptionbenchResourcesServer":
    from resources_servers.perceptionbench.app import (
        PerceptionbenchResourcesServer,
        PerceptionbenchResourcesServerConfig,
    )

    config = PerceptionbenchResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="perceptionbench",
        judge_model_server=ModelServerRef(type="responses_api_models", name="perceptionbench_judge_model"),
    )
    return PerceptionbenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_with_fixture_judge(
    server: "PerceptionbenchResourcesServer",
    request: "PerceptionbenchVerifyRequest",
) -> "PerceptionbenchVerifyResponse":
    """Verify using ``request.fixture_judge_text`` when set; cases without it must not call the judge."""
    judge_text = getattr(request, "fixture_judge_text", None)
    server.server_client.post = AsyncMock(return_value=fixture_judge_reply(judge_text or ""))
    result = await server.verify(request)
    assert server.server_client.post.await_count == (1 if judge_text is not None else 0)
    return result
