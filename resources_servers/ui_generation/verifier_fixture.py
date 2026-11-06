# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier fixture: the real verify() with the judge's reply fixed by each case."""

from unittest.mock import AsyncMock, MagicMock

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion


def fixture_judge_reply(text: str, *, finish_reason: str = "stop", reply_id: str = "fixture_judge") -> MagicMock:
    """An HTTP reply of the judge model server carrying one chat completion whose content is ``text``."""
    completion = NeMoGymChatCompletion.model_validate(
        {
            "id": reply_id,
            "object": "chat.completion",
            "created": 0,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": text},
                }
            ],
        }
    )
    reply = MagicMock(ok=True)
    reply.read = AsyncMock(return_value=completion.model_dump_json())
    return reply


def create_fixture_server():
    from resources_servers.ui_generation.app import UiGenerationResourcesServer, UiGenerationResourcesServerConfig

    config = UiGenerationResourcesServerConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        name="ui_generation",
        judge_model_server=ModelServerRef(type="responses_api_models", name="ui_generation_judge_model"),
        judge_chat_completions_create_params={"messages": [], "temperature": 1.0, "top_p": 1.0, "max_tokens": 65536},
    )
    return UiGenerationResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_with_fixture_judge(server, request):
    """Verify with the judge replying ``request.fixture_judge_text``; a case without one must not call the judge."""
    judge_text = getattr(request, "fixture_judge_text", None)
    server.server_client.post = AsyncMock(return_value=fixture_judge_reply(judge_text or ""))
    result = await server.verify(request)
    assert server.server_client.post.await_count == (1 if judge_text is not None else 0)
    return result
