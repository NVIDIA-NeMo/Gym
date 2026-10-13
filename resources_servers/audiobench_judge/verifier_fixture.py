# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier checks using a fixed judge reply.

Each fixture case carries the judge's reply text in ``fixture_judge_text``; ``invoke`` serves it from a
mocked judge model server, so the verifier runs without network access.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.audiobench_judge.app import (
        AudioBenchJudgeResourcesServer,
        AudioBenchJudgeVerifyRequest,
        AudioBenchJudgeVerifyResponse,
    )


def create_server() -> "AudioBenchJudgeResourcesServer":
    from resources_servers.audiobench_judge.app import AudioBenchJudgeConfig, AudioBenchJudgeResourcesServer

    return AudioBenchJudgeResourcesServer(
        config=AudioBenchJudgeConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="audiobench_judge",
            judge_model_server={"type": "responses_api_models", "name": "judge_model"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


async def invoke_with_fixed_judge(
    server: "AudioBenchJudgeResourcesServer", request: "AudioBenchJudgeVerifyRequest"
) -> "AudioBenchJudgeVerifyResponse":
    judge_response = NeMoGymResponse(
        id="fixture_judge",
        created_at=0,
        model="fixture",
        object="response",
        status="completed",
        output=[
            {
                "id": "fixture_message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": request.fixture_judge_text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    reply = MagicMock(ok=True)
    reply.read = AsyncMock(return_value=judge_response.model_dump_json())
    server.server_client.post = AsyncMock(return_value=reply)
    return await server.verify(request)
