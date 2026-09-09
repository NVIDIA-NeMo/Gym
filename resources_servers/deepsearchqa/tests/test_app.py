# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import orjson
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.deepsearchqa.app import (
    DeepSearchQAConfig,
    DeepSearchQAServer,
    DeepSearchQAVerifyRequest,
    parse_judge,
)


def response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def test_parse_published_output_format() -> None:
    parsed = parse_judge(
        '```json\n{"Answer Correctness":{"Explanation":"ok","Correctness Details":{"A":true},'
        '"Excessive Answers":[]}}\n```'
    )
    assert parsed["Correctness Details"] == {"A": True}


async def test_verify_set_f1() -> None:
    client = MagicMock(spec=ServerClient)
    judged = response(
        '{"Answer Correctness":{"Explanation":"one missing, one extra",'
        '"Correctness Details":{"A":true,"B":false},"Excessive Answers":["C"]}}'
    )
    http_response = AsyncMock()
    http_response.json = AsyncMock(return_value=judged.model_dump())
    http_response.read = AsyncMock(return_value=orjson.dumps(judged.model_dump()))
    client.post = AsyncMock(return_value=http_response)
    server = DeepSearchQAServer(
        config=DeepSearchQAConfig(
            host="0.0.0.0",
            port=0,
            entrypoint="app.py",
            name="deepsearchqa",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        ),
        server_client=client,
    )
    result = await server.verify(
        DeepSearchQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="question")]
            ),
            example_id="1",
            problem="question",
            answer="A, B",
            answer_type="Set Answer",
            problem_category="test",
            response=response("A and C"),
        )
    )
    assert result.precision == approx(0.5)
    assert result.recall == approx(0.5)
    assert result.f1 == approx(0.5)
    assert result.reward == approx(0.5)
