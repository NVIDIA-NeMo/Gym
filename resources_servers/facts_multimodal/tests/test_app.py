# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.facts_multimodal.app import (
    FACTSMultimodalConfig,
    FACTSMultimodalServer,
    FACTSMultimodalVerifyRequest,
    parse_coverage_score,
    parse_factuality_verdict,
)


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
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
        status="completed",
    )


@pytest.mark.parametrize(
    ("text", "essential_fact_count", "expected"),
    [
        ('{"Fact 1": "Yes", "Fact 2": "No"}', 2, 0.5),
        ('Analysis\n{"Fact 1": "Yes"}', 1, 1.0),
        ('{"Fact 1": "Maybe"}', 1, 0.0),
        ("not JSON", 1, 0.0),
    ],
)
def test_parse_coverage_score(text: str, essential_fact_count: int, expected: float) -> None:
    assert parse_coverage_score(text, essential_fact_count) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FINAL VERDICT: NO CLEAR CONTRADICTION.", True),
        ("FINAL VERDICT: HAS CLEAR CONTRADICTION(s).", False),
        ("malformed", False),
    ],
)
def test_parse_factuality_verdict(text: str, expected: bool) -> None:
    assert parse_factuality_verdict(text) is expected


async def test_verify_reports_component_scores() -> None:
    config = FACTSMultimodalConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    server = FACTSMultimodalServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._call_judge = AsyncMock(
        side_effect=['{"Fact 1": "Yes", "Fact 2": "No"}', "FINAL VERDICT: NO CLEAR CONTRADICTION."]
    )
    request = FACTSMultimodalVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("A blue sedan is parked under a tree."),
        prompt="Describe the image.",
        image_data_url="data:image/jpeg;base64,/9j/",
        rubric_items=[
            {"fact": "There is a blue sedan.", "tags": ["Sentence", "essential"]},
            {"fact": "It is parked under a tree.", "tags": ["Sentence", "essential"]},
        ],
    )

    result = await server.verify(request)

    assert result.reward == 0.0
    assert result.coverage == 0.5
    assert result.factuality == 1.0
    assert result.accuracy == 0.0
    assert result.response_status == "completed"
    assert result.model_incomplete is False
    assert result.empty_generation is False
    assert result.coverage_parse_failed is False
    assert result.factuality_parse_failed is False
    assert result.coverage_judge_output.startswith('{"Fact 1"')
    assert result.factuality_judge_output.startswith("FINAL VERDICT")
    assert server._call_judge.await_args_list[0].args[1:] == ()
    assert server._call_judge.await_args_list[1].args[1:] == ("data:image/jpeg;base64,/9j/",)


async def test_empty_generation_skips_the_judge() -> None:
    config = FACTSMultimodalConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )
    server = FACTSMultimodalServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._call_judge = AsyncMock()
    request = FACTSMultimodalVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response(""),
    )

    result = await server.verify(request)

    assert result.reward == 0.0
    assert result.empty_generation is True
    server._call_judge.assert_not_awaited()


async def test_malformed_judge_outputs_are_retried() -> None:
    config = FACTSMultimodalConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )
    server = FACTSMultimodalServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._call_judge = AsyncMock(side_effect=["not json", '{"Fact 1": "Yes"}'])

    assert await server._judge_coverage("prompt", 1) == ('{"Fact 1": "Yes"}', 1.0, False)
    assert server._call_judge.await_count == 2

    server._call_judge = AsyncMock(side_effect=["missing verdict", "FINAL VERDICT: HAS CLEAR CONTRADICTION(s)."])
    assert await server._judge_factuality("prompt", "data:image/jpeg;base64,/9j/") == (
        "FINAL VERDICT: HAS CLEAR CONTRADICTION(s).",
        False,
        False,
    )
    assert server._call_judge.await_count == 2


async def test_verify_uses_source_url_for_factuality_when_configured() -> None:
    config = FACTSMultimodalConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        use_base64_images=False,
    )
    server = FACTSMultimodalServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._call_judge = AsyncMock(side_effect=['{"Fact 1": "Yes"}', "FINAL VERDICT: NO CLEAR CONTRADICTION."])
    request = FACTSMultimodalVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("A blue sedan."),
        prompt="Describe the image.",
        image_url="https://example.test/image.jpg",
        rubric_items=[{"fact": "There is a blue sedan.", "tags": ["Sentence", "essential"]}],
    )

    result = await server.verify(request)

    assert result.factuality == 1.0
    assert server._call_judge.await_args_list[1].args[1:] == ("https://example.test/image.jpg",)
