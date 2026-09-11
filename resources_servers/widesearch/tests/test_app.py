# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from pytest import approx

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.widesearch.app import (
    WideSearchConfig,
    WideSearchServer,
    WideSearchVerifyRequest,
    extract_dataframe,
)


def test_reverify_mode_is_stateless() -> None:
    assert WideSearchConfig.REVERIFY_MODE is ReverifyMode.STATELESS


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


def test_extract_markdown_table() -> None:
    frame = extract_dataframe("```markdown\n| Name | Rank |\n| --- | --- |\n| Alpha | 1 |\n```")
    assert frame is not None
    assert frame.to_dict(orient="records") == [{"Name": "Alpha", "Rank": 1}]


def test_extract_markdown_table_ignores_earlier_placeholder() -> None:
    frame = extract_dataframe(
        """```markdown
{data_content}
```
```markdown
| Name | Rank |
| --- | --- |
| Alpha | 1 |
```"""
    )
    assert frame is not None
    assert frame.to_dict(orient="records") == [{"Name": "Alpha", "Rank": 1}]


async def test_perfect_table_receives_official_strict_score() -> None:
    server = WideSearchServer(
        config=WideSearchConfig(
            host="0.0.0.0",
            port=0,
            entrypoint="app.py",
            name="widesearch",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    result = await server.verify(
        WideSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="Return the table")]
            ),
            instance_id="ws_test_001",
            query="Return the table",
            evaluation={
                "unique_columns": ["name"],
                "required": ["name", "rank"],
                "eval_pipeline": {
                    "name": {"preprocess": ["norm_str"], "metric": ["exact_match"]},
                    "rank": {"preprocess": ["extract_number"], "metric": ["number_near"]},
                },
            },
            gold_answer=[{"Name": "Alpha", "Rank": "1"}],
            language="en",
            response=response("| Name | Rank |\n| --- | --- |\n| Alpha | 1 |"),
        )
    )
    assert result.reward == 1.0
    assert result.f1_by_row == approx(1.0)
    assert result.f1_by_item == approx(1.0)
