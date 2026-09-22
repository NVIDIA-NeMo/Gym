# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.gpqa_diamond.app import (
    GPQADiamondResourcesServer,
    GPQADiamondResourcesServerConfig,
    extract_official_answer,
)
from resources_servers.mcqa.app import MCQAVerifyRequest


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The correct answer is (B)", "B"),
        ("Answer: (C)", "C"),
        ("answer: (D)", "D"),
        ("answer (A)", "A"),
        ("(C)", "C"),
        ("B", None),
        (r"\boxed{B}", None),
        ("(b)", None),
        ("(E)", None),
        ("(e) then (B)", None),
        ("(A) is tempting; the correct answer is (D)", "D"),
        ("The correct answer is (A). The correct answer is (B).", "A"),
        ("Answer: (C); the answer is (B)", "B"),
        ("<think>The correct answer is (A)</think>The correct answer is (B)", "B"),
        ("<thinking>(A)</thinking>Answer: (C)", "C"),
        ("(A)</think>(D)", "D"),
        ("<think>(B)", None),
    ],
)
def test_official_pattern_order_and_inline_thinking(text, expected):
    assert extract_official_answer(text) == expected


def response(text):
    return NeMoGymResponse.model_validate(
        {
            "id": "test",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "text,reward,extracted",
    [
        ("The correct answer is (B)", 1.0, "B"),
        ("Answer: (A)", 0.0, "A"),
        (r"\boxed{B}", 0.0, None),
        ("", 0.0, None),
    ],
)
async def test_official_verifier_bypasses_generic_regex(text, reward, extracted):
    config = GPQADiamondResourcesServerConfig(
        host="127.0.0.1",
        port=0,
        name="test",
        entrypoint="app.py",
        use_official_parser=True,
    )
    server = GPQADiamondResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    body = MCQAVerifyRequest(
        responses_create_params={"input": "synthetic"},
        response=response(text),
        options=[{letter: letter} for letter in "ABCD"],
        expected_answer="B",
        template_metadata={"output_regex": r"(B)"},
    )

    result = await server.verify(body)

    assert result.reward == reward
    assert result.extracted_answer == extracted
