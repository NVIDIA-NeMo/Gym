# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"Explanation. The answer is \boxed{B}.", "B"),
        (r"\boxed { * b * }", "B"),
        ("The answer is *(b)*", "B"),
        ("THE ANSWER IS **B**", "B"),
        (r"\boxed{A} then \boxed{B}", "A"),
        (r"The answer is C, then \boxed{B}", "C"),
        (r"\boxed{E} then \boxed{B}", "B"),
        (r"\boxed{\text{B}}", None),
        (r"\boxed{B: option text}", None),
        (r"\boxed{B", None),
        (r"\boxed{E}", None),
        ("Answer: B", None),
        ("B", None),
        ("", None),
        ("   ", None),
    ],
)
@pytest.mark.parametrize("mode_from_config", [True, False])
async def test_reference_extraction_and_reward(text, expected, mode_from_config):
    server = MCQAResourcesServer(
        config=MCQAResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="test",
            grading_mode="milu_v2" if mode_from_config else None,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    response = NeMoGymResponse(
        id="resp_test",
        created_at=0,
        model="test",
        object="response",
        output=[
            {
                "id": "msg_test",
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    result = await server.verify(
        MCQAVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Question"}]},
            response=response,
            options=[{letter: f"option {letter}"} for letter in "ABCD"],
            expected_answer="B",
            grading_mode="milu_v2" if not mode_from_config else "strict_single_letter_boxed",
        )
    )
    assert result.extracted_answer == expected
    assert result.reward == float(expected == "B")
    assert result.expected_answer == "B"
