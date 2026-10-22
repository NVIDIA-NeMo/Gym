# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.string_match.app import StringMatchResourcesServerConfig, StringMatchVerifyRequest
from resources_servers.string_match.app_passrate import PassrateStringMatchResourcesServer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_answer", "case_sensitive", "extracted_answer", "reward"),
    [
        ("42<|im_end|>", "42", False, "42", 1.0),
        ("42<|endoftext|>", "42", False, "42", 1.0),
        ("42<|eot_id|>", "42", False, "42", 1.0),
        ("42 <|im_end|> \n<|eot_id|>  ", "42", False, "42", 1.0),
        ("42", "43", False, "42", 0.0),
        ("Blue<|im_end|>", "blue", True, "Blue", 0.0),
        ("Blue<|im_end|>", "blue", False, "Blue", 1.0),
        ("a<|im_end|>b", "a<|im_end|>b", True, "a<|im_end|>b", 1.0),
    ],
)
async def test_verify_preserves_response(
    text: str, expected_answer: str, case_sensitive: bool, extracted_answer: str, reward: float
) -> None:
    server = PassrateStringMatchResourcesServer(
        config=StringMatchResourcesServerConfig(
            host="127.0.0.1", port=8080, entrypoint="app_passrate.py", name="test"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    body = StringMatchVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "expected_answer": expected_answer,
            "extraction_mode": "full_response",
            "case_sensitive": case_sensitive,
            "response": {
                "id": "response",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
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
            },
        }
    )
    original = body.model_dump()
    result = await server.verify(body)
    assert result.reward == reward
    assert result.extracted_answer == extracted_answer
    assert body.model_dump() == original
    assert result.response.model_dump() == body.response.model_dump()
