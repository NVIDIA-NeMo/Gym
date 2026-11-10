# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.string_match.app import (
    StringMatchResourcesServer,
    StringMatchResourcesServerConfig,
    StringMatchVerifyRequest,
)
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
@pytest.mark.parametrize("configured", [False, True])
async def test_verify_preserves_response(
    text: str, expected_answer: str, case_sensitive: bool, extracted_answer: str, reward: float, configured: bool
) -> None:
    server_class = StringMatchResourcesServer if configured else PassrateStringMatchResourcesServer
    server = server_class(
        config=StringMatchResourcesServerConfig(
            host="127.0.0.1", port=8080, entrypoint="app_passrate.py", name="test", strip_trailing_eos=configured
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
    default_server = StringMatchResourcesServer(
        config=StringMatchResourcesServerConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="plain"),
        server_client=MagicMock(spec=ServerClient),
    )
    assert default_server._grading_text(body) == text.strip()
    original = body.model_dump()
    result = await server.verify(body)
    assert result.reward == reward
    assert result.extracted_answer == extracted_answer
    assert body.model_dump() == original
    assert result.response.model_dump() == body.response.model_dump()


@pytest.mark.parametrize("tail", [" " * 100000, (" <|im_end|>" * 10000)])
def test_long_trailing_markers_and_whitespace(tail):
    from resources_servers.string_match.app import strip_trailing_eos

    assert strip_trailing_eos("42" + tail) == "42"
    assert strip_trailing_eos("a<|im_end|>b") == "a<|im_end|>b"
