# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
from unittest.mock import MagicMock

import pytest
from app import (
    StringMatchResourcesServer,
    StringMatchResourcesServerConfig,
    StringMatchVerifyRequest,
    _answers_match,
    _extract_answer,
    _grade_string_match,
)

from nemo_gym.client import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = StringMatchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        StringMatchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.parametrize(
        ("gt_answer", "pred_answer", "expected_score"),
        [
            ("oven", "oven", 1.0),
            ("oven", "Oven", 1.0),
            ("oven", "  oven  ", 1.0),
            ("oven", "stove", 0.0),
        ],
    )
    def test_grade_string_match(self, gt_answer: str, pred_answer: str, expected_score: float) -> None:
        assert _grade_string_match(gt_answer, pred_answer) == expected_score

    @pytest.mark.parametrize(
        ("extracted", "expected", "case_sensitive", "expected_score"),
        [
            ("Blue", "blue", False, 1.0),
            ("Blue", "blue", True, 0.0),
            ("blue", "blue", True, 1.0),
        ],
    )
    def test_answers_match_case_sensitivity(
        self, extracted: str, expected: str, case_sensitive: bool, expected_score: float
    ) -> None:
        assert _answers_match(extracted, expected, case_sensitive) == expected_score

    def test_extract_answer_prefers_boxed(self) -> None:
        text = "Some reasoning here. The answer is \\boxed{42}."
        assert _extract_answer(text, "boxed") == "42"

    def test_extract_answer_last_line(self) -> None:
        text = "First line of reasoning.\nSecond line.\noven"
        assert _extract_answer(text, "last_line") == "oven"

    def test_verify_preserves_json_schema_alias(self) -> None:
        config = StringMatchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = StringMatchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        body = StringMatchVerifyRequest.model_validate(
            {
                "responses_create_params": {"input": "question"},
                "response": {
                    "id": "response-1",
                    "created_at": 1.0,
                    "model": "model",
                    "object": "response",
                    "output": [
                        {
                            "id": "message-1",
                            "content": [{"annotations": [], "text": '{"answer": "yes"}', "type": "output_text"}],
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                        }
                    ],
                    "parallel_tool_calls": True,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "answer",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    "tool_choice": "auto",
                    "tools": [],
                },
                "expected_answer": "yes",
            }
        )

        verified = asyncio.run(server.verify(body))
        response_format = verified.model_dump(mode="json", by_alias=True)["response"]["text"]["format"]

        assert response_format["schema"] == schema
        assert "schema_" not in response_format
