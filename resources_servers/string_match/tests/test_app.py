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

    @pytest.mark.parametrize(
        ("gt_answer", "pred_answer"),
        [
            ("mother", "other"),
            ("team", "tea"),
            ("Rome", "Roe"),
            ("warm", "war"),
            ("Tom", "To"),
        ],
    )
    def test_units_are_not_stripped_from_inside_words(self, gt_answer: str, pred_answer: str) -> None:
        assert _grade_string_match(gt_answer, pred_answer) == 0.0
        assert _grade_string_match(pred_answer, gt_answer) == 0.0

    @pytest.mark.parametrize(
        ("gt_answer", "pred_answer", "expected_score"),
        [
            ("5", "5 m", 0.98),
            ("5", "5kg", 0.98),
            ("10", "10 mm", 0.98),
            ("90", "90°", 0.98),
            ("90", "90^\\circ", 0.98),
            ("5", "5\\text{ m}", 1.0),
        ],
    )
    def test_units_are_still_stripped_after_numbers(
        self, gt_answer: str, pred_answer: str, expected_score: float
    ) -> None:
        assert _grade_string_match(gt_answer, pred_answer) == expected_score

    async def test_verify_units_inside_words_do_not_match(self) -> None:
        config = StringMatchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = StringMatchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        body = StringMatchVerifyRequest.model_validate(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "Who is Ana's mother?"}]},
                "response": {
                    "id": "resp_1",
                    "created_at": 0.0,
                    "model": "test-model",
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Final answer: other", "annotations": []}],
                        }
                    ],
                    "parallel_tool_calls": False,
                    "tool_choice": "none",
                    "tools": [],
                },
                "expected_answer": "mother",
                "extraction_mode": "final_answer",
            }
        )

        res = await server.verify(body)

        assert res.extracted_answer == "other"
        assert res.reward == 0.0
