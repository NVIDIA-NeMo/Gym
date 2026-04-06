# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from app import GPQADiamondResourcesServer

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.resources_servers.mcqa.app import MCQAResourcesServerConfig, MCQAVerifyRequest
from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        GPQADiamondResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_verify_gpqa_diamond_template_metadata_priority(self) -> None:
        server = GPQADiamondResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        regex_response = NeMoGymResponse(
            id="resp_regex",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_regex",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Answer: A\nFinal Choice: c",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = MCQAVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Question?\nA: optA\nB: optB\nC: optC\nD: optD"}]
            },
            response=regex_response,
            options=[{"A": "optA"}, {"B": "optB"}, {"C": "optC"}, {"D": "optD"}],
            expected_answer="C",
            grading_mode="strict_single_letter_boxed",
            template_metadata={"output_regex": r"Final Choice:\s*([A-Za-z])"},
        )
        result = await server.verify(verify_request)

        assert result.reward == 1.0
        assert result.extracted_answer == "C"

    async def test_verify_gpqa_diamond_format(self) -> None:
        server = GPQADiamondResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        answer_response = NeMoGymResponse(
            id="resp_answer",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_answer",
                    "content": [{"annotations": [], "text": "Reasoning...\nAnswer: C", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = MCQAVerifyRequest(
            responses_create_params={
                "input": [
                    {
                        "role": "user",
                        "content": (
                            "The last line should be of the format 'Answer: LETTER'. "
                            "Question?\nA: optA\nB: optB\nC: optC\nD: optD"
                        ),
                    }
                ]
            },
            response=answer_response,
            options=[{"A": "optA"}, {"B": "optB"}, {"C": "optC"}, {"D": "optD"}],
            expected_answer="C",
            grading_mode="strict_single_letter_boxed",
        )
        result_answer = await server.verify(verify_request)
        assert result_answer.reward == 1.0
        assert result_answer.extracted_answer == "C"

        boxed_response = NeMoGymResponse(
            id="resp_boxed",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_boxed",
                    "content": [{"annotations": [], "text": "Final: \\boxed{C}", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        verify_request_boxed = MCQAVerifyRequest(
            responses_create_params=verify_request.responses_create_params.model_dump(exclude_none=True),
            response=boxed_response,
            options=verify_request.options,
            expected_answer=verify_request.expected_answer,
            grading_mode=verify_request.grading_mode,
        )
        result_boxed = await server.verify(verify_request_boxed)

        assert result_boxed.reward == 1.0
        assert result_boxed.extracted_answer == "C"

    async def test_verify_gpqa_diamond_rejects_invalid_letter(self) -> None:
        server = GPQADiamondResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        invalid_response = NeMoGymResponse(
            id="resp_invalid",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_invalid",
                    "content": [{"annotations": [], "text": "Answer: E", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = MCQAVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Question?\nA: optA\nB: optB\nC: optC\nD: optD"}]
            },
            response=invalid_response,
            options=[{"A": "optA"}, {"B": "optB"}, {"C": "optC"}, {"D": "optD"}],
            expected_answer="C",
            grading_mode="strict_single_letter_boxed",
        )
        result = await server.verify(verify_request)

        assert result.reward == 0.0
        assert result.extracted_answer is None
