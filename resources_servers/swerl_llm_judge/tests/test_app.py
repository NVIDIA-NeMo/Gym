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

from app import SWEJudgeResourcesServer, SWEJudgeResourcesServerConfig, SWEJudgeVerifyRequest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = SWEJudgeResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        SWEJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_verify_correct(self) -> None:
        # Build a NeMoGymResponse with a valid OpenAI Responses shape and the assistant message including letter C
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": "The answer is C.",
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

        server = SWEJudgeResourcesServer(
            config=SWEJudgeResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = SWEJudgeVerifyRequest(
            responses_create_params={
                "input": [
                    {
                        "role": "user",
                        "content": "Q?\nA: optA\nB: optB\nC: optC\nD: optD",
                    },
                ],
                "parallel_tool_calls": False,
                "temperature": 0,
            },
            response=response,
            options=[{"A": "optA"}, {"B": "optB"}, {"C": "optC"}, {"D": "optD"}],
            expected_answer="C",
            grading_mode="strict",
        )

        # strict requires <solution>C</solution>; plain C should fail
        result = await server.verify(verify_request)
        assert result.reward == 0.0

        # Now send <solution>C</solution> (strict)
        response_boxed = NeMoGymResponse(
            id="resp_test2",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test2",
                    "content": [
                        {
                            "annotations": [],
                            "text": "<solution>C</solution>",
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

        verify_request_boxed = verify_request.model_copy(update={"response": response_boxed})
        result2 = await server.verify(verify_request_boxed)
        assert result2.reward == 1.0

        # Still strict, but with different format
        response_bracketed = NeMoGymResponse(
            id="resp_test3",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "<solution>[C]</solution>",
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

        verify_request_bracketed = verify_request.model_copy(
            update={"response": response_bracketed, "grading_mode": "strict"}
        )
        result3 = await server.verify(verify_request_bracketed)
        assert result3.reward == 1.0

        # Strict: lowercase letter is accepted
        response_lowercase = NeMoGymResponse(
            id="resp_test4",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test4",
                    "content": [
                        {
                            "annotations": [],
                            "text": "<solution>c</solution>",
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
        verify_request_lowercase = verify_request.model_copy(
            update={
                "response": response_lowercase,
                "grading_mode": "strict",
            }
        )
        result4 = await server.verify(verify_request_lowercase)
        assert result4.reward == 1.0

        # Strict: extra text is not allowed
        response_extra_text = NeMoGymResponse(
            id="resp_test5",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test5",
                    "content": [
                        {
                            "annotations": [],
                            "text": "<solution>Answer: C</solution>",
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
        verify_request_extra_text = verify_request.model_copy(
            update={
                "response": response_extra_text,
                "grading_mode": "strict",
            }
        )
        result5 = await server.verify(verify_request_extra_text)
        assert result5.reward == 0.0

        # Lenient: allow matching option text within solution content
        response_extra_text_lenient = NeMoGymResponse(
            id="resp_test6",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test6",
                    "content": [
                        {
                            "annotations": [],
                            "text": "<solution>Answer: C</solution>",
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
        verify_request_extra_text_lenient = verify_request.model_copy(
            update={
                "response": response_extra_text_lenient,
                "grading_mode": "lenient",
            }
        )
        result6 = await server.verify(verify_request_extra_text_lenient)
        assert result6.reward == 1.0
