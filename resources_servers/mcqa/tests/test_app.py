# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock

from app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

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

        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = MCQAVerifyRequest(
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
            grading_mode="strict_single_letter_boxed",
        )

        # strict requires boxed; plain C should fail
        result = await server.verify(verify_request)
        assert result.reward == 0.0

        # Now send boxed C (strict)
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
                            "text": "Final: \\boxed{ [C] }",
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

        # Lenient: allow matching option text within boxed content
        response_boxed_text = NeMoGymResponse(
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
                            "text": "Final: \\boxed{ optC }",
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

        verify_request_lenient = verify_request.model_copy(
            update={"response": response_boxed_text, "grading_mode": "lenient_boxed"}
        )
        result3 = await server.verify(verify_request_lenient)
        assert result3.reward == 1.0

        # Lenient answer colon: letter
        response_answer_colon = NeMoGymResponse(
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
                            "text": "Answer: c",
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
        verify_request_answer_colon = verify_request.model_copy(
            update={
                "response": response_answer_colon,
                "grading_mode": "lenient_answer_colon",
            }
        )
        result4 = await server.verify(verify_request_answer_colon)
        assert result4.reward == 1.0

        # Lenient answer colon: exact option text
        response_answer_colon_text = NeMoGymResponse(
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
                            "text": "Answer: optC",
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
        verify_request_answer_colon_text = verify_request.model_copy(
            update={
                "response": response_answer_colon_text,
                "grading_mode": "lenient_answer_colon",
            }
        )
        result5 = await server.verify(verify_request_answer_colon_text)
        assert result5.reward == 1.0

    async def test_template_metadata_basic(self) -> None:
        """Test basic template_metadata with custom regex"""
        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        # Test custom regex: "Option Selected: X"
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "Option Selected: B", "type": "output_text"}],
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
                "input": [{"role": "user", "content": "Question?\nA: optA\nB: optB"}],
                "parallel_tool_calls": False,
                "temperature": 0,
            },
            response=response,
            options=[{"A": "optA"}, {"B": "optB"}],
            expected_answer="B",
            grading_mode="strict_single_letter_boxed",  # Will be overridden by template_metadata
            template_metadata={"output_regex": r"Option Selected:\s*([A-Za-z])"},
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.extracted_answer == "B"

    async def test_template_metadata_case_insensitive(self) -> None:
        """Test that template_metadata regex is case-insensitive"""
        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        # Model outputs lowercase 'b', should match uppercase 'B'
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "ANSWER IS b", "type": "output_text"}],
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
                "input": [{"role": "user", "content": "Question?\nA: optA\nB: optB"}],
            },
            response=response,
            options=[{"A": "optA"}, {"B": "optB"}],
            expected_answer="B",
            template_metadata={"output_regex": r"ANSWER IS\s*([A-Za-z])"},
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.extracted_answer == "B"

    async def test_template_metadata_rightmost_match(self) -> None:
        """Test that rightmost (last) match is used when multiple matches exist"""
        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        # Model mentions A first, then concludes with B
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
                            "text": "Maybe Answer: A? Let me reconsider. Final Answer: B",
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
                "input": [{"role": "user", "content": "Question?\nA: optA\nB: optB"}],
            },
            response=response,
            options=[{"A": "optA"}, {"B": "optB"}],
            expected_answer="B",
            template_metadata={"output_regex": r"Answer:\s*([A-Za-z])"},
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.extracted_answer == "B"

    async def test_template_metadata_priority_over_grading_mode(self) -> None:
        """Test that template_metadata takes priority over grading_mode"""
        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        # Model outputs "Final Choice: B" (not boxed format)
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "Final Choice: B", "type": "output_text"}],
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
                "input": [{"role": "user", "content": "Question?\nA: optA\nB: optB"}],
            },
            response=response,
            options=[{"A": "optA"}, {"B": "optB"}],
            expected_answer="B",
            grading_mode="strict_single_letter_boxed",  # Would fail without boxed
            template_metadata={"output_regex": r"Final Choice:\s*([A-Za-z])"},  # Should use this instead
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0  # Should succeed via template_metadata
        assert result.extracted_answer == "B"

    async def test_template_metadata_invalid_regex(self) -> None:
        """Test that invalid regex patterns are handled gracefully"""
        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "\\boxed{B}", "type": "output_text"}],
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
                "input": [{"role": "user", "content": "Question?\nA: optA\nB: optB"}],
            },
            response=response,
            options=[{"A": "optA"}, {"B": "optB"}],
            expected_answer="B",
            grading_mode="strict_single_letter_boxed",  # Should fallback to this
            template_metadata={"output_regex": r"(["},  # Invalid regex
        )

        # Should fallback to grading_mode and succeed
        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.extracted_answer == "B"

    async def test_template_metadata_without_options(self) -> None:
        """Test template_metadata works even with incomplete options metadata"""
        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "Selected: B", "type": "output_text"}],
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
                "input": [{"role": "user", "content": "Question?"}],
            },
            response=response,
            options=[],  # Empty options
            expected_answer="B",
            template_metadata={"output_regex": r"Selected:\s*([A-Za-z])"},
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.extracted_answer == "B"
