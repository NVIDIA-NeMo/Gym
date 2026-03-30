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

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.mcqa.app import (
    MCQAResourcesServer,
    MCQAResourcesServerConfig,
    MCQAVerifyRequest,
)


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


def _make_verify_request(text: str, expected: str = "B", grading_mode: str = "strict_single_letter_boxed"):
    """Helper to build a MCQAVerifyRequest with proper schema."""
    response = NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return MCQAVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Q?"}]},
        response=response,
        options=[{"A": "opt1"}, {"B": "opt2"}, {"C": "opt3"}, {"D": "opt4"}],
        expected_answer=expected,
        grading_mode=grading_mode,
    )


class TestGradingModeConfig:
    """Test that MCQAResourcesServerConfig.grading_mode overrides per-row grading_mode."""

    async def test_config_grading_mode_overrides_row(self) -> None:
        config = MCQAResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="mcqa",
            grading_mode="lenient_answer_colon",
        )
        server = MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        body = _make_verify_request(
            text="I think the answer is B.\n\nAnswer: B",
            expected="B",
            grading_mode="strict_single_letter_boxed",
        )
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    async def test_no_config_grading_mode_uses_row_default(self) -> None:
        config = MCQAResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="mcqa",
        )
        server = MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        body = _make_verify_request(
            text="I think the answer is B.\n\nAnswer: B",
            expected="B",
            grading_mode="strict_single_letter_boxed",
        )
        result = await server.verify(body)
        assert result.extracted_answer is None
        assert result.reward == 0.0


class TestGradingModeAnswerColonMD:
    """Test lenient_answer_colon_md grading mode (markdown-aware Answer: extraction)."""

    def _make_server(self, grading_mode="lenient_answer_colon_md"):
        config = MCQAResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="mcqa",
            grading_mode=grading_mode,
        )
        return MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_plain_answer(self) -> None:
        server = self._make_server()
        body = _make_verify_request(text="The answer is B.\n\nAnswer: B", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    async def test_markdown_bold_answer(self) -> None:
        server = self._make_server()
        body = _make_verify_request(text="Reasoning here.\n\n**Answer: C**", expected="C")
        result = await server.verify(body)
        assert result.extracted_answer == "C"
        assert result.reward == 1.0

    async def test_markdown_bold_no_match_old_regex(self) -> None:
        """Verify that lenient_answer_colon does NOT extract **Answer: C** (old behavior preserved)."""
        server = self._make_server(grading_mode="lenient_answer_colon")
        body = _make_verify_request(text="**Answer: C**", expected="C")
        result = await server.verify(body)
        assert result.extracted_answer is None
        assert result.reward == 0.0

    async def test_markdown_underscore_answer(self) -> None:
        server = self._make_server()
        body = _make_verify_request(text="__Answer__: A", expected="A")
        result = await server.verify(body)
        assert result.extracted_answer == "A"
        assert result.reward == 1.0

    async def test_no_answer_pattern(self) -> None:
        server = self._make_server()
        body = _make_verify_request(text="I think it might be B but I'm not sure", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer is None
        assert result.reward == 0.0

    async def test_verify_preserves_metadata_for_aggregate_metrics(self) -> None:
        server = self._make_server()
        body = _make_verify_request(
            text="Answer: C",
            expected="C",
            grading_mode="lenient_answer_colon",
        ).model_copy(
            update={
                "uuid": "gpqa-task-1",
                "metadata": {"subset_for_metrics": "Organic Chemistry"},
                "template_metadata": {"output_regex": r"Answer:\s*([A-Za-z])"},
            }
        )

        result = await server.verify(body)

        assert result.reward == 1.0
        assert result.uuid == "gpqa-task-1"
        assert result.metadata == {"subset_for_metrics": "Organic Chemistry"}
        assert result.template_metadata == {"output_regex": r"Answer:\s*([A-Za-z])"}


class TestComputeMetrics:
    async def test_mcqa_server_returns_pass_majority_metrics(self) -> None:
        """MCQA server overrides compute_metrics to compute pass@k and majority@k."""
        from nemo_gym.base_resources_server import AggregateMetricsRequest
        from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME

        config = MCQAResourcesServerConfig(host="127.0.0.1", port=12345, entrypoint="app.py", name="mcqa")
        server = MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        responses = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "extracted_answer": "A"},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 1.0, "extracted_answer": "A"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, "extracted_answer": "B"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 1.0, "extracted_answer": "C"},
        ]
        body = AggregateMetricsRequest(verify_responses=responses)
        result = await server.aggregate_metrics(body)

        assert "pass@2/accuracy" in result.agent_metrics
        assert "pass@1[avg-of-2]/accuracy" in result.agent_metrics
        assert "majority@2/accuracy" in result.agent_metrics
        assert result.key_metrics == {
            "pass@1/accuracy": result.agent_metrics["pass@1/accuracy"],
            "pass@1[avg-of-2]/accuracy": result.agent_metrics["pass@1[avg-of-2]/accuracy"],
            "pass@1[avg-of-2]/no_answer": result.agent_metrics["pass@1[avg-of-2]/no_answer"],
            "majority@2/accuracy": result.agent_metrics["majority@2/accuracy"],
            "pass@2/no_answer": result.agent_metrics["pass@2/no_answer"],
            "mean/reward": result.agent_metrics["mean/reward"],
        }
