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
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import fixture

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.terminus_judge.app import (
    FailureCode,
    TerminusJudgeResourcesServer,
    TerminusJudgeResourcesServerConfig,
    TerminusJudgeVerifyRequest,
    _extract_last_assistant_text,
    check_task_complete,
    command_similarity,
    extract_keystrokes,
    text_similarity,
)


def create_terminus_1_response(commands: list, is_task_complete: bool = False) -> dict:
    """Create a valid terminus_1 schema response."""
    return {
        "state_analysis": "Analyzing the current state",
        "explanation": "Explanation of the commands",
        "commands": [
            {
                "keystrokes": cmd["keystrokes"],
                "is_blocking": cmd.get("is_blocking", True),
                "timeout_sec": cmd.get("timeout_sec", 5.0),
            }
            for cmd in commands
        ],
        "is_task_complete": is_task_complete,
    }


def create_terminus_2_response(commands: list, task_complete: bool = False) -> dict:
    """Create a valid terminus_2 schema response."""
    return {
        "analysis": "Analyzing the current state",
        "plan": "Plan for the next steps",
        "commands": [
            {
                "keystrokes": cmd["keystrokes"],
                "duration": cmd.get("duration", 1.0),
            }
            for cmd in commands
        ],
        "task_complete": task_complete,
    }


class TestExtractKeystrokes:
    """Tests for the extract_keystrokes helper function."""

    def test_extract_single_keystroke(self):
        """Test extracting keystrokes from a single command."""
        data = {"commands": [{"keystrokes": "ls -la"}]}
        result = extract_keystrokes(data)
        assert result == ["ls -la"]

    def test_extract_multiple_keystrokes(self):
        """Test extracting keystrokes from multiple commands."""
        data = {"commands": [{"keystrokes": "cd /home"}, {"keystrokes": "ls"}]}
        result = extract_keystrokes(data)
        assert result == ["cd /home", "ls"]

    def test_extract_empty_commands(self):
        """Test extracting from empty commands list."""
        data = {"commands": []}
        result = extract_keystrokes(data)
        assert result == []

    def test_extract_missing_keystrokes(self):
        """Test extracting when some commands lack keystrokes."""
        data = {"commands": [{"keystrokes": "ls"}, {"other": "field"}]}
        result = extract_keystrokes(data)
        assert result == ["ls"]


class TestTextSimilarity:
    """Tests for the text_similarity function."""

    def test_identical_strings(self):
        """Test similarity of identical strings."""
        assert text_similarity("hello", "hello") == 1.0

    def test_completely_different_strings(self):
        """Test similarity of completely different strings."""
        result = text_similarity("abc", "xyz")
        assert 0.0 <= result < 0.5

    def test_similar_strings(self):
        """Test similarity of similar strings."""
        result = text_similarity("hello world", "hello world!")
        assert 0.9 < result < 1.0

    def test_empty_strings(self):
        """Test similarity of empty strings."""
        assert text_similarity("", "") == 1.0


class TestCommandSimilarity:
    """Tests for the command_similarity function."""

    def test_identical_commands(self):
        """Test similarity of identical commands."""
        gt = {"commands": [{"keystrokes": "ls -la"}]}
        pred = {"commands": [{"keystrokes": "ls -la"}]}
        assert command_similarity(gt, pred) == 1.0

    def test_different_commands(self):
        """Test similarity of different commands."""
        gt = {"commands": [{"keystrokes": "ls"}]}
        pred = {"commands": [{"keystrokes": "pwd"}]}
        result = command_similarity(gt, pred)
        assert result < 0.5

    def test_both_empty(self):
        """Test similarity when both have empty commands."""
        gt = {"commands": []}
        pred = {"commands": []}
        assert command_similarity(gt, pred) == 1.0

    def test_one_empty(self):
        """Test similarity when one is empty."""
        gt = {"commands": [{"keystrokes": "ls"}]}
        pred = {"commands": []}
        assert command_similarity(gt, pred) == 0.0

    def test_multiple_commands_concatenation(self):
        """Test concatenation of multiple commands."""
        gt = {"commands": [{"keystrokes": "cd /home"}, {"keystrokes": "ls"}]}
        pred = {"commands": [{"keystrokes": "cd /home"}, {"keystrokes": "ls"}]}
        assert command_similarity(gt, pred) == 1.0

    def test_command_order_matters(self):
        """Test that command order affects similarity."""
        gt = {"commands": [{"keystrokes": "ls"}, {"keystrokes": "pwd"}]}
        pred = {"commands": [{"keystrokes": "pwd"}, {"keystrokes": "ls"}]}
        result = command_similarity(gt, pred)
        assert result < 1.0


class TestCheckTaskComplete:
    """Tests for the check_task_complete function."""

    def test_task_complete_true_when_both_true(self):
        """Test when both pred and expected have task_complete=True."""
        pred = {"task_complete": True}
        expected = {"task_complete": True}
        assert check_task_complete(pred, expected) is True

    def test_task_complete_false_when_pred_missing(self):
        """Test when pred is missing task_complete but expected has it."""
        pred = {}
        expected = {"task_complete": True}
        assert check_task_complete(pred, expected) is False

    def test_task_complete_false_when_pred_false(self):
        """Test when pred has task_complete=False but expected has True."""
        pred = {"task_complete": False}
        expected = {"task_complete": True}
        assert check_task_complete(pred, expected) is False

    def test_is_task_complete_true_when_both_true(self):
        """Test when both pred and expected have is_task_complete=True."""
        pred = {"is_task_complete": True}
        expected = {"is_task_complete": True}
        assert check_task_complete(pred, expected) is True

    def test_is_task_complete_false_when_pred_missing(self):
        """Test when pred is missing is_task_complete but expected has it."""
        pred = {}
        expected = {"is_task_complete": True}
        assert check_task_complete(pred, expected) is False

    def test_task_complete_true_when_expected_false(self):
        """Test passes when expected task_complete is False."""
        pred = {}
        expected = {"task_complete": False}
        assert check_task_complete(pred, expected) is True

    def test_task_complete_true_when_not_in_expected(self):
        """Test passes when task_complete is not in expected answer."""
        pred = {}
        expected = {}
        assert check_task_complete(pred, expected) is True


class TestExtractLastAssistantText:
    """Tests for the _extract_last_assistant_text helper function."""

    def _create_verify_request_with_output(self, output_items: list) -> TerminusJudgeVerifyRequest:
        """Helper to create a TerminusJudgeVerifyRequest with specified output items."""
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=output_items,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        return TerminusJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            response=response,
            expected_answer='{"commands": []}',
            metadata={"harness": "terminus_1"},
        )

    def test_extract_single_assistant_message(self):
        """Test extracting text from a single assistant message."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text="Hello response.")],
        )
        body = self._create_verify_request_with_output([output_message])
        result = _extract_last_assistant_text(body)
        assert result == "Hello response."

    def test_extract_multiple_content_parts(self):
        """Test extracting text from message with multiple content parts."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[
                NeMoGymResponseOutputText(annotations=[], text="Part 1."),
                NeMoGymResponseOutputText(annotations=[], text="Part 2."),
            ],
        )
        body = self._create_verify_request_with_output([output_message])
        result = _extract_last_assistant_text(body)
        assert result == "Part 1.\nPart 2."

    def test_extract_ignores_reasoning_items(self):
        """Test that reasoning items are ignored."""
        output_items = [
            NeMoGymResponseReasoningItem(
                id="reasoning_1",
                summary=[NeMoGymSummary(type="summary_text", text="thinking...")],
            ),
            NeMoGymResponseOutputMessage(
                id="msg_1",
                content=[NeMoGymResponseOutputText(annotations=[], text="Actual response.")],
            ),
        ]
        body = self._create_verify_request_with_output(output_items)
        result = _extract_last_assistant_text(body)
        assert result == "Actual response."

    def test_extract_empty_output(self):
        """Test extracting from empty output."""
        body = self._create_verify_request_with_output([])
        result = _extract_last_assistant_text(body)
        assert result == ""


class TestTerminusJudgeResourcesServerVerify:
    """Tests for the TerminusJudgeResourcesServer.verify method."""

    @fixture
    def resources_server(self) -> TerminusJudgeResourcesServer:
        """Create a TerminusJudgeResourcesServer instance for testing."""
        config = TerminusJudgeResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminus_judge_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            string_similarity_threshold=0.8,
        )

        with patch("builtins.open", MagicMock()):
            server = TerminusJudgeResourcesServer(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            server._judge_prompt_template = "Expected: {expected_answer}\nGenerated: {generated_answer}"
            return server

    def _create_verify_request(
        self,
        model_output: str,
        expected_answer: dict,
        harness: str = "terminus_1",
        threshold: float = None,
    ) -> TerminusJudgeVerifyRequest:
        """Helper to create a TerminusJudgeVerifyRequest."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text=model_output)],
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=[output_message],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        return TerminusJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            response=response,
            expected_answer=json.dumps(expected_answer),
            metadata={"harness": harness},
            threshold=threshold,
        )

    @pytest.mark.asyncio
    async def test_verify_correct_prediction_terminus_1(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify returns reward=1.0 for correct terminus_1 prediction."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la"}])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE
        assert response.schema_check_passed is True
        assert response.task_complete_check_passed is True
        assert response.string_similarity_passed is True

    @pytest.mark.asyncio
    async def test_verify_correct_prediction_terminus_2(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify returns reward=1.0 for correct terminus_2 prediction."""
        expected_answer = create_terminus_2_response([{"keystrokes": "ls -la\n"}])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_2")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE

    @pytest.mark.asyncio
    async def test_verify_with_think_tag(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify handles </think> tag correctly."""
        expected_answer = create_terminus_1_response([{"keystrokes": "pwd"}])
        model_output = "<think>Let me think...</think>" + json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.model_output == json.dumps(expected_answer)

    @pytest.mark.asyncio
    async def test_verify_json_parsing_failed(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify returns reward=0.0 for invalid JSON."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        model_output = "not valid json"
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.JSON_PARSING_FAILED

    @pytest.mark.asyncio
    async def test_verify_unknown_harness(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify returns reward=0.0 for unknown harness."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, harness="unknown")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.UNKNOWN_HARNESS

    @pytest.mark.asyncio
    async def test_verify_schema_check_failed_terminus_1(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify returns reward=0.0 for schema validation failure."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        invalid_output = json.dumps({"commands": [{"keystrokes": "ls"}]})
        request = self._create_verify_request(invalid_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.SCHEMA_CHECK_FAILED
        assert response.schema_check_passed is False

    @pytest.mark.asyncio
    async def test_verify_task_complete_check_failed(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify returns reward=0.0 when task_complete check fails."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}], is_task_complete=True)
        pred_answer = create_terminus_1_response([{"keystrokes": "ls"}], is_task_complete=False)
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.TASK_COMPLETE_CHECK_FAILED
        assert response.task_complete_check_passed is False

    @pytest.mark.asyncio
    async def test_verify_string_similarity_below_threshold(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify invokes judge when string similarity is below threshold."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "pwd"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # Mock judge to return not equal
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A!=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.JUDGE_EVALUATION_FAILED
        assert response.string_similarity_passed is False
        assert len(response.judge_evaluations) == 1

    @pytest.mark.asyncio
    async def test_verify_judge_passes_without_swap(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify succeeds when judge passes without swap check."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "completely different command"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # Mock judge to return equal
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE
        assert response.judge_passed is True
        assert len(response.judge_evaluations) == 1

    @pytest.mark.asyncio
    async def test_verify_judge_with_swap_check(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify with swap check enabled."""
        resources_server.config.check_twice_swap = True

        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "completely different command"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # Mock judge to return equal for both calls
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.judge_passed is True
        assert len(response.judge_evaluations) == 2

    @pytest.mark.asyncio
    async def test_verify_custom_threshold(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify uses custom threshold from request."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1", threshold=0.9)

        response = await resources_server.verify(request)

        assert response.threshold == 0.9
        assert response.reward == 1.0

    @pytest.mark.asyncio
    async def test_verify_missing_expected_answer(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify raises error when expected answer is missing."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text="test")],
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=[output_message],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        request = TerminusJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            response=response,
        )

        with pytest.raises(ValueError, match="Expected answer is required"):
            await resources_server.verify(request)
