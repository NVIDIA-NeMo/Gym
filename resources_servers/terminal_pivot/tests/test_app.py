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
from unittest.mock import MagicMock

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
from resources_servers.terminal_pivot.app import (
    FailureCode,
    TBResourcesServer,
    TBResourcesServerConfig,
    TBVerifyRequest,
    _extract_last_assistant_text,
    check_command_correctness,
    check_schema,
    check_task_complete,
)


# Helper functions to create valid schema-compliant test data
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


class TestExtractLastAssistantText:
    """Tests for the _extract_last_assistant_text helper function."""

    def _create_verify_request_with_output(
        self, output_items: list
    ) -> TBVerifyRequest:
        """Helper to create a TBVerifyRequest with specified output items."""
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
        return TBVerifyRequest(
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
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="Hello, this is the assistant response.",
                )
            ],
        )
        body = self._create_verify_request_with_output([output_message])
        result = _extract_last_assistant_text(body)
        assert result == "Hello, this is the assistant response."

    def test_extract_multiple_assistant_messages(self):
        """Test extracting text from multiple assistant messages."""
        output_messages = [
            NeMoGymResponseOutputMessage(
                id="msg_1",
                content=[
                    NeMoGymResponseOutputText(annotations=[], text="First message.")
                ],
            ),
            NeMoGymResponseOutputMessage(
                id="msg_2",
                content=[
                    NeMoGymResponseOutputText(annotations=[], text="Second message.")
                ],
            ),
        ]
        body = self._create_verify_request_with_output(output_messages)
        result = _extract_last_assistant_text(body)
        assert result == "First message.\nSecond message."

    def test_extract_with_multiple_content_parts(self):
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
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[], text="Actual response."
                    )
                ],
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


class TestCheckTaskComplete:
    """Tests for the check_task_complete function."""

    def test_task_complete_true_when_both_true(self):
        """Test when both pred and expected have task_complete=True."""
        pred = {"task_complete": True, "commands": []}
        expected = {"task_complete": True, "commands": []}
        assert check_task_complete(pred, expected) is True

    def test_task_complete_false_when_pred_missing(self):
        """Test when pred is missing task_complete but expected has it."""
        pred = {"commands": []}
        expected = {"task_complete": True, "commands": []}
        assert check_task_complete(pred, expected) is False

    def test_task_complete_false_when_pred_false(self):
        """Test when pred has task_complete=False but expected has True."""
        pred = {"task_complete": False, "commands": []}
        expected = {"task_complete": True, "commands": []}
        assert check_task_complete(pred, expected) is False

    def test_is_task_complete_true_when_both_true(self):
        """Test when both pred and expected have is_task_complete=True."""
        pred = {"is_task_complete": True, "commands": []}
        expected = {"is_task_complete": True, "commands": []}
        assert check_task_complete(pred, expected) is True

    def test_is_task_complete_false_when_pred_missing(self):
        """Test when pred is missing is_task_complete but expected has it."""
        pred = {"commands": []}
        expected = {"is_task_complete": True, "commands": []}
        assert check_task_complete(pred, expected) is False

    def test_task_complete_true_when_expected_false(self):
        """Test passes when expected task_complete is False."""
        pred = {"commands": []}
        expected = {"task_complete": False, "commands": []}
        assert check_task_complete(pred, expected) is True

    def test_task_complete_true_when_not_in_expected(self):
        """Test passes when task_complete is not in expected answer."""
        pred = {"commands": []}
        expected = {"commands": []}
        assert check_task_complete(pred, expected) is True


class TestCheckSchema:
    """Tests for the check_schema function."""

    def test_valid_schema(self):
        """Test valid schema passes."""
        pred = {"commands": [{"keystrokes": "ls -la"}]}
        expected = {"commands": [{"keystrokes": "ls -la"}]}
        assert check_schema(pred, expected) is True

    def test_missing_commands_key(self):
        """Test schema fails when commands key is missing."""
        pred = {"other": "value"}
        expected = {"commands": []}
        assert check_schema(pred, expected) is False

    def test_commands_not_list(self):
        """Test schema fails when commands is not a list."""
        pred = {"commands": "not a list"}
        expected = {"commands": []}
        assert check_schema(pred, expected) is False

    def test_command_not_dict(self):
        """Test schema fails when a command is not a dict."""
        pred = {"commands": ["not a dict"]}
        expected = {"commands": []}
        assert check_schema(pred, expected) is False

    def test_command_missing_keystrokes(self):
        """Test schema fails when command is missing keystrokes."""
        pred = {"commands": [{"other": "value"}]}
        expected = {"commands": []}
        assert check_schema(pred, expected) is False

    def test_keystrokes_not_string(self):
        """Test schema fails when keystrokes is not a string."""
        pred = {"commands": [{"keystrokes": 123}]}
        expected = {"commands": []}
        assert check_schema(pred, expected) is False

    def test_empty_commands_list(self):
        """Test valid schema with empty commands list."""
        pred = {"commands": []}
        expected = {"commands": []}
        assert check_schema(pred, expected) is True

    def test_multiple_valid_commands(self):
        """Test valid schema with multiple commands."""
        pred = {
            "commands": [
                {"keystrokes": "cd /home"},
                {"keystrokes": "ls"},
                {"keystrokes": "pwd"},
            ]
        }
        expected = {"commands": []}
        assert check_schema(pred, expected) is True


class TestCheckCommandCorrectness:
    """Tests for the check_command_correctness function."""

    def test_matching_commands(self):
        """Test matching commands return True."""
        pred = {"commands": [{"keystrokes": "ls -la"}]}
        expected = {"commands": [{"keystrokes": "ls -la"}]}
        assert check_command_correctness(pred, expected) is True

    def test_different_keystrokes(self):
        """Test different keystrokes return False."""
        pred = {"commands": [{"keystrokes": "ls"}]}
        expected = {"commands": [{"keystrokes": "ls -la"}]}
        assert check_command_correctness(pred, expected) is False

    def test_different_length(self):
        """Test different number of commands return False."""
        pred = {"commands": [{"keystrokes": "ls"}]}
        expected = {"commands": [{"keystrokes": "ls"}, {"keystrokes": "pwd"}]}
        assert check_command_correctness(pred, expected) is False

    def test_empty_commands(self):
        """Test empty commands match."""
        pred = {"commands": []}
        expected = {"commands": []}
        assert check_command_correctness(pred, expected) is True

    def test_multiple_matching_commands(self):
        """Test multiple matching commands."""
        pred = {
            "commands": [
                {"keystrokes": "cd /home"},
                {"keystrokes": "ls -la"},
                {"keystrokes": "cat file.txt"},
            ]
        }
        expected = {
            "commands": [
                {"keystrokes": "cd /home"},
                {"keystrokes": "ls -la"},
                {"keystrokes": "cat file.txt"},
            ]
        }
        assert check_command_correctness(pred, expected) is True

    def test_order_matters(self):
        """Test that command order matters."""
        pred = {
            "commands": [{"keystrokes": "pwd"}, {"keystrokes": "ls"}]
        }
        expected = {
            "commands": [{"keystrokes": "ls"}, {"keystrokes": "pwd"}]
        }
        assert check_command_correctness(pred, expected) is False


class TestTBResourcesServerVerify:
    """Tests for the TBResourcesServer.verify method."""

    @fixture
    def resources_server(self) -> TBResourcesServer:
        """Create a TBResourcesServer instance for testing."""
        config = TBResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminal_pivot_test_server",
        )
        return TBResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

    def _create_verify_request(
        self,
        model_output: str,
        expected_answer: dict,
        harness: str = "terminus_1",
    ) -> TBVerifyRequest:
        """Helper to create a TBVerifyRequest."""
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
        return TBVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            response=response,
            expected_answer=json.dumps(expected_answer),
            metadata={"harness": harness},
        )

    @pytest.mark.asyncio
    async def test_verify_correct_prediction_terminus_1(self, resources_server: TBResourcesServer):
        """Test verify returns reward=1.0 for correct terminus_1 prediction."""
        expected_answer = create_terminus_1_response(
            [{"keystrokes": "ls -la"}], is_task_complete=False
        )
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason is None

    @pytest.mark.asyncio
    async def test_verify_correct_prediction_terminus_2(self, resources_server: TBResourcesServer):
        """Test verify returns reward=1.0 for correct terminus_2 prediction."""
        expected_answer = create_terminus_2_response(
            [{"keystrokes": "ls -la\n"}], task_complete=False
        )
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_2")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason is None

    @pytest.mark.asyncio
    async def test_verify_with_think_tag(self, resources_server: TBResourcesServer):
        """Test verify handles </think> tag correctly."""
        expected_answer = create_terminus_1_response([{"keystrokes": "pwd"}])
        model_output = "<think>Let me think about this...</think>" + json.dumps(
            expected_answer
        )
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason is None

    @pytest.mark.asyncio
    async def test_verify_json_parsing_failed(self, resources_server: TBResourcesServer):
        """Test verify returns reward=0.0 for invalid JSON."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        model_output = "not valid json"
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.JSON_PARSING_FAILED

    @pytest.mark.asyncio
    async def test_verify_unknown_harness(self, resources_server: TBResourcesServer):
        """Test verify returns reward=0.0 for unknown harness."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(
            model_output, expected_answer, harness="unknown_harness"
        )

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.UNKNOWN_HARNESS

    @pytest.mark.asyncio
    async def test_verify_schema_check_failed_terminus_1(self, resources_server: TBResourcesServer):
        """Test verify returns reward=0.0 for schema validation failure in terminus_1."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        # Invalid schema - missing required fields for terminus_1
        invalid_output = json.dumps({"commands": [{"keystrokes": "ls"}]})
        request = self._create_verify_request(invalid_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.SCHEMA_CHECK_FAILED

    @pytest.mark.asyncio
    async def test_verify_schema_check_failed_terminus_2(self, resources_server: TBResourcesServer):
        """Test verify returns reward=0.0 for schema validation failure in terminus_2."""
        expected_answer = create_terminus_2_response([{"keystrokes": "ls"}])
        # Invalid schema - missing required fields for terminus_2
        invalid_output = json.dumps({"commands": [{"keystrokes": "ls"}]})
        request = self._create_verify_request(invalid_output, expected_answer, "terminus_2")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.SCHEMA_CHECK_FAILED

    @pytest.mark.asyncio
    async def test_verify_task_complete_check_failed_terminus_1(
        self, resources_server: TBResourcesServer
    ):
        """Test verify returns reward=0.0 when is_task_complete check fails in terminus_1."""
        # Expected has is_task_complete=True
        expected_answer = create_terminus_1_response(
            [{"keystrokes": "ls"}], is_task_complete=True
        )
        # Prediction has is_task_complete=False
        pred_answer = create_terminus_1_response(
            [{"keystrokes": "ls"}], is_task_complete=False
        )
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.TASK_COMPLETE_CHECK_FAILED

    @pytest.mark.asyncio
    async def test_verify_task_complete_check_failed_terminus_2(
        self, resources_server: TBResourcesServer
    ):
        """Test verify returns reward=0.0 when task_complete check fails in terminus_2."""
        # Expected has task_complete=True
        expected_answer = create_terminus_2_response(
            [{"keystrokes": "ls"}], task_complete=True
        )
        # Prediction has task_complete=False
        pred_answer = create_terminus_2_response(
            [{"keystrokes": "ls"}], task_complete=False
        )
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_2")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.TASK_COMPLETE_CHECK_FAILED

    @pytest.mark.asyncio
    async def test_verify_command_correctness_failed(
        self, resources_server: TBResourcesServer
    ):
        """Test verify returns reward=0.0 when commands don't match."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "pwd"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.COMMAND_CORRECTNESS_FAILED

    @pytest.mark.asyncio
    async def test_verify_empty_commands(self, resources_server: TBResourcesServer):
        """Test verify with empty commands list."""
        expected_answer = create_terminus_1_response([])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason is None

    @pytest.mark.asyncio
    async def test_verify_multiple_commands(self, resources_server: TBResourcesServer):
        """Test verify with multiple commands."""
        expected_answer = create_terminus_1_response([
            {"keystrokes": "cd /home"},
            {"keystrokes": "ls"},
            {"keystrokes": "cat file.txt"},
        ])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason is None

    @pytest.mark.asyncio
    async def test_verify_response_contains_input_data(
        self, resources_server: TBResourcesServer
    ):
        """Test verify response contains original input data."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.expected_answer == json.dumps(expected_answer)
        assert response.model_output == model_output
        assert response.metadata == {"harness": "terminus_1"}

    @pytest.mark.asyncio
    async def test_verify_missing_harness(self, resources_server: TBResourcesServer):
        """Test verify handles missing harness in metadata."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls"}])
        model_output = json.dumps(expected_answer)
        
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
        request = TBVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            response=response,
            expected_answer=json.dumps(expected_answer),
            metadata={},  # Missing harness
        )

        verify_response = await resources_server.verify(request)

        assert verify_response.reward == 0.0
        assert verify_response.failure_reason == FailureCode.UNKNOWN_HARNESS

    @pytest.mark.asyncio
    async def test_verify_terminus_2_with_duration(self, resources_server: TBResourcesServer):
        """Test verify works with terminus_2 schema including duration."""
        expected_answer = create_terminus_2_response([
            {"keystrokes": "ls -la\n", "duration": 0.1},
            {"keystrokes": "pwd\n", "duration": 0.5},
        ])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_2")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason is None

    @pytest.mark.asyncio
    async def test_verify_terminus_1_with_blocking_commands(
        self, resources_server: TBResourcesServer
    ):
        """Test verify works with terminus_1 schema including blocking commands."""
        expected_answer = create_terminus_1_response([
            {"keystrokes": "make build", "is_blocking": True, "timeout_sec": 60.0},
            {"keystrokes": "ls output/", "is_blocking": False, "timeout_sec": 5.0},
        ])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason is None

    @pytest.mark.asyncio
    async def test_verify_wrong_command_count(self, resources_server: TBResourcesServer):
        """Test verify fails when prediction has different number of commands."""
        expected_answer = create_terminus_1_response([
            {"keystrokes": "cd /home"},
            {"keystrokes": "ls"},
        ])
        pred_answer = create_terminus_1_response([{"keystrokes": "cd /home"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.COMMAND_CORRECTNESS_FAILED
