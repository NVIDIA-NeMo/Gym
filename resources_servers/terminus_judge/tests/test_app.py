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


# Sample rubrics v4 structured output for testing (5-step format, task_complete checked in code)
RUBRICS_V4_EQUAL_RESPONSE = """1. **EXECUTION STAGE**: GOLD=Stage 3 (Execute) vs CANDIDATE=Stage 3 (Execute) - Match
2. **ACTION TYPE**: GOLD=Execute vs CANDIDATE=Execute - Match
3. **SCOPE**: GOLD=runs single script vs CANDIDATE=runs single script - Comparable
4. **FUNCTIONAL EQUIVALENCE**: E1=TRUE, E2=TRUE, E3=TRUE
5. **Final verdict**: [[A=B]]"""

RUBRICS_V4_NOT_EQUAL_RESPONSE = """1. **EXECUTION STAGE**: GOLD=Stage 4 (Verify) vs CANDIDATE=Stage 3 (Execute) - Mismatch
2. **ACTION TYPE**: GOLD=Verify vs CANDIDATE=Execute - Mismatch
3. **SCOPE**: GOLD=verifies 3 outputs vs CANDIDATE=runs script once - Different
4. **FUNCTIONAL EQUIVALENCE**: Not evaluated due to earlier failures
5. **Final verdict**: [[A!=B]]"""

RUBRICS_V4_MULTILINE_REASONING = """Let me analyze these two responses step by step.

1. **EXECUTION STAGE**:
   - GOLD: Stage 3 (Execute) - runs `python reproduce_issue.py`
   - CANDIDATE: Stage 3 (Execute) - also runs `python reproduce_issue.py`
   - Result: Match

2. **ACTION TYPE**:
   - GOLD: Execute/Run script
   - CANDIDATE: Execute/Run script
   - Result: Match

3. **SCOPE**:
   - GOLD: Runs single script with timeout_sec=3
   - CANDIDATE: Runs single script with timeout_sec=5
   - Result: Comparable (timeout difference doesn't affect functional outcome)

4. **FUNCTIONAL EQUIVALENCE**:
   - E1: Same operation type performed? TRUE - both execute the same Python script
   - E2: Same functional outcome achieved? TRUE - both will trigger the NameError
   - E3: Same purpose in task workflow? TRUE - both aim to reproduce the issue

All checks pass.

5. **Final verdict**: [[A=B]]"""


def create_terminus_1_response(
    commands: list,
    is_task_complete: bool = False,
    state_analysis: str = "The terminal shows the current working directory. Previous command completed successfully.",
    explanation: str = "Running the next command to progress the task.",
) -> dict:
    """Create a valid terminus_1 schema response.

    Schema: state_analysis, explanation, commands [{keystrokes, is_blocking, timeout_sec}], is_task_complete
    """
    return {
        "state_analysis": state_analysis,
        "explanation": explanation,
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


def create_terminus_2_response(
    commands: list,
    task_complete: bool = False,
    analysis: str = "The terminal output shows the command executed successfully. Current directory contains the expected files.",
    plan: str = "1. Execute the next command. 2. Verify the output. 3. Complete the task if successful.",
) -> dict:
    """Create a valid terminus_2 schema response.

    Schema: analysis, plan, commands [{keystrokes, duration (optional)}], task_complete (optional)
    """
    return {
        "analysis": analysis,
        "plan": plan,
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
        """Create a TerminusJudgeResourcesServer instance for testing.

        Uses string similarity + LLM judge fallback configuration.
        """
        config = TerminusJudgeResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminus_judge_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            # Verification options: string sim as fast-path, LLM judge as fallback
            enable_string_similarity=True,
            string_similarity_threshold=0.8,
            enable_llm_judge=True,
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
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID

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
        assert response.failure_reason == FailureCode.JUDGE_NOT_EQUIVALENT
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
        """Test verify returns failure code when expected answer is missing."""
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

        result = await resources_server.verify(request)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID


class TestParseVerdictFromText:
    """Tests for the _parse_verdict_from_text method."""

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
            # Verification options (not used in these tests, just for config validity)
            enable_string_similarity=False,
            string_similarity_threshold=0.8,
            enable_llm_judge=True,
        )

        with patch("builtins.open", MagicMock()):
            server = TerminusJudgeResourcesServer(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            server._judge_prompt_template = "Expected: {expected_answer}\nGenerated: {generated_answer}"
            return server

    def test_parse_simple_equal(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing simple [[A=B]] verdict."""
        text = "The commands are equivalent. [[A=B]]"
        is_equal, verdict = resources_server._parse_verdict_from_text(text, "[[A=B]]", "[[A!=B]]")
        assert is_equal is True
        assert verdict == "[[A=B]]"

    def test_parse_simple_not_equal(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing simple [[A!=B]] verdict."""
        text = "The commands are not equivalent. [[A!=B]]"
        is_equal, verdict = resources_server._parse_verdict_from_text(text, "[[A=B]]", "[[A!=B]]")
        assert is_equal is False
        assert verdict == "[[A!=B]]"

    def test_parse_rubrics_v4_equal_structured(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing rubrics v4 structured output with [[A=B]] verdict."""
        is_equal, verdict = resources_server._parse_verdict_from_text(
            RUBRICS_V4_EQUAL_RESPONSE, "[[A=B]]", "[[A!=B]]"
        )
        assert is_equal is True
        assert verdict == "[[A=B]]"

    def test_parse_rubrics_v4_not_equal_structured(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing rubrics v4 structured output with [[A!=B]] verdict."""
        is_equal, verdict = resources_server._parse_verdict_from_text(
            RUBRICS_V4_NOT_EQUAL_RESPONSE, "[[A=B]]", "[[A!=B]]"
        )
        assert is_equal is False
        assert verdict == "[[A!=B]]"

    def test_parse_rubrics_v4_multiline_reasoning(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing rubrics v4 with extensive multi-line reasoning."""
        is_equal, verdict = resources_server._parse_verdict_from_text(
            RUBRICS_V4_MULTILINE_REASONING, "[[A=B]]", "[[A!=B]]"
        )
        assert is_equal is True
        assert verdict == "[[A=B]]"

    def test_parse_verdict_at_very_end(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing when verdict is at the very end after long reasoning."""
        long_reasoning = "A" * 500 + "\n\nFinal verdict: [[A=B]]"
        is_equal, verdict = resources_server._parse_verdict_from_text(long_reasoning, "[[A=B]]", "[[A!=B]]")
        assert is_equal is True
        assert verdict == "[[A=B]]"

    def test_parse_no_verdict_found(self, resources_server: TerminusJudgeResourcesServer):
        """Test handling when no verdict is found in text."""
        text = "The judge failed to provide a clear verdict."
        is_equal, verdict = resources_server._parse_verdict_from_text(text, "[[A=B]]", "[[A!=B]]")
        assert is_equal is False
        assert verdict is None

    def test_parse_both_markers_last_wins(self, resources_server: TerminusJudgeResourcesServer):
        """Test that when both markers appear, the last one (in final section) wins."""
        # This could happen if the judge mentions a previous example
        text = "In some cases [[A!=B]] but in this case [[A=B]]"
        is_equal, verdict = resources_server._parse_verdict_from_text(text, "[[A=B]]", "[[A!=B]]")
        assert is_equal is True
        assert verdict == "[[A=B]]"

    def test_parse_verdict_in_middle_not_end(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing when verdict appears in the middle, not at end."""
        text = "The verdict is [[A=B]] because the commands match. Additional notes follow."
        is_equal, verdict = resources_server._parse_verdict_from_text(text, "[[A=B]]", "[[A!=B]]")
        assert is_equal is True
        assert verdict == "[[A=B]]"

    def test_parse_empty_text(self, resources_server: TerminusJudgeResourcesServer):
        """Test parsing empty text returns no verdict."""
        is_equal, verdict = resources_server._parse_verdict_from_text("", "[[A=B]]", "[[A!=B]]")
        assert is_equal is False
        assert verdict is None


class TestVerifyWithRubricsV4JudgeResponse:
    """Tests for verify method with rubrics v4 style judge responses."""

    @fixture
    def resources_server(self) -> TerminusJudgeResourcesServer:
        """Create a TerminusJudgeResourcesServer instance for testing.

        Uses LLM judge only (no string similarity) to test judge behavior.
        """
        config = TerminusJudgeResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminus_judge_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            # LLM judge only - no string similarity fast-path
            enable_string_similarity=False,
            string_similarity_threshold=0.8,
            enable_llm_judge=True,
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
        )

    @pytest.mark.asyncio
    async def test_verify_with_rubrics_v4_equal_response(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify succeeds with rubrics v4 structured [[A=B]] response."""
        # Use very different keystrokes to ensure similarity is below threshold and judge is called
        expected_answer = create_terminus_1_response([{"keystrokes": "python reproduce_issue.py --verbose\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "python3 reproduce.py -v\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # Mock judge to return rubrics v4 structured response
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
                        "content": [{"type": "output_text", "text": RUBRICS_V4_EQUAL_RESPONSE, "annotations": []}],
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
        assert response.judge_evaluations[0].verdict_label == "[[A=B]]"

    @pytest.mark.asyncio
    async def test_verify_with_rubrics_v4_not_equal_response(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify fails with rubrics v4 structured [[A!=B]] response."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "pwd\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # Mock judge to return rubrics v4 structured not-equal response
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
                        "content": [{"type": "output_text", "text": RUBRICS_V4_NOT_EQUAL_RESPONSE, "annotations": []}],
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
        assert response.failure_reason == FailureCode.JUDGE_NOT_EQUIVALENT
        assert response.judge_passed is False
        assert len(response.judge_evaluations) == 1
        assert response.judge_evaluations[0].verdict_label == "[[A!=B]]"

    @pytest.mark.asyncio
    async def test_verify_with_rubrics_v4_multiline_response(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify handles rubrics v4 multi-line reasoning response."""
        # Use very different keystrokes to ensure similarity is below threshold and judge is called
        expected_answer = create_terminus_1_response([{"keystrokes": "python reproduce_issue.py --timeout 30\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "python3 repro.py -t 30\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # Mock judge to return rubrics v4 multiline reasoning response
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
                        "content": [{"type": "output_text", "text": RUBRICS_V4_MULTILINE_REASONING, "annotations": []}],
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
        assert response.judge_evaluations[0].verdict_label == "[[A=B]]"

    @pytest.mark.asyncio
    async def test_verify_with_no_verdict_in_judge_response(self, resources_server: TerminusJudgeResourcesServer):
        """Test verify handles judge response with no verdict (judge parsing failed)."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "pwd\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # Mock judge to return response without verdict markers
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
                        "content": [{"type": "output_text", "text": "I cannot determine equivalence.", "annotations": []}],
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
        assert response.failure_reason == FailureCode.JUDGE_PARSING_FAILED
        assert response.judge_passed is None  # None because judge didn't return meaningful result
        assert response.judge_evaluations[0].verdict_label is None


class TestVerifyAdditionalScenarios:
    """Additional test scenarios for full coverage."""

    @fixture
    def resources_server_string_sim_only(self) -> TerminusJudgeResourcesServer:
        """Create server with string similarity only (no LLM judge)."""
        config = TerminusJudgeResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminus_judge_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            enable_string_similarity=True,
            string_similarity_threshold=0.8,
            enable_llm_judge=False,
        )

        with patch("builtins.open", MagicMock()):
            server = TerminusJudgeResourcesServer(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            server._judge_prompt_template = "Expected: {expected_answer}\nGenerated: {generated_answer}"
            return server

    @fixture
    def resources_server_neither_enabled(self) -> TerminusJudgeResourcesServer:
        """Create server with neither string sim nor LLM judge enabled."""
        config = TerminusJudgeResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminus_judge_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            enable_string_similarity=False,
            string_similarity_threshold=0.8,
            enable_llm_judge=False,
        )

        with patch("builtins.open", MagicMock()):
            server = TerminusJudgeResourcesServer(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            server._judge_prompt_template = "Expected: {expected_answer}\nGenerated: {generated_answer}"
            return server

    @fixture
    def resources_server_with_swap(self) -> TerminusJudgeResourcesServer:
        """Create server with LLM judge and swap check enabled."""
        config = TerminusJudgeResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="terminus_judge_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            enable_string_similarity=False,
            string_similarity_threshold=0.8,
            enable_llm_judge=True,
            check_twice_swap=True,
            reward_if_swap_fails=0.0,
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
        )

    def _create_verify_request_raw_expected(
        self,
        model_output: str,
        expected_answer_str: str,
        harness: str = "terminus_1",
    ) -> TerminusJudgeVerifyRequest:
        """Helper to create request with raw expected answer string."""
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
            expected_answer=expected_answer_str,
            metadata={"harness": harness},
        )

    # --- String Similarity Only Mode Tests ---

    @pytest.mark.asyncio
    async def test_string_sim_only_passes(self, resources_server_string_sim_only: TerminusJudgeResourcesServer):
        """Test string similarity only mode passes when commands match."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la\n"}])
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server_string_sim_only.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE
        assert response.string_similarity_passed is True
        assert response.judge_passed is None  # Judge not called
        assert len(response.judge_evaluations) == 0

    @pytest.mark.asyncio
    async def test_string_sim_only_fails(self, resources_server_string_sim_only: TerminusJudgeResourcesServer):
        """Test string similarity only mode fails when commands differ (no judge fallback)."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "pwd\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server_string_sim_only.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.STRING_SIMILARITY_BELOW_THRESHOLD
        assert response.string_similarity_passed is False
        assert response.judge_passed is None  # Judge not called
        assert len(response.judge_evaluations) == 0

    # --- Neither Enabled Mode Tests ---

    @pytest.mark.asyncio
    async def test_neither_enabled_passes_on_schema_task_complete(
        self, resources_server_neither_enabled: TerminusJudgeResourcesServer
    ):
        """Test neither enabled mode passes if schema and task_complete pass."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "completely different command\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server_neither_enabled.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE
        assert response.string_similarity_passed is None  # Not attempted
        assert response.judge_passed is None  # Not attempted

    # --- terminus_2 Schema Tests ---

    @pytest.mark.asyncio
    async def test_verify_schema_check_failed_terminus_2(
        self, resources_server_string_sim_only: TerminusJudgeResourcesServer
    ):
        """Test verify returns failure for terminus_2 schema validation failure."""
        expected_answer = create_terminus_2_response([{"keystrokes": "ls\n"}])
        # Missing required 'analysis' field
        invalid_output = json.dumps({"plan": "test", "commands": [{"keystrokes": "ls\n"}]})
        request = self._create_verify_request(invalid_output, expected_answer, "terminus_2")

        response = await resources_server_string_sim_only.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.SCHEMA_CHECK_FAILED
        assert response.schema_check_passed is False

    @pytest.mark.asyncio
    async def test_verify_terminus_2_task_complete_optional(
        self, resources_server_string_sim_only: TerminusJudgeResourcesServer
    ):
        """Test terminus_2 works without task_complete field (it's optional)."""
        expected_answer = {
            "analysis": "Test analysis",
            "plan": "Test plan",
            "commands": [{"keystrokes": "ls\n"}],
            # No task_complete field
        }
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_2")

        response = await resources_server_string_sim_only.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE

    # --- Swap Check Failure Tests ---

    @pytest.mark.asyncio
    async def test_swap_check_first_passes_second_fails(
        self, resources_server_with_swap: TerminusJudgeResourcesServer
    ):
        """Test swap check: first judge passes, second (swap) fails."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "different command\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # First call returns equal, second returns not equal
        call_count = [0]

        async def mock_json():
            call_count[0] += 1
            verdict = "[[A=B]]" if call_count[0] == 1 else "[[A!=B]]"
            return {
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": verdict, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }

        mock_response = MagicMock()
        mock_response.json = mock_json
        resources_server_with_swap.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server_with_swap.verify(request)

        assert response.reward == 0.0  # reward_if_swap_fails = 0.0
        assert response.failure_reason == FailureCode.JUDGE_NOT_EQUIVALENT
        assert response.judge_passed is False
        assert len(response.judge_evaluations) == 2

    @pytest.mark.asyncio
    async def test_swap_check_second_parsing_fails(
        self, resources_server_with_swap: TerminusJudgeResourcesServer
    ):
        """Test swap check: first judge passes, second judge parsing fails."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls -la\n"}])
        pred_answer = create_terminus_1_response([{"keystrokes": "different command\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        # First call returns equal, second returns no verdict
        call_count = [0]

        async def mock_json():
            call_count[0] += 1
            text = "[[A=B]]" if call_count[0] == 1 else "No clear verdict here."
            return {
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }

        mock_response = MagicMock()
        mock_response.json = mock_json
        resources_server_with_swap.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server_with_swap.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.JUDGE_SWAP_PARSING_FAILED
        assert response.judge_passed is None  # Second judge didn't return meaningful result
        assert len(response.judge_evaluations) == 2

    # --- Invalid Expected Answer Tests ---

    @pytest.mark.asyncio
    async def test_expected_answer_invalid_json(
        self, resources_server_string_sim_only: TerminusJudgeResourcesServer
    ):
        """Test verify fails gracefully when expected_answer is invalid JSON."""
        pred_answer = create_terminus_1_response([{"keystrokes": "ls\n"}])
        model_output = json.dumps(pred_answer)
        request = self._create_verify_request_raw_expected(
            model_output, "not valid json {{{", "terminus_1"
        )

        response = await resources_server_string_sim_only.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID

    # --- Empty Model Output Tests ---

    @pytest.mark.asyncio
    async def test_model_output_empty(self, resources_server_string_sim_only: TerminusJudgeResourcesServer):
        """Test verify fails when model output is empty."""
        expected_answer = create_terminus_1_response([{"keystrokes": "ls\n"}])

        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text="")],
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
            expected_answer=json.dumps(expected_answer),
            metadata={"harness": "terminus_1"},
        )

        result = await resources_server_string_sim_only.verify(request)

        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.MODEL_OUTPUT_INVALID

    # --- Multiple Commands Tests ---

    @pytest.mark.asyncio
    async def test_multiple_commands_match(self, resources_server_string_sim_only: TerminusJudgeResourcesServer):
        """Test verify with multiple commands that match."""
        commands = [
            {"keystrokes": "cd /app\n"},
            {"keystrokes": "ls -la\n"},
            {"keystrokes": "cat README.md\n"},
        ]
        expected_answer = create_terminus_1_response(commands)
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_1")

        response = await resources_server_string_sim_only.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE

    @pytest.mark.asyncio
    async def test_multiple_commands_terminus_2(self, resources_server_string_sim_only: TerminusJudgeResourcesServer):
        """Test verify with multiple terminus_2 commands."""
        commands = [
            {"keystrokes": "cd /app\n", "duration": 0.1},
            {"keystrokes": "python3 script.py\n", "duration": 5.0},
            {"keystrokes": "echo 'done'\n", "duration": 0.1},
        ]
        expected_answer = create_terminus_2_response(commands, task_complete=True)
        model_output = json.dumps(expected_answer)
        request = self._create_verify_request(model_output, expected_answer, "terminus_2")

        response = await resources_server_string_sim_only.verify(request)

        assert response.reward == 1.0
        assert response.failure_reason == FailureCode.NONE


class TestNonDictJsonParsing:
    """Tests for the bug where json.loads succeeds but returns a non-dict value.

    Policy models can output weird things like just "True", "123", "[1,2,3]",
    or '"hello"'. These are all valid JSON, so json.loads won't raise
    JSONDecodeError, but the result is not a dict. Without the isinstance check,
    the code would proceed and crash at pydantic validation or .get() calls.

    Same issue applies to expected_answer (ground truth) — if the data has a
    non-dict JSON value, we should fail gracefully with EXPECTED_ANSWER_INVALID.
    """

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
            enable_string_similarity=True,
            string_similarity_threshold=0.8,
            enable_llm_judge=False,
        )

        with patch("builtins.open", MagicMock()):
            server = TerminusJudgeResourcesServer(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            server._judge_prompt_template = "Expected: {expected_answer}\nGenerated: {generated_answer}"
            return server

    def _create_verify_request_raw(
        self,
        model_output: str,
        expected_answer_str: str,
        harness: str = "terminus_1",
    ) -> TerminusJudgeVerifyRequest:
        """Helper to create request with raw strings for both model output and expected answer."""
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
            expected_answer=expected_answer_str,
            metadata={"harness": harness},
        )

    # --- Model output is valid JSON but not a dict ---

    @pytest.mark.asyncio
    async def test_model_output_json_bool_true(self, resources_server: TerminusJudgeResourcesServer):
        """json.loads('true') -> True (bool). This is the most common policy failure."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw("true", json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_json_bool_false(self, resources_server: TerminusJudgeResourcesServer):
        """json.loads('false') -> False (bool)."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw("false", json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_json_null(self, resources_server: TerminusJudgeResourcesServer):
        """json.loads('null') -> None (NoneType)."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw("null", json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_json_number_int(self, resources_server: TerminusJudgeResourcesServer):
        """json.loads('42') -> 42 (int)."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw("42", json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_json_number_float(self, resources_server: TerminusJudgeResourcesServer):
        """json.loads('1.5') -> 1.5 (float)."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw("1.5", json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_json_string(self, resources_server: TerminusJudgeResourcesServer):
        """json.loads('"hello"') -> 'hello' (str). Model just wraps output in quotes."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw('"hello"', json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_json_array(self, resources_server: TerminusJudgeResourcesServer):
        """json.loads('[1, 2, 3]') -> [1, 2, 3] (list). Model outputs a JSON array instead of object."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw("[1, 2, 3]", json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_json_array_of_dicts(self, resources_server: TerminusJudgeResourcesServer):
        """Model outputs a JSON array of dicts — still not a dict at top level."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw('[{"keystrokes": "ls"}]', json.dumps(expected))

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_bool_after_think_tags(self, resources_server: TerminusJudgeResourcesServer):
        """Model outputs <think>...</think>true — after stripping think tags, 'true' parses to bool."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw(
            "<think>Let me think about this...</think>true", json.dumps(expected)
        )

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_model_output_number_after_think_tags(self, resources_server: TerminusJudgeResourcesServer):
        """Model outputs <think>reasoning</think>42 — bare number after think tags."""
        expected = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw(
            "<think>The answer is 42</think>42", json.dumps(expected)
        )

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        assert response.parsed_output is None

    # --- Expected answer is valid JSON but not a dict ---

    @pytest.mark.asyncio
    async def test_expected_answer_json_bool(self, resources_server: TerminusJudgeResourcesServer):
        """Expected answer is 'true' — valid JSON, not a dict (data issue)."""
        pred = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw(json.dumps(pred), "true")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID

    @pytest.mark.asyncio
    async def test_expected_answer_json_null(self, resources_server: TerminusJudgeResourcesServer):
        """Expected answer is 'null' — valid JSON, not a dict (data issue)."""
        pred = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw(json.dumps(pred), "null")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID

    @pytest.mark.asyncio
    async def test_expected_answer_json_number(self, resources_server: TerminusJudgeResourcesServer):
        """Expected answer is '42' — valid JSON, not a dict (data issue)."""
        pred = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw(json.dumps(pred), "42")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID

    @pytest.mark.asyncio
    async def test_expected_answer_json_string(self, resources_server: TerminusJudgeResourcesServer):
        """Expected answer is '"hello"' — valid JSON string, not a dict (data issue)."""
        pred = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw(json.dumps(pred), '"hello"')

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID

    @pytest.mark.asyncio
    async def test_expected_answer_json_array(self, resources_server: TerminusJudgeResourcesServer):
        """Expected answer is '[1, 2, 3]' — valid JSON array, not a dict (data issue)."""
        pred = create_terminus_1_response([{"keystrokes": "ls\n"}])
        request = self._create_verify_request_raw(json.dumps(pred), "[1, 2, 3]")

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID

    # --- Both non-dict: model output AND expected answer ---

    @pytest.mark.asyncio
    async def test_both_non_dict_expected_fails_first(self, resources_server: TerminusJudgeResourcesServer):
        """When both are non-dict JSON, expected_answer check fails first."""
        request = self._create_verify_request_raw("true", "true")

        response = await resources_server.verify(request)

        # Expected answer is checked before model output
        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.EXPECTED_ANSWER_INVALID
