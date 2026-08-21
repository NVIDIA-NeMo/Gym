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
import json
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
)
from responses_api_agents.prorl_agent.run_prorl import (
    call_prorl_process,
    convert_prorl_messages_to_output_items,
    initialize_prorl_server,
)


class TestConvertProrlMessagesToOutputItems:
    """Tests for convert_prorl_messages_to_output_items."""

    def test_empty_messages(self) -> None:
        """Test with empty messages list."""
        result = convert_prorl_messages_to_output_items([])
        assert result == []

    def test_system_message(self) -> None:
        """Test conversion of system message."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 1
        assert isinstance(result[0], NeMoGymMessage)
        assert result[0].role == "system"
        assert result[0].content[0]["type"] == "input_text"
        assert result[0].content[0]["text"] == "You are a helpful assistant."

    def test_user_message(self) -> None:
        """Test conversion of user message."""
        messages = [{"role": "user", "content": "Fix this bug."}]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 1
        assert isinstance(result[0], NeMoGymMessage)
        assert result[0].role == "user"

    def test_developer_message(self) -> None:
        """Test conversion of developer message."""
        messages = [{"role": "developer", "content": "Developer instructions."}]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 1
        assert isinstance(result[0], NeMoGymMessage)
        assert result[0].role == "developer"

    def test_system_message_empty_content_skipped(self) -> None:
        """Test that system messages with empty content are skipped."""
        messages = [{"role": "system", "content": ""}]
        result = convert_prorl_messages_to_output_items(messages)
        assert result == []

    def test_assistant_message(self) -> None:
        """Test conversion of assistant message."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll fix that.",
                "token_ids": [10, 20, 30],
                "input_ids": [1, 2, 3],
                "logprobs": [-0.1, -0.2, -0.3],
            }
        ]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 1
        assert isinstance(result[0], NeMoGymResponseOutputMessageForTraining)
        assert result[0].role == "assistant"
        assert result[0].content[0].text == "I'll fix that."
        assert result[0].generation_token_ids == [10, 20, 30]
        assert result[0].prompt_token_ids == [1, 2, 3]
        assert result[0].generation_log_probs == [-0.1, -0.2, -0.3]

    def test_assistant_message_without_token_ids(self) -> None:
        """Test assistant message without token ID fields."""
        messages = [{"role": "assistant", "content": "response"}]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 1
        assert isinstance(result[0], NeMoGymResponseOutputMessageForTraining)
        assert result[0].generation_token_ids == []
        assert result[0].prompt_token_ids == []
        assert result[0].generation_log_probs == []

    def test_assistant_message_with_tool_calls(self) -> None:
        """Test assistant message with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me run a command.",
                "tool_calls": [
                    {"name": "bash", "arguments": '{"command": "ls"}'},
                ],
            }
        ]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 2
        # First is the assistant message
        assert isinstance(result[0], NeMoGymResponseOutputMessageForTraining)
        assert result[0].role == "assistant"
        # Second is the tool call
        assert isinstance(result[1], NeMoGymResponseFunctionToolCall)
        assert result[1].name == "bash"
        assert result[1].arguments == '{"command": "ls"}'
        assert result[1].call_id == "call_1"
        assert result[1].type == "function_call"

    def test_assistant_message_with_dict_arguments(self) -> None:
        """Test that dict arguments in tool calls are JSON-serialized."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "edit", "arguments": {"file": "test.py", "content": "print()"}},
                ],
            }
        ]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 2
        tool_call = result[1]
        assert isinstance(tool_call, NeMoGymResponseFunctionToolCall)
        parsed = json.loads(tool_call.arguments)
        assert parsed["file"] == "test.py"

    def test_tool_message_matched_to_pending_call(self) -> None:
        """Test tool message is matched to pending call ID."""
        messages = [
            {
                "role": "assistant",
                "content": "Running command.",
                "tool_calls": [{"name": "bash", "arguments": "{}"}],
            },
            {"role": "tool", "content": "command output"},
        ]
        result = convert_prorl_messages_to_output_items(messages)
        # assistant msg + tool call + tool output
        assert len(result) == 3
        assert isinstance(result[2], NeMoGymFunctionCallOutput)
        assert result[2].call_id == "call_1"
        assert result[2].output == "command output"

    def test_tool_message_with_explicit_tool_call_id(self) -> None:
        """Test tool message with explicit tool_call_id."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "bash", "arguments": "{}"}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 3
        assert isinstance(result[2], NeMoGymFunctionCallOutput)
        assert result[2].call_id == "call_1"

    def test_tool_message_without_matching_call_id_skipped(self) -> None:
        """Test that tool messages without a matching call ID are skipped."""
        messages = [{"role": "tool", "content": "orphan output"}]
        result = convert_prorl_messages_to_output_items(messages)
        assert result == []

    def test_multiple_tool_calls_and_responses(self) -> None:
        """Test multi-turn interaction with multiple tool calls."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"name": "bash", "arguments": '{"cmd": "cat file.py"}'},
                    {"name": "edit", "arguments": '{"file": "file.py"}'},
                ],
                "token_ids": [100, 200],
                "input_ids": [1, 2],
                "logprobs": [-0.5, -0.6],
            },
            {"role": "tool", "content": "file contents"},
            {"role": "tool", "content": "edit success"},
            {
                "role": "assistant",
                "content": "Done.",
                "token_ids": [300],
                "input_ids": [3],
                "logprobs": [-0.1],
            },
        ]
        result = convert_prorl_messages_to_output_items(messages)

        # system(1) + user(1) + assistant(1) + 2 tool calls + 2 tool outputs + assistant(1) = 8
        assert len(result) == 8
        assert isinstance(result[0], NeMoGymMessage)  # system
        assert isinstance(result[1], NeMoGymMessage)  # user
        assert isinstance(result[2], NeMoGymResponseOutputMessageForTraining)  # assistant
        assert isinstance(result[3], NeMoGymResponseFunctionToolCall)  # bash call
        assert isinstance(result[4], NeMoGymResponseFunctionToolCall)  # edit call
        assert isinstance(result[5], NeMoGymFunctionCallOutput)  # bash output
        assert isinstance(result[6], NeMoGymFunctionCallOutput)  # edit output
        assert isinstance(result[7], NeMoGymResponseOutputMessageForTraining)  # final assistant

        # Verify call IDs match
        assert result[3].call_id == "call_1"
        assert result[4].call_id == "call_2"
        assert result[5].call_id == "call_1"
        assert result[6].call_id == "call_2"

    def test_none_content_treated_as_empty(self) -> None:
        """Test that None content is treated as empty string."""
        messages = [{"role": "assistant", "content": None}]
        result = convert_prorl_messages_to_output_items(messages)
        assert len(result) == 1
        assert result[0].content[0].text == ""

    def test_assistant_message_id_increments(self) -> None:
        """Test that message IDs increment correctly."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        result = convert_prorl_messages_to_output_items(messages)
        assert result[0].content[0]["text"] == "sys"
        assert result[1].id == "msg-1"
        assert result[2].id == "msg-2"


class TestCallProrlProcess:
    """Tests for call_prorl_process."""

    @pytest.mark.asyncio
    async def test_call_prorl_process_success(self) -> None:
        """Test successful call to ProRL /process endpoint."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = lambda: self._async_return({"resolved": True, "messages": []})

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__aenter__ = lambda s: self._async_return(mock_session)
            mock_session.__aexit__ = lambda s, *a: self._async_return(None)

            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = lambda s: self._async_return(mock_response)
            mock_post_ctx.__aexit__ = lambda s, *a: self._async_return(None)

            mock_session.post.return_value = mock_post_ctx
            mock_session_cls.return_value = mock_session

            result = await call_prorl_process(
                prorl_url="http://localhost:8006",
                instance={"instance_id": "test"},
                sampling_params={"temperature": 0.6},
                job_id="test-job",
            )
            assert result == {"resolved": True, "messages": []}

    @staticmethod
    async def _async_return(value):
        return value

    @pytest.mark.asyncio
    async def test_call_prorl_process_includes_job_id(self) -> None:
        """Test that job_id is included in request when provided."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = lambda: self._async_return({"resolved": False})

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__aenter__ = lambda s: self._async_return(mock_session)
            mock_session.__aexit__ = lambda s, *a: self._async_return(None)

            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = lambda s: self._async_return(mock_response)
            mock_post_ctx.__aexit__ = lambda s, *a: self._async_return(None)

            mock_session.post.return_value = mock_post_ctx
            mock_session_cls.return_value = mock_session

            await call_prorl_process(
                prorl_url="http://localhost:8006",
                instance={"instance_id": "test"},
                sampling_params={},
                job_id="my-job-id",
            )
            # Verify post was called with correct URL and body containing job_id
            call_args = mock_session.post.call_args
            assert call_args[0][0] == "http://localhost:8006/process"
            assert call_args[1]["json"]["job_id"] == "my-job-id"

    @pytest.mark.asyncio
    async def test_call_prorl_process_no_job_id(self) -> None:
        """Test that job_id is omitted from request when not provided."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = lambda: self._async_return({"resolved": False})

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__aenter__ = lambda s: self._async_return(mock_session)
            mock_session.__aexit__ = lambda s, *a: self._async_return(None)

            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = lambda s: self._async_return(mock_response)
            mock_post_ctx.__aexit__ = lambda s, *a: self._async_return(None)

            mock_session.post.return_value = mock_post_ctx
            mock_session_cls.return_value = mock_session

            await call_prorl_process(
                prorl_url="http://localhost:8006",
                instance={"instance_id": "test"},
                sampling_params={},
                job_id=None,
            )
            call_args = mock_session.post.call_args
            assert "job_id" not in call_args[1]["json"]


class TestInitializeProrlServer:
    """Tests for initialize_prorl_server."""

    def _mock_requests_module(self, mock_post):
        """Create a mock requests module with the given post mock."""
        mock_requests = MagicMock()
        mock_requests.post = mock_post
        return mock_requests

    def test_initialize_success(self) -> None:
        """Test successful initialization."""
        mock_start_resp = MagicMock()
        mock_start_resp.status_code = 200
        mock_start_resp.raise_for_status = MagicMock()

        mock_add_resp = MagicMock()
        mock_add_resp.raise_for_status = MagicMock()

        mock_post = MagicMock(side_effect=[mock_start_resp, mock_add_resp])
        mock_requests = self._mock_requests_module(mock_post)

        with patch.dict("sys.modules", {"requests": mock_requests}):
            initialize_prorl_server("http://localhost:8006", "http://localhost:8000/v1")

        assert mock_post.call_count == 2
        # First call: /start
        assert mock_post.call_args_list[0][0][0] == "http://localhost:8006/start"
        # Second call: /add_llm_server
        assert mock_post.call_args_list[1][0][0] == "http://localhost:8006/add_llm_server"
        assert mock_post.call_args_list[1][1]["json"] == {"address": "http://localhost:8000/v1"}

    def test_initialize_already_running(self) -> None:
        """Test initialization when server is already running (400 from /start)."""
        mock_start_resp = MagicMock()
        mock_start_resp.status_code = 400

        mock_add_resp = MagicMock()
        mock_add_resp.raise_for_status = MagicMock()

        mock_post = MagicMock(side_effect=[mock_start_resp, mock_add_resp])
        mock_requests = self._mock_requests_module(mock_post)

        with patch.dict("sys.modules", {"requests": mock_requests}):
            # Should not raise
            initialize_prorl_server("http://localhost:8006", "http://localhost:8000/v1")

        assert mock_post.call_count == 2

    def test_initialize_start_fails(self) -> None:
        """Test initialization when /start fails with non-400 error."""
        mock_start_resp = MagicMock()
        mock_start_resp.status_code = 500
        mock_start_resp.raise_for_status.side_effect = Exception("Server error")

        mock_add_resp = MagicMock()
        mock_add_resp.raise_for_status = MagicMock()

        mock_post = MagicMock(side_effect=[mock_start_resp, mock_add_resp])
        mock_requests = self._mock_requests_module(mock_post)

        with patch.dict("sys.modules", {"requests": mock_requests}):
            # Should not raise (prints warning instead)
            initialize_prorl_server("http://localhost:8006", "http://localhost:8000/v1")

        assert mock_post.call_count == 2

    def test_initialize_add_llm_fails(self) -> None:
        """Test initialization when /add_llm_server fails."""
        mock_start_resp = MagicMock()
        mock_start_resp.status_code = 200
        mock_start_resp.raise_for_status = MagicMock()

        mock_add_resp = MagicMock()
        mock_add_resp.raise_for_status.side_effect = Exception("Connection refused")

        mock_post = MagicMock(side_effect=[mock_start_resp, mock_add_resp])
        mock_requests = self._mock_requests_module(mock_post)

        with patch.dict("sys.modules", {"requests": mock_requests}):
            # Should not raise (prints warning instead)
            initialize_prorl_server("http://localhost:8006", "http://localhost:8000/v1")

        assert mock_post.call_count == 2
