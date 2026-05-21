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

"""Tests for the multi-turn agent server.

The multi-turn agent orchestrates a conversation between a policy model and a
user model (LLM simulating the human). These tests focus on:

- Config validation (e.g. user_model_tool_choice field)
- _generate_user_response behavior: tool_choice passthrough, text extraction,
  fallback when user model only makes tool calls, and input construction.

All model/resources server calls are mocked via ServerClient. The mock helpers
below build minimal OpenAI Responses API payloads that the agent code parses.
"""

import json
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.multi_turn_agent.app import (
    MultiTurnAgent,
    MultiTurnAgentConfig,
    MultiTurnAgentRunRequest,
)


# ---------------------------------------------------------------------------
# Helpers: config, request, and mock response factories
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> MultiTurnAgentConfig:
    """Create a MultiTurnAgentConfig with sensible test defaults.

    Pass keyword arguments to override any field, e.g.
    _make_config(user_model_tool_choice="required").
    """
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server={"type": "resources_servers", "name": "test_resources"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        user_model_server={"type": "responses_api_models", "name": "user_model"},
        max_turns=5,
        user_model_system_prompt="You are a test user.",
    )
    return MultiTurnAgentConfig(**(defaults | overrides))


# Minimal tool definition matching the OpenAI FunctionToolParam schema.
# Used in run requests so the agent has tools to pass to the user model.
_TEST_TOOL = {
    "name": "make_move",
    "type": "function",
    "parameters": {"type": "object", "properties": {"position": {"type": "integer"}}, "required": ["position"]},
    "strict": True,
    "description": "Place a mark",
}


def _make_run_request(**overrides) -> MultiTurnAgentRunRequest:
    """Create a run request with a single user message and one tool."""
    defaults = dict(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Hello"}],
            tools=[_TEST_TOOL],
        )
    )
    return MultiTurnAgentRunRequest(**(defaults | overrides))


def _user_model_text_response(text: str) -> dict:
    """Mock a user model API response that contains only a text message."""
    return {
        "output": [
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
    }


def _user_model_tool_call_response(call_id: str, name: str, arguments: str) -> dict:
    """Mock a user model API response that contains only a tool call (no text)."""
    return {
        "output": [
            {
                "arguments": arguments,
                "call_id": call_id,
                "name": name,
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
            }
        ],
    }


def _user_model_tool_call_and_text_response(call_id: str, name: str, arguments: str, text: str) -> dict:
    """Mock a user model API response that contains both a tool call and text."""
    return {
        "output": [
            {
                "arguments": arguments,
                "call_id": call_id,
                "name": name,
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
            },
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestApp:
    def test_sanity(self) -> None:
        config = _make_config()
        MultiTurnAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_user_model_tool_choice_default(self) -> None:
        config = _make_config()
        assert config.user_model_tool_choice is None

    def test_config_user_model_tool_choice_required(self) -> None:
        config = _make_config(user_model_tool_choice="required")
        assert config.user_model_tool_choice == "required"


# ---------------------------------------------------------------------------
# _generate_user_response tests
#
# These test the user model interaction method directly. Each test mocks
# server_client.post to simulate the user model's API responses and (when
# tool calls are involved) the resources server's tool execution responses.
#
# The call sequence for server_client.post in _generate_user_response is:
#   1. POST to user_model_server /v1/responses  (get model output)
#   2. POST to resources_server /<tool_name>     (execute tool, if any)
#   3. Back to step 1 (loop until text or safety limit)
# ---------------------------------------------------------------------------


class TestGenerateUserResponse:
    async def _call_generate(self, server: MultiTurnAgent, body=None) -> str | None:
        """Shorthand to call _generate_user_response with minimal arguments."""
        if body is None:
            body = _make_run_request()
        return await server._generate_user_response(
            body=body,
            original_input=[{"role": "user", "content": "Hello"}],
            all_turn_outputs=[],
            cookies={},
        )

    async def test_text_only_response(self) -> None:
        """User model returns text — should be returned directly."""
        server = MultiTurnAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        mock_resp = AsyncMock()
        mock_resp.read.return_value = json.dumps(_user_model_text_response("Your turn!"))
        mock_resp.cookies = {}
        server.server_client.post.return_value = mock_resp

        result = await self._call_generate(server)
        assert result == "Your turn!"

    async def test_tool_choice_passed_when_configured(self) -> None:
        """When user_model_tool_choice="required", it appears in the API call JSON."""
        server = MultiTurnAgent(
            config=_make_config(user_model_tool_choice="required"),
            server_client=MagicMock(spec=ServerClient),
        )

        mock_resp = AsyncMock()
        mock_resp.read.return_value = json.dumps(_user_model_text_response("Done."))
        mock_resp.cookies = {}
        server.server_client.post.return_value = mock_resp

        await self._call_generate(server)

        # Inspect the JSON body sent to the user model server
        call_kwargs = server.server_client.post.call_args
        sent_json = call_kwargs.kwargs["json"]
        assert sent_json["tool_choice"] == "required"

    async def test_tool_choice_not_passed_when_none(self) -> None:
        """When user_model_tool_choice is None (default), tool_choice is omitted
        so the API uses its own default ("auto")."""
        server = MultiTurnAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        mock_resp = AsyncMock()
        mock_resp.read.return_value = json.dumps(_user_model_text_response("Hi."))
        mock_resp.cookies = {}
        server.server_client.post.return_value = mock_resp

        await self._call_generate(server)

        call_kwargs = server.server_client.post.call_args
        sent_json = call_kwargs.kwargs["json"]
        assert "tool_choice" not in sent_json

    async def test_fallback_to_tool_result_when_no_text(self) -> None:
        """When user model makes tool calls but never produces text, the last
        tool call result is returned as the user message. This is the key
        behavior for environments like tic-tac-toe where the user model only
        needs to take an action (make_move) without generating commentary.

        Call sequence:
          1. User model returns a tool call (make_move)
          2. Tool is executed against the resources server
          3. Loop continues — user model returns empty output (breaks loop)
          4. Fallback: last tool result becomes the user message
        """
        server = MultiTurnAgent(
            config=_make_config(user_model_tool_choice="required"),
            server_client=MagicMock(spec=ServerClient),
        )

        tool_result = json.dumps({"success": True, "board": "X | O | ...", "message": "O placed at position 1."})

        # Step 1: User model returns a tool call
        mock_user_resp = AsyncMock()
        mock_user_resp.read.return_value = json.dumps(
            _user_model_tool_call_response("call_1", "make_move", '{"position": 1}')
        )
        mock_user_resp.cookies = {}

        # Step 2: Resources server executes the tool call
        mock_tool_resp = AsyncMock()
        mock_tool_resp.cookies = {}
        mock_tool_resp.content = AsyncMock()
        mock_tool_resp.content.read.return_value = tool_result.encode()

        # Step 3: On the next loop iteration, user model returns nothing (breaks loop)
        mock_empty_resp = AsyncMock()
        mock_empty_resp.read.return_value = json.dumps({"output": []})
        mock_empty_resp.cookies = {}

        server.server_client.post.side_effect = [mock_user_resp, mock_tool_resp, mock_empty_resp]

        result = await self._call_generate(server)
        assert result == tool_result

    async def test_tool_call_then_text(self) -> None:
        """User model first makes a tool call, then produces text on the next
        iteration. The text is preferred over the tool result fallback.

        Call sequence:
          1. User model returns make_move tool call
          2. Tool is executed against resources server
          3. User model returns text "I placed O. Your move!"
          4. Text is returned (not the tool result)
        """
        server = MultiTurnAgent(
            config=_make_config(user_model_tool_choice="required"),
            server_client=MagicMock(spec=ServerClient),
        )

        tool_result = json.dumps({"success": True, "message": "O placed."})

        mock_tool_call_resp = AsyncMock()
        mock_tool_call_resp.read.return_value = json.dumps(
            _user_model_tool_call_response("call_1", "make_move", '{"position": 4}')
        )
        mock_tool_call_resp.cookies = {}

        mock_resource_resp = AsyncMock()
        mock_resource_resp.cookies = {}
        mock_resource_resp.content = AsyncMock()
        mock_resource_resp.content.read.return_value = tool_result.encode()

        mock_text_resp = AsyncMock()
        mock_text_resp.read.return_value = json.dumps(_user_model_text_response("I placed O. Your move!"))
        mock_text_resp.cookies = {}

        server.server_client.post.side_effect = [mock_tool_call_resp, mock_resource_resp, mock_text_resp]

        result = await self._call_generate(server)
        assert result == "I placed O. Your move!"

    async def test_returns_none_when_no_output(self) -> None:
        """When user model produces neither text nor tool calls, returns None.
        The outer loop in run() treats this as a stop signal."""
        server = MultiTurnAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        mock_resp = AsyncMock()
        mock_resp.read.return_value = json.dumps({"output": []})
        mock_resp.cookies = {}
        server.server_client.post.return_value = mock_resp

        result = await self._call_generate(server)
        assert result is None

    async def test_max_steps_per_turn_limits_user_model_loop(self) -> None:
        """When max_steps_per_turn=1, the user model loop executes at most one
        tool call. This prevents environments like tic-tac-toe from having the
        user model make multiple moves in a single turn (which would place marks
        for both sides since the server alternates turns)."""
        server = MultiTurnAgent(
            config=_make_config(max_steps_per_turn=1, user_model_tool_choice="required"),
            server_client=MagicMock(spec=ServerClient),
        )

        tool_result = json.dumps({"success": True, "board": "X | O | ...", "game_over": False})

        mock_user_resp = AsyncMock()
        mock_user_resp.read.return_value = json.dumps(
            _user_model_tool_call_response("call_1", "make_move", '{"position": 4}')
        )
        mock_user_resp.cookies = {}

        mock_tool_resp = AsyncMock()
        mock_tool_resp.cookies = {}
        mock_tool_resp.content = AsyncMock()
        mock_tool_resp.content.read.return_value = tool_result.encode()

        server.server_client.post.side_effect = [mock_user_resp, mock_tool_resp]

        result = await self._call_generate(server)

        # Only 2 calls: one to the user model, one to the resources server.
        # Without the limit, the loop would call the user model again.
        assert server.server_client.post.call_count == 2
        assert result == tool_result

    async def test_tools_passed_from_jsonl(self) -> None:
        """Tools defined in responses_create_params (from JSONL data) are
        forwarded to the user model so it can interact with the environment."""
        server = MultiTurnAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        mock_resp = AsyncMock()
        mock_resp.read.return_value = json.dumps(_user_model_text_response("Ok"))
        mock_resp.cookies = {}
        server.server_client.post.return_value = mock_resp

        await self._call_generate(server)

        call_kwargs = server.server_client.post.call_args
        sent_json = call_kwargs.kwargs["json"]
        assert "tools" in sent_json
        assert sent_json["tools"][0]["name"] == "make_move"

    async def test_system_prompt_stripped_from_user_model_input(self) -> None:
        """The policy model's system/developer prompt must NOT appear in the
        user model's input — only the user model's own system prompt should
        be present. This prevents the user model from seeing instructions
        meant for the policy (e.g. "You are X, play to win")."""
        server = MultiTurnAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        mock_resp = AsyncMock()
        mock_resp.read.return_value = json.dumps(_user_model_text_response("Hi"))
        mock_resp.cookies = {}
        server.server_client.post.return_value = mock_resp

        body = _make_run_request(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {"role": "developer", "content": "You are X. Play to win."},
                    {"role": "user", "content": "Let's play!"},
                ],
            )
        )
        await self._call_generate(server, body=body)

        call_kwargs = server.server_client.post.call_args
        sent_input = call_kwargs.kwargs["json"]["input"]

        roles = [m.get("role") if isinstance(m, dict) else m.role for m in sent_input]
        assert "developer" not in roles
        # The user model's own system prompt should be first
        assert "system" in roles
        assert sent_input[0]["content"] == "You are a test user."
