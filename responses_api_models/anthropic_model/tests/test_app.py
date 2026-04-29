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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytest.importorskip("anthropic", reason="anthropic SDK not installed")

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.anthropic_model.app import (
    AnthropicModelServer,
    AnthropicModelServerConfig,
    _extract_b64_from_data_url,
)


def _make_config(**overrides):
    defaults = {
        "host": "0.0.0.0",
        "port": 8082,
        "entrypoint": "",
        "name": "",
        "anthropic_api_key": "test-key",  # pragma: allowlist secret
        "anthropic_model": "claude-sonnet-4-20250514",
    }
    defaults.update(overrides)
    return AnthropicModelServerConfig(**defaults)


class TestAnthropicModelServerConfig:
    def test_defaults(self):
        cfg = _make_config()
        assert cfg.anthropic_model == "claude-sonnet-4-20250514"
        assert cfg.anthropic_timeout == 300.0
        assert cfg.effort_level == "high"
        assert cfg.computer_tool_version == "computer_20250124"
        assert cfg.computer_betas == ["computer-use-2025-01-24", "token-efficient-tools-2025-02-19"]
        assert cfg.zoom_enabled is False

    def test_override_model(self):
        cfg = _make_config(anthropic_model="claude-opus-4-20250514")
        assert cfg.anthropic_model == "claude-opus-4-20250514"

    def test_effort_level_override(self):
        cfg = _make_config(effort_level="medium")
        assert cfg.effort_level == "medium"

    def test_opus_style_config(self):
        cfg = _make_config(
            anthropic_model="claude-opus-4-6",
            computer_tool_version="computer_20251124",
            computer_betas=["computer-use-2025-11-24", "token-efficient-tools-2025-02-19"],
            zoom_enabled=True,
        )
        assert cfg.computer_tool_version == "computer_20251124"
        assert cfg.zoom_enabled is True
        assert "computer-use-2025-11-24" in cfg.computer_betas


class TestAnthropicModelServer:
    def _make_server(self, **config_overrides):
        with patch("responses_api_models.anthropic_model.app.AsyncAnthropic"):
            cfg = _make_config(**config_overrides)
            return AnthropicModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_instantiation(self):
        server = self._make_server()
        assert server.config.anthropic_api_key == "test-key"

    def test_derive_cua_tools_and_betas(self):
        server = self._make_server()
        body = MagicMock()
        body.tools = [
            {
                "type": "computer_use_preview",
                "display_width": 1920,
                "display_height": 1080,
                "environment": "browser",
            }
        ]
        tools, betas = server._derive_tools_and_betas(body)
        assert len(tools) == 1
        assert tools[0]["type"] == "computer_20250124"
        assert tools[0]["display_width_px"] == 1920
        assert tools[0]["display_height_px"] == 1080
        assert "computer-use-2025-01-24" in betas
        assert "token-efficient-tools-2025-02-19" in betas

    def test_derive_no_tools_generic_mode(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(input="test")
        tools, betas = server._derive_tools_and_betas(body)
        assert tools == []
        assert betas == []

    def test_derive_ignores_non_cua_tools(self):
        server = self._make_server()
        body = MagicMock()
        body.tools = [{"type": "web_search_preview"}]
        tools, betas = server._derive_tools_and_betas(body)
        assert tools == []
        assert betas == []

    def test_derive_viewport_defaults(self):
        server = self._make_server()
        body = MagicMock()
        body.tools = [{"type": "computer_use_preview"}]
        tools, _ = server._derive_tools_and_betas(body)
        assert tools[0]["display_width_px"] == 1280
        assert tools[0]["display_height_px"] == 720

    def test_derive_opus_uses_config_tool_version(self):
        server = self._make_server(
            computer_tool_version="computer_20251124",
            computer_betas=["computer-use-2025-11-24", "token-efficient-tools-2025-02-19"],
            zoom_enabled=True,
        )
        body = MagicMock()
        body.tools = [{"type": "computer_use_preview", "display_width": 1280, "display_height": 720}]
        tools, betas = server._derive_tools_and_betas(body)
        assert tools[0]["type"] == "computer_20251124"
        assert tools[0]["enable_zoom"] is True
        assert "computer-use-2025-11-24" in betas

    def test_derive_sonnet_no_zoom(self):
        server = self._make_server()
        body = MagicMock()
        body.tools = [{"type": "computer_use_preview", "display_width": 1280, "display_height": 720}]
        tools, _ = server._derive_tools_and_betas(body)
        assert "enable_zoom" not in tools[0]

    @pytest.mark.asyncio
    async def test_responses_generic_uses_regular_api(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        server._client = MagicMock()
        server._client.messages.create = AsyncMock(return_value=mock_response)

        body = NeMoGymResponseCreateParamsNonStreaming(
            input="Hello",
            instructions="You are a helpful assistant",
        )
        result = await server.responses(body)

        server._client.messages.create.assert_called_once()
        call_kwargs = server._client.messages.create.call_args[1]
        assert "betas" not in call_kwargs
        assert "tools" not in call_kwargs
        assert result.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_responses_cua_uses_beta_api(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        server._client = MagicMock()
        server._client.beta.messages.create = AsyncMock(return_value=mock_response)

        cua_tools = [
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": 1280,
                "display_height_px": 720,
                "display_number": 1,
            }
        ]
        cua_betas = ["computer-use-2025-01-24", "token-efficient-tools-2025-02-19"]
        with patch.object(server, "_derive_tools_and_betas", return_value=(cua_tools, cua_betas)):
            body = NeMoGymResponseCreateParamsNonStreaming(input=[], instructions="test")
            result = await server.responses(body)

        server._client.beta.messages.create.assert_called_once()
        call_kwargs = server._client.beta.messages.create.call_args[1]
        assert call_kwargs["betas"] == cua_betas
        assert call_kwargs["tools"] == cua_tools
        assert result.object == "response"

    @pytest.mark.asyncio
    async def test_responses_api_error_propagates(self):
        server = self._make_server()

        server._client = MagicMock()
        server._client.messages.create = AsyncMock(side_effect=RuntimeError("API error"))

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        with pytest.raises(RuntimeError, match="API error"):
            await server.responses(body)

    @pytest.mark.asyncio
    async def test_chat_completions_not_implemented(self):
        from fastapi import HTTPException

        server = self._make_server()

        with pytest.raises(HTTPException) as exc_info:
            await server.chat_completions()
        assert exc_info.value.status_code == 501


class TestAnthropicTranslateInput:
    def _make_server(self):
        with patch("responses_api_models.anthropic_model.app.AsyncAnthropic"):
            cfg = _make_config()
            return AnthropicModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_string_input(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(input="Hello world")
        messages = server._translate_input_to_messages(body)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "Hello world"

    def test_message_with_text(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Click the button"}]}
            ]
        )
        messages = server._translate_input_to_messages(body)
        assert len(messages) == 1
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Click the button"

    def test_computer_call_translates_to_tool_use(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call",
                    "call_id": "call_abc",
                    "action": {"action": "left_click", "coordinate": [100, 200]},
                    "id": "cu_1",
                }
            ]
        )
        messages = server._translate_input_to_messages(body)
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["type"] == "tool_use"
        assert messages[0]["content"][0]["id"] == "call_abc"
        assert messages[0]["content"][0]["input"]["action"] == "left_click"

    def test_computer_call_output_translates_to_tool_result(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_abc",
                    "output": {
                        "image_url": "data:image/png;base64,abc123",
                    },
                }
            ]
        )
        messages = server._translate_input_to_messages(body)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "tool_result"
        assert messages[0]["content"][0]["tool_use_id"] == "call_abc"


class TestAnthropicTranslateResponse:
    def _make_server(self):
        with patch("responses_api_models.anthropic_model.app.AsyncAnthropic"):
            cfg = _make_config()
            return AnthropicModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_tool_use_becomes_computer_call(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tu_123"
        mock_tool_block.input = {"action": "left_click", "coordinate": [100, 200]}
        mock_response.content = [mock_tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert len(result.output) == 1
        output_item = result.output[0]
        assert output_item.type == "computer_call"
        assert output_item.action["action"] == "left_click"

    def test_text_becomes_message(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Task completed"
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=20)

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert len(result.output) == 1
        output_item = result.output[0]
        assert output_item.type == "message"
        assert output_item.content[0].text == "Task completed"

    def test_usage_translated(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert result.usage is not None
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 200
        assert result.usage.total_tokens == 700

    def test_max_tokens_incomplete_details(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "max_tokens"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert result.incomplete_details.reason == "max_output_tokens"


class TestAnthropicTranslateInputEdgeCases:
    def _make_server(self):
        with patch("responses_api_models.anthropic_model.app.AsyncAnthropic"):
            cfg = _make_config()
            return AnthropicModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_message_with_string_content(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(input=[{"type": "message", "role": "user", "content": "Hello"}])
        messages = server._translate_input_to_messages(body)
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello"

    def test_message_with_input_image(self):
        server = self._make_server()
        body = MagicMock()
        body.input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,abc123"}],
            }
        ]
        messages = server._translate_input_to_messages(body)
        assert messages[0]["content"][0]["type"] == "image"
        assert messages[0]["content"][0]["source"]["data"] == "abc123"

    def test_message_with_empty_content_list(self):
        server = self._make_server()
        body = MagicMock()
        body.input = [{"type": "message", "role": "user", "content": []}]
        messages = server._translate_input_to_messages(body)
        assert messages[0]["content"][0]["type"] == "text"

    def test_computer_call_output_with_error_url(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {
                        "image_url": "data:image/png;base64,abc",
                        "current_url": "error: page not found",
                    },
                }
            ]
        )
        messages = server._translate_input_to_messages(body)
        tool_result = messages[0]["content"][0]
        assert tool_result["is_error"] is True
        assert tool_result["content"][0]["type"] == "text"
        assert "Action failed" in tool_result["content"][0]["text"]


class TestAnthropicResponsesEffort:
    def _make_server(self, **overrides):
        with patch("responses_api_models.anthropic_model.app.AsyncAnthropic"):
            cfg = _make_config(**overrides)
            return AnthropicModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_effort_level_medium_sets_output_config(self):
        server = self._make_server(effort_level="medium")

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        server._client = MagicMock()
        server._client.messages.create = AsyncMock(return_value=mock_response)

        body = NeMoGymResponseCreateParamsNonStreaming(input="test")
        await server.responses(body)

        call_kwargs = server._client.messages.create.call_args[1]
        assert call_kwargs["output_config"] == {"effort": "medium"}


class TestExtractB64FromDataUrl:
    def test_data_url(self):
        assert _extract_b64_from_data_url("data:image/png;base64,abc123") == "abc123"

    def test_data_url_no_comma(self):
        assert _extract_b64_from_data_url("data:image/png") is None

    def test_raw_b64_passthrough(self):
        assert _extract_b64_from_data_url("abc123") == "abc123"

    def test_empty_string(self):
        assert _extract_b64_from_data_url("") is None
