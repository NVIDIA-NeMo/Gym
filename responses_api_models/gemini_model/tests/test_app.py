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


pytest.importorskip("google.genai", reason="google-genai SDK not installed")

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.gemini_model.app import (
    GeminiModelServer,
    GeminiModelServerConfig,
    _extract_b64_from_data_url,
    _is_retryable_gemini_error,
)


def _make_config(**overrides):
    defaults = {
        "host": "0.0.0.0",
        "port": 8083,
        "entrypoint": "",
        "name": "",
        "gemini_api_key": "test-key",  # pragma: allowlist secret
        "gemini_model": "gemini-2.5-computer-use-preview-10-2025",
    }
    defaults.update(overrides)
    return GeminiModelServerConfig(**defaults)


class TestGeminiModelServerConfig:
    def test_defaults(self):
        cfg = _make_config()
        assert cfg.gemini_model == "gemini-2.5-computer-use-preview-10-2025"
        assert cfg.gemini_timeout == 300.0

    def test_override_model(self):
        cfg = _make_config(gemini_model="gemini-3-pro-preview")
        assert cfg.gemini_model == "gemini-3-pro-preview"


class TestGeminiModelServer:
    def _make_server(self, **config_overrides):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config(**config_overrides)
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_instantiation(self):
        server = self._make_server()
        assert server.config.gemini_api_key == "test-key"

    @pytest.mark.asyncio
    async def test_responses_calls_gemini_api(self):
        server = self._make_server()

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content.role = "model"
        mock_candidate.content.parts = []

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100, candidates_token_count=50, total_token_count=150
        )

        server._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        body = NeMoGymResponseCreateParamsNonStreaming(
            input="Hello world",
            instructions="You are a browser automation agent",
        )
        result = await server.responses(body)

        server._client.aio.models.generate_content.assert_called_once()
        assert result.model == "gemini-2.5-computer-use-preview-10-2025"
        assert result.object == "response"

    @pytest.mark.asyncio
    async def test_responses_api_error_propagates(self):
        server = self._make_server()

        server._client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API error"))

        body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
        with pytest.raises(RuntimeError, match="API error"):
            await server.responses(body)

    @pytest.mark.asyncio
    async def test_chat_completions_not_implemented(self):
        from fastapi import HTTPException

        server = self._make_server()

        with pytest.raises(HTTPException) as exc_info:
            await server.chat_completions()
        assert exc_info.value.status_code == 501


class TestGeminiTranslateInput:
    def _make_server(self):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config()
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_string_input(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(input="Hello world")
        contents = server._translate_input_to_contents(body)
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello world"

    def test_message_with_text(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Click the button"}]}
            ]
        )
        contents = server._translate_input_to_contents(body)
        assert len(contents) == 1
        assert contents[0].parts[0].text == "Click the button"

    def test_computer_call_translates_to_function_call(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call",
                    "call_id": "call_abc",
                    "action": {"type": "click_at", "x": 500, "y": 300},
                    "id": "cu_1",
                }
            ]
        )
        contents = server._translate_input_to_contents(body)
        assert contents[0].role == "model"
        fc = contents[0].parts[0].function_call
        assert fc.name == "click_at"
        assert fc.args == {"x": 500, "y": 300}


class TestGeminiTranslateResponse:
    def _make_server(self):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config()
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_function_call_becomes_computer_call(self):
        server = self._make_server()

        mock_fc = MagicMock()
        mock_fc.name = "click_at"
        mock_fc.args = {"x": 500, "y": 300}

        mock_part = MagicMock()
        mock_part.function_call = mock_fc
        mock_part.text = None
        mock_part.thought = False

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=50)

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert len(result.output) == 1
        output_item = result.output[0]
        assert output_item.type == "computer_call"
        assert output_item.action["type"] == "click_at"
        assert output_item.action["x"] == 500

    def test_text_becomes_message(self):
        server = self._make_server()

        mock_part = MagicMock()
        mock_part.function_call = None
        mock_part.text = "Task completed"
        mock_part.thought = False

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert len(result.output) == 1
        output_item = result.output[0]
        assert output_item.type == "message"
        assert output_item.content[0].text == "Task completed"

    def test_usage_translated(self):
        server = self._make_server()

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = []

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(prompt_token_count=500, candidates_token_count=200)

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert result.usage is not None
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 200
        assert result.usage.total_tokens == 700

    def test_thought_text_excluded_from_output(self):
        server = self._make_server()

        mock_part = MagicMock()
        mock_part.function_call = None
        mock_part.text = "thinking..."
        mock_part.thought = True

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert len(result.output) == 1
        assert result.output[0].content[0].text == ""


class TestGeminiBuildConfig:
    def _make_server(self, **config_overrides):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config(**config_overrides)
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_includes_system_instruction_from_body(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(input="test", instructions="Be helpful")
        config = server._build_generate_config(body)
        assert config.system_instruction == "Be helpful"

    def test_includes_computer_use_tool_when_cua(self):
        server = self._make_server()
        body = MagicMock()
        body.tools = [{"type": "computer_use_preview", "display_width": 1280, "display_height": 720}]
        body.instructions = None
        body.temperature = None
        body.max_output_tokens = None
        body.top_p = None
        config = server._build_generate_config(body)
        assert config.tools is not None
        assert len(config.tools) == 1

    def test_no_tools_in_generic_mode(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(input="test")
        config = server._build_generate_config(body)
        assert config.tools is None or config.tools == []

    def test_derive_tools_ignores_non_cua(self):
        server = self._make_server()
        body = MagicMock()
        body.tools = [{"type": "web_search_preview"}]
        tools = server._derive_tools(body)
        assert tools == []

    def test_gemini_3_enables_thinking(self):
        server = self._make_server(gemini_model="gemini-3-pro-preview")
        body = NeMoGymResponseCreateParamsNonStreaming(input="test")
        config = server._build_generate_config(body)
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == "MEDIUM"
        assert config.temperature == 1.0

    def test_non_gemini_3_defaults_temp_zero(self):
        server = self._make_server(gemini_model="gemini-2.5-computer-use-preview-10-2025")
        body = NeMoGymResponseCreateParamsNonStreaming(input="test")
        config = server._build_generate_config(body)
        assert config.temperature == 0.0


class TestGeminiRetryableErrors:
    def test_connection_error_retryable(self):
        assert _is_retryable_gemini_error(ConnectionError("lost")) is True

    def test_timeout_error_retryable(self):
        assert _is_retryable_gemini_error(TimeoutError("timed out")) is True

    def test_named_exception_retryable(self):
        exc = type("ResourceExhausted", (Exception,), {})()
        assert _is_retryable_gemini_error(exc) is True

    def test_status_code_retryable(self):
        exc = Exception("server error")
        exc.status_code = 429
        assert _is_retryable_gemini_error(exc) is True

    def test_non_retryable(self):
        assert _is_retryable_gemini_error(ValueError("bad input")) is False


class TestGeminiTranslateInputEdgeCases:
    def _make_server(self):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config()
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_computer_call_output_with_screenshot(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "click_at", "x": 100, "y": 200},
                    "id": "cu_1",
                },
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {
                        "image_url": "data:image/png;base64,aWdub3Jl",
                        "current_url": "https://example.com",
                    },
                },
            ]
        )
        contents = server._translate_input_to_contents(body)
        assert len(contents) == 2
        assert contents[1].role == "user"

    def test_computer_call_output_error_url(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {"current_url": "error: page crashed"},
                },
            ]
        )
        contents = server._translate_input_to_contents(body)
        func_resp = contents[0].parts[0].function_response
        assert func_resp.response["status"] == "error"

    def test_message_with_string_content(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(input=[{"type": "message", "role": "user", "content": "Hello"}])
        contents = server._translate_input_to_contents(body)
        assert contents[0].parts[0].text == "Hello"

    def test_message_with_image_content(self):
        server = self._make_server()
        body = MagicMock()
        body.input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,aWdub3Jl"}],
            }
        ]
        body.instructions = None
        contents = server._translate_input_to_contents(body)
        assert len(contents) == 1

    def test_message_empty_content_fallback(self):
        server = self._make_server()
        body = MagicMock()
        body.input = [{"type": "message", "role": "user", "content": []}]
        body.instructions = None
        contents = server._translate_input_to_contents(body)
        assert contents[0].parts[0].text is not None

    def test_assistant_message_maps_to_model_role(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"type": "message", "role": "assistant", "content": "I will click"}]
        )
        contents = server._translate_input_to_contents(body)
        assert contents[0].role == "model"


class TestGeminiTranslateResponseEdgeCases:
    def _make_server(self):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config()
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_no_candidates_returns_empty_message(self):
        server = self._make_server()
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_response.usage_metadata = None

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        assert len(result.output) == 1
        assert result.output[0].type == "message"
        assert result.output[0].content[0].text == ""

    def test_max_tokens_incomplete_details(self):
        server = self._make_server()
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "MAX_TOKENS"
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = []

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)
        assert result.incomplete_details.reason == "max_output_tokens"


class TestGeminiRetryLoop:
    def _make_server(self, **overrides):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config(gemini_max_retries=1, **overrides)
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_retries_on_transient_error_then_succeeds(self):
        server = self._make_server()

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content.parts = []
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        server._client.aio.models.generate_content = AsyncMock(side_effect=[ConnectionError("lost"), mock_response])

        body = NeMoGymResponseCreateParamsNonStreaming(input="test")
        with patch("responses_api_models.gemini_model.app.asyncio.sleep", new_callable=AsyncMock):
            result = await server.responses(body)

        assert result.object == "response"
        assert server._client.aio.models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_raises(self):
        server = self._make_server()
        server._client.aio.models.generate_content = AsyncMock(side_effect=ConnectionError("always fails"))

        body = NeMoGymResponseCreateParamsNonStreaming(input="test")
        with patch("responses_api_models.gemini_model.app.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError):
                await server.responses(body)


class TestGeminiExtractB64:
    def test_data_url(self):
        assert _extract_b64_from_data_url("data:image/png;base64,abc") == "abc"

    def test_data_url_no_comma(self):
        assert _extract_b64_from_data_url("data:image/png") is None

    def test_raw_b64(self):
        assert _extract_b64_from_data_url("abc123") == "abc123"

    def test_empty(self):
        assert _extract_b64_from_data_url("") is None


class TestGeminiFindFunctionName:
    def _make_server(self):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config()
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_finds_function_name_from_prior_content(self):
        from google.genai.types import Content, FunctionCall, Part

        server = self._make_server()
        contents = [
            Content(role="model", parts=[Part(function_call=FunctionCall(name="type_text_at", args={}))]),
        ]
        result = server._find_function_name_for_call_id(contents, "call_1", "click_at")
        assert result == "type_text_at"

    def test_returns_fallback_when_no_match(self):
        from google.genai.types import Content, Part

        server = self._make_server()
        contents = [Content(role="user", parts=[Part(text="hello")])]
        result = server._find_function_name_for_call_id(contents, "call_1", "click_at")
        assert result == "click_at"


class TestGeminiSafetyDecision:
    """Tests for safety_decision propagation in translate_input and translate_response."""

    def _make_server(self):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config()
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_safety_decision_propagated_in_input_translation(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "id": "cu_1",
                    "action": {
                        "type": "click_at",
                        "x": 100,
                        "y": 200,
                        "safety_decision": {"id": "sd_1", "code": "malicious_content", "message": "Warning"},
                    },
                },
            ]
        )
        contents = server._translate_input_to_contents(body)
        fn_call = contents[0].parts[0].function_call
        assert "safety_decision" in fn_call.args
        assert fn_call.args["safety_decision"]["id"] == "sd_1"

    def test_safety_acknowledgement_in_function_response(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {"current_url": "https://example.com"},
                    "acknowledged_safety_checks": [{"id": "sd_1"}],
                },
            ]
        )
        contents = server._translate_input_to_contents(body)
        fn_resp = contents[0].parts[0].function_response
        assert fn_resp.response["safety_acknowledgement"] == "true"

    def test_no_safety_acknowledgement_when_empty(self):
        server = self._make_server()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {"current_url": "https://example.com"},
                },
            ]
        )
        contents = server._translate_input_to_contents(body)
        fn_resp = contents[0].parts[0].function_response
        assert "safety_acknowledgement" not in fn_resp.response

    def test_translate_response_extracts_safety_decision(self):
        from google.genai.types import Content, FunctionCall, Part

        server = self._make_server()
        safety_val = {"id": "sd_1", "code": "malicious", "message": "Risky"}
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="click_at",
                        args={"x": 100, "y": 200, "safety_decision": safety_val},
                    )
                )
            ],
        )
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        computer_call = result.output[0]
        assert computer_call.type == "computer_call"
        assert computer_call.action["safety_decision"] == safety_val
        assert len(computer_call.pending_safety_checks) == 1

    def test_translate_response_no_safety_decision(self):
        from google.genai.types import Content, FunctionCall, Part

        server = self._make_server()
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="click_at",
                        args={"x": 50, "y": 60},
                    )
                )
            ],
        )
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        body = NeMoGymResponseCreateParamsNonStreaming(input=[])
        result = server._translate_response(mock_response, body)

        computer_call = result.output[0]
        assert "safety_decision" not in computer_call.action
        assert computer_call.pending_safety_checks == []
