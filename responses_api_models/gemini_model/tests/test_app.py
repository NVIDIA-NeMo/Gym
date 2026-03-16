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
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytest.importorskip("google.genai", reason="google-genai SDK not installed")

from nemo_gym.server_utils import ServerClient
from responses_api_models.gemini_model.app import (
    GeminiModelServer,
    GeminiModelServerConfig,
    GeminiProxyRequest,
)


def _make_config(**overrides):
    defaults = {
        "host": "0.0.0.0",
        "port": 8083,
        "entrypoint": "",
        "name": "",
        "gemini_api_key": "test-key",
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


class TestGeminiProxyRequest:
    def test_minimal(self):
        req = GeminiProxyRequest(contents=[{"role": "user", "parts": [{"text": "hi"}]}])
        assert len(req.contents) == 1
        assert req.config == {}
        assert req.model is None

    def test_with_model_override(self):
        req = GeminiProxyRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            model="gemini-3-pro-preview",
        )
        assert req.model == "gemini-3-pro-preview"

    def test_extra_fields_allowed(self):
        req = GeminiProxyRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            custom_field="value",
        )
        assert req.custom_field == "value"


class TestGeminiModelServer:
    def _make_server(self, **config_overrides):
        with patch("google.genai.Client", return_value=MagicMock()):
            cfg = _make_config(**config_overrides)
            return GeminiModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_instantiation(self):
        server = self._make_server()
        assert server.config.gemini_api_key == "test-key"

    @pytest.mark.asyncio
    async def test_responses_uses_config_model_when_body_model_is_none(self):
        server = self._make_server()

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content.role = "model"
        mock_candidate.content.parts = []

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_response):
            body = GeminiProxyRequest(
                contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            )
            result = await server.responses(body)
            assert "candidates" in result

    @pytest.mark.asyncio
    async def test_responses_prefers_body_model(self):
        server = self._make_server()

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content.role = "model"
        mock_candidate.content.parts = []

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_response) as mock_to_thread:
            body = GeminiProxyRequest(
                contents=[{"role": "user", "parts": [{"text": "hello"}]}],
                model="gemini-3-pro-preview",
            )
            await server.responses(body)

    @pytest.mark.asyncio
    async def test_responses_api_error_propagates(self):
        server = self._make_server()

        with patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=RuntimeError("API error")):
            body = GeminiProxyRequest(
                contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            )
            with pytest.raises(RuntimeError, match="API error"):
                await server.responses(body)

    @pytest.mark.asyncio
    async def test_chat_completions_not_implemented(self):
        server = self._make_server()

        with pytest.raises(NotImplementedError):
            await server.chat_completions()


class TestGeminiDeserializeContents:
    def test_text_content(self):
        raw = [{"role": "user", "parts": [{"text": "hello"}]}]
        contents = GeminiModelServer._deserialize_contents(raw)
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "hello"

    def test_function_call_content(self):
        raw = [
            {
                "role": "model",
                "parts": [{"function_call": {"name": "click_at", "args": {"x": 100, "y": 200}}}],
            }
        ]
        contents = GeminiModelServer._deserialize_contents(raw)
        assert len(contents) == 1
        fc = contents[0].parts[0].function_call
        assert fc.name == "click_at"
        assert fc.args == {"x": 100, "y": 200}

    def test_function_response_content(self):
        raw = [
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "click_at",
                            "response": {"url": "https://example.com", "status": "success"},
                        }
                    }
                ],
            }
        ]
        contents = GeminiModelServer._deserialize_contents(raw)
        fr = contents[0].parts[0].function_response
        assert fr.name == "click_at"
        assert fr.response["url"] == "https://example.com"

    def test_inline_data_content(self):
        png_b64 = base64.b64encode(b"fake-png-data").decode()
        raw = [
            {
                "role": "user",
                "parts": [{"inline_data": {"mime_type": "image/png", "data": png_b64}}],
            }
        ]
        contents = GeminiModelServer._deserialize_contents(raw)
        assert len(contents[0].parts) == 1

    def test_thought_content(self):
        raw = [
            {
                "role": "model",
                "parts": [{"text": "thinking...", "thought": True}],
            }
        ]
        contents = GeminiModelServer._deserialize_contents(raw)
        part = contents[0].parts[0]
        assert part.text == "thinking..."


class TestGeminiDeserializeConfig:
    def test_empty_config_returns_default(self):
        config = GeminiModelServer._deserialize_config({})
        assert config.temperature == 0.0

    def test_with_thinking_config(self):
        raw = {
            "tools": [{"computer_use": {"environment": "ENVIRONMENT_BROWSER"}}],
            "temperature": 0.5,
            "thinking_config": {
                "thinking_level": "THINKING_LEVEL_MEDIUM",
                "include_thoughts": True,
            },
        }
        config = GeminiModelServer._deserialize_config(raw)
        assert config.temperature == 0.5
        assert config.thinking_config is not None

    def test_with_system_instruction(self):
        raw = {
            "tools": [{"computer_use": {"environment": "ENVIRONMENT_BROWSER"}}],
            "system_instruction": "You are a helpful assistant.",
        }
        config = GeminiModelServer._deserialize_config(raw)
        assert config.system_instruction == "You are a helpful assistant."


class TestGeminiSerializeResponse:
    def test_serializes_text_part(self):
        mock_part = MagicMock()
        mock_part.text = "Hello"
        mock_part.thought = False
        mock_part.function_call = None
        mock_part.function_response = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        result = GeminiModelServer._serialize_response(mock_response)
        assert len(result["candidates"]) == 1
        parts = result["candidates"][0]["content"]["parts"]
        assert parts[0]["text"] == "Hello"

    def test_serializes_function_call(self):
        mock_fc = MagicMock()
        mock_fc.name = "click_at"
        mock_fc.args = {"x": 100, "y": 200}

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_fc
        mock_part.function_response = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        result = GeminiModelServer._serialize_response(mock_response)
        fc_data = result["candidates"][0]["content"]["parts"][0]["function_call"]
        assert fc_data["name"] == "click_at"
        assert fc_data["args"] == {"x": 100, "y": 200}

    def test_serializes_usage_metadata(self):
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content = MagicMock()
        mock_candidate.content.role = "model"
        mock_candidate.content.parts = []

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50
        mock_usage.total_token_count = 150
        mock_usage.cached_content_token_count = 0
        mock_usage.thoughts_token_count = 10

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        result = GeminiModelServer._serialize_response(mock_response)
        assert result["usage_metadata"]["prompt_token_count"] == 100
        assert result["usage_metadata"]["total_token_count"] == 150

    def test_no_candidates(self):
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_response.usage_metadata = None

        result = GeminiModelServer._serialize_response(mock_response)
        assert result["candidates"] == []
