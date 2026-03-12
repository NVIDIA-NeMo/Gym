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

from nemo_gym.server_utils import ServerClient
from responses_api_models.anthropic_model.app import (
    AnthropicModelServer,
    AnthropicModelServerConfig,
    AnthropicProxyRequest,
)


def _make_config(**overrides):
    defaults = {
        "host": "0.0.0.0",
        "port": 8082,
        "entrypoint": "",
        "name": "",
        "anthropic_api_key": "test-key",
        "anthropic_model": "claude-sonnet-4-20250514",
    }
    defaults.update(overrides)
    return AnthropicModelServerConfig(**defaults)


class TestAnthropicModelServerConfig:
    def test_defaults(self):
        cfg = _make_config()
        assert cfg.anthropic_model == "claude-sonnet-4-20250514"
        assert cfg.anthropic_timeout == 300.0

    def test_override_model(self):
        cfg = _make_config(anthropic_model="claude-opus-4-20250514")
        assert cfg.anthropic_model == "claude-opus-4-20250514"


class TestAnthropicProxyRequest:
    def test_minimal(self):
        req = AnthropicProxyRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.messages == [{"role": "user", "content": "hi"}]
        assert req.system == ""
        assert req.tools == []
        assert req.betas == []
        assert req.max_tokens == 4096
        assert req.model is None

    def test_with_model_override(self):
        req = AnthropicProxyRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-20250514",
        )
        assert req.model == "claude-opus-4-20250514"

    def test_extra_fields_allowed(self):
        req = AnthropicProxyRequest(
            messages=[{"role": "user", "content": "hi"}],
            custom_field="value",
        )
        assert req.custom_field == "value"


class TestAnthropicModelServer:
    def _make_server(self, **config_overrides):
        with patch("responses_api_models.anthropic_model.app.AsyncAnthropic"):
            cfg = _make_config(**config_overrides)
            return AnthropicModelServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    def test_instantiation(self):
        server = self._make_server()
        assert server.config.anthropic_api_key == "test-key"

    @pytest.mark.asyncio
    async def test_responses_uses_config_model_when_body_model_is_none(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.model_dump.return_value = {
            "stop_reason": "end_turn",
            "content": [],
        }

        server._client = MagicMock()
        server._client.beta.messages.create = AsyncMock(return_value=mock_response)

        body = AnthropicProxyRequest(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            system="You are a helpful assistant",
        )
        result = await server.responses(body)

        call_kwargs = server._client.beta.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_responses_uses_body_model_when_provided(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.model_dump.return_value = {"stop_reason": "end_turn", "content": []}

        server._client = MagicMock()
        server._client.beta.messages.create = AsyncMock(return_value=mock_response)

        body = AnthropicProxyRequest(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            model="claude-opus-4-20250514",
        )
        result = await server.responses(body)

        call_kwargs = server._client.beta.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_responses_propagates_output_config(self):
        server = self._make_server()

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.model_dump.return_value = {"stop_reason": "end_turn", "content": []}

        server._client = MagicMock()
        server._client.beta.messages.create = AsyncMock(return_value=mock_response)

        body = AnthropicProxyRequest(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            output_config={"format": "json"},
        )
        await server.responses(body)

        call_kwargs = server._client.beta.messages.create.call_args[1]
        assert call_kwargs["output_config"] == {"format": "json"}

    @pytest.mark.asyncio
    async def test_responses_api_error_propagates(self):
        server = self._make_server()

        server._client = MagicMock()
        server._client.beta.messages.create = AsyncMock(side_effect=RuntimeError("API error"))

        body = AnthropicProxyRequest(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )
        with pytest.raises(RuntimeError, match="API error"):
            await server.responses(body)

    @pytest.mark.asyncio
    async def test_chat_completions_not_implemented(self):
        server = self._make_server()

        with pytest.raises(NotImplementedError):
            await server.chat_completions()
