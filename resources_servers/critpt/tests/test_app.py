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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.critpt.app import (
    CritPtResourcesServer,
    CritPtResourcesServerConfig,
    CritPtVerifyRequest,
    _extract_code,
)


def _make_config(**kwargs) -> CritPtResourcesServerConfig:
    return CritPtResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        api_key="test-key",
        **kwargs,
    )


def _make_server(config: CritPtResourcesServerConfig | None = None) -> CritPtResourcesServer:
    return CritPtResourcesServer(
        config=config or _make_config(),
        server_client=MagicMock(spec=ServerClient),
    )


def _make_verify_request(output_text: str, problem_id: str = "1") -> CritPtVerifyRequest:
    response = NeMoGymResponse(
        id="test-id",
        created_at=1234.5,
        model="test-model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg-id",
                content=[NeMoGymResponseOutputText(annotations=[], text=output_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    return CritPtVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        problem_id=problem_id,
    )


class TestExtractCode:
    def test_fenced_python_block(self):
        text = "Here is the answer:\n```python\ndef solve():\n    return 42\n```"
        assert _extract_code(text) == "def solve():\n    return 42"

    def test_fenced_block_no_language(self):
        text = "```\ndef solve():\n    return 42\n```"
        assert _extract_code(text) == "def solve():\n    return 42"

    def test_multiple_blocks_returns_last(self):
        text = "```python\ndef first():\n    pass\n```\nThen:\n```python\ndef last():\n    return 1\n```"
        assert _extract_code(text) == "def last():\n    return 1"

    def test_no_fence_returns_stripped_text(self):
        text = "  def solve():\n    return 42  "
        assert _extract_code(text) == "def solve():\n    return 42"

    def test_empty_string_returns_empty(self):
        assert _extract_code("") == ""


class TestApp:
    def test_sanity(self):
        _make_server()

    @pytest.mark.asyncio
    async def test_verify_correct(self):
        server = _make_server()
        body = _make_verify_request("```python\ndef solve():\n    return 1.23\n```", problem_id="5")

        api_result = {"accuracy": 1.0, "timeout_rate": 0.0, "server_timeout_count": 0, "judge_error_count": 0}

        with (
            patch("resources_servers.critpt.app.request") as mock_request,
            patch("resources_servers.critpt.app.raise_for_status", new_callable=AsyncMock),
        ):
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=api_result)
            mock_request.return_value = mock_response

            result = await server.verify(body)

        assert result.reward == 1.0
        assert result.accuracy == 1.0
        assert result.timeout_rate == 0.0
        assert result.problem_id == "5"

        call_kwargs = mock_request.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["submissions"][0]["problem_id"] == "5"
        assert payload["submissions"][0]["generated_code"].startswith("```python\n")
        assert payload["submissions"][0]["model"] == "unknown"
        assert call_kwargs.kwargs["headers"]["x-api-key"] == "test-key"

    @pytest.mark.asyncio
    async def test_verify_incorrect(self):
        server = _make_server()
        body = _make_verify_request("```python\ndef solve():\n    return 9.99\n```")

        api_result = {"accuracy": 0.0, "timeout_rate": 0.0, "server_timeout_count": 0, "judge_error_count": 0}

        with (
            patch("resources_servers.critpt.app.request") as mock_request,
            patch("resources_servers.critpt.app.raise_for_status", new_callable=AsyncMock),
        ):
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=api_result)
            mock_request.return_value = mock_response

            result = await server.verify(body)

        assert result.reward == 0.0
        assert result.accuracy == 0.0

    @pytest.mark.asyncio
    async def test_verify_no_code_in_response(self):
        server = _make_server()
        body = _make_verify_request("")

        with patch("resources_servers.critpt.app.request") as mock_request:
            result = await server.verify(body)

        mock_request.assert_not_called()
        assert result.reward == 0.0
        assert result.accuracy == 0.0
