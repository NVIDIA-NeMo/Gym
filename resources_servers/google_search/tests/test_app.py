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

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.google_search.app import (
    BaseGetPageContentRequest,
    BaseSearchQueryRequest,
    GoogleSearchResourcesServer,
    GoogleSearchResourcesServerConfig,
    GoogleSearchVerifyRequest,
    box_parser,
)


class TestApp:
    @staticmethod
    def _make_server() -> GoogleSearchResourcesServer:
        config = GoogleSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            google_api_key="dummy_key",  # pragma: allowlist secret
            google_cx="dummy_cx",
        )
        return GoogleSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @staticmethod
    def _make_verify_request(output: list[dict]) -> GoogleSearchVerifyRequest:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=output,
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        return GoogleSearchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Question"}]},
            response=response,
            expected_answer="B",
            task_difficulty_qwen3_32b_avg_8=0.5,
        )

    def test_sanity(self) -> None:
        self._make_server()

    def test_box_parser_valid_content(self) -> None:
        """Test box_parser with valid boxed content"""
        # Test basic boxed content
        result = box_parser("The answer is \\boxed{42}")
        assert result == "42"

        # Test with complex content
        result = box_parser("After calculation: \\boxed{x + y = 10}")
        assert result == "x + y = 10"

        # Test with no boxed content
        result = box_parser("No boxed content here")
        assert result is None

        # Test with empty string
        result = box_parser("")
        assert result is None

    async def test_verify_extracts_standard_response_content(self) -> None:
        request = self._make_verify_request(
            [
                {
                    "id": "msg_test",
                    "content": [
                        {"annotations": [], "text": "Reasoning. ", "type": "output_text"},
                        {"annotations": [], "text": "\\boxed{B}", "type": "output_text"},
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        )

        result = await self._make_server().verify(request)

        assert result.reward == 1.0
        assert result.parsed_option == "B"

    async def test_verify_returns_zero_for_unparseable_output(self) -> None:
        request = self._make_verify_request(
            [
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "No boxed answer", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        )

        result = await self._make_server().verify(request)

        assert result.reward == 0.0
        assert result.parsed_option is None

    async def test_verify_returns_zero_for_empty_output(self) -> None:
        result = await self._make_server().verify(self._make_verify_request([]))

        assert result.reward == 0.0
        assert result.parsed_option is None

    async def test_search_uses_shared_async_client(self) -> None:
        response = MagicMock(ok=True)
        response.json = AsyncMock(return_value={"items": [{"title": "result"}]})

        with patch("resources_servers.google_search.app.request", AsyncMock(return_value=response)) as mock_request:
            result = await self._make_server().search(BaseSearchQueryRequest(query="test query"))

        assert json.loads(result.search_results) == {"items": [{"title": "result"}]}
        assert mock_request.await_args.kwargs["method"] == "GET"
        assert mock_request.await_args.kwargs["params"]["q"] == "test query"

    async def test_browse_fetches_page_without_blocking_event_loop(self) -> None:
        response = MagicMock(ok=True)
        response.text = AsyncMock(return_value="<html><body>Page</body></html>")

        with (
            patch("resources_servers.google_search.app.request", AsyncMock(return_value=response)) as mock_request,
            patch("resources_servers.google_search.app.trafilatura.extract", return_value="Page text"),
        ):
            result = await self._make_server().browse(BaseGetPageContentRequest(url="https://example.com"))

        assert result.page_content == "Page text"
        assert mock_request.await_args.kwargs["url"] == "https://example.com"
