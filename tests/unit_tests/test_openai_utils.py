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

from aiohttp import ClientPayloadError

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


class TestOpenAIUtils:
    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")

    @patch("nemo_gym.openai_utils.request")
    @patch("nemo_gym.openai_utils.get_response_json")
    async def test_create_response_retries_on_body_payload_error(self, mock_get_json, mock_request):
        """ClientPayloadError during response body read should retry the full request."""
        client = NeMoGymAsyncOpenAI(api_key="test", base_url="http://fake/v1")

        ok_response = MagicMock(status=200, ok=True)
        mock_request.return_value = ok_response

        mock_get_json.side_effect = [
            ClientPayloadError("connection dropped"),
            {"id": "resp_1", "output": []},
        ]

        result = await client.create_response(input="hello")
        assert result == {"id": "resp_1", "output": []}
        assert mock_request.call_count == 2
        assert mock_get_json.call_count == 2

    @patch("nemo_gym.openai_utils.request")
    async def test_request_survives_payload_error_on_error_body(self, mock_request):
        """ClientPayloadError reading a 502 error body should not abort the retry loop."""
        client = NeMoGymAsyncOpenAI(api_key="test", base_url="http://fake/v1", internal=True)

        bad_response = MagicMock(status=502, ok=False)
        bad_response.content.read = AsyncMock(side_effect=ClientPayloadError("body truncated"))

        ok_response = MagicMock(status=200, ok=True)

        mock_request.side_effect = [bad_response, ok_response]

        result = await client._request(method="POST", url="http://fake/v1/responses")
        assert result is ok_response
        assert mock_request.call_count == 2

    @patch("nemo_gym.openai_utils.request")
    @patch("nemo_gym.openai_utils.get_response_json")
    async def test_create_response_raises_after_max_retries(self, mock_get_json, mock_request):
        """Persistent ClientPayloadError should raise after exhausting retries."""
        client = NeMoGymAsyncOpenAI(api_key="test", base_url="http://fake/v1")

        ok_response = MagicMock(status=200, ok=True)
        mock_request.return_value = ok_response

        mock_get_json.side_effect = ClientPayloadError("persistent failure")

        try:
            await client.create_response(input="hello")
            assert False, "Should have raised ClientPayloadError"
        except ClientPayloadError:
            pass

        assert mock_request.call_count == 3
        assert mock_get_json.call_count == 3
