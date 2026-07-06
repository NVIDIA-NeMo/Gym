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
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, NeMoGymResponseCreateParamsNonStreaming


class TestOpenAIUtils:
    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")

    async def test_chat_completion_request_id_is_stable_across_retries(
        self, monkeypatch
    ) -> None:
        request_id = "00000000-0000-4000-8000-000000000001"

        class FakeUUID:
            def __str__(self) -> str:
                return request_id

        retry_response = MagicMock()
        retry_response.status = 500
        retry_response.content.read = AsyncMock(return_value=b"temporary error")

        success_response = MagicMock()
        success_response.status = 200
        success_response.ok = True
        success_response.read = AsyncMock(return_value=b'{"id":"completion-id"}')

        request_mock = AsyncMock(side_effect=[retry_response, success_response])
        monkeypatch.setattr("nemo_gym.openai_utils.request", request_mock)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", AsyncMock())
        monkeypatch.setattr("nemo_gym.openai_utils.uuid4", lambda: FakeUUID())

        client = NeMoGymAsyncOpenAI(
            api_key="abc",
            base_url="https://example.test/v1",
            default_headers={"X-Default": "present"},
        )
        result = await client.create_chat_completion(
            model="test-model", messages=[]
        )

        assert result == {"id": "completion-id"}
        assert request_mock.await_count == 2
        for request_call in request_mock.await_args_list:
            assert request_call.kwargs["headers"] == {
                "Authorization": "Bearer abc",
                "X-Default": "present",
                "X-Request-ID": request_id,
            }


class TestNeMoGymResponseCreateParamsNonStreaming:
    def test_seed_rejected_at_top_level(self) -> None:
        """seed is not part of the OpenAI Responses schema; it must be passed via metadata.extra_body."""
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", seed=42)

    def test_seed_via_metadata_extra_body(self) -> None:
        """seed passed through metadata.extra_body round-trips through the strict schema."""
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello", metadata={"extra_body": '{"seed": 42}'})
        assert params.metadata["extra_body"] == '{"seed": 42}'

    def test_unknown_field_still_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", not_a_real_field=1)
