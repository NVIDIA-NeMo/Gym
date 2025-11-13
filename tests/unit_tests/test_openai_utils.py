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
import uuid
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    empty_response,
)


class TestOpenAIUtils:
    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")

    def test_empty_response(self, monkeypatch: MonkeyPatch) -> None:
        uuid_value = uuid.UUID("12345678123456781234567812345678")
        uuid4_mock = MagicMock()
        uuid4_mock.return_value = uuid_value
        monkeypatch.setattr(uuid, "uuid4", uuid4_mock)
        expected_response_id = f"resp_{uuid_value.hex}"
        actual_response = empty_response()
        assert actual_response.id == expected_response_id
        assert actual_response.output[0].role == "assistant"
        assert len(actual_response.output[0].content) == 0
