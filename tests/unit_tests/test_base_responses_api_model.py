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
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


class StubResponsesAPIModel(SimpleResponsesAPIModel):
    async def chat_completions(self, request: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymChatCompletion:
        raise NotImplementedError

    async def responses(self, request: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError


class TestBaseResponsesAPIModel:
    def test_BaseResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        BaseResponsesAPIModel(config=config)

    def test_SimpleResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        model = StubResponsesAPIModel(config=config, server_client=MagicMock(spec=ServerClient))
        model.setup_webserver()

    def test_embeddings_default_not_implemented(self) -> None:
        # A policy-only server exposes /v1/embeddings but the base default rejects it with 501,
        # since it has no embeddings backend to pass the request through to.
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        model = StubResponsesAPIModel(config=config, server_client=MagicMock(spec=ServerClient))
        client = TestClient(model.setup_webserver())

        response = client.post("/v1/embeddings", json={"input": "hello", "model": "text-embedding-3-small"})

        assert response.status_code == 501
        assert "does not support the embeddings endpoint" in response.json()["detail"]
