# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Smoke tests to verify inference provider API keys are valid.

Each test creates a real ChatCompletionsModel server, wraps it in a TestClient, and makes
a single request to the provider. This validates that the API key works and the full server
code path functions (config, NeMoGymAsyncOpenAI, ResponsesConverter).

Tests skip automatically when the corresponding API key env var is not set.
Set env vars to enable: OPENROUTER_API_KEY, FRIENDLIAI_API_KEY, HUGGINGFACE_API_KEY.
"""

import os
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from responses_api_models.chat_completions_model.app import (
    ChatCompletionsModel,
    ChatCompletionsModelConfig,
)


PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.1-8b-instruct",
    },
    "friendli": {
        "base_url": "https://api.friendli.ai/serverless/v1",
        "env_var": "FRIENDLIAI_API_KEY",
        "model": "meta-llama-3.1-8b-instruct",
    },
    "hf_inference": {
        "base_url": "https://router.huggingface.co/v1",
        "env_var": "HUGGINGFACE_API_KEY",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    },
}


def _get_provider_params():
    return [
        pytest.param(
            name,
            cfg["base_url"],
            os.environ.get(cfg["env_var"], ""),
            cfg["model"],
            id=name,
            marks=pytest.mark.skipif(
                not os.environ.get(cfg["env_var"]),
                reason=f"{cfg['env_var']} not set",
            ),
        )
        for name, cfg in PROVIDERS.items()
    ]


@contextmanager
def _integration_client(base_url: str, api_key: str, model: str):
    config = ChatCompletionsModelConfig(
        host="0.0.0.0",
        port=8081,
        base_url=base_url,
        api_key=api_key,
        model=model,
        entrypoint="",
        name="",
    )
    server = ChatCompletionsModel(config=config, server_client=MagicMock(spec=ServerClient))
    with TestClient(server.setup_webserver()) as client:
        yield client


@pytest.mark.integration
class TestProviderKeyValidation:
    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    async def test_api_key_works(self, provider, base_url, api_key, model):
        with _integration_client(base_url, api_key, model) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                    "max_tokens": 16,
                    "temperature": 0,
                },
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["choices"][0]["message"]["content"]
            assert data["choices"][0]["message"]["role"] == "assistant"
