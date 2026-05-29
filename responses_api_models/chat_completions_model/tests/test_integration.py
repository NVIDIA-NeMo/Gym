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
"""Integration tests for ChatCompletionsModel against real inference providers.

Tests skip automatically when the corresponding API key env var is not set.
Set env vars to enable: OPENROUTER_API_KEY, FRIENDLIAI_API_KEY, HUGGINGFACE_API_KEY,
FIREWORKS_API_KEY, DEEPINFRA_API_KEY.
"""

import os
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
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_var": "FIREWORKS_API_KEY",
        "model": "accounts/fireworks/models/llama4-scout-instruct-basic",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "env_var": "DEEPINFRA_API_KEY",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
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


def _make_client(base_url: str, api_key: str, model: str) -> tuple[TestClient, str]:
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
    app = server.setup_webserver()
    return TestClient(app), model


@pytest.mark.integration
class TestChatCompletionsIntegration:
    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_basic_chat_completion(self, provider, base_url, api_key, model):
        client, model_name = _make_client(base_url, api_key, model)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, f"{provider}: HTTP {resp.status_code} — {resp.text}"
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] in ("stop", "length")

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_chat_completion_with_system_message(self, provider, base_url, api_key, model):
        client, model_name = _make_client(base_url, api_key, model)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You only respond with the word 'yes'."},
                    {"role": "user", "content": "Can you help me?"},
                ],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, f"{provider}: HTTP {resp.status_code} — {resp.text}"
        data = resp.json()
        assert data["choices"][0]["message"]["content"]

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_chat_completion_returns_usage(self, provider, base_url, api_key, model):
        client, model_name = _make_client(base_url, api_key, model)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say 'hello'."}],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        if data.get("usage"):
            assert data["usage"]["prompt_tokens"] > 0
            assert data["usage"]["completion_tokens"] > 0


@pytest.mark.integration
class TestResponsesIntegration:
    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_basic_responses(self, provider, base_url, api_key, model):
        client, model_name = _make_client(base_url, api_key, model)

        resp = client.post(
            "/v1/responses",
            json={
                "model": model_name,
                "input": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "max_output_tokens": 16,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, f"{provider}: HTTP {resp.status_code} — {resp.text}"
        data = resp.json()
        assert data["object"] == "response"
        assert data["id"].startswith("resp_")
        assert len(data["output"]) > 0

        message_items = [o for o in data["output"] if o["type"] == "message"]
        assert len(message_items) > 0
        assert message_items[0]["content"][0]["type"] == "output_text"
        assert message_items[0]["content"][0]["text"]

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_responses_string_input(self, provider, base_url, api_key, model):
        client, model_name = _make_client(base_url, api_key, model)

        resp = client.post(
            "/v1/responses",
            json={
                "model": model_name,
                "input": "Say 'hello' and nothing else.",
                "max_output_tokens": 16,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, f"{provider}: HTTP {resp.status_code} — {resp.text}"
        data = resp.json()
        assert data["object"] == "response"
        assert len(data["output"]) > 0

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_responses_with_instructions(self, provider, base_url, api_key, model):
        client, model_name = _make_client(base_url, api_key, model)

        resp = client.post(
            "/v1/responses",
            json={
                "model": model_name,
                "input": [{"role": "user", "content": "Can you help me?"}],
                "instructions": "You only respond with the word 'yes'.",
                "max_output_tokens": 16,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, f"{provider}: HTTP {resp.status_code} — {resp.text}"
        data = resp.json()
        assert len(data["output"]) > 0

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_responses_returns_usage(self, provider, base_url, api_key, model):
        client, model_name = _make_client(base_url, api_key, model)

        resp = client.post(
            "/v1/responses",
            json={
                "model": model_name,
                "input": "Say 'hello'.",
                "max_output_tokens": 16,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        if data.get("usage"):
            assert data["usage"]["input_tokens"] > 0
            assert data["usage"]["output_tokens"] > 0
            assert data["usage"]["total_tokens"] > 0
