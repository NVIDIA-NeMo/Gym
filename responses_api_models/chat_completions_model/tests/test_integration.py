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
"""Integration tests for hosted inference providers via OpenAI-compatible chat completions.

Tests skip automatically when the corresponding API key env var is not set.
Set env vars to enable: OPENROUTER_API_KEY, FRIENDLIAI_API_KEY, HUGGINGFACE_API_KEY.
"""

import json
import os
import time

import httpx
import pytest
from openai import OpenAI

REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 3

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
    # "fireworks": {
    #     "base_url": "https://api.fireworks.ai/inference/v1",
    #     "env_var": "FIREWORKS_API_KEY",
    #     "model": "accounts/fireworks/models/llama4-scout-instruct-basic",
    # },
    # "deepinfra": {
    #     "base_url": "https://api.deepinfra.com/v1/openai",
    #     "env_var": "DEEPINFRA_API_KEY",
    #     "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # },
    # "baseten": {
    #     "base_url": "https://bridge.baseten.co/v1",
    #     "env_var": "BASETEN_API_KEY",
    #     "model": "<deployment-model-id>",
    # },
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


def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        max_retries=MAX_RETRIES,
    )


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
}


@pytest.mark.integration
class TestChatCompletionsIntegration:
    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_basic_chat_completion(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=16,
            temperature=0,
        )
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason in ("stop", "length")

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_chat_completion_with_system_message(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You only respond with the word 'yes'."},
                {"role": "user", "content": "Can you help me?"},
            ],
            max_tokens=16,
            temperature=0,
        )
        assert len(response.choices) > 0
        assert response.choices[0].message.content

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_chat_completion_returns_usage(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello'."}],
            max_tokens=16,
            temperature=0,
        )
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_chat_completion_multi_turn(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is 2+2? Reply with just the number."},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "Now add 1 to that. Reply with just the number."},
            ],
            max_tokens=16,
            temperature=0,
        )
        assert len(response.choices) > 0
        assert response.choices[0].message.content

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_chat_completion_with_temperature(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello'."}],
            max_tokens=16,
            temperature=0.5,
        )
        assert len(response.choices) > 0
        assert response.choices[0].message.content

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_chat_completion_max_tokens_respected(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write a 500 word essay about AI."}],
            max_tokens=5,
            temperature=0,
        )
        assert len(response.choices) > 0
        assert response.choices[0].finish_reason in ("stop", "length")


@pytest.mark.integration
class TestToolCallingIntegration:
    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_single_tool_call(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
            tools=[WEATHER_TOOL],
            max_tokens=128,
            temperature=0,
        )
        assert len(response.choices) > 0
        choice = response.choices[0]
        assert choice.finish_reason in ("tool_calls", "stop")
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) > 0

        tool_call = choice.message.tool_calls[0]
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert "location" in args

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_tool_call_with_tool_choice_auto(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
            temperature=0,
        )
        assert len(response.choices) > 0
        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        tool_call = choice.message.tool_calls[0]
        assert tool_call.function.name == "get_weather"

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_tool_call_with_tool_choice_forced(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_tokens=128,
            temperature=0,
        )
        assert len(response.choices) > 0
        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        assert choice.message.tool_calls[0].function.name == "get_weather"

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_tool_call_arguments_are_valid_json(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 123 * 456?"}],
            tools=[CALCULATOR_TOOL],
            max_tokens=128,
            temperature=0,
        )
        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        for tool_call in choice.message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_multi_turn_with_tool_result(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            max_tokens=128,
            temperature=0,
        )
        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        tool_call = choice.message.tool_calls[0]

        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                choice.message.model_dump(),
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": '{"temperature": 18, "condition": "sunny"}',
                },
            ],
            tools=[WEATHER_TOOL],
            max_tokens=128,
            temperature=0,
        )
        choice2 = response2.choices[0]
        assert choice2.message.content
        assert choice2.message.role == "assistant"

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_multiple_tools_available(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            max_tokens=128,
            temperature=0,
        )
        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        assert choice.message.tool_calls[0].function.name == "get_weather"

    @pytest.mark.parametrize("provider,base_url,api_key,model", _get_provider_params())
    def test_no_tool_call_when_not_needed(self, provider, base_url, api_key, model):
        client = _make_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=16,
            temperature=0,
        )
        choice = response.choices[0]
        assert choice.message.content
        assert choice.finish_reason == "stop"
