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
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import create_tool_from_function
from pytest import MonkeyPatch

from nemo_gym.server_utils import ServerClient
from responses_api_agents.haystack_agent import chat_generator as chat_generator_module
from responses_api_agents.haystack_agent.app import (
    HaystackAgent,
    HaystackAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
)
from responses_api_agents.haystack_agent.chat_generator import NeMoGymResponsesChatGenerator
from responses_api_agents.haystack_agent.example_tools import get_weather


SYSTEM_PROMPT = "You are a helpful assistant. Use get_weather when asked about weather, then answer."


def _make_response(payload: dict) -> AsyncMock:
    """Build a mock aiohttp ClientResponse returning ``payload`` as JSON."""
    resp = AsyncMock()
    resp.ok = True
    resp.read = AsyncMock(return_value=json.dumps(payload))
    resp.cookies = {}
    return resp


def _envelope(output: list[dict], usage: dict | None = None) -> dict:
    payload = {
        "id": "resp_1",
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _function_call_item(arguments: str = '{"city": "San Francisco"}') -> dict:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": arguments,
        "status": "completed",
    }


def _text_item(text: str = "It is sunny in San Francisco.") -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


_USAGE = {
    "input_tokens": 10,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 15,
}


def _weather_tool():
    return create_tool_from_function(get_weather, name="get_weather", description="Get the weather for a city.")


def _pipeline_yaml(raise_on_tool_invocation_failure: bool = False) -> str:
    agent = Agent(
        chat_generator=NeMoGymResponsesChatGenerator(server_name="policy_model"),
        tools=[_weather_tool()],
        system_prompt=SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=6,
        raise_on_tool_invocation_failure=raise_on_tool_invocation_failure,
    )
    pipe = Pipeline()
    pipe.add_component("agent", agent)
    return pipe.dumps()


def _build_agent(tmp_path, monkeypatch: MonkeyPatch, model_responses: list[dict], *, raise_on_fail: bool = False):
    """Create a HaystackAgent whose loaded pipeline's generator uses a mocked model server."""
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(_pipeline_yaml(raise_on_tool_invocation_failure=raise_on_fail))

    client = MagicMock()
    client.post = AsyncMock(side_effect=[_make_response(p) for p in model_responses])
    monkeypatch.setattr(chat_generator_module, "_server_client", client)

    config = HaystackAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="haystack_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="res"),
        pipeline_yaml=str(pipeline_path),
    )
    return HaystackAgent(config=config, server_client=MagicMock(spec=ServerClient)), client


class TestChatGenerator:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        gen = NeMoGymResponsesChatGenerator(server_name="policy_model", generation_kwargs={"temperature": 0.5})
        data = gen.to_dict()
        assert data["init_parameters"]["server_name"] == "policy_model"
        restored = NeMoGymResponsesChatGenerator.from_dict(data)
        assert restored.server_name == "policy_model"
        assert restored.generation_kwargs == {"temperature": 0.5}

    def test_agent_accepts_generator_with_tools(self) -> None:
        # The Agent inspects run() for a `tools` parameter; this must not raise a TypeError.
        Agent(chat_generator=NeMoGymResponsesChatGenerator(server_name="policy_model"), tools=[_weather_tool()])

    async def test_run_async_converts_messages_and_tool_call(self, monkeypatch: MonkeyPatch) -> None:
        client = MagicMock()
        client.post = AsyncMock(return_value=_make_response(_envelope([_function_call_item()], _USAGE)))
        monkeypatch.setattr(chat_generator_module, "_server_client", client)

        gen = NeMoGymResponsesChatGenerator(server_name="policy_model")
        out = await gen.run_async(
            messages=[ChatMessage.from_system("sys"), ChatMessage.from_user("weather in SF?")],
            tools=[_weather_tool()],
        )

        replies = out["replies"]
        assert len(replies) == 1
        assert replies[0].tool_call.tool_name == "get_weather"
        assert replies[0].tool_call.arguments == {"city": "San Francisco"}

        # The request body carries the converted input + tools.
        body = client.post.call_args.kwargs["json"]
        assert body.input[0].role == "system"
        assert body.input[1].content == "weather in SF?"
        assert body.tools[0]["name"] == "get_weather"
        # Usage was captured on the generator for later aggregation.
        assert gen._usage.total_tokens == 15

    async def test_run_async_streaming_unsupported(self, monkeypatch: MonkeyPatch) -> None:
        gen = NeMoGymResponsesChatGenerator(server_name="policy_model")
        try:
            await gen.run_async(messages=[ChatMessage.from_user("hi")], streaming_callback=lambda _c: None)
            raised = False
        except NotImplementedError:
            raised = True
        assert raised


class TestApp:
    def test_sanity(self, tmp_path) -> None:
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(_pipeline_yaml())
        config = HaystackAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="haystack_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="res"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            pipeline_yaml=str(pipeline_path),
        )
        HaystackAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses_runs_haystack_agent_loop(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        # Model call 1: request the tool. Model call 2: final text answer.
        server, client = _build_agent(
            tmp_path,
            monkeypatch,
            model_responses=[
                _envelope([_function_call_item()], _USAGE),
                _envelope([_text_item()], _USAGE),
            ],
        )
        app = server.setup_webserver()
        http = TestClient(app)

        res = http.post("/v1/responses", json={"input": [{"role": "user", "content": "weather in SF?"}]})
        assert res.status_code == 200

        # The model was called twice (Haystack Agent looped).
        assert client.post.call_count == 2

        body = res.json()
        output_types = [item["type"] for item in body["output"]]
        # Generated trajectory: the tool call, the tool's output, then the assistant's text.
        assert output_types == ["function_call", "function_call_output", "message"]
        weather_output = body["output"][1]["output"]
        assert "22 degrees" in weather_output
        # Usage summed across the two model calls.
        assert body["usage"]["total_tokens"] == 30

    async def test_responses_tool_failure_does_not_crash(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        # Malformed tool-call arguments -> the Haystack tool invocation fails, but with
        # raise_on_tool_invocation_failure=False the rollout continues to a final text answer.
        server, client = _build_agent(
            tmp_path,
            monkeypatch,
            model_responses=[
                _envelope([_function_call_item(arguments="{not json")]),
                _envelope([_text_item("Sorry, I could not fetch the weather.")]),
            ],
            raise_on_fail=False,
        )
        app = server.setup_webserver()
        http = TestClient(app)

        res = http.post("/v1/responses", json={"input": [{"role": "user", "content": "weather?"}]})
        assert res.status_code == 200
        assert client.post.call_count == 2
        output_types = [item["type"] for item in res.json()["output"]]
        assert output_types[-1] == "message"
        assert "function_call_output" in output_types
