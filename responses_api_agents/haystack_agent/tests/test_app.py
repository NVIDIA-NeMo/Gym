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
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import create_tool_from_function
from haystack_integrations.tools.mcp import StreamableHttpServerInfo
from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.haystack_agent import chat_generator as chat_generator_module
from responses_api_agents.haystack_agent.app import (
    HaystackAgent,
    HaystackAgentConfig,
    HaystackAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
)
from responses_api_agents.haystack_agent.chat_generator import (
    NeMoGymResponsesChatGenerator,
    _content_to_text,
    _current_run_state,
    _GenRunState,
    _stringify,
    chat_messages_to_responses,
    response_to_chat_messages,
    responses_input_to_messages,
)
from responses_api_agents.haystack_agent.http_tool import HTTPTool
from responses_api_agents.haystack_agent.mcp_toolset import (
    ContextAwareMCPToolset,
)


SYSTEM_PROMPT = "You are a helpful assistant. Use get_weather when asked about weather, then answer."


async def get_weather(city: str) -> str:
    """Return deterministic weather data for agent tests."""
    return f"The weather in {city} is sunny and 22 degrees."


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


def _function_call_item(name: str = "get_weather", arguments: str = '{"city": "San Francisco"}') -> dict:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
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


def _reasoning_item(text: str = "I should call the tool.") -> dict:
    return {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": text}],
        "encrypted_content": "opaque-reasoning-state",
    }


def _with_training_metadata(item: dict, *, offset: int = 0) -> dict:
    return {
        **item,
        "prompt_token_ids": [10 + offset, 11 + offset],
        "generation_token_ids": [20 + offset, 21 + offset],
        "generation_log_probs": [-0.1, -0.2],
        "routed_experts": [[[offset]]],
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


def _http_response(body: str, cookies: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.cookies = cookies or {}
    response.content.read = AsyncMock(return_value=body.encode())
    return response


async def _post_responses(app, body: dict, *, cookies: dict | None = None):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/responses", json=body, cookies=cookies)


def _pipeline_yaml(
    raise_on_tool_invocation_failure: bool = False,
    generation_kwargs: dict | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> str:
    agent_kwargs = dict(
        chat_generator=NeMoGymResponsesChatGenerator(server_name="policy_model", generation_kwargs=generation_kwargs),
        tools=[_weather_tool()],
        exit_conditions=["text"],
        max_agent_steps=6,
        raise_on_tool_invocation_failure=raise_on_tool_invocation_failure,
    )
    if system_prompt is not None:
        agent_kwargs["system_prompt"] = system_prompt
    agent = Agent(**agent_kwargs)
    pipe = Pipeline()
    pipe.add_component("agent", agent)
    return pipe.dumps()


def _config(pipeline_path) -> HaystackAgentConfig:
    return HaystackAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="haystack_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="res"),
        pipeline_yaml=str(pipeline_path),
    )


def _build_agent(
    tmp_path,
    monkeypatch: MonkeyPatch,
    model_responses: list[dict],
    *,
    raise_on_fail: bool = False,
    generation_kwargs: dict | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
):
    """Create a HaystackAgent whose loaded pipeline's generator uses a mocked model server."""
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        _pipeline_yaml(
            raise_on_tool_invocation_failure=raise_on_fail,
            generation_kwargs=generation_kwargs,
            system_prompt=system_prompt,
        )
    )

    client = MagicMock()
    client.post = AsyncMock(side_effect=[_make_response(p) for p in model_responses])
    monkeypatch.setattr(chat_generator_module, "_server_client", client)

    return HaystackAgent(config=_config(pipeline_path), server_client=MagicMock(spec=ServerClient)), client


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

    def test_round_trips_training_metadata_on_the_terminal_output_item(self) -> None:
        response = chat_generator_module.NeMoGymResponse.model_validate(
            _envelope(
                [_reasoning_item(), _with_training_metadata(_function_call_item())],
                _USAGE,
            )
        )

        messages = response_to_chat_messages(response)
        assert messages[0].meta["__ng_training__"]["generation_token_ids"] == [20, 21]
        assert messages[0].meta["__ng_usage__"]["total_tokens"] == 15
        assert messages[0].meta["__ng_reasoning_encrypted__"] == "opaque-reasoning-state"

        output = chat_messages_to_responses(messages, output=True)
        assert [item.type for item in output] == ["reasoning", "function_call"]
        assert output[0].id == "rs_1"
        assert output[0].encrypted_content == "opaque-reasoning-state"
        assert not hasattr(output[0], "generation_token_ids")
        assert output[1].id == "fc_1"
        assert output[1].generation_token_ids == [20, 21]
        assert output[1].generation_log_probs == [-0.1, -0.2]
        assert output[1].routed_experts == [[[0]]]

    def test_preserves_all_model_output_item_ids(self) -> None:
        second_reasoning = {**_reasoning_item("I should explain the result."), "id": "rs_2"}
        second_message = {**_text_item("Done."), "id": "msg_2"}
        second_tool_call = {
            **_function_call_item(name="get_forecast", arguments='{"city": "New York"}'),
            "id": "fc_2",
            "call_id": "call_2",
        }
        response = chat_generator_module.NeMoGymResponse.model_validate(
            _envelope(
                [
                    _reasoning_item(),
                    _text_item(),
                    _function_call_item(),
                    second_reasoning,
                    second_message,
                    second_tool_call,
                ]
            )
        )

        messages = response_to_chat_messages(response)

        for output in (False, True):
            reconstructed = chat_messages_to_responses(messages, output=output)
            assert [item.id for item in reconstructed] == ["rs_1", "msg_1", "fc_1", "rs_2", "msg_2", "fc_2"]
            assert [item.type for item in reconstructed] == [
                "reasoning",
                "message",
                "function_call",
                "reasoning",
                "message",
                "function_call",
            ]

        seeded_messages = responses_input_to_messages(response.output)
        seeded_items = chat_messages_to_responses(seeded_messages)
        assert [item.id for item in seeded_items] == ["rs_1", "msg_1", "fc_1", "rs_2", "msg_2", "fc_2"]

    async def test_run_async_does_not_replace_resource_cookies(self, monkeypatch: MonkeyPatch) -> None:
        client = MagicMock()
        model_response = _make_response(_envelope([_text_item()], _USAGE))
        model_response.cookies = {"model_session": "next"}
        client.post = AsyncMock(return_value=model_response)
        monkeypatch.setattr(chat_generator_module, "_server_client", client)

        state = chat_generator_module._GenRunState(resources_server_cookies={"resource_session": "seeded"})
        context_token = chat_generator_module._current_run_state.set(state)
        try:
            await NeMoGymResponsesChatGenerator(server_name="policy_model").run_async(
                messages=[ChatMessage.from_user("hi")]
            )
        finally:
            chat_generator_module._current_run_state.reset(context_token)

        assert state.resources_server_cookies == {"resource_session": "seeded"}
        assert state.model_server_cookies == {"model_session": "next"}

    async def test_run_async_streaming_unsupported(self, monkeypatch: MonkeyPatch) -> None:
        gen = NeMoGymResponsesChatGenerator(server_name="policy_model")
        try:
            await gen.run_async(messages=[ChatMessage.from_user("hi")], streaming_callback=lambda _c: None)
            raised = False
        except NotImplementedError:
            raised = True
        assert raised


class TestContextAwareMCPToolset:
    def test_agent_retargets_mcp_toolset_once_at_startup(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        warm_up = MagicMock()
        monkeypatch.setattr(ContextAwareMCPToolset, "warm_up", warm_up)
        monkeypatch.setattr(HaystackAgent, "_resources_mcp_url", lambda self: "http://resources:19724/mcp")
        pipeline = Pipeline()
        pipeline.add_component(
            "agent",
            Agent(
                chat_generator=NeMoGymResponsesChatGenerator(server_name="policy_model"),
                tools=[ContextAwareMCPToolset(server_info=StreamableHttpServerInfo(url="http://unused/mcp"))],
            ),
        )
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(pipeline.dumps())
        config = HaystackAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="haystack_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="res"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            pipeline_yaml=str(pipeline_path),
        )

        agent = HaystackAgent(config=config, server_client=MagicMock(spec=ServerClient))
        toolset = next(tool for tool in agent._agent.tools if isinstance(tool, ContextAwareMCPToolset))
        assert toolset.server_info.url == "http://resources:19724/mcp"
        warm_up.assert_not_called()

    def test_requires_seeded_mcp_token(self) -> None:
        toolset = ContextAwareMCPToolset.__new__(ContextAwareMCPToolset)
        state = chat_generator_module._GenRunState()
        context_token = chat_generator_module._current_run_state.set(state)
        try:
            with pytest.raises(RuntimeError, match="X-NeMo-Gym-Session-Token"):
                toolset._client_for_current_rollout()
        finally:
            chat_generator_module._current_run_state.reset(context_token)

    async def test_rejects_mcp_toolset_when_mcp_is_disabled(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        server, _ = _build_agent(tmp_path, monkeypatch, model_responses=[])
        server._agent.tools = [ContextAwareMCPToolset.__new__(ContextAwareMCPToolset)]
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "hi"})

        with pytest.raises(RuntimeError, match="did not enable MCP"):
            await server.responses(request=MagicMock(headers={}, cookies={}), response=MagicMock(), body=body)

    async def test_warns_when_mcp_is_enabled_without_mcp_toolset(
        self, tmp_path, monkeypatch: MonkeyPatch, caplog
    ) -> None:
        server, _ = _build_agent(tmp_path, monkeypatch, model_responses=[_envelope([_text_item()], _USAGE)])
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "hi"})

        with caplog.at_level("WARNING"):
            await server.responses(
                request=MagicMock(
                    headers={"X-NeMo-Gym-Session-Token": "token"},
                    cookies={},
                ),
                response=MagicMock(),
                body=body,
            )

        assert "has no ContextAwareMCPToolset" in caplog.text


class TestHTTPTool:
    def test_drops_null_placeholder_properties(self) -> None:
        tool = HTTPTool(
            {
                "type": "function",
                "name": "send_email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string"},
                        "subject": None,
                    },
                    "required": ["recipient", "subject"],
                },
            },
            MagicMock(spec=ServerClient),
            "resources",
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {"recipient": {"type": "string"}},
            "required": ["recipient"],
        }

    async def test_posts_arguments_and_preserves_response_body(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=_http_response('{"error": "bad request"}', {"sid": "next"}))
        tool = HTTPTool(
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up an item.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            server_client,
            "resources",
        )
        state = chat_generator_module._GenRunState(resources_server_cookies={"sid": "current"})
        context_token = chat_generator_module._current_run_state.set(state)
        try:
            assert await tool.invoke_async(query="Ada") == '{"error": "bad request"}'
        finally:
            chat_generator_module._current_run_state.reset(context_token)

        server_client.post.assert_awaited_once_with(
            server_name="resources",
            url_path="/lookup",
            json={"query": "Ada"},
            cookies={"sid": "current"},
        )
        assert state.resources_server_cookies == {"sid": "next"}

    def test_rejects_non_function_schema(self) -> None:
        with pytest.raises(ValueError, match="type 'function'"):
            HTTPTool({"type": "web_search_preview"}, MagicMock(spec=ServerClient), "resources")


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

    async def test_rejects_request_system_message_when_pipeline_has_system_prompt(
        self, tmp_path, monkeypatch: MonkeyPatch
    ) -> None:
        server, client = _build_agent(tmp_path, monkeypatch, model_responses=[])
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [
                    {"type": "message", "role": "system", "content": "Environment instructions."},
                    {"type": "message", "role": "user", "content": "Hello."},
                ]
            }
        )

        with pytest.raises(ValueError, match="only one system instruction source"):
            await server.responses(request=MagicMock(headers={}, cookies={}), response=MagicMock(), body=body)
        client.post.assert_not_awaited()

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
        res = await _post_responses(
            server.setup_webserver(),
            {"input": [{"role": "user", "content": "weather in SF?"}]},
        )
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

    async def test_responses_preserves_training_metadata_through_agent_loop(
        self, tmp_path, monkeypatch: MonkeyPatch
    ) -> None:
        server, client = _build_agent(
            tmp_path,
            monkeypatch,
            model_responses=[
                _envelope([_with_training_metadata(_function_call_item(), offset=0)], _USAGE),
                _envelope([_with_training_metadata(_text_item("Done."), offset=100)], _USAGE),
            ],
        )
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "weather in SF?"})
        model_response = await server.responses(
            request=MagicMock(headers={}, cookies={}), response=MagicMock(), body=body
        )

        function_call = next(item for item in model_response.output if item.type == "function_call")
        final_message = next(item for item in model_response.output if item.type == "message")
        assert function_call.generation_token_ids == [20, 21]
        assert final_message.generation_token_ids == [120, 121]
        assert final_message.routed_experts == [[[100]]]
        assert model_response.usage.total_tokens == 30

        second_model_request = client.post.call_args_list[1].kwargs["json"].model_dump(mode="json")
        assert "__ng_training__" not in json.dumps(second_model_request)
        assert "__ng_usage__" not in json.dumps(second_model_request)

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
        res = await _post_responses(
            server.setup_webserver(),
            {"input": [{"role": "user", "content": "weather?"}]},
        )
        assert res.status_code == 200
        assert client.post.call_count == 2
        output_types = [item["type"] for item in res.json()["output"]]
        assert output_types[-1] == "message"
        assert "function_call_output" in output_types

    async def test_responses_forwards_sampling_params(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        # The row's sampling params reach the model call; request tools are runtime Haystack tools.
        server, client = _build_agent(tmp_path, monkeypatch, model_responses=[_envelope([_text_item()])])
        res = await _post_responses(
            server.setup_webserver(),
            {
                "input": [{"role": "user", "content": "weather in SF?"}],
                "temperature": 0.9,
                "max_output_tokens": 123,
                "top_p": 0.5,
                "instructions": "ignored: pipeline system_prompt is authoritative",
                "tools": [
                    {"type": "function", "name": "ignored_tool", "description": "x", "parameters": {}, "strict": False}
                ],
            },
        )
        assert res.status_code == 200

        out_body = client.post.call_args_list[0].kwargs["json"]
        assert out_body.temperature == 0.9
        assert out_body.max_output_tokens == 123
        assert out_body.top_p == 0.5
        # instructions is dropped; request tools are combined with the pipeline's local tools.
        assert out_body.instructions is None
        assert [tool["name"] for tool in out_body.tools] == ["get_weather", "ignored_tool"]

    async def test_responses_invokes_request_http_tool(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        server, model_client = _build_agent(
            tmp_path,
            monkeypatch,
            model_responses=[
                _envelope([_function_call_item(name="environment_tool", arguments='{"value": 3}')], _USAGE),
                _envelope([_text_item("Done.")], _USAGE),
            ],
        )
        server.server_client.post = AsyncMock(return_value=_http_response('{"result": 6}', {"sid": "updated"}))
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [{"role": "user", "content": "Double 3."}],
                "tools": [
                    {
                        "type": "function",
                        "name": "environment_tool",
                        "description": "Double a value.",
                        "strict": False,
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                        },
                    }
                ],
            }
        )
        request = MagicMock(headers={}, cookies={})
        response = MagicMock()
        model_response = await server.responses(request=request, response=response, body=body)

        assert [item.type for item in model_response.output] == ["function_call", "function_call_output", "message"]
        server.server_client.post.assert_awaited_once_with(
            server_name="res", url_path="/environment_tool", json={"value": 3}, cookies={}
        )
        assert [tool["name"] for tool in model_client.post.call_args_list[0].kwargs["json"].tools] == [
            "get_weather",
            "environment_tool",
        ]

    def test_request_http_tool_overrides_configured_tool(self, tmp_path, monkeypatch: MonkeyPatch, caplog) -> None:
        server, _ = _build_agent(tmp_path, monkeypatch, model_responses=[])

        with caplog.at_level("WARNING"):
            runtime_tools = server._tools_for_http_request(
                [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Environment weather.",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            )

        assert len(runtime_tools) == 1
        assert isinstance(runtime_tools[0], HTTPTool)
        assert "overrides a configured local tool with the same name" in caplog.text

    def test_configured_mcp_tool_overrides_request_http_tool(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        server, _ = _build_agent(tmp_path, monkeypatch, model_responses=[])
        mcp_toolset = ContextAwareMCPToolset(server_info=StreamableHttpServerInfo(url="http://unused/mcp"))
        mcp_toolset.tools = [_weather_tool()]
        mcp_toolset._warmup_called = True
        server._agent.tools = [mcp_toolset]

        runtime_tools = server._tools_for_http_request(
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Environment weather.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        )

        assert [tool.name for tool in runtime_tools] == ["get_weather"]
        assert not isinstance(runtime_tools[0], HTTPTool)

    async def test_responses_request_param_overrides_static_generation_kwargs(
        self, tmp_path, monkeypatch: MonkeyPatch
    ) -> None:
        # The pipeline configures temperature=0.1; the request's temperature=0.9 wins.
        server, client = _build_agent(
            tmp_path,
            monkeypatch,
            model_responses=[_envelope([_text_item()])],
            generation_kwargs={"temperature": 0.1},
        )
        res = await _post_responses(
            server.setup_webserver(),
            {"input": [{"role": "user", "content": "hi"}], "temperature": 0.9},
        )
        assert res.status_code == 200
        assert client.post.call_args_list[0].kwargs["json"].temperature == 0.9


def test_responses_input_to_messages_multiturn() -> None:
    items = [
        NeMoGymEasyInputMessage(type="message", role="developer", content="be terse"),
        NeMoGymEasyInputMessage(type="message", role="user", content="weather in SF?"),
        NeMoGymResponseFunctionToolCall(
            type="function_call",
            call_id="call_1",
            name="get_weather",
            arguments='{"city": "SF"}',
        ),
        NeMoGymFunctionCallOutput(type="function_call_output", call_id="call_1", output="sunny"),
        NeMoGymEasyInputMessage(type="message", role="assistant", content="It is sunny."),
        NeMoGymResponseReasoningItem(
            type="reasoning",
            id="rs_1",
            summary=[NeMoGymSummary(type="summary_text", text="think")],
            encrypted_content="enc",
        ),
    ]

    messages = responses_input_to_messages(items)

    assert messages[0].is_from(ChatRole.SYSTEM)
    assert messages[0].meta["__ng_role__"] == "developer"
    assert messages[1].is_from(ChatRole.USER)
    assert messages[1].text == "weather in SF?"
    assert messages[2].tool_call.tool_name == "get_weather"
    assert messages[2].tool_call.arguments == {"city": "SF"}
    assert messages[2].tool_call.id == "call_1"
    assert messages[3].is_from(ChatRole.TOOL)
    assert messages[3].tool_call_result.result == "sunny"
    assert messages[3].tool_call_result.origin.id == "call_1"
    assert messages[4].is_from(ChatRole.ASSISTANT)
    assert messages[4].text == "It is sunny."
    assert messages[5].reasoning.reasoning_text == "think"
    assert messages[5].meta["__ng_reasoning_id__"] == "rs_1"
    assert messages[5].meta["__ng_reasoning_encrypted__"] == "enc"


def test_function_call_output_without_prior_call_synthesizes_origin() -> None:
    items = [NeMoGymFunctionCallOutput(type="function_call_output", call_id="orphan", output="x")]

    messages = responses_input_to_messages(items)

    assert messages[0].tool_call_result.origin.id == "orphan"
    assert messages[0].tool_call_result.origin.tool_name == ""


def test_function_call_empty_arguments() -> None:
    items = [NeMoGymResponseFunctionToolCall(type="function_call", call_id="c", name="f", arguments="")]

    messages = responses_input_to_messages(items)

    assert messages[0].tool_call.arguments == {}


def test_chat_messages_to_responses_output_and_reasoning() -> None:
    message = ChatMessage.from_assistant(text="hello", reasoning="because")

    output_items = chat_messages_to_responses([message], output=True)
    assert [item.type for item in output_items] == ["reasoning", "message"]
    assert output_items[1].content[0].text == "hello"
    assert output_items[1].content[0].type == "output_text"

    input_items = chat_messages_to_responses([message], output=False)
    assistant_message = next(item for item in input_items if item.type == "message")
    assert isinstance(assistant_message, NeMoGymEasyInputMessage)
    assert assistant_message.role == "assistant"
    assert assistant_message.content == "hello"


async def test_responses_without_system_prompt(tmp_path, monkeypatch: MonkeyPatch) -> None:
    server, _ = _build_agent(
        tmp_path,
        monkeypatch,
        [_envelope([_text_item("hi there")], _USAGE)],
        system_prompt=None,
    )
    response = await _post_responses(
        server.setup_webserver(),
        {"input": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert [item["type"] for item in response.json()["output"]] == ["message"]


async def test_responses_string_input_and_cookie_propagation(tmp_path, monkeypatch: MonkeyPatch) -> None:
    model_response = _make_response(_envelope([_text_item("hi")], _USAGE))
    model_response.cookies = {"model_cookie": "mv"}
    server, _ = _build_agent(tmp_path, monkeypatch, [])
    client = MagicMock()
    client.post = AsyncMock(return_value=model_response)
    monkeypatch.setattr(chat_generator_module, "_server_client", client)
    response = await _post_responses(
        server.setup_webserver(),
        {"input": "just say hi"},
        cookies={"session_cookie": "sv"},
    )

    assert response.status_code == 200
    assert response.cookies.get("session_cookie") == "sv"
    assert response.cookies.get("model_cookie") == "mv"


def test_generator_sync_run(monkeypatch: MonkeyPatch) -> None:
    client = MagicMock()
    client.post = AsyncMock(return_value=_make_response(_envelope([_text_item("done")], _USAGE)))
    monkeypatch.setattr(chat_generator_module, "_server_client", client)
    generator = NeMoGymResponsesChatGenerator(server_name="policy_model")

    result = generator.run(messages=[ChatMessage.from_user("hi")])

    assert result["replies"][0].text == "done"


async def test_run_seeds_session_and_verifies(tmp_path) -> None:
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(_pipeline_yaml())
    server = HaystackAgent(config=_config(pipeline_path), server_client=MagicMock(spec=ServerClient))

    responses_create_params = {"input": [{"role": "user", "content": "hi"}]}
    model_response = _envelope([_text_item("done")], _USAGE)
    verify_payload = {
        "responses_create_params": responses_create_params,
        "response": model_response,
        "reward": 1.0,
    }
    server.server_client.post = AsyncMock(
        side_effect=[
            _make_response({}),
            _make_response(model_response),
            _make_response(verify_payload),
        ]
    )
    request = MagicMock()
    request.cookies = {}

    result = await server.run(
        request,
        HaystackAgentRunRequest(responses_create_params=responses_create_params),
    )

    assert result.reward == 1.0
    assert server.server_client.post.call_count == 3
    assert [call.kwargs["url_path"] for call in server.server_client.post.call_args_list] == [
        "/seed_session",
        "/v1/responses",
        "/verify",
    ]


async def test_aggregate_metrics_proxied(tmp_path) -> None:
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(_pipeline_yaml())
    server = HaystackAgent(config=_config(pipeline_path), server_client=MagicMock(spec=ServerClient))
    server.server_client.post = AsyncMock(return_value=_make_response({"key_metrics": {"reward": 0.5}}))

    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))

    assert result.key_metrics == {"reward": 0.5}
    assert server.server_client.post.call_args.kwargs["url_path"] == "/aggregate_metrics"


async def test_run_async_reasoning_reply(monkeypatch: MonkeyPatch) -> None:
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_9",
        "summary": [{"type": "summary_text", "text": "let me think"}],
        "encrypted_content": None,
    }
    client = MagicMock()
    client.post = AsyncMock(return_value=_make_response(_envelope([reasoning_item, _text_item("answer")], _USAGE)))
    monkeypatch.setattr(chat_generator_module, "_server_client", client)
    generator = NeMoGymResponsesChatGenerator(server_name="policy_model")

    result = await generator.run_async(messages=[ChatMessage.from_user("q")])

    reply = result["replies"][0]
    assert reply.reasoning.reasoning_text == "let me think"
    assert reply.text == "answer"
    assert reply.meta["__ng_reasoning_id__"] == "rs_9"


def test_response_to_chat_messages_reasoning_only() -> None:
    response = NeMoGymResponse.model_validate(
        _envelope(
            [
                {
                    "type": "reasoning",
                    "id": "r",
                    "summary": [{"type": "summary_text", "text": "x"}],
                    "encrypted_content": None,
                }
            ]
        )
    )

    messages = response_to_chat_messages(response)

    assert messages[0].reasoning.reasoning_text == "x"


def test_stringify_and_content_to_text_helpers() -> None:
    parts = [
        {"type": "text", "text": "hello "},
        {"type": "output_text", "text": "world"},
    ]
    assert _stringify(parts) == "hello world"
    assert _stringify("plain") == "plain"
    assert _stringify(42) == "42"

    assert (
        _content_to_text(
            [
                {"type": "output_text", "text": "A"},
                {"type": "input_text", "text": "B"},
            ]
        )
        == "AB"
    )
    assert _content_to_text("str") == "str"
    with pytest.raises(ValueError, match="text-only"):
        _content_to_text(123)


def test_relative_pipeline_path(tmp_path, monkeypatch: MonkeyPatch) -> None:
    import responses_api_agents.haystack_agent.app as app_module

    pipeline_path = tmp_path / "relative_pipeline.yaml"
    pipeline_path.write_text(_pipeline_yaml())
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))

    HaystackAgent(config=_config(pipeline_path.name), server_client=MagicMock(spec=ServerClient))


async def test_concurrent_rollouts_do_not_clobber_state(monkeypatch: MonkeyPatch) -> None:
    gate = asyncio.Event()

    async def fake_post(*args, **kwargs):
        await gate.wait()
        response = _make_response(_envelope([_text_item("ok")], _USAGE))
        response.cookies = {"who": kwargs["cookies"]["who"]}
        return response

    client = MagicMock()
    client.post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(chat_generator_module, "_server_client", client)
    generator = NeMoGymResponsesChatGenerator(server_name="policy_model")

    async def rollout(tag: str) -> _GenRunState:
        state = _GenRunState(model_server_cookies={"who": tag})
        token = _current_run_state.set(state)
        try:
            await generator.run_async(messages=[ChatMessage.from_user(tag)])
        finally:
            _current_run_state.reset(token)
        return state

    first_task = asyncio.create_task(rollout("a"))
    second_task = asyncio.create_task(rollout("b"))
    await asyncio.sleep(0)
    gate.set()
    first_state, second_state = await asyncio.gather(first_task, second_task)

    assert first_state.usage.total_tokens == 15
    assert second_state.usage.total_tokens == 15
    assert first_state.model_server_cookies == {"who": "a"}
    assert second_state.model_server_cookies == {"who": "b"}
    assert first_state.last_response is not second_state.last_response
