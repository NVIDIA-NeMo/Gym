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
"""Tests for the streaming Responses dialect on ``SimpleResponsesAPIModel``.

Every Gym model server's ``/v1/responses`` accepts the wire dialect streaming harnesses
(e.g. the Codex CLI) speak: ``stream: true`` plus extra bookkeeping fields and ``namespace``
tool specs. The request is sanitized onto the strict params model and the complete response
is re-emitted as a synthesized SSE event stream. Non-streaming requests keep the historical
strict-validation behavior.
"""

import json
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_streaming import (
    flatten_namespace_tools,
    sanitize_streaming_responses_body,
    synthesize_responses_failure_sse,
    synthesize_responses_sse,
    validate_streaming_responses_params,
)
from nemo_gym.server_utils import ServerClient


def _build_response(output: list) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model="downstream-model",
        object="response",
        output=output,
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        usage={
            "input_tokens": 7,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10,
        },
    )


def _message_item(text: str) -> dict:
    return {
        "type": "message",
        "id": f"msg_{uuid4().hex}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call_item(name: str) -> dict:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": "{}",
        "status": "completed",
    }


NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "mcp__weather",
    "description": "Tools in the mcp__weather namespace.",
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather.",
            "strict": False,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ],
}


class TestSanitizeStreamingBody:
    def test_drops_unknown_top_level_fields(self) -> None:
        cleaned, _ = sanitize_streaming_responses_body(
            {"input": [], "stream": True, "client_metadata": {"x": 1}, "prompt_cache_key": "abc", "store": False}
        )
        assert set(cleaned) == {"input", "store"}
        # the cleaned body validates against the strict params model
        NeMoGymResponseCreateParamsNonStreaming.model_validate(cleaned)

    def test_flattens_namespace_tools(self) -> None:
        flat, ns_map = flatten_namespace_tools([NAMESPACE_TOOL])
        assert len(flat) == 1
        assert flat[0]["type"] == "function"
        assert flat[0]["name"] == "mcp__weather__get_weather"
        assert ns_map == {"mcp__weather__get_weather": ("mcp__weather", "get_weather")}

    def test_sanitize_keeps_function_tools_and_flattens_namespaces(self) -> None:
        function_tool = {
            "type": "function",
            "name": "exec_command",
            "description": "Run a command.",
            "strict": False,
            "parameters": {"type": "object", "properties": {}},
        }
        cleaned, ns_map = sanitize_streaming_responses_body(
            {"input": [], "stream": True, "tools": [function_tool, NAMESPACE_TOOL]}
        )
        names = [t["name"] for t in cleaned["tools"]]
        assert names == ["exec_command", "mcp__weather__get_weather"]
        assert "mcp__weather__get_weather" in ns_map
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(cleaned)
        assert len(params.tools) == 2

    def test_drops_unsupported_tool_specs(self) -> None:
        cleaned, _ = sanitize_streaming_responses_body(
            {"input": [], "stream": True, "tools": [{"type": "totally_unknown_tool_kind", "config": 1}]}
        )
        assert cleaned["tools"] == []

    def test_rewrites_namespaced_calls_in_input_history(self) -> None:
        cleaned, _ = sanitize_streaming_responses_body(
            {
                "stream": True,
                "input": [
                    {
                        "type": "function_call",
                        "namespace": "mcp__weather",
                        "name": "get_weather",
                        "arguments": "{}",
                        "call_id": "call_1",
                    },
                    {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
                ],
            }
        )
        call = cleaned["input"][0]
        assert call["name"] == "mcp__weather__get_weather"
        assert "namespace" not in call
        NeMoGymResponseCreateParamsNonStreaming.model_validate(cleaned)

    def test_drops_unsupported_input_items(self) -> None:
        # Codex's code_mode interleaves an `additional_tools` carrier item into the input history;
        # the Gym input union has no representation for it, so it is dropped item-by-item.
        cleaned, _ = sanitize_streaming_responses_body(
            {
                "stream": True,
                "input": [
                    {"type": "additional_tools", "role": "developer", "tools": [{"type": "custom", "name": "exec"}]},
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                ],
            }
        )
        assert [i["type"] for i in cleaned["input"]] == ["message"]
        NeMoGymResponseCreateParamsNonStreaming.model_validate(cleaned)

    def test_additional_tools_carrier_functions_hoisted_into_tools(self) -> None:
        # Codex's code mode ships tools inside an `additional_tools` input item; plain and
        # namespaced function tools are hoisted into `tools`, non-function tools are dropped.
        cleaned, ns_map = sanitize_streaming_responses_body(
            {
                "stream": True,
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [
                            {"type": "custom", "name": "exec", "description": "JS orchestrator", "format": {}},
                            {
                                "type": "function",
                                "name": "wait",
                                "description": "Wait.",
                                "strict": False,
                                "parameters": {"type": "object", "properties": {}},
                            },
                            NAMESPACE_TOOL,
                        ],
                    },
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                ],
            }
        )
        names = [t["name"] for t in cleaned["tools"]]
        assert "wait" in names
        assert "mcp__weather__get_weather" in names
        assert "exec" not in names
        assert [i.get("type") for i in cleaned["input"]] == ["message"]
        assert "mcp__weather__get_weather" in ns_map
        NeMoGymResponseCreateParamsNonStreaming.model_validate(cleaned)

    def test_hoists_leading_developer_messages_into_instructions(self) -> None:
        # Codex may open with several developer messages and no `instructions`; strict chat
        # backends admit a single leading system message, so they are hoisted into instructions.
        cleaned, _ = sanitize_streaming_responses_body(
            {
                "stream": True,
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "You are Codex."}],
                    },
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "<permissions>"}],
                    },
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                ],
            }
        )
        assert cleaned["instructions"] == "You are Codex.\n\n<permissions>"
        assert [i["role"] for i in cleaned["input"]] == ["user"]

    def test_hoisting_prepends_existing_instructions(self) -> None:
        cleaned, _ = sanitize_streaming_responses_body(
            {
                "stream": True,
                "instructions": "base instructions",
                "input": [
                    {"type": "message", "role": "developer", "content": "perms"},
                    {"type": "message", "role": "user", "content": "hi"},
                ],
            }
        )
        assert cleaned["instructions"] == "base instructions\n\nperms"

    def test_mid_conversation_developer_messages_not_hoisted(self) -> None:
        cleaned, _ = sanitize_streaming_responses_body(
            {
                "stream": True,
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {"type": "message", "role": "developer", "content": "mid-run steer"},
                ],
            }
        )
        assert "instructions" not in cleaned
        assert [i["role"] for i in cleaned["input"]] == ["user", "developer"]

    def test_does_not_mutate_caller_body(self) -> None:
        body = {"input": [], "stream": True, "tools": [NAMESPACE_TOOL]}
        sanitize_streaming_responses_body(body)
        assert body["tools"] == [NAMESPACE_TOOL]
        assert body["stream"] is True


class TestValidateStreamingParams:
    def test_prunes_nested_extra_fields(self) -> None:
        # Codex sends `reasoning.context`, which the pinned SDK's Reasoning model forbids.
        params = validate_streaming_responses_params(
            {"input": [], "reasoning": {"effort": "medium", "context": "all_turns"}}
        )
        assert params.reasoning == {"effort": "medium"}

    def test_unfixable_errors_still_raise(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            validate_streaming_responses_params({"input": [], "temperature": "not-a-number"})


class TestSynthesizeSSE:
    def _events(self, sse_text: str) -> list[dict]:
        events = []
        for block in sse_text.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))
        return events

    def test_event_sequence(self) -> None:
        response = _build_response([_message_item("hello")]).model_dump(mode="json")
        events = self._events("".join(synthesize_responses_sse(response)))
        assert [e["type"] for e in events] == ["response.created", "response.output_item.done", "response.completed"]
        assert events[0]["response"]["status"] == "in_progress"
        assert events[0]["response"]["output"] == []
        assert events[1]["output_index"] == 0
        assert events[1]["item"]["content"][0]["text"] == "hello"
        completed = events[2]["response"]
        assert completed["id"] == response["id"]
        assert completed["usage"]["input_tokens"] == 7
        assert len(completed["output"]) == 1

    def test_namespaced_call_names_restored(self) -> None:
        response = _build_response([_function_call_item("mcp__weather__get_weather")]).model_dump(mode="json")
        ns_map = {"mcp__weather__get_weather": ("mcp__weather", "get_weather")}
        events = self._events("".join(synthesize_responses_sse(response, ns_map)))
        item = events[1]["item"]
        assert item["namespace"] == "mcp__weather"
        assert item["name"] == "get_weather"
        # the terminal envelope carries the same rewritten item
        assert events[2]["response"]["output"][0]["name"] == "get_weather"

    def test_unmapped_calls_left_alone(self) -> None:
        response = _build_response([_function_call_item("exec_command")]).model_dump(mode="json")
        events = self._events("".join(synthesize_responses_sse(response, {"other__tool": ("other", "tool")})))
        assert events[1]["item"]["name"] == "exec_command"
        assert "namespace" not in events[1]["item"]

    def test_failure_stream_is_terminal_response_failed(self) -> None:
        events = self._events("".join(synthesize_responses_failure_sse("boom", code="server_error")))
        assert [e["type"] for e in events] == ["response.created", "response.failed"]
        failed = events[-1]["response"]
        assert failed["status"] == "failed"
        assert failed["error"] == {"code": "server_error", "message": "boom"}
        assert failed["output"] == []


class _EchoModel(SimpleResponsesAPIModel):
    """Fake model server capturing the params its responses() receives."""

    config: BaseResponsesAPIModelConfig
    last_params: object = None
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        object.__setattr__(self, "last_params", body)
        output = [_message_item("hi")]
        if body.tools:
            output.insert(0, _function_call_item(body.tools[0].get("name", "")))
        return _build_response(output)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


class _RequestAwareEchoModel(_EchoModel):
    saw_request: bool = False

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        object.__setattr__(self, "saw_request", isinstance(request, Request))
        return await super().responses(body)


class _FailingModel(_EchoModel):
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise RuntimeError("backend exploded")


def _client(model_cls) -> tuple[TestClient, SimpleResponsesAPIModel]:
    server = model_cls(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    return TestClient(server.setup_webserver()), server


class TestResponsesDispatchRoute:
    def test_non_streaming_request_returns_plain_json(self) -> None:
        client, server = _client(_EchoModel)
        resp = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["output"][-1]["content"][0]["text"] == "hi"
        assert server.last_params.input[0].content == "hi"

    def test_non_streaming_request_still_validates_strictly(self) -> None:
        client, _ = _client(_EchoModel)
        resp = client.post("/v1/responses", json={"input": [], "client_metadata": {"x": 1}})
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"][0] == "body"

    def test_streaming_request_returns_synthesized_sse(self) -> None:
        client, server = _client(_EchoModel)
        resp = client.post(
            "/v1/responses",
            json={
                "stream": True,
                "client_metadata": {"cli": "codex"},
                "prompt_cache_key": "abc",
                "input": [{"role": "user", "content": "hi"}],
                "tools": [NAMESPACE_TOOL],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "event: response.completed" in resp.text
        # the server saw sanitized params: flattened tools, no bookkeeping fields
        assert server.last_params.tools[0]["name"] == "mcp__weather__get_weather"
        # and the synthesized items restore the namespaced call shape
        done_events = [line for line in resp.text.splitlines() if '"response.output_item.done"' in line]
        first_item = json.loads(done_events[0][len("data: ") :])["item"]
        assert first_item["namespace"] == "mcp__weather"
        assert first_item["name"] == "get_weather"

    def test_streaming_request_aware_signature(self) -> None:
        client, server = _client(_RequestAwareEchoModel)
        resp = client.post("/v1/responses", json={"stream": True, "input": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        assert server.saw_request is True

    def test_streaming_invalid_body_is_422(self) -> None:
        client, _ = _client(_EchoModel)
        resp = client.post("/v1/responses", json={"stream": True})  # no input at all
        assert resp.status_code == 422

    def test_streaming_backend_error_yields_response_failed(self) -> None:
        # A responses() failure after the streaming contract is committed becomes a terminal
        # response.failed event (HTTP 200 SSE), not a broken-stream HTTP 500.
        client, _ = _client(_FailingModel)
        resp = client.post("/v1/responses", json={"stream": True, "input": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "event: response.failed" in resp.text
        assert "event: response.completed" not in resp.text
        failed = [line for line in resp.text.splitlines() if line.startswith("data: ") and "response.failed" in line]
        payload = json.loads(failed[0][len("data: ") :])
        assert payload["response"]["status"] == "failed"
        assert "backend exploded" in payload["response"]["error"]["message"]

    def test_non_streaming_backend_error_still_raises(self) -> None:
        # Without the streaming contract, a backend failure is a normal exception (HTTP 500), not a
        # synthesized response.failed — only the stream path swallows it into a terminal event.
        client, _ = _client(_FailingModel)
        with pytest.raises(RuntimeError, match="backend exploded"):
            client.post("/v1/responses", json={"input": [{"role": "user", "content": "hi"}]})
