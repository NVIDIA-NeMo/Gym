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

from __future__ import annotations

import json
from typing import Any

from nooa.unifiedllm import LLMResponse, Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


class PolicyCallBudgetExceeded(RuntimeError):
    """Raised when one rollout exceeds its configured policy-call budget."""


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json", exclude_none=True) if isinstance(value, BaseModel) else value


def _responses_input(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    instructions: list[str] = []
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            if content := message.get("content"):
                instructions.append(str(content))
            continue
        if "_batch" in message:
            batch = message["_batch"]
            if not isinstance(batch, list):
                raise ValueError("NOOA assistant _batch must be a list of Responses items")
            result.extend(_dump(item) for item in batch)
            continue
        if "type" in message:
            result.append(_dump(message))
            continue
        if message.get("role") == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content", ""),
                }
            )
            continue
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if message.get("content"):
                result.append({"role": "assistant", "content": message["content"]})
            for call in message["tool_calls"]:
                function = call.get("function", {})
                result.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    }
                )
            continue
        result.append(_dump(message))
    return result, "\n\n".join(instructions) or None


def _tool_schema(tool: Tool) -> dict[str, Any]:
    schema = tool.get_parameter_schema()
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
        "strict": set(schema.get("required", [])) == set(schema.get("properties", {})),
    }


def _output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if isinstance(item, NeMoGymResponseOutputMessage):
            parts.extend(part.text for part in item.content if part.type == "output_text")
    return "\n".join(parts)


class GymResponsesLLM(UnifiedLLM):
    """NOOA LLM implementation backed exclusively by a Gym Responses model server."""

    def __init__(
        self,
        *,
        server_client: ServerClient,
        model_server_name: str,
        model_url_path: str,
        max_steps: int,
        response_collector: list[NeMoGymResponse],
        cookies: dict[str, str],
        model: str = "gym-policy",
    ) -> None:
        super().__init__(model=model)
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._model_url_path = model_url_path
        self._max_steps = max_steps
        self._response_collector = response_collector
        self._cookies = cookies
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise RuntimeError("GymResponsesLLM supports async NOOA entrypoints only")

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._calls >= self._max_steps:
            raise PolicyCallBudgetExceeded(f"NOOA policy call budget exhausted after {self._max_steps} calls")
        self._calls += 1

        input_items, instructions = _responses_input(messages)
        request: dict[str, Any] = {
            "input": input_items,
            "instructions": instructions,
            "model": None,
            "parallel_tool_calls": False,
            "tools": [_tool_schema(tool) for tool in tools or []],
        }
        if output_model is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_model.__name__,
                    "schema": output_model.model_json_schema(),
                    "strict": True,
                }
            }

        aliases = {"max_tokens": "max_output_tokens"}
        supported = set(NeMoGymResponseCreateParamsNonStreaming.model_fields)
        for name, value in kwargs.items():
            destination = aliases.get(name, name)
            if destination in supported and value is not None:
                request[destination] = value

        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(request)
        http_response = await self._server_client.post(
            server_name=self._model_server_name,
            url_path=self._model_url_path,
            json=body,
            cookies=self._cookies,
        )
        await raise_for_status(http_response)
        raw = await get_response_json(http_response)
        response = NeMoGymResponse.model_validate(raw)
        self._cookies.update({name: morsel.value for name, morsel in http_response.cookies.items()})
        self._response_collector.append(response)

        dumped_output = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        function_calls = [item for item in response.output if isinstance(item, NeMoGymResponseFunctionToolCall)]
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        if function_calls:
            return LLMResponse(
                raw_response=response,
                content="",
                tool_calls=[
                    ToolCall(id=item.call_id, name=item.name, arguments=item.arguments) for item in function_calls
                ],
                finish_reason="tool_calls",
                assistant_message={"_batch": dumped_output},
                usage=usage,
            )

        content: str | BaseModel = _output_text(response)
        if output_model is not None:
            try:
                content = output_model.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"Gym model returned invalid {output_model.__name__} JSON") from error

        reasoning = [
            item.model_dump(mode="json", exclude_none=True) for item in response.output if item.type == "reasoning"
        ]
        return LLMResponse(
            raw_response=response,
            content=content,
            tool_calls=[],
            finish_reason="length" if response.incomplete_details else "stop",
            assistant_message={"role": "assistant", "content": _output_text(response)},
            reasoning=json.dumps(reasoning) if reasoning else None,
            usage=usage,
        )
