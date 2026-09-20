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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from nooa.unifiedllm import CacheBoundary, LLMResponse, Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.rollout_observability import ModelCallRef, ObservationGap
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


class PolicyCallBudgetExceeded(RuntimeError):
    """Raised when one rollout exceeds its configured policy-call budget."""


class InvalidPolicyOutputError(ValueError):
    """A successful model request whose output does not satisfy the method contract."""


@dataclass(slots=True)
class GymModelCall:
    """Exact Gym request/response evidence for one NOOA policy call."""

    model_ref: ModelServerRef
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse | None = None
    invocation_id: str | None = None


@dataclass(slots=True)
class RolloutLLMState:
    """Gym-owned budget and exact model evidence shared by this rollout's clients."""

    max_policy_calls: int
    used: int = 0
    calls: list[GymModelCall] = field(default_factory=list)
    gaps: list[ObservationGap] = field(default_factory=list)

    def charge(self) -> None:
        # No await between check and increment: atomic for the async rollout task tree.
        if self.used >= self.max_policy_calls:
            raise PolicyCallBudgetExceeded(f"NOOA policy call budget exhausted after {self.max_policy_calls} calls")
        self.used += 1

    @property
    def model_calls(self) -> list[ModelCallRef]:
        return [
            ModelCallRef(model_ref=call.model_ref, response_id=call.response.id)
            for call in self.calls
            if call.response is not None
        ]


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json", exclude_none=True) if isinstance(value, BaseModel) else value


def _portable_assistant_message(response: LLMResponse) -> dict[str, Any]:
    """Project a foreign LLMResponse onto the portable chat-shaped dict."""
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "function": {"name": call.name, "arguments": call.arguments}}
            for call in response.tool_calls
        ]
    return message


def _responses_input(
    messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
    gaps: list[ObservationGap] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    instructions: list[str] = []
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, CacheBoundary):
            # Stable-prefix marker, never a model input.
            continue
        if isinstance(message, LLMResponse):
            # The rewritten unifiedllm passes prior turns back as the stored
            # LLMResponse object. This adapter's responses carry the full Gym
            # output (including training token metadata) on raw_response.
            if isinstance(message.raw_response, NeMoGymResponse):
                result.extend(_dump(item) for item in message.raw_response.output)
                continue
            # Foreign or detached turns (per-method model aliases, edited turns,
            # snapshot-restored sessions) carry no Gym raw output; replay only
            # their portable public fields and record the gap instead of guessing
            # at training metadata.
            if gaps is not None:
                gaps.append(
                    ObservationGap(
                        code="foreign_turn_projected_portable",
                        detail=(
                            "A stored LLMResponse without a Gym raw_response was projected from its portable "
                            "public fields; training metadata was not guessed."
                        ),
                    )
                )
            message = _portable_assistant_message(message)
        if message.get("role") == "system":
            if content := message.get("content"):
                instructions.append(str(content))
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


def _responses_tool_schema(tool: Tool) -> dict[str, Any]:
    schema = tool.get_parameter_schema()
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
        "strict": (
            schema.get("additionalProperties") is False
            and set(schema.get("required", [])) == set(schema.get("properties", {}))
        ),
    }


def _output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if isinstance(item, NeMoGymResponseOutputMessage):
            parts.extend(part.text for part in item.content if part.type == "output_text")
    return "\n".join(parts)


def _finish_reason(response: NeMoGymResponse) -> Literal["stop", "length", "error"]:
    if response.incomplete_details is None:
        return "stop"
    if response.incomplete_details.reason == "max_output_tokens":
        return "length"
    return "error"


class GymResponsesLLM(UnifiedLLM):
    """NOOA LLM implementation backed exclusively by a Gym Responses model server."""

    def __init__(
        self,
        *,
        server_client: ServerClient,
        model_server_name: str,
        model_url_path: str,
        state: RolloutLLMState,
        cookies: dict[str, str],
        model: str = "gym-policy",
        on_call: Callable[[GymModelCall], None] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model)
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._model_url_path = model_url_path
        self._state = state
        self._on_call = on_call
        self._sampling_overrides = dict(sampling_overrides or {})
        self._cookies = cookies
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def call(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise RuntimeError("GymResponsesLLM supports async NOOA entrypoints only")

    async def acall(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._state.charge()
        self._calls += 1

        input_items, instructions = _responses_input(messages, gaps=self._state.gaps)
        request: dict[str, Any] = {
            "input": input_items,
            "instructions": instructions,
            "model": None,
            "parallel_tool_calls": False,
            "tools": [_responses_tool_schema(tool) for tool in tools or []],
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
        supported = set(NeMoGymResponseCreateParamsNonStreaming.model_fields) - {"model"}
        for name, value in kwargs.items():
            destination = aliases.get(name, name)
            if destination in supported and value is not None:
                request[destination] = value

        # Explicit Gym rollout controls take precedence over NOOA's per-call settings.
        request.update(self._sampling_overrides)
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(request)
        call = GymModelCall(
            model_ref=ModelServerRef(name=self._model_server_name, type="responses_api_models"),
            request=body.model_copy(deep=True),
        )
        self._state.calls.append(call)
        if self._on_call is not None:
            self._on_call(call)
        http_response = await self._server_client.post(
            server_name=self._model_server_name,
            url_path=self._model_url_path,
            json=body,
            cookies=self._cookies,
        )
        await raise_for_status(http_response)
        raw = await get_response_json(http_response)
        response = NeMoGymResponse.model_validate(raw)
        call.response = response
        self._cookies.update({name: morsel.value for name, morsel in http_response.cookies.items()})

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
                usage=usage,
            )

        content: str | BaseModel = _output_text(response)
        if output_model is not None:
            try:
                content = output_model.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise InvalidPolicyOutputError(f"Gym model returned invalid {output_model.__name__} JSON") from error

        reasoning = [
            item.model_dump(mode="json", exclude_none=True) for item in response.output if item.type == "reasoning"
        ]
        return LLMResponse(
            raw_response=response,
            content=content,
            tool_calls=[],
            finish_reason=_finish_reason(response),
            reasoning=json.dumps(reasoning) if reasoning else None,
            usage=usage,
        )
