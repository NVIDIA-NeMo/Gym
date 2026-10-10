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

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import aiohttp
from nooa.agents.summarization import summary_fork_active
from nooa.unifiedllm import (
    AssistantPart,
    AssistantReasoning,
    AssistantText,
    CacheBoundary,
    LLMResponse,
    Tool,
    ToolCall,
    UnifiedLLM,
)
from nooa.unifiedllm.limits import REPLY_CAP_KEYS, ContextLimits
from pydantic import BaseModel

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.rollout_observability import ModelCallRef, ObservationGap
from nemo_gym.server_utils import get_response_json, raise_for_status


class GymModelClient(Protocol):
    """Model transport usable in the agent server or a sandbox process."""

    async def post(
        self,
        *,
        server_name: str,
        url_path: str,
        json: NeMoGymResponseCreateParamsNonStreaming,
        cookies: dict[str, str],
        headers: dict[str, str],
    ) -> aiohttp.ClientResponse: ...


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

    max_policy_calls: int | None
    fatal_error: Exception | None = None
    used: int = 0
    calls: list[GymModelCall] = field(default_factory=list)
    gaps: list[ObservationGap] = field(default_factory=list)

    def charge(self) -> None:
        # No await between check and increment: atomic for the async rollout task tree.
        if self.max_policy_calls is not None and self.used >= self.max_policy_calls:
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


_GYM_REPLAY_SCOPE = "nemo-gym/responses/v1:"


def _assistant_parts(response: NeMoGymResponse) -> tuple[AssistantPart, ...]:
    """Keep readable NOOA history and the exact Gym wire items in the same order."""
    parts: list[AssistantPart] = []
    for item in response.output:
        # NOOA freezes native JSON, persists it, and strips it on public edits.
        # Keeping the complete item deliberately duplicates readable text so Gym
        # can preserve block boundaries, opaque reasoning, and training metadata.
        native = item.model_dump(mode="json", exclude_none=True)
        if isinstance(item, NeMoGymResponseFunctionToolCall):
            parts.append(ToolCall(id=item.call_id, name=item.name, arguments=item.arguments, native=native))
        elif isinstance(item, NeMoGymResponseReasoningItem):
            text = "\n".join(block.text for block in item.content or [])
            if not text:
                text = "\n".join(block.text for block in item.summary)
            parts.append(AssistantReasoning(text=text, native=native))
        elif isinstance(item, NeMoGymResponseOutputMessage):
            text = "".join(block.text if block.type == "output_text" else block.refusal for block in item.content)
            parts.append(AssistantText(text=text, native=native))
        else:
            # NOOA has no public part for hosted-tool/other opaque output items.
            # Retain them for exact Gym replay without making them executable.
            parts.append(AssistantText(text="", native=native))
    return tuple(parts)


def _portable_assistant_items(response: LLMResponse) -> list[dict[str, Any]]:
    """Replay ordered public parts without another client's opaque state."""
    items: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, ToolCall):
            items.append({"type": "function_call", "call_id": part.id, "name": part.name, "arguments": part.arguments})
        elif part.text:
            # Like NOOA's portable projection, readable reasoning becomes text;
            # provider reasoning IDs/signatures cannot be invented across routes.
            items.append({"role": "assistant", "content": part.text})
    return items


def _responses_input(
    messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
    gaps: list[ObservationGap] | None = None,
    *,
    replay_scope: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    instructions: list[str] = []
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, CacheBoundary):
            # Stable-prefix marker, never a model input.
            continue
        if isinstance(message, LLMResponse):
            if (
                message.replay_scope is not None
                and message.replay_scope.startswith(_GYM_REPLAY_SCOPE)
                and (replay_scope is None or message.replay_scope == replay_scope)
                and all(part.native is not None for part in message.parts)
            ):
                result.extend(part.model_dump(mode="json", include={"native"})["native"] for part in message.parts)
                continue
            # Foreign/edited turns and old archives have only portable authority.
            # A live raw_response may predate public edits and cannot authorize replay.
            # Preserve readable parts, but do not guess opaque/training metadata.
            if gaps is not None:
                gaps.append(
                    ObservationGap(
                        code="foreign_turn_projected_portable",
                        detail=(
                            "An LLMResponse without compatible Gym replay state was projected from its portable "
                            "public fields; training metadata was not guessed."
                        ),
                    )
                )
            result.extend(_portable_assistant_items(message))
            continue
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
        if message.get("role") == "assistant" and (message.get("tool_calls") or message.get("reasoning_content")):
            # NOOA renders edited turns as public dictionaries. Match its portable
            # Responses projection without restoring stale provider/training state.
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("Assistant reasoning_content must be a string.")
            if reasoning:
                result.append({"role": "assistant", "content": reasoning})
            if message.get("content"):
                result.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls") or []:
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
    schema = tool.get_parameter_schema(strict=True)
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
        "strict": _schema_is_strict(schema),
    }


def _schema_is_strict(value: Any, root: dict[str, Any] | None = None) -> bool:
    if not isinstance(value, dict):
        return True
    root = root or value
    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        definition = root.get("$defs", {}).get(reference.removeprefix("#/$defs/"))
        return _schema_is_strict(definition, root) if isinstance(definition, dict) else False
    if value.get("type") == "object":
        properties = value.get("properties", {})
        if value.get("additionalProperties") is not False or set(value.get("required", [])) != set(properties):
            return False
        return all(_schema_is_strict(child, root) for child in properties.values())
    if value.get("type") == "array":
        return _schema_is_strict(value.get("items", {}), root)
    return all(_schema_is_strict(child, root) for child in value.values() if isinstance(child, dict))


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
        server_client: GymModelClient,
        model_server_name: str,
        model_url_path: str,
        state: RolloutLLMState,
        cookies: dict[str, str],
        model: str = "gym-policy",
        on_call: Callable[[GymModelCall], None] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
        context_window: int | None = None,
    ) -> None:
        super().__init__(model=model, context_window=context_window)
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._model_url_path = model_url_path
        self._state = state
        self._on_call = on_call
        self._sampling_overrides = dict(sampling_overrides or {})
        # Rollout IDs in model_url_path change across restores; the configured
        # model server and alias identify the compatible inference route.
        self._gym_replay_scope = _GYM_REPLAY_SCOPE + json.dumps([model_server_name, model])
        self._cookies = cookies
        self._calls = 0
        self._lock = asyncio.Lock()

    def get_context_limits(
        self, overrides: dict[str, Any] | None = None, *, fallback_reserve: int = 0
    ) -> ContextLimits:
        """Reserve the same reply cap that the Gym request will actually send."""
        params = dict(overrides or {})
        cap = self._sampling_overrides.get("max_output_tokens")
        if cap is not None:
            params = {key: value for key, value in params.items() if key not in REPLY_CAP_KEYS}
            if params.get("extra_body"):
                params["extra_body"] = {
                    key: value for key, value in params["extra_body"].items() if key not in REPLY_CAP_KEYS
                }
            params["max_tokens"] = cap
        return super().get_context_limits(params, fallback_reserve=fallback_reserve)

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
        async with self._lock:
            return await self._acall(messages, tools, output_model, **kwargs)

    async def _acall(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._state.charge()
        self._calls += 1

        input_items, instructions = _responses_input(
            messages, gaps=self._state.gaps, replay_scope=self._gym_replay_scope
        )
        request: dict[str, Any] = {
            "input": input_items,
            "instructions": instructions,
            "model": None,
            "parallel_tool_calls": False,
            "tools": [_responses_tool_schema(tool) for tool in tools or []],
        }
        if output_model is not None:
            output_schema = output_model.model_json_schema()
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_model.__name__,
                    "schema": output_schema,
                    "strict": _schema_is_strict(output_schema),
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
        try:
            http_response = await self._server_client.post(
                server_name=self._model_server_name,
                url_path=self._model_url_path,
                json=body,
                cookies=self._cookies,
                headers={"x-session-id": call.invocation_id} if call.invocation_id is not None else {},
            )
            try:
                await raise_for_status(http_response)
            except aiohttp.ClientResponseError as error:
                # Expose the response body for NOOA context-overflow detection.
                content = getattr(error, "response_content", b"")
                if content:
                    if isinstance(content, bytes):
                        content = content.decode(errors="replace")
                    error.message = f"{error.message}: {content}"
                raise
            raw = await get_response_json(http_response)
            response = NeMoGymResponse.model_validate(raw)
            call.response = response
            self._cookies.update({name: morsel.value for name, morsel in http_response.cookies.items()})
        except Exception as error:
            # Summary forks contain their own errors; their optional model calls
            # must neither poison nor clear the main invocation's failure state.
            if not summary_fork_active():
                self._state.fatal_error = error
            raise
        if not summary_fork_active():
            # A recovered model call no longer vetoes successful completion.
            # Earlier requests and responses remain in the rollout evidence.
            self._state.fatal_error = None

        function_calls = [item for item in response.output if isinstance(item, NeMoGymResponseFunctionToolCall)]
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        parsed: BaseModel | None = None
        if output_model is not None and not function_calls:
            try:
                parsed = output_model.model_validate(json.loads(_output_text(response)))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise InvalidPolicyOutputError(f"Gym model returned invalid {output_model.__name__} JSON") from error

        return LLMResponse(
            raw_response=response,
            parts=_assistant_parts(response),
            replay_scope=self._gym_replay_scope,
            parsed=parsed,
            finish_reason="tool_calls" if function_calls else _finish_reason(response),
            usage=usage,
        )
