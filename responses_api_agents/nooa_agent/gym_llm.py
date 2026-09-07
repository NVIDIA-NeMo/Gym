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
import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

from nooa.unifiedllm import LLMResponse, Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.rollout_observability import ObservationGap
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status
from responses_api_agents.nooa_agent.observability import GymTraceHooks


class PolicyCallBudgetExceeded(RuntimeError):
    """Raised when one rollout exceeds its configured policy-call budget."""


def _journal_callbacks() -> list[Any]:
    """Return NOOA's installed litellm journal callbacks, if tracing is enabled.

    NOOA's LLM-message journal (and the viewer's LLM turns) are fed by a litellm
    callback installed by ``nooa.tracing``. This LLM calls the Gym model server
    directly, bypassing litellm, so the callback has to be driven by hand.
    """
    try:
        import litellm

        from nooa.tracing._litellm_journal import MessageJournalCallback
    except Exception:  # pragma: no cover - litellm optional in unit tests
        return []
    return [cb for cb in litellm.callbacks if isinstance(cb, MessageJournalCallback)]


@contextmanager
def _llm_span(model: str) -> Iterator[Any]:
    """Emit the LLM span litellm's instrumentor would have produced.

    The viewer keys its LLM-turn rendering on ``openinference.span.kind = LLM``
    spans and reconstructs message content from the journal by span id.
    """
    try:
        from opentelemetry import trace as otel_trace

        tracer = otel_trace.get_tracer("openinference.instrumentation.litellm")
    except Exception:  # pragma: no cover - otel always present in practice
        yield None
        return
    with tracer.start_as_current_span("litellm.completion") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", model)
        span.set_attribute("gen_ai.operation.name", "chat")
        yield span


def _content_text(content: Any) -> str | None:
    """Flatten a Responses content value to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("output_text")
                if text:
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts) if parts else None
    if isinstance(content, dict):
        return content.get("text") or content.get("output_text")
    return None


def _chat_items(items: list[Any]) -> list[dict[str, Any]]:
    """Responses-API items -> chat-shaped messages the trace journal can render."""
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            out.append({"role": "user", "content": str(item)})
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            out.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": _content_text(item.get("output")),
                }
            )
        else:
            role = item.get("role") or ("assistant" if item_type == "message" else "user")
            message: dict[str, Any] = {"role": role}
            text = _content_text(item.get("content"))
            if text is not None:
                message["content"] = text
            out.append(message)
    return out


class _JournalResponseView:
    """Show a Gym response to the journal callback in litellm's response shape."""

    __slots__ = ("_output_messages", "_usage")

    def __init__(self, response: NeMoGymResponse) -> None:
        self._output_messages = _chat_items(
            [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        )
        usage = response.usage
        self._usage = (
            None
            if usage is None
            else SimpleNamespace(
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0,
                prompt_tokens_details=usage.input_tokens_details,
            )
        )

    @property
    def output(self) -> Any:
        return self._output_messages

    @property
    def usage(self) -> Any:
        return self._usage


class InvalidPolicyOutputError(ValueError):
    """Raised when a valid model response contains unusable generated output."""


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


def _assistant_content_matches(content: Any, expected: str) -> bool:
    if isinstance(content, str):
        return content == expected
    if not isinstance(content, list):
        return False
    texts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}
    ]
    return bool(texts) and expected in {"".join(texts), "\n".join(texts)}


class GymResponsesLLM(UnifiedLLM):
    """NOOA LLM implementation backed exclusively by a Gym Responses model server."""

    def __init__(
        self,
        *,
        server_client: ServerClient,
        model_server_name: str,
        model_url_path: str,
        max_steps: int,
        request_collector: list[NeMoGymResponseCreateParamsNonStreaming],
        response_collector: list[NeMoGymResponse],
        cookies: dict[str, str],
        trace_hooks: GymTraceHooks | None = None,
        observation_gaps: list[ObservationGap] | None = None,
        model: str = "gym-policy",
    ) -> None:
        super().__init__(model=model)
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._model_url_path = model_url_path
        self._max_steps = max_steps
        self._request_collector = request_collector
        self._response_collector = response_collector
        self._cookies = cookies
        self._trace_hooks = trace_hooks
        self._observation_gaps = observation_gaps
        self._prior_outputs: list[dict[str, Any]] = []
        self._reported_unrestored_outputs: set[int] = set()
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
        self._restore_prior_output_metadata(input_items)
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
        self._request_collector.append(body.model_copy(deep=True))
        # NOOA's message journal is a litellm callback; this LLM bypasses litellm, so drive the
        # callback by hand so the viewer receives LLM turns (same events litellm delivers).
        callbacks = _journal_callbacks()
        litellm_call_id = f"gym-{uuid.uuid4().hex}" if callbacks else ""
        chat_messages = _chat_items(input_items) if callbacks else []
        if callbacks and instructions:
            # litellm carries the system message as messages[0]; the Responses API keeps it in
            # `instructions`. Mirror the native shape so the journal records the system prompt.
            chat_messages.insert(0, {"role": "system", "content": instructions})
        with _llm_span(self.model) as llm_span:
            for callback in callbacks:
                callback.log_pre_api_call(self.model, chat_messages, {"litellm_call_id": litellm_call_id})
            started = time.time()
            try:
                http_response = await self._server_client.post(
                    server_name=self._model_server_name,
                    url_path=self._model_url_path,
                    json=body,
                    cookies=self._cookies,
                )
                await raise_for_status(http_response)
                raw = await get_response_json(http_response)
                response = NeMoGymResponse.model_validate(raw)
            except BaseException:
                for callback in callbacks:
                    callback.log_failure_event(
                        {"litellm_call_id": litellm_call_id}, None, started, time.time()
                    )
                raise
            completed = time.time()
            self._cookies.update({name: morsel.value for name, morsel in http_response.cookies.items()})
            self._response_collector.append(response)
            if self._trace_hooks is not None:
                self._trace_hooks.record_model_response(response)
            if llm_span is not None:
                llm_span.set_attribute("llm.model_name", response.model or self.model)
            for callback in callbacks:
                callback.log_success_event(
                    {"litellm_call_id": litellm_call_id, "model": response.model or self.model},
                    _JournalResponseView(response),
                    started,
                    completed,
                )

        dumped_output = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        self._prior_outputs.extend(dumped_output)
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
                raise InvalidPolicyOutputError(f"Gym model returned invalid {output_model.__name__} JSON") from error

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

    def _restore_prior_output_metadata(self, input_items: list[dict[str, Any]]) -> None:
        """Replace NOOA's normalized history items with the exact prior Gym outputs."""

        consumed_indices: set[int] = set()
        for output_index, raw in enumerate(self._prior_outputs):
            item_type = raw.get("type")
            identity = raw.get("call_id") or raw.get("id")
            replacement_index = next(
                (
                    index
                    for index, item in enumerate(input_items)
                    if index not in consumed_indices
                    and item.get("type") == item_type
                    and identity is not None
                    and (item.get("call_id") or item.get("id")) == identity
                ),
                None,
            )
            if replacement_index is None and item_type == "message":
                raw_text = "\n".join(
                    part.get("text", "")
                    for part in raw.get("content", [])
                    if isinstance(part, dict) and part.get("type") == "output_text"
                )
                replacement_index = next(
                    (
                        index
                        for index, item in enumerate(input_items)
                        if index not in consumed_indices
                        and item.get("role") == "assistant"
                        and _assistant_content_matches(item.get("content"), raw_text)
                    ),
                    None,
                )
            if replacement_index is not None:
                input_items[replacement_index] = raw
                consumed_indices.add(replacement_index)
            elif (
                self._observation_gaps is not None
                and "prompt_token_ids" in raw
                and output_index not in self._reported_unrestored_outputs
            ):
                self._reported_unrestored_outputs.add(output_index)
                self._observation_gaps.append(
                    ObservationGap(
                        code="prior_output_metadata_unrestored",
                        detail=(
                            f"Could not restore training metadata for prior {item_type!r} output "
                            f"at index {output_index}; its generated tokens may be masked."
                        ),
                    )
                )
