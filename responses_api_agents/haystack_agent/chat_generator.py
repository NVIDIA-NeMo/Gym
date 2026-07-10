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
"""A Haystack ``ChatGenerator`` backed by a native NeMo Gym model server.

``NeMoGymResponsesChatGenerator`` implements the Haystack chat-generator contract
(``run(messages, ..., tools=None, ...) -> {"replies": [ChatMessage]}``) so it can be
dropped into ``haystack.components.agents.agent.Agent`` (or any pipeline) in place of a
provider generator such as ``NvidiaChatGenerator``. Instead of talking to an external
provider it POSTs to a NeMo Gym model server's ``/v1/responses`` endpoint, resolving the
server by name through Gym's global config. Haystack's ``Agent`` drives the repeated
tool-calling loop; this component is the single per-step model call.
"""

import asyncio
import contextvars
import json
from dataclasses import dataclass
from typing import Any, Optional

from haystack import component
from haystack.core.serialization import allow_deserialization_module, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


# Allow this generator to be deserialized from a pipeline YAML (Haystack blocks non-allowlisted
# modules by default). Safe: anyone loading such a YAML must already import this module to use it.
allow_deserialization_module("responses_api_agents.haystack_agent.chat_generator")

# Lazily-constructed, process-wide client. We don't build it at import time so the module
# can be imported (e.g. by Haystack deserialization or in unit tests) without a running
# Gym head server. Tests may monkeypatch this attribute with a mock.
_server_client: Optional[ServerClient] = None


def _get_server_client() -> ServerClient:
    global _server_client
    if _server_client is None:
        _server_client = ServerClient.load_from_global_config()
    return _server_client


@dataclass
class _GenRunState:
    """Per-rollout session state for a (potentially shared) generator instance.

    The generator is deserialized once and shared across concurrent requests (see
    ``HaystackAgent.model_post_init``), so its per-run state cannot live as plain instance
    attributes. Instead each request installs a fresh ``_GenRunState`` in ``_current_run_state``;
    reads/writes of ``_cookies``/``_usage``/``_last_response`` resolve through it.
    """

    cookies: Any = None
    usage: Any = None
    last_response: Optional["NeMoGymResponse"] = None


# Request-scoped generator state. Set by ``HaystackAgent.responses`` (and left unset for standalone
# ``run``/``run_async`` use, in which case the generator falls back to a per-instance holder).
# Isolated per asyncio Task, so concurrent rollouts never clobber each other's cookies/usage.
_current_run_state: contextvars.ContextVar[Optional[_GenRunState]] = contextvars.ContextVar(
    "nemogym_gen_run_state", default=None
)


def _stringify(value: Any) -> str:
    """Coerce a tool result (str, or list of content parts) into a plain string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [getattr(part, "text", None) or str(part) for part in value]
        return "".join(p for p in parts if p is not None)
    return str(value)


def _content_to_text(content: Any) -> str:
    """Extract plain text from a Responses message ``content`` (str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(getattr(part, "text", "") or "")
        return "".join(parts)
    return ""


def responses_input_to_messages(input_items: list[Any]) -> list[ChatMessage]:
    """Convert NeMo Gym Responses input items into Haystack ``ChatMessage``s.

    Used to seed a Haystack pipeline from a request's ``input`` (which may be a full
    multi-turn trajectory: prior assistant messages, function calls, tool outputs, and
    reasoning items).
    """
    messages: list[ChatMessage] = []
    # Track tool calls by call_id so a later function_call_output can reconstruct its origin.
    calls_by_id: dict[str, ToolCall] = {}

    for item in input_items:
        item_type = getattr(item, "type", "message")
        role = getattr(item, "role", None)

        if item_type == "function_call":
            tool_call = ToolCall(
                tool_name=item.name,
                arguments=json.loads(item.arguments) if item.arguments else {},
                id=item.call_id,
            )
            calls_by_id[item.call_id] = tool_call
            messages.append(ChatMessage.from_assistant(tool_calls=[tool_call]))
        elif item_type == "function_call_output":
            origin = calls_by_id.get(item.call_id) or ToolCall(tool_name="", arguments={}, id=item.call_id)
            messages.append(ChatMessage.from_tool(tool_result=item.output, origin=origin))
        elif item_type == "reasoning":
            summary = "".join(s.text for s in item.summary)
            messages.append(
                ChatMessage.from_assistant(
                    reasoning=summary,
                    meta={"__ng_reasoning_id__": item.id, "__ng_reasoning_encrypted__": item.encrypted_content},
                )
            )
        elif role == "assistant":
            messages.append(ChatMessage.from_assistant(text=_content_to_text(item.content) or None))
        elif role == "system" or role == "developer":
            messages.append(ChatMessage.from_system(_content_to_text(item.content), meta={"__ng_role__": role}))
        else:  # user (default)
            messages.append(ChatMessage.from_user(_content_to_text(item.content)))
    return messages


def _tool_to_responses_param(tool: Any, tools_strict: bool) -> dict[str, Any]:
    """Convert a Haystack ``Tool`` to a Responses-API function tool definition."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": tools_strict,
    }


def messages_to_responses_input(messages: list[ChatMessage], *, output: bool = False) -> list[Any]:
    """Convert Haystack ``ChatMessage``s to NeMo Gym Responses input/output items.

    :param output: When ``True``, assistant text is emitted as ``NeMoGymResponseOutputMessage``
        (the shape a verifier/trainer expects in ``response.output``). When ``False`` (the
        default, used to seed the model on each step) assistant text is emitted as the lighter
        ``NeMoGymEasyInputMessage`` which does not require synthesized ids.
    """
    items: list[Any] = []
    for index, message in enumerate(messages):
        # Tool result messages carry one or more ToolCallResults.
        if message.tool_call_results:
            for result in message.tool_call_results:
                items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=result.origin.id or f"call_{index}",
                        output=_stringify(result.result),
                    )
                )
            continue

        role = message.role
        if role == ChatRole.ASSISTANT:
            reasoning = message.reasoning
            if reasoning is not None:
                items.append(
                    NeMoGymResponseReasoningItem(
                        type="reasoning",
                        id=message.meta.get("__ng_reasoning_id__", f"rs_{index}"),
                        summary=[NeMoGymSummary(text=reasoning.reasoning_text, type="summary_text")],
                        encrypted_content=message.meta.get("__ng_reasoning_encrypted__"),
                    )
                )
            text = message.text
            if text:
                if output:
                    items.append(
                        NeMoGymResponseOutputMessage(
                            type="message",
                            id=message.meta.get("__ng_message_id__", f"msg_{index}"),
                            content=[NeMoGymResponseOutputText(type="output_text", annotations=[], text=text)],
                        )
                    )
                else:
                    items.append(NeMoGymEasyInputMessage(type="message", role="assistant", content=text))
            for call_index, tool_call in enumerate(message.tool_calls):
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        type="function_call",
                        call_id=tool_call.id or f"call_{index}_{call_index}",
                        name=tool_call.tool_name,
                        arguments=json.dumps(tool_call.arguments or {}),
                    )
                )
        else:
            # user / system / developer (developer is preserved via meta since Haystack has no such role)
            ng_role = message.meta.get("__ng_role__", role.value)
            items.append(NeMoGymEasyInputMessage(type="message", role=ng_role, content=message.text or ""))
    return items


def response_to_chat_messages(ng_response: NeMoGymResponse) -> list[ChatMessage]:
    """Fold a ``NeMoGymResponse``'s output items into Haystack assistant ``ChatMessage``(s)."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_id: Optional[str] = None
    tool_calls: list[ToolCall] = []

    for item in ng_response.output:
        item_type = getattr(item, "type", None)
        if item_type == "message" and getattr(item, "role", None) == "assistant":
            for content in item.content:
                if getattr(content, "type", None) == "output_text":
                    text_parts.append(content.text)
        elif item_type == "function_call":
            try:
                arguments = json.loads(item.arguments) if item.arguments else {}
            except (json.JSONDecodeError, TypeError):
                # Preserve malformed arguments as a raw string so the downstream tool
                # invocation surfaces the error instead of us crashing here.
                arguments = {"__raw_arguments__": item.arguments}
            tool_calls.append(ToolCall(tool_name=item.name, arguments=arguments, id=item.call_id))
        elif item_type == "reasoning":
            reasoning_id = item.id
            reasoning_parts.extend(summary.text for summary in item.summary)

    meta: dict[str, Any] = {}
    if reasoning_id is not None:
        meta["__ng_reasoning_id__"] = reasoning_id

    message = ChatMessage.from_assistant(
        text="".join(text_parts) or None,
        tool_calls=tool_calls or None,
        reasoning="".join(reasoning_parts) or None,
        meta=meta or None,
    )
    return [message]


@component
class NeMoGymResponsesChatGenerator:
    """Haystack chat generator that calls a NeMo Gym model server's ``/v1/responses``."""

    def __init__(self, server_name: str, generation_kwargs: Optional[dict[str, Any]] = None) -> None:
        """
        :param server_name: Name of the NeMo Gym model server to call, resolved via Gym's
            global config (e.g. ``"policy_model"``).
        :param generation_kwargs: Default Responses-API parameters merged into every request
            (e.g. ``temperature``, ``max_output_tokens``, ``tool_choice``).
        """
        self.server_name = server_name
        self.generation_kwargs = generation_kwargs or {}
        # Per-run session state resolves through ``_current_run_state`` when a request has installed
        # one (shared-instance use); otherwise it lands on this fallback holder (standalone use).
        self._standalone_state = _GenRunState()

    def _run_state(self) -> _GenRunState:
        state = _current_run_state.get()
        return state if state is not None else self._standalone_state

    @property
    def _cookies(self) -> Any:
        return self._run_state().cookies

    @_cookies.setter
    def _cookies(self, value: Any) -> None:
        self._run_state().cookies = value

    @property
    def _usage(self) -> Any:
        return self._run_state().usage

    @_usage.setter
    def _usage(self, value: Any) -> None:
        self._run_state().usage = value

    @property
    def _last_response(self) -> Optional[NeMoGymResponse]:
        return self._run_state().last_response

    @_last_response.setter
    def _last_response(self, value: Optional[NeMoGymResponse]) -> None:
        self._run_state().last_response = value

    def warm_up(self) -> None:  # pragma: no cover - trivial
        """No-op: the model client is resolved lazily on first call."""

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, server_name=self.server_name, generation_kwargs=self.generation_kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeMoGymResponsesChatGenerator":
        return default_from_dict(cls, data)

    def _build_params(
        self, messages: list[ChatMessage], tools: Any, generation_kwargs: Optional[dict[str, Any]], tools_strict: bool
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        params: dict[str, Any] = {"input": messages_to_responses_input(messages, output=False)}
        if tools:
            params["tools"] = [_tool_to_responses_param(tool, tools_strict) for tool in tools]
        params.update(self.generation_kwargs)
        params.update(generation_kwargs or {})
        return NeMoGymResponseCreateParamsNonStreaming(**params)

    def _accumulate_usage(self, ng_response: NeMoGymResponse) -> None:
        usage = ng_response.usage
        if usage is None:
            return
        if self._usage is None:
            self._usage = usage.model_copy(deep=True)
            return
        self._usage.input_tokens += usage.input_tokens
        self._usage.output_tokens += usage.output_tokens
        self._usage.total_tokens += usage.total_tokens
        # TODO support more advanced token details.
        self._usage.input_tokens_details.cached_tokens = 0
        self._usage.output_tokens_details.reasoning_tokens = 0

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        tools: Any = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        streaming_callback: Any = None,
        tools_strict: bool = False,
    ) -> dict[str, list[ChatMessage]]:
        """Synchronous entrypoint (standalone use). The Gym async endpoint uses ``run_async``."""
        return asyncio.run(
            self.run_async(
                messages=messages,
                tools=tools,
                generation_kwargs=generation_kwargs,
                streaming_callback=streaming_callback,
                tools_strict=tools_strict,
            )
        )

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        tools: Any = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        streaming_callback: Any = None,
        tools_strict: bool = False,
    ) -> dict[str, list[ChatMessage]]:
        if streaming_callback is not None:
            raise NotImplementedError("NeMoGymResponsesChatGenerator does not support streaming.")

        body = self._build_params(messages, tools, generation_kwargs, tools_strict)
        response = await _get_server_client().post(
            server_name=self.server_name,
            url_path="/v1/responses",
            json=body,
            cookies=self._cookies,
        )
        # We raise for status here since we expect model calls to always succeed.
        await raise_for_status(response)
        self._cookies = response.cookies
        response_json = await get_response_json(response)
        ng_response = NeMoGymResponse.model_validate(response_json)

        self._accumulate_usage(ng_response)
        self._last_response = ng_response
        return {"replies": response_to_chat_messages(ng_response)}
