# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge synchronous AgentDojo-family pipelines to NeMo Gym's async model server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Literal

from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM, _function_to_openai, _message_to_openai
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from openai._types import NOT_GIVEN, NotGiven

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    accumulate_response_usage,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


def _content_text(message: ChatMessage) -> str:
    content = message.get("content")
    return get_text_content_as_str(content) if content else ""


def agentdojo_messages_to_responses_input(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert AgentDojo's complete conversation to Responses API input items."""
    items: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        role = message["role"]
        if role in {"system", "user"}:
            # AgentDojo's OpenAI adapter maps its system role to OpenAI's developer role.
            items.append(
                {
                    "type": "message",
                    "role": "developer" if role == "system" else "user",
                    "content": [{"type": "input_text", "text": _content_text(message)}],
                }
            )
            continue
        if role == "assistant":
            text = _content_text(message)
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
            for call_index, tool_call in enumerate(message.get("tool_calls") or []):
                call_id = tool_call.id or f"agentdojo-{message_index}-{call_index}"
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_call.function,
                        "arguments": json.dumps(tool_call.args),
                    }
                )
            continue
        if role == "tool":
            tool_call = message["tool_call"]
            call_id = message.get("tool_call_id") or tool_call.id
            if call_id is None:
                raise ValueError("AgentDojo tool results require a call id for Responses API replay.")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.get("error") or _content_text(message),
                }
            )
            continue
        raise ValueError(f"Unsupported AgentDojo message role: {role!r}")
    return items


def agentdojo_tools_to_responses_tools(runtime: FunctionsRuntime) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": function.name,
            "description": function.description,
            "parameters": function.parameters.model_json_schema(),
        }
        for function in runtime.functions.values()
    ]


def response_to_agentdojo_message(response: NeMoGymResponse) -> ChatAssistantMessage:
    text_parts: list[str] = []
    tool_calls: list[FunctionCall] = []
    for item in response.output:
        if isinstance(item, NeMoGymResponseOutputMessage):
            text_parts.extend(part.text for part in item.content if getattr(part, "text", None))
        elif isinstance(item, NeMoGymResponseFunctionToolCall):
            try:
                arguments = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Model returned invalid arguments for {item.name!r}: {item.arguments!r}") from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"Model returned non-object arguments for {item.name!r}.")
            tool_calls.append(FunctionCall(function=item.name, args=arguments, id=item.call_id))
    text = "\n".join(part for part in text_parts if part)
    return ChatAssistantMessage(
        role="assistant",
        content=[text_content_block_from_string(text)] if text else None,
        tool_calls=tool_calls or None,
    )


def agentdojo_messages_to_response_output(messages: Sequence[ChatMessage]) -> list[Any]:
    """Serialize AgentDojo's generated assistant/tool trajectory as response output."""
    output: list[Any] = []
    pending_call_ids: dict[str, list[str]] = {}
    for message_index, message in enumerate(messages):
        if message["role"] == "assistant":
            text = _content_text(message)
            if text:
                output.append(
                    NeMoGymResponseOutputMessage(
                        id=f"agentdojo-message-{message_index}",
                        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                    )
                )
            for call_index, tool_call in enumerate(message.get("tool_calls") or []):
                call_id = tool_call.id or f"agentdojo-{message_index}-{call_index}"
                pending_call_ids.setdefault(tool_call.function, []).append(call_id)
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        call_id=call_id,
                        name=tool_call.function,
                        arguments=json.dumps(tool_call.args),
                    )
                )
        elif message["role"] == "tool":
            tool_call = message["tool_call"]
            call_id = message.get("tool_call_id") or tool_call.id
            if call_id is None:
                pending = pending_call_ids.get(tool_call.function, [])
                call_id = pending.pop(0) if pending else f"agentdojo-tool-{message_index}"
            output.append(
                NeMoGymResponseFunctionCallOutput(
                    id=f"agentdojo-tool-output-{message_index}",
                    call_id=call_id,
                    output=message.get("error") or _content_text(message),
                    status="completed",
                    type="function_call_output",
                )
            )
    return output


class NeMoGymAgentDojoLLM(OpenAILLM):
    """Synchronous AgentDojo pipeline element backed by a NeMo Gym model server.

    AgentDojo's pipeline is synchronous. The adapter executes it in a worker thread;
    each model query is submitted back to the agent server's main asyncio loop so the
    existing ``ServerClient`` and its shared aiohttp transport remain authoritative.

    It subclasses ``OpenAILLM`` rather than ``BasePipelineElement`` for one reason: the
    ``tool_filter`` defense refuses to build unless ``isinstance(llm, OpenAILLM)``. Upstream's
    ``__init__`` only assigns four attributes and opens no connection, so standing in for it
    costs nothing and ``query`` below still owns every policy call.

    The alternative -- handing upstream a real ``openai.OpenAI`` pointed at the serving
    endpoint -- would make this the one defense that bypasses the model server: no token
    accounting, no per-rollout attribution, the OpenAI SDK's httpx transport that this repo
    bans at concurrency, and no `developer`->`system` rewrite, which some OpenAI-compatible
    servers (Qwen3.5 on SGLang, for one) reject outright. ``self.client`` is the proxy instead, so `tool_filter`'s
    `client.chat.completions.create` lands on the same path as every other call.
    """

    name = "nemo-gym-policy-model"

    def __init__(
        self,
        *,
        pipeline_name: str,
        event_loop: asyncio.AbstractEventLoop,
        server_client: ServerClient,
        model_server_name: str,
        model_url_path: str,
        request_options: dict[str, Any],
        system_role: Literal["developer", "system"] = "developer",
    ) -> None:
        self.name = pipeline_name
        # The OpenAILLM surface tool_filter reads: a client it can call, and a model name it
        # puts in the request. Both point back here, not at the serving endpoint.
        self.client = NeMoGymOpenAIProxy(self)
        self.model = pipeline_name
        self.temperature = request_options.get("temperature", 0.0)
        self.reasoning_effort = None
        self._event_loop = event_loop
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._model_url_path = model_url_path
        self._request_options = request_options
        self._system_role = system_role
        self._cookies: Any = None
        self.responses: list[NeMoGymResponse] = []
        self.last_messages: Sequence[ChatMessage] = []
        # Set only under the tool_filter defense: the tool names its selection call kept.
        self.tool_filter_kept_tools: list[str] | None = None

    async def _request_chat(self, payload: dict[str, Any]) -> NeMoGymChatCompletion:
        response = await self._server_client.post(
            server_name=self._model_server_name,
            url_path=self._model_url_path,
            json=payload,
            cookies=self._cookies,
        )
        await raise_for_status(response)
        self._cookies = response.cookies
        return NeMoGymChatCompletion.model_validate(await get_response_json(response))

    def _chat_to_response(
        self,
        payload: dict[str, Any],
        chat_completion: NeMoGymChatCompletion,
    ) -> NeMoGymResponse:
        chat_params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
        converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=True)
        # We only need the normalized response envelope and usage here; app.py replaces
        # output with the complete AgentDojo transcript. Building this small envelope
        # avoids round-tripping AgentDojo's Chat content/tool schema through Responses.
        response_params = NeMoGymResponseCreateParamsNonStreaming(
            input="",
            max_output_tokens=chat_params.max_tokens or chat_params.max_completion_tokens,
            model=chat_completion.model,
            parallel_tool_calls=chat_params.parallel_tool_calls,
            temperature=chat_params.temperature,
            top_p=chat_params.top_p,
        )
        return converter.chat_completion_to_response(
            response_params,
            chat_completion,
            preserve_envelope_id=True,
        )

    async def _request(self, payload: dict[str, Any]) -> NeMoGymResponse:
        chat_completion = await self._request_chat(payload)
        return self._chat_to_response(payload, chat_completion)

    def create_chat_completion(self, **kwargs: Any) -> NeMoGymChatCompletion:
        """Synchronous OpenAI-compatible entrypoint for upstream defense clients."""

        def drop_not_given(value: Any) -> Any:
            if value is NOT_GIVEN or isinstance(value, NotGiven):
                return None
            if isinstance(value, dict):
                return {key: cleaned for key, item in value.items() if (cleaned := drop_not_given(item)) is not None}
            if isinstance(value, list):
                return [cleaned for item in value if (cleaned := drop_not_given(item)) is not None]
            return value

        payload = drop_not_given(kwargs)
        for message in payload.get("messages", []):
            if self._system_role == "system" and message.get("role") == "developer":
                message["role"] = "system"
            if message.get("role") == "tool":
                message.pop("name", None)
        future = asyncio.run_coroutine_threadsafe(self._request_chat(payload), self._event_loop)
        chat_completion = future.result()
        # Record first: the transcript keeps the reasoning, which the converter lifts into
        # its own item. Only the copy handed to upstream has the envelope removed.
        self.responses.append(self._chat_to_response(payload, chat_completion))
        return self._without_reasoning_envelope(chat_completion)

    @staticmethod
    def _without_reasoning_envelope(chat_completion: NeMoGymChatCompletion) -> NeMoGymChatCompletion:
        """Hand upstream the model's answer, not Gym's think-tag envelope.

        On the Chat Completions path the model server re-wraps a provider's
        `reasoning_content` into `<think>...</think>` and prepends it to `content`, because
        that is how reasoning travels through Gym's Chat schema; the Responses conversion
        unwraps it again. The defenses that own their OpenAI client -- CaMeL, Progent,
        DRIFT -- read `content` directly and were written against models that never emit
        it: Progent runs `json.loads` over the whole string and raises
        `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` on the leading tag,
        which masks the rollout. So the envelope comes off before upstream sees it.

        Content with no think tag is returned untouched. Content that is *only* reasoning
        becomes empty, which is a real model failure and is graded as one rather than
        being disguised as a parse error.
        """
        converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=True)
        normalized = chat_completion.model_copy(deep=True)
        for choice in normalized.choices:
            content = choice.message.content
            if not isinstance(content, str) or not content:
                continue
            reasoning, remaining = converter._extract_reasoning_from_content(content)
            if reasoning:
                choice.message.content = remaining.strip()
        return normalized

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        tools = [dict(_function_to_openai(function)) for function in runtime.functions.values()]
        chat_messages = [dict(_message_to_openai(message, self.name)) for message in messages]
        for chat_message in chat_messages:
            if self._system_role == "system" and chat_message.get("role") == "developer":
                chat_message["role"] = "system"
            # AgentDojo emits the optional legacy `name` field on tool results.
            # Gym's current Chat schema rejects it; call_id already identifies the
            # function unambiguously, so removing it is semantics-preserving.
            if chat_message.get("role") == "tool":
                chat_message.pop("name", None)
        payload = {
            **self._request_options,
            "messages": chat_messages,
            "tools": tools,
        }
        if tools:
            payload["tool_choice"] = "auto"
        future = asyncio.run_coroutine_threadsafe(self._request(payload), self._event_loop)
        response = future.result()
        output = response_to_agentdojo_message(response)
        updated_messages = [*messages, output]
        self.responses.append(response)
        self.last_messages = updated_messages
        return query, runtime, env, updated_messages, extra_args or {}

    def accumulated_usage(self):
        usage = None
        for response in self.responses:
            usage = accumulate_response_usage(usage, response.usage)
        return usage


class NeMoGymOpenAIProxy:
    """Minimal OpenAI client surface used by AgentDyn's bundled defenses."""

    def __init__(self, bridge: NeMoGymAgentDojoLLM) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=bridge.create_chat_completion))
