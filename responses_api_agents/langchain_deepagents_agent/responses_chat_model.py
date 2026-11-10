# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The LangChain chat model that a deepagents graph calls, as Gym Responses API requests.

`ResponsesChatModel` implements the part of LangChain's `BaseChatModel` that deepagents uses (`bind_tools()` and
`_agenerate()`) and leaves sending each request to a subclass. Two subclasses send them:
`GymResponsesChatModel` (responses_langchain_bridge.py) posts through the Agent Server's `server_client` for the
in-process graph, and `SandboxResponsesChatModel` (sandbox_runner.py) posts from the task sandbox.

The sandbox runner imports this module inside the task sandbox, whose runtime has deepagents and its dependencies but
no Gym, so it imports nothing from Gym.
"""

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


def is_context_limit(error_body: str) -> bool:
    """Whether a Model Server error body reports a context limit. HTTP status alone cannot prove one, so this reads
    the provider's explicit error code."""
    try:
        document = json.loads(error_body)
    except (TypeError, ValueError):
        return False
    error = document.get("error") if isinstance(document, dict) else None
    return isinstance(error, dict) and error.get("code") == "context_length_exceeded"


def message_text(content: Any) -> str:
    """The text of a LangChain message's or a Responses input item's content, which may be a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def to_gym_input(messages: list) -> list[dict]:
    """LangChain messages -> Gym Responses API input items, for one internal deepagents model call."""
    items: list[dict] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message_text(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            # Round-trips reasoning items stashed by ai_message_from_response() so they aren't silently
            # dropped on the next internal turn. This resends reasoning content on every subsequent
            # internal model call (the full growing input list is resent each turn), which grows token
            # usage — comment out this line to stop round-tripping and only pay for reasoning once, on
            # the turn it was produced.
            items.extend(message.additional_kwargs.get("reasoning_items", []))
            if message.content:
                items.append({"type": "message", "role": "assistant", "content": message_text(message.content)})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(call["args"]),
                    }
                )
        elif isinstance(message, SystemMessage):
            items.append({"type": "message", "role": "system", "content": message_text(message.content)})
        else:
            items.append({"type": "message", "role": "user", "content": message_text(message.content)})
    return items


def ai_message_from_response(response: Mapping[str, Any]) -> AIMessage:
    """A Gym Responses API response body -> a single LangChain AIMessage, for one internal deepagents model call."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    # Stashed separately from text_parts (not shown as visible assistant text) so to_gym_input() can
    # round-trip each one back as its own `reasoning` item on the next internal turn instead of silently
    # dropping it. Kept as the original item dict (full fidelity: id, summary, encrypted_content, ...) so
    # to_gym_input() can pass it straight through unchanged.
    reasoning_items: list[dict] = []
    for item in response.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            text_parts.append(
                "".join(
                    part.get("text", "") for part in item.get("content") or [] if part.get("type") == "output_text"
                )
            )
        elif item_type == "function_call":
            try:
                args = json.loads(item["arguments"])
            except (json.JSONDecodeError, TypeError):
                # Malformed args from the model server. Pass through as {} rather than crashing the
                # request — deepagents' underlying ToolNode (handle_tool_errors=True by default) will
                # fail the tool invocation against the real schema and feed the model an error
                # ToolMessage, same graceful-degradation outcome as simple_agent's explicit guard.
                args = {}
            tool_calls.append(tool_call(name=item["name"], args=args, id=item["call_id"]))
        elif item_type == "reasoning":
            reasoning_items.append(dict(item))
        else:
            # Anything else (mcp_call, web_search_call, code_interpreter_call, computer_call, ...) has no
            # client-side execution path in deepagents and, for hosted/server-executed items, represents a
            # tool result the model already produced — dropping it silently would leave the agent's next
            # turn with no memory of something that already happened. Fail loudly instead of guessing a
            # textual fallback for a type with no observed repro yet.
            raise NotImplementedError(
                f"The DeepAgents chat model does not support Gym Responses output item type {item_type!r} "
                f"(response id={response.get('id')!r}). Only 'message', 'function_call', and 'reasoning' are "
                "handled today."
            )
    return AIMessage(
        content="".join(text_parts),
        tool_calls=tool_calls,
        id=response.get("id"),
        response_metadata={"id": response.get("id")},
        additional_kwargs={"reasoning_items": reasoning_items} if reasoning_items else {},
    )


class ResponsesChatModel(BaseChatModel):
    """A LangChain chat model whose every call is one Gym Responses API request; a subclass sends it."""

    @property
    def _llm_type(self) -> str:
        return "gym-responses-api"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError(f"{type(self).__name__} is async-only; use ainvoke/astream_events.")

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # Gym's FunctionToolParam (nemo_gym/client/openai_client.py, a passthrough of the real OpenAI Responses
        # API type) requires "strict" as a present-but-nullable field — omitting it entirely fails
        # request validation ("Field required"), confirmed by actually running this against a live
        # model server. convert_to_openai_tool()'s "function" dict doesn't include it by default.
        formatted = [{"type": "function", "strict": False, **convert_to_openai_tool(t)["function"]} for t in tools]
        bind_kwargs: dict[str, Any] = {"tools": formatted}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        request_body: dict[str, Any] = {"input": to_gym_input(messages)}
        if "tools" in kwargs:
            request_body["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            request_body["tool_choice"] = kwargs["tool_choice"]
        response = await self.create_response(request_body)
        return ChatResult(generations=[ChatGeneration(message=ai_message_from_response(response))])

    async def create_response(self, request_body: dict[str, Any]) -> Mapping[str, Any]:
        """Send one Responses API request to the Gym Model Server and return the response body."""
        raise NotImplementedError
