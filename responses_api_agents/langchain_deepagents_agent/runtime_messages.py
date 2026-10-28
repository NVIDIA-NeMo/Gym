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
"""Message conversion shared by in-process and sandboxed DeepAgents runtimes."""

import json
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def to_responses(new_messages: list, model_name: str) -> dict:
    """LangChain messages -> a Gym Responses API response dict, preserving the full tool-call/tool-result
    trace (not just the final assistant text). Reuses to_gym_input()'s message-type mapping: its
    function_call/function_call_output dicts already satisfy NeMoGymResponseOutputItem's required fields
    (NeMoGymResponseFunctionToolCall and NeMoGymFunctionCallOutput are the same classes on the input and
    output sides), but its message dicts need an `id` and output_text-shaped `content` added, since
    NeMoGymResponseOutputMessage (output-only) requires both and to_gym_input()'s input-side message dicts
    don't carry them."""
    output_items = []
    for item in to_gym_input(new_messages):
        if item["type"] == "message":
            item = {
                **item,
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "content": [{"type": "output_text", "text": item["content"], "annotations": []}],
            }
        output_items.append(item)
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "created_at": time.time(),
        "model": model_name,
        "object": "response",
        "output": output_items,
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
    }


def to_gym_input(messages: list) -> list[dict]:
    """LangChain messages -> Gym Responses API input items, for one internal deepagents model call."""
    items: list[dict] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _text(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            # Round-trips reasoning items stashed by to_langchain_ai_message() so they aren't silently
            # dropped on the next internal turn. This resends reasoning content on every subsequent
            # internal model call (the full growing input list is resent each turn), which grows token
            # usage — comment out this line to stop round-tripping and only pay for reasoning once, on
            # the turn it was produced.
            items.extend(message.additional_kwargs.get("reasoning_items", []))
            if message.content:
                items.append({"type": "message", "role": "assistant", "content": _text(message.content)})
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
            items.append({"type": "message", "role": "system", "content": _text(message.content)})
        else:
            items.append({"type": "message", "role": "user", "content": _text(message.content)})
    return items


def from_items(items: str | list[dict[str, Any]]) -> list:
    """Decode the text/function-call input supported by both execution modes."""
    if isinstance(items, str):
        return [HumanMessage(content=items)]
    messages = []
    roles = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage, "developer": SystemMessage}
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            content = item["content"]
            if isinstance(content, list) and any(
                p.get("type") not in {"input_text", "output_text", "text"} for p in content
            ):
                raise ValueError("Sandboxed DeepAgents currently supports text input only")
            messages.append(roles[item["role"]](content=_text(content)))
        elif kind == "function_call":
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": item["name"], "args": json.loads(item["arguments"]), "id": item["call_id"]}],
                )
            )
        elif kind == "function_call_output":
            messages.append(ToolMessage(content=_text(item["output"]), tool_call_id=item["call_id"]))
        elif kind == "reasoning":
            messages.append(AIMessage(content="", additional_kwargs={"reasoning_items": [item]}))
        else:
            raise ValueError(f"Unsupported DeepAgents input item: {kind}")
    return messages


def from_response(response: dict[str, Any]) -> AIMessage:
    """Decode one model response, retaining reasoning and native usage metadata."""
    texts, calls, invalid, reasoning = [], [], [], []
    for item in response["output"]:
        kind = item["type"]
        if kind == "message":
            texts.append(_text(item["content"]))
        elif kind == "reasoning":
            reasoning.append(item)
        elif kind == "function_call":
            try:
                args = json.loads(item["arguments"])
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (ValueError, TypeError) as error:
                invalid.append(
                    {"name": item["name"], "args": item["arguments"], "id": item["call_id"], "error": str(error)}
                )
            else:
                calls.append({"name": item["name"], "args": args, "id": item["call_id"]})
        else:
            raise ValueError(f"Unsupported DeepAgents model output item: {kind}")
    usage = response.get("usage")
    native_usage = (
        None
        if usage is None
        else {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    )
    return AIMessage(
        content="".join(texts),
        tool_calls=calls,
        invalid_tool_calls=invalid,
        id=response["id"],
        response_metadata={"id": response["id"]},
        usage_metadata=native_usage,
        additional_kwargs={"reasoning_items": reasoning} if reasoning else {},
    )
