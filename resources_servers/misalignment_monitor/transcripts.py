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
"""Model calls and rollouts in any of Gym's API dialects, as plain text for a reviewer."""

import json
from typing import Any


# Training token fields are large and mean nothing to a reviewer.
_TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")


def _text(content: Any) -> str:
    """Text of a message's content: a string, or a list of Responses, Chat or Anthropic parts."""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        return json.dumps(content, default=str)
    parts = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text") or part.get("refusal") or part.get("thinking")
            parts.append(text if isinstance(text, str) else _text(part.get("content")) or f"[{part.get('type')}]")
        else:
            parts.append(str(part))
    return "\n".join(parts)


def _chat_items(message: dict) -> list[dict]:
    """A Chat Completions message as Responses-style items."""
    role = message.get("role", "user")
    if role == "tool":
        return [{"type": "function_call_output", "output": _text(message.get("content"))}]
    items = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        items.append({"type": "reasoning", "summary": [{"text": reasoning}]})
    if message.get("content"):
        items.append({"type": "message", "role": role, "content": _text(message["content"])})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        items.append({"type": "function_call", "name": function.get("name"), "arguments": function.get("arguments")})
    return items


def _block_items(role: str, content: Any) -> list[dict]:
    """Anthropic Messages content as Responses-style items."""
    if not isinstance(content, list):
        return [{"type": "message", "role": role, "content": _text(content)}]
    items = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "thinking":
            items.append({"type": "reasoning", "summary": [{"text": block.get("thinking", "")}]})
        elif kind == "tool_use":
            items.append(
                {"type": "function_call", "name": block.get("name"), "arguments": json.dumps(block.get("input"))}
            )
        elif kind == "tool_result":
            items.append({"type": "function_call_output", "output": _text(block.get("content"))})
        else:
            items.append({"type": "message", "role": role, "content": _text([block])})
    return items


def _responses_items(request: dict) -> list[dict]:
    task = request.get("input") or []
    items = [{"role": "user", "content": task}] if isinstance(task, str) else list(task)
    if request.get("instructions"):
        items.insert(0, {"role": "system", "content": request["instructions"]})
    return items


def turn_items(dialect: str, request: dict, response: dict) -> tuple[list[dict], list[dict]]:
    """One model call as (context, output) items, whichever API the agent used."""
    if dialect == "chat":
        context = [item for message in request.get("messages") or [] for item in _chat_items(message)]
        choices = response.get("choices") or [{}]
        return context, _chat_items({"role": "assistant", **(choices[0].get("message") or {})})
    if dialect == "messages":
        context = _block_items("system", request["system"]) if request.get("system") else []
        for message in request.get("messages") or []:
            context += _block_items(message.get("role", "user"), message.get("content"))
        return context, _block_items("assistant", response.get("content"))
    return _responses_items(request), list(response.get("output") or [])


def _item_text(item: dict) -> tuple[str, str]:
    kind = item.get("type", "message")
    if kind == "message":
        return str(item.get("role", "user")), _text(item.get("content"))
    if kind == "reasoning":
        parts = [*(item.get("summary") or []), *(item.get("content") or [])]
        return "reasoning", "\n".join(str(part.get("text", "")) for part in parts)
    if kind == "function_call":
        return "tool_call", f"{item.get('name')}({item.get('arguments')})"
    if kind == "function_call_output":
        return "tool_output", _text(item.get("output"))
    return str(kind), json.dumps({k: v for k, v in item.items() if k not in _TOKEN_FIELDS}, default=str)


def render_items(items: list[dict], max_chars: int) -> str:
    """Items as labeled plain text. A longer text loses its middle."""
    text = "\n\n".join("[{}]\n{}".format(*_item_text(item)) for item in items)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n[... {len(text) - 2 * half} characters omitted ...]\n\n{text[-half:]}"


def render_transcript(rollout: dict, max_chars: int) -> str:
    """A finished rollout's instructions, task input and every output item."""
    items = _responses_items(rollout.get("responses_create_params") or {})
    return render_items(items + list((rollout.get("response") or {}).get("output") or []), max_chars)
