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
"""Pure-function translation between the Anthropic Messages API and OpenAI Chat Completions.

Direction in : Anthropic ``/v1/messages`` request  -> Chat Completions messages/tools.
Direction out: Chat Completions assistant message   -> Anthropic content blocks + SSE.

The vLLM reasoning parser surfaces reasoning wrapped in ``<think>...</think>`` inside the
assistant content; on the way out we keep it inline as text (we do not synthesize separate
``thinking`` blocks, which would need signatures the open model can't produce). On the way
in, prior ``thinking`` blocks are folded back inline so the model sees its own reasoning.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


def _blocks_to_text(content: Any) -> str:
    """Flatten an Anthropic content value (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", "") or "")
        elif btype == "thinking":
            think = block.get("thinking") or ""
            if think:
                parts.append(f"<think>{think}</think>")
    return "".join(parts)


def anthropic_messages_to_chat(
    system: Any,
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert an Anthropic request's ``system`` + ``messages`` to Chat Completions messages.

    Anthropic carries tool results in the *user* turn following an assistant ``tool_use``;
    Chat Completions expects ``role="tool"`` messages right after the assistant message that
    issued the calls. We preserve that ordering by emitting tool messages before any user
    text within a user turn.
    """
    out: List[Dict[str, Any]] = []

    sys_text = _blocks_to_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for m in messages:
        role = m.get("role")
        content = m.get("content")

        if role == "user":
            text_parts: List[str] = []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", "") or "")
                elif btype == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _blocks_to_text(block.get("content")),
                        }
                    )
            text = "".join(text_parts)
            if text:
                out.append({"role": "user", "content": text})

        elif role == "assistant":
            text_parts = []
            tool_calls: List[Dict[str, Any]] = []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", "") or "")
                elif btype == "thinking":
                    think = block.get("thinking") or ""
                    if think:
                        text_parts.append(f"<think>{think}</think>")
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
            msg: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)

    return out


def anthropic_tools_to_chat(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", "") or "",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Chat Completions assistant message -> Anthropic content blocks
# ---------------------------------------------------------------------------

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def chat_message_to_anthropic_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build Anthropic content blocks (text + tool_use) from a chat completion message."""
    blocks: List[Dict[str, Any]] = []

    text = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    if text or not tool_calls:
        # Always emit at least one (possibly empty) text block when there are no tool calls,
        # so the response is never block-less.
        blocks.append({"type": "text", "text": text})

    for tc in tool_calls:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": parsed,
            }
        )

    return blocks


def anthropic_stop_reason(finish_reason: Optional[str], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_use"
    return _FINISH_TO_STOP.get(finish_reason or "stop", "end_turn")


def build_anthropic_message(
    message: Dict[str, Any],
    finish_reason: Optional[str],
    model: str,
    input_tokens: int,
    output_tokens: int,
    message_id: str,
) -> Dict[str, Any]:
    """Assemble a complete (non-streaming) Anthropic Messages response object."""
    blocks = chat_message_to_anthropic_blocks(message)
    has_tools = bool(message.get("tool_calls"))
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": anthropic_stop_reason(finish_reason, has_tools),
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def sse_events_for_message(anthropic_message: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand a complete Anthropic message into the ordered SSE (event, data) pairs.

    Each content block is streamed as start -> single delta -> stop. Sending whole blocks
    in one delta (rather than token-by-token) is accepted by the claude CLI.
    """
    msg_id = anthropic_message["id"]
    model = anthropic_message["model"]
    blocks = anthropic_message["content"]
    usage = anthropic_message["usage"]

    events: List[Tuple[str, Dict[str, Any]]] = []
    events.append(
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 0},
                },
            },
        )
    )

    for index, block in enumerate(blocks):
        if block["type"] == "text":
            events.append(
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                )
            )
        elif block["type"] == "tool_use":
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}},
                    },
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
                    },
                )
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))

    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": anthropic_message["stop_reason"],
                    "stop_sequence": anthropic_message["stop_sequence"],
                },
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return events
