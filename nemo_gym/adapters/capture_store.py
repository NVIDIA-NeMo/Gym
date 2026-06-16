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
"""Session-keyed capture store and trajectory assembly.

The sandbox-bound model proxy records every model exchange (request + response,
including token-ids when the policy is the Gym model server) into a per-session
JSONL file. Writing one file per session — keyed by the rollout's identity, i.e.
"one box = one session" — means a sandbox reaped mid-run cannot lose turns that
were already streamed out (durable gather). After the run the harness reads the
exchanges back and assembles a NeMoGym trajectory (for eval) carrying per-turn
token-ids (for RL).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


# Per-message token-id / logprob fields a Gym model server returns when
# ``return_token_id_information`` is enabled. We look for them on either the
# choice or the message object, since servers differ on placement.
_TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")


def _sanitize(session_id: str) -> str:
    cleaned = "".join(c for c in session_id if c.isalnum() or c in ("-", "_", "."))
    return cleaned or "session"


class CaptureStore:
    """Append-only, session-keyed JSONL sink for model exchanges."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, session_id: str) -> Path:
        return self._root / f"{_sanitize(session_id)}.capture.jsonl"

    def record(self, session_id: str, exchange: dict[str, Any]) -> None:
        """Append one exchange and fsync, so a killed box can't lose it."""
        line = json.dumps(exchange, default=str, ensure_ascii=False)
        path = self.path_for(session_id)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def read(self, session_id: str) -> list[dict[str, Any]]:
        path = self.path_for(session_id)
        if not path.exists():
            return []
        exchanges: list[dict[str, Any]] = []
        # Stream line-by-line: a capture can be hundreds of MB (token-ids/logprobs), so avoid
        # read_text().splitlines() which would hold the whole file as a string + a list of lines.
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    exchanges.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
        return exchanges


def _token_fields(obj: Any) -> dict[str, list]:
    if not isinstance(obj, dict):
        return {}
    return {key: obj[key] for key in _TOKEN_FIELDS if isinstance(obj.get(key), list)}


def _response_message(response: dict[str, Any]) -> dict[str, Any]:
    """First-choice assistant message, with token-ids hoisted onto it."""
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    choice = choices[0]
    message = choice.get("message") or choice.get("delta") or {}
    if not isinstance(message, dict):
        return {}
    merged = dict(message)
    merged.update(_token_fields(choice))
    merged.update(_token_fields(message))
    return merged


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            (part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")) for part in content
        )
    return "" if content is None else str(content)


def _assistant_message(index: int, text: str, token_fields: dict[str, Any]) -> NeMoGymResponseOutputMessageForTraining:
    """A training assistant message, carrying token-ids/logprobs when the policy returned them."""
    return NeMoGymResponseOutputMessageForTraining(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
        prompt_token_ids=token_fields.get("prompt_token_ids") or [],
        generation_token_ids=token_fields.get("generation_token_ids") or [],
        generation_log_probs=token_fields.get("generation_log_probs") or [],
    )


def assemble_trajectory(exchanges: list[dict[str, Any]], wire: str = "chat") -> list[Any]:
    """Reconstruct ordered NeMoGym output items from captured model exchanges.

    ``wire`` selects how the captured request/response bodies are shaped:
    ``"chat"`` (OpenAI Chat Completions / the Anthropic-translated path) or
    ``"responses"`` (OpenAI Responses API, e.g. codex). Assistant messages carry
    token-ids/logprobs when the policy returned them.
    """
    if wire == "responses":
        return _assemble_responses(exchanges)
    return _assemble_chat(exchanges)


def _assemble_chat(exchanges: list[dict[str, Any]]) -> list[Any]:
    """Chat-Completions wire: a turn's tool *results* arrive as ``role: tool``
    messages in the *next* request, emitted just before the following assistant
    message to yield the natural ``assistant -> tool_output -> assistant`` order."""
    output: list[Any] = []
    seen_tool_call_ids: set[str] = set()
    assistant_index = 0

    for exchange in exchanges:
        request = exchange.get("request") or {}
        response = exchange.get("response") or {}

        for message in request.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id") or ""
            if call_id and call_id in seen_tool_call_ids:
                continue
            if call_id:
                seen_tool_call_ids.add(call_id)
            output.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=_content_text(message.get("content")),
                    status="completed",
                )
            )

        message = _response_message(response)
        if not message:
            continue
        output.append(
            _assistant_message(assistant_index, _content_text(message.get("content")), _token_fields(message))
        )
        assistant_index += 1

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not function:
                continue
            output.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=function.get("arguments", ""),
                    call_id=tool_call.get("id", ""),
                    name=function.get("name", ""),
                    type="function_call",
                    id=tool_call.get("id"),
                    status="completed",
                )
            )

    return output


def _assemble_responses(exchanges: list[dict[str, Any]]) -> list[Any]:
    """Responses-API wire: each response carries an ``output`` list of items
    (``message`` / ``reasoning`` / ``function_call``); token-ids ride on the
    assistant ``message`` item when the policy is a Gym model. A turn's tool
    *results* arrive as ``function_call_output`` items in the *next* request's
    ``input``, so we emit those just before that turn's response — yielding the
    natural ``assistant -> function_call -> function_call_output -> assistant``
    interleave (mirrors the chat-wire assembler)."""
    output: list[Any] = []
    assistant_index = 0
    seen_output_call_ids: set[str] = set()
    for exchange in exchanges:
        request = exchange.get("request") or {}
        request_input = request.get("input")
        if isinstance(request_input, list):
            for item in request_input:
                if not isinstance(item, dict) or item.get("type") != "function_call_output":
                    continue
                call_id = item.get("call_id") or item.get("id") or ""
                if call_id and call_id in seen_output_call_ids:
                    continue
                if call_id:
                    seen_output_call_ids.add(call_id)
                output.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=_content_text(item.get("output")),
                        status="completed",
                    )
                )

        response = exchange.get("response") or {}
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message" and item.get("role") == "assistant":
                text = "".join(
                    block.get("text", "")
                    for block in (item.get("content") or [])
                    if isinstance(block, dict) and block.get("type") == "output_text"
                )
                output.append(_assistant_message(assistant_index, text, _token_fields(item)))
                assistant_index += 1
            elif kind in ("function_call", "tool_call"):
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=item.get("arguments", "") or "",
                        call_id=item.get("call_id") or item.get("id") or "",
                        name=item.get("name", ""),
                        type="function_call",
                        id=item.get("id"),
                        status="completed",
                    )
                )
    return output


def has_token_ids(output_items: list[Any]) -> bool:
    """True if any assistant item carries generation token-ids (RL-usable)."""
    for item in output_items:
        if getattr(item, "generation_token_ids", None):
            return True
    return False
