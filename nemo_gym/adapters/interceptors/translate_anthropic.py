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
"""Anthropic Messages <-> OpenAI Chat Completions translation.

Claude Code speaks the Anthropic Messages API (``POST /v1/messages``). To run it
against **any backend** — i.e. the Gym/vLLM model server, which speaks the
OpenAI Chat Completions API — we translate the request on the way out and the
response on the way back. Placed *before* the ``capture`` interceptor in a
sandbox-bound proxy, the captured exchange is the OpenAI-shaped one (so token-id
capture and trajectory assembly stay uniform), while the agent still receives an
Anthropic-shaped response.

NOTE: this is the non-streaming mapping — the request is forced to
``stream: false`` so the backend returns a single JSON body the proxy can buffer.
Streaming (SSE) translation is the remaining piece for clients that require it
(see the proxy's buffering limitation).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)


logger = logging.getLogger(__name__)

_TRANSLATE_FLAG = "_anthropic_translate"
_MODEL_KEY = "_anthropic_model"

_FINISH_TO_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "end_turn"}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


def _system_text(system: Any) -> str:
    if isinstance(system, list):
        return "\n".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
    return system or ""


def _map_tool_choice(tc: Any) -> Any:
    if not isinstance(tc, dict):
        return tc
    kind = tc.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "tool" and tc.get("name"):
        return {"type": "function", "function": {"name": tc["name"]}}
    return "auto"


def anthropic_to_openai_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request body into an OpenAI Chat body."""
    out: dict[str, Any] = {"model": body.get("model"), "messages": []}

    system = body.get("system")
    if system:
        out["messages"].append({"role": "system", "content": _system_text(system)})

    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            out["messages"].append({"role": role, "content": content})
            continue
        blocks = content if isinstance(content, list) else []

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
            message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            out["messages"].append(message)
        else:
            # user turn: text becomes a user message; tool_result blocks become
            # separate OpenAI ``tool`` messages keyed by the originating call id.
            text_parts = []
            tool_msgs: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _text_of(block.get("content")),
                        }
                    )
            if text_parts or not tool_msgs:
                out["messages"].append({"role": "user", "content": "".join(text_parts)})
            out["messages"].extend(tool_msgs)

    if body.get("tools"):
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
            for tool in body["tools"]
            if isinstance(tool, dict)
        ]
    if body.get("tool_choice") is not None:
        out["tool_choice"] = _map_tool_choice(body["tool_choice"])

    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]

    # The proxy buffers a single JSON body; force non-streaming upstream.
    out["stream"] = False
    return out


def openai_to_anthropic_response(body: dict[str, Any], model: str | None) -> dict[str, Any]:
    """Translate an OpenAI Chat Completion response into an Anthropic Messages response."""
    choices = body.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}

    content_blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        try:
            tool_input = json.loads(function.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid4().hex[:16]}",
                "name": function.get("name", ""),
                "input": tool_input,
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = body.get("usage") or {}
    return {
        "id": body.get("id") or f"msg_{uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model or body.get("model"),
        "content": content_blocks,
        "stop_reason": _FINISH_TO_STOP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


_STATUS_TO_ANTHROPIC_ERROR = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
}


def openai_to_anthropic_error(body: dict[str, Any], status_code: int | None = None) -> dict[str, Any]:
    """Reshape an OpenAI-style error body into the Anthropic error envelope.

    Upstream failures (401/429/validation) come back OpenAI-shaped; without this the
    Claude CLI hits an opaque parse error instead of a typed Anthropic error."""
    err = body.get("error") if isinstance(body.get("error"), dict) else {}
    message = (err.get("message") if isinstance(err, dict) else None) or body.get("message") or "upstream error"
    err_type = (err.get("type") if isinstance(err, dict) else None) or _STATUS_TO_ANTHROPIC_ERROR.get(
        int(status_code or 0), "api_error"
    )
    return {"type": "error", "error": {"type": str(err_type), "message": str(message)}}


class Interceptor(RequestInterceptor, ResponseInterceptor):
    """Adapt Anthropic ``/v1/messages`` calls to an OpenAI Chat Completions backend."""

    def __init__(self, *, model_override: str | None = None) -> None:
        self._model_override = model_override

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        if req.path.rstrip("/").endswith("/messages") and isinstance(req.body, dict):
            req.ctx.extra[_TRANSLATE_FLAG] = True
            req.ctx.extra[_MODEL_KEY] = req.body.get("model")
            # /v1/messages -> /v1/chat/completions
            req.path = req.path.rstrip("/")[: -len("/messages")] + "/chat/completions"
            req.body = anthropic_to_openai_request(req.body)
        return req

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        if resp.ctx.extra.get(_TRANSLATE_FLAG) and isinstance(resp.body, dict):
            if "choices" in resp.body:
                model = self._model_override or resp.ctx.extra.get(_MODEL_KEY)
                resp.body = openai_to_anthropic_response(resp.body, model)
            else:
                # Non-completion body (error / 4xx-5xx): reshape to the Anthropic error
                # envelope so the CLI surfaces a typed error, not an opaque parse failure.
                resp.body = openai_to_anthropic_error(resp.body, resp.status_code)
        return resp
