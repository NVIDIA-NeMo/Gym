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
"""Streaming Chat Completions-dialect support shared by every Gym model server.

Blackbox harnesses that speak the OpenAI Chat Completions API over SSE (e.g. the OpenClaw
agent PinchBench runs) send requests the strict ``NeMoGymChatCompletionCreateParamsNonStreaming``
model rejects: a ``stream: true`` flag (its ``stream`` field is typed ``Literal[False]``, since a
Gym model server calls its backend non-streaming and buffers the whole response) plus a
``stream_options`` block, which is only meaningful alongside ``stream: true``.

A Gym model server can still serve these clients by computing the complete Chat Completion with
its existing non-streaming backend call and re-emitting it as an SSE stream. This module provides:

- the request-side sanitizer that maps the streaming wire body onto the strict params shape
  (drops ``stream``/``stream_options``, remembers whether usage was requested);
- the response-side synthesizer that re-emits a complete ``NeMoGymChatCompletion`` as the
  ``chat.completion.chunk`` SSE sequence a streaming client expects, terminated by ``data: [DONE]``.

Only the SSE envelope is synthesized -- there is no true token-by-token streaming. The backend
call completes before the first byte is emitted, so the model server's retry and
error-normalization behavior is fully preserved on this path. This path is intended for eval-only
streaming clients: token ids and logprobs from the backend response are not carried in the
``chat.completion.chunk`` schema, and a client that does not set ``stream_options.include_usage``
gets no usage chunk, so a model-call record reconstructed from this stream will lack token counts.
"""

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any, AsyncIterator, Awaitable, Iterator

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming


LOG = logging.getLogger(__name__)

# An SSE comment: ignored by every conforming client parser, but still bytes on the wire, so it
# refreshes read/idle timers while the buffered backend call is still running.
SSE_HEARTBEAT = ": keepalive\n\n"

# Well under the 120s idle timeout the OpenClaw agent applies to its model endpoint.
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0

_PARAM_FIELDS = frozenset(NeMoGymChatCompletionCreateParamsNonStreaming.model_fields)


def _wants_usage(stream_options: Any) -> bool:
    """Whether the client asked for a terminal usage chunk (``stream_options.include_usage``)."""
    return bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))


def sanitize_streaming_chat_body(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Map a streaming-dialect chat body onto the strict non-streaming params shape.

    Returns the cleaned body dict (ready for ``NeMoGymChatCompletionCreateParamsNonStreaming``
    validation) and whether a terminal usage chunk was requested via
    ``stream_options.include_usage``.

    ``stream`` and ``stream_options`` are removed: the params model's ``stream`` field is typed
    ``Literal[False]``, and ``stream_options`` is only meaningful with ``stream: true``, so it has
    no effect on the non-streaming backend call. Remaining fields are filtered to the known params
    fields, so a harness's extra bookkeeping never reaches the backend.
    """
    body = deepcopy(body)
    include_usage = _wants_usage(body.get("stream_options"))
    body.pop("stream", None)
    body.pop("stream_options", None)

    dropped = sorted(set(body) - _PARAM_FIELDS)
    if dropped:
        LOG.debug("Dropping unsupported fields from a streaming /v1/chat/completions request: %s", dropped)
    return {key: value for key, value in body.items() if key in _PARAM_FIELDS}, include_usage


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse_error(exc: Exception) -> str:
    payload = {"error": {"message": str(exc), "type": type(exc).__name__}}
    return f"event: error\ndata: {json.dumps(payload)}\n\n"


def _chunk(completion: dict[str, Any], choices: list[dict[str, Any]], usage: Any = None) -> dict[str, Any]:
    """Build one ``chat.completion.chunk`` object sharing the completion's id/created/model."""
    chunk: dict[str, Any] = {
        "id": completion.get("id"),
        "object": "chat.completion.chunk",
        "created": completion.get("created"),
        "model": completion.get("model"),
        "choices": choices,
    }
    system_fingerprint = completion.get("system_fingerprint")
    if system_fingerprint is not None:
        chunk["system_fingerprint"] = system_fingerprint
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _choice_deltas(index: int, choice: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the ``chat.completion.chunk`` choice deltas for one completed choice.

    A role delta opens the choice; reasoning, content, and tool-call deltas follow (each emitted
    only when present); a terminal delta carries the ``finish_reason``. Splitting the message this
    way keeps every field a client tracks (role, reasoning, content, tool-call name/arguments,
    finish reason) in the delta position that client expects, even though it is a single logical
    chunk sequence rather than incremental tokens.
    """
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    yield {"index": index, "delta": {"role": message.get("role") or "assistant"}, "finish_reason": None}

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        yield {"index": index, "delta": {"reasoning_content": reasoning}, "finish_reason": None}

    content = message.get("content")
    if content:
        yield {"index": index, "delta": {"content": content}, "finish_reason": None}

    tool_calls = message.get("tool_calls")
    if tool_calls:
        delta_tool_calls = []
        for tool_index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            delta_tool_calls.append(
                {
                    "index": tool_index,
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type") or "function",
                    "function": {"name": function.get("name"), "arguments": function.get("arguments") or ""},
                }
            )
        yield {"index": index, "delta": {"tool_calls": delta_tool_calls}, "finish_reason": None}

    yield {"index": index, "delta": {}, "finish_reason": finish_reason}


def synthesize_chat_completion_sse(completion: dict[str, Any], include_usage: bool = False) -> Iterator[str]:
    """Re-emit a complete Chat Completion object as a ``chat.completion.chunk`` SSE stream.

    Emits, per choice, a role chunk -> optional reasoning/content/tool-call chunks -> a terminal
    chunk carrying ``finish_reason``. When ``include_usage`` is set and the completion reports
    usage, a final ``choices: []`` chunk carries the usage block (OpenAI's contract). The stream
    always ends with the ``data: [DONE]`` sentinel streaming clients treat as terminal.
    """
    for index, choice in enumerate(completion.get("choices") or []):
        for delta in _choice_deltas(index, choice):
            yield _sse_data(_chunk(completion, [delta]))

    usage = completion.get("usage")
    if include_usage and usage is not None:
        yield _sse_data(_chunk(completion, [], usage=usage))

    yield "data: [DONE]\n\n"


async def synthesize_chat_completion_sse_with_heartbeat(
    pending: Awaitable[dict[str, Any]],
    include_usage: bool = False,
    interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
) -> AsyncIterator[str]:
    """Emit SSE heartbeats until ``pending`` resolves, then the synthesized completion stream.

    The backend call is non-streaming, so nothing can be emitted until it returns. On a slow
    model that silence is indistinguishable from a hung endpoint: the OpenClaw agent aborts the
    session after 120s of quiet ("LLM idle timeout"), and the rollout is graded as an empty
    failure even though the model is still working. Heartbeating keeps the socket alive for the
    whole generation without changing what the client ultimately parses.

    A failure after the first heartbeat cannot be reported as an HTTP status (the response has
    already begun), so it is logged and emitted as a terminal SSE error event. Callers should
    therefore await a short grace window first, so fast failures still surface with their normal
    status code.
    """
    while True:
        try:
            completion = await asyncio.wait_for(asyncio.shield(pending), timeout=interval_s)
            break
        except asyncio.TimeoutError:
            yield SSE_HEARTBEAT
        except Exception as exc:
            LOG.exception("Chat completion failed after the SSE stream had already started")
            yield _sse_error(exc)
            return

    for chunk in synthesize_chat_completion_sse(completion, include_usage=include_usage):
        yield chunk
