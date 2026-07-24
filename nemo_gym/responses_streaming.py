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
"""Streaming Responses-dialect support shared by every Gym model server.

Blackbox harnesses that speak the OpenAI Responses API over SSE (e.g. the Codex CLI) send
requests the strict ``NeMoGymResponseCreateParamsNonStreaming`` model would reject: a
``stream: true`` flag, client-bookkeeping fields (``client_metadata``, ``prompt_cache_key``),
and ``namespace`` tool specs — functions grouped under a namespace that the model calls back
with separate ``namespace`` + ``name`` fields on the ``function_call`` item. Backends behind a
Gym model server (chat-completions endpoints, vLLM) only understand flat function tools, so
this module provides:

- the request-side sanitizer that flattens namespace tools into plain functions (joined as
  ``<namespace>__<name>``), rewrites replayed namespaced calls in the input history the same
  way, and drops the fields the params model does not know;
- the response-side synthesizer that re-emits a complete ``NeMoGymResponse`` as the minimal
  Responses SSE event sequence streaming clients require (``response.created`` ->
  ``response.output_item.done`` per output item -> ``response.completed``), splitting flattened
  function-call names back into ``namespace`` + ``name`` on the way out.
"""

import json
import logging
from copy import deepcopy
from typing import Any, Iterator, Optional
from uuid import uuid4

from openai.types.responses.response_create_params import ToolParam
from pydantic import TypeAdapter, ValidationError

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseInputItem


LOG = logging.getLogger(__name__)

NAMESPACE_TOOL_DELIMITER = "__"

_PARAM_FIELDS = frozenset(NeMoGymResponseCreateParamsNonStreaming.model_fields)
_TOOL_ADAPTER = TypeAdapter(ToolParam)
_INPUT_ITEM_ADAPTER = TypeAdapter(NeMoGymResponseInputItem)

# ns_map: flattened function name -> (namespace, name) for the round trip back to the client.
NamespaceMap = dict[str, tuple[str, str]]


def flatten_namespace_tools(tools: Any) -> tuple[list[Any], NamespaceMap]:
    """Flatten ``namespace`` tool specs into plain function tools with joined names."""
    flat: list[Any] = []
    ns_map: NamespaceMap = {}
    for tool in tools or []:
        if not (isinstance(tool, dict) and tool.get("type") == "namespace"):
            flat.append(tool)
            continue
        namespace = str(tool.get("name") or "")
        for sub in tool.get("tools") or []:
            if not isinstance(sub, dict) or not sub.get("name"):
                continue
            joined = f"{namespace}{NAMESPACE_TOOL_DELIMITER}{sub['name']}"
            flat.append({**sub, "type": "function", "name": joined})
            ns_map[joined] = (namespace, str(sub["name"]))
    return flat, ns_map


def _tool_valid(tool: Any) -> bool:
    """Whether a tool spec validates against the params model's tool union; failures are logged."""
    try:
        _TOOL_ADAPTER.validate_python(tool)
        return True
    except ValidationError:
        LOG.warning(
            "Dropping unsupported tool spec of type %r from a streaming /v1/responses request.",
            tool.get("type") if isinstance(tool, dict) else type(tool).__name__,
        )
        return False


def _is_system_like_message(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") in (None, "message")
        and item.get("role") in ("system", "developer")
    )


def _input_message_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
            parts.append(part.get("text") or "")
    return "".join(parts)


def sanitize_streaming_responses_body(body: dict[str, Any]) -> tuple[dict[str, Any], NamespaceMap]:
    """Map a streaming-dialect request body onto the strict non-streaming params shape.

    Returns the cleaned body dict (ready for ``NeMoGymResponseCreateParamsNonStreaming``
    validation) and the namespace map needed to restore namespaced call names in the
    synthesized SSE response. Tool entries that still fail per-entry validation after
    flattening are dropped with a warning rather than failing the whole request, since a
    harness's exotic tool is better lost than the rollout.
    """
    body = deepcopy(body)
    # The params model only admits `stream: false` (Gym responses are non-streaming internally);
    # the streaming envelope is synthesized by the caller, so the flag is dropped here.
    body.pop("stream", None)

    ns_map: NamespaceMap = {}
    if "tools" in body:
        tools, ns_map = flatten_namespace_tools(body.get("tools"))
        body["tools"] = [tool for tool in tools if _tool_valid(tool)]

    # Replayed history: a namespaced call the client echoes back must match the flattened tool
    # name the model actually saw, so the conversation stays self-consistent for the backend.
    input_items = body.get("input")
    if isinstance(input_items, list):
        kept_items = []
        carrier_tools: list[Any] = []
        for item in input_items:
            if isinstance(item, dict) and item.get("type") == "function_call" and item.get("namespace"):
                item["name"] = f"{item.pop('namespace')}{NAMESPACE_TOOL_DELIMITER}{item.get('name')}"
            # Codex's code mode ships tools inside an `additional_tools` input item instead of the
            # `tools` param. Hoist what a Gym backend can express (plain and namespaced function
            # tools) into `tools`; the rest (e.g. the freeform JS `exec` tool) has no function-call
            # representation and is dropped with a warning.
            if isinstance(item, dict) and item.get("type") == "additional_tools":
                carrier_tools.extend(item.get("tools") or [])
                continue
            # Streaming harnesses interleave item kinds the Gym input union has no representation
            # for. Drop those individually -- a lost carrier item is recoverable, a 422 on the
            # whole request kills the rollout.
            try:
                _INPUT_ITEM_ADAPTER.validate_python(item)
            except ValidationError:
                LOG.warning(
                    "Dropping unsupported input item of type %r from a streaming /v1/responses request.",
                    item.get("type") if isinstance(item, dict) else type(item).__name__,
                )
                continue
            kept_items.append(item)

        if carrier_tools:
            flat, carrier_map = flatten_namespace_tools(carrier_tools)
            ns_map.update(carrier_map)
            usable = [t for t in flat if isinstance(t, dict) and t.get("type") == "function" and _tool_valid(t)]
            skipped = [t.get("name") for t in flat if not (isinstance(t, dict) and t.get("type") == "function")]
            if skipped:
                LOG.warning(
                    "Dropping non-function tools %s from an additional_tools item; a Gym backend can only "
                    "express function calls. Configure the harness with an explicit unknown model name so "
                    "it advertises classic function tools instead.",
                    skipped,
                )
            if usable:
                body["tools"] = [*(body.get("tools") or []), *usable]

        # Hoist the leading run of system/developer messages into `instructions` (prepended by any
        # instructions already present). Streaming harnesses may open with several developer
        # messages and no instructions (Codex does, for newer model families); downstream
        # Responses -> Chat conversion renders `instructions` as the single leading system message
        # strict chat backends require.
        leading_parts = [body.get("instructions") or ""]
        while kept_items and _is_system_like_message(kept_items[0]):
            leading_parts.append(_input_message_text(kept_items.pop(0)))
        hoisted = "\n\n".join(part for part in leading_parts if part)
        if hoisted:
            body["instructions"] = hoisted

        body["input"] = kept_items

    dropped = sorted(set(body) - _PARAM_FIELDS)
    if dropped:
        LOG.debug("Dropping unsupported fields from a streaming /v1/responses request: %s", dropped)
    return {key: value for key, value in body.items() if key in _PARAM_FIELDS}, ns_map


def _delete_loc(body: Any, loc: tuple) -> bool:
    """Delete the value at a pydantic error loc from a nested dict/list structure.

    Returns False when the loc cannot be walked literally (e.g. it contains a union-arm label
    rather than a real key), in which case nothing is deleted.
    """
    node = body
    for part in loc[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and part < len(node):
            node = node[part]
        else:
            return False
    last = loc[-1]
    if isinstance(node, dict) and last in node:
        del node[last]
        return True
    return False


def validate_streaming_responses_params(body: dict[str, Any]) -> NeMoGymResponseCreateParamsNonStreaming:
    """Validate a sanitized streaming-dialect body, pruning fields newer than the params model.

    Harness wire formats evolve faster than the pinned OpenAI SDK types (e.g. Codex sending
    ``reasoning.context``, which the SDK's ``Reasoning`` model forbids). Any field pydantic flags
    as ``extra_forbidden`` is removed and validation retried, so only errors that cannot be fixed
    by dropping an unknown field surface to the client.
    """
    body = deepcopy(body)
    while True:
        try:
            return NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
        except ValidationError as exc:
            removed = False
            for error in exc.errors():
                if error["type"] == "extra_forbidden" and _delete_loc(body, tuple(error["loc"])):
                    LOG.warning(
                        "Dropping unsupported field %s from a streaming /v1/responses request.",
                        ".".join(str(part) for part in error["loc"]),
                    )
                    removed = True
            if not removed:
                raise


def _sse_event(payload: dict[str, Any]) -> str:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"


def synthesize_responses_sse(response_json: dict[str, Any], ns_map: Optional[NamespaceMap] = None) -> Iterator[str]:
    """Re-emit a complete Responses API response object as an SSE event stream.

    Streaming clients build their view of the turn from ``response.output_item.done`` events and
    treat ``response.completed`` (which carries the response id and usage) as the terminal event,
    so those two are the required minimum; ``response.created`` is included for clients that wait
    for an acknowledgement before reading items.
    """
    output_items = []
    for item in response_json.get("output") or []:
        if ns_map and isinstance(item, dict) and item.get("type") == "function_call" and item.get("name") in ns_map:
            namespace, name = ns_map[item["name"]]
            item = {**item, "namespace": namespace, "name": name}
        output_items.append(item)

    yield _sse_event(
        {"type": "response.created", "response": {**response_json, "status": "in_progress", "output": []}}
    )
    for index, item in enumerate(output_items):
        yield _sse_event({"type": "response.output_item.done", "output_index": index, "item": item})
    yield _sse_event({"type": "response.completed", "response": {**response_json, "output": output_items}})


def synthesize_responses_failure_sse(message: str, *, code: str = "server_error") -> Iterator[str]:
    """Emit a terminal Responses SSE stream for a backend failure.

    Once the streaming contract is committed (HTTP 200 + ``text/event-stream``), a ``responses()``
    error can no longer surface as an HTTP 500. Streaming clients (e.g. Codex) expect a terminal
    ``response.failed`` event; without one they see a truncated stream and cannot tell an
    application error from a transport failure. Emitting ``response.failed`` lets the client report
    a clean turn failure, and lets model-call capture classify it as an upstream error (its
    terminal-SSE table maps ``response.failed`` to an error) rather than a stream truncation.
    """
    response = {
        "id": f"resp_{uuid4().hex}",
        "object": "response",
        "status": "failed",
        "output": [],
        "error": {"code": code, "message": message},
    }
    yield _sse_event({"type": "response.created", "response": {**response, "status": "in_progress"}})
    yield _sse_event({"type": "response.failed", "response": response})
