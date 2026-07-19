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
"""Validation and reconstruction of Switchyard token-capture sessions.

Switchyard's `GET /v1/sessions/{session_id}/completions` returns one lossless
record per policy call: the OpenAI chat messages it proxied (prompt plus the
generated assistant turn) and the exact vLLM token triple
(`prompt_token_ids`, `generation_token_ids`, `generation_log_probs`).

This module validates a retrieved session fail-closed and reconstructs it into
one ordinary Gym rollout: Responses API input/output items where each captured
assistant generation carries its token triple on the final generated item.
Any schema or history divergence raises `SwitchyardTraceError` — the caller
masks the sample rather than emitting partially annotated training data.
"""

import json
import math
from typing import Any, Dict, List, NamedTuple

from openai.types.responses.function_tool import FunctionTool

from nemo_gym.responses_converter import ResponsesConverter, split_responses_input_output_items


SWITCHYARD_SCHEMA_VERSION = 1

# Message roles OpenHands can legitimately send through Switchyard.
_KNOWN_ROLES = {"system", "developer", "user", "assistant", "tool"}
# Roles allowed in the between-call suffix a later prompt appends: environment
# feedback only. An assistant message here means a policy call was not captured.
_ENV_ROLES = {"user", "tool"}


class SwitchyardTraceError(Exception):
    """A Switchyard session failed validation or reconstruction."""


class SwitchyardTrace(NamedTuple):
    input_items: List[Any]
    output_items: List[Any]
    tools: List[FunctionTool]
    record_uuids: List[str]
    model: str


def map_switchyard_tools(raw_tools: Any) -> List[FunctionTool]:
    """Map Switchyard trace tools `{id, description, inputSchema.jsonSchema}` to Gym function tools."""
    if not isinstance(raw_tools, list):
        raise SwitchyardTraceError(f"tools is not a list: {type(raw_tools).__name__}")

    tools = []
    for i, entry in enumerate(raw_tools):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise SwitchyardTraceError(f"tool {i} has no usable id")
        input_schema = entry.get("inputSchema")
        parameters = input_schema.get("jsonSchema") if isinstance(input_schema, dict) else None
        tools.append(
            FunctionTool(
                type="function",
                name=entry["id"],
                description=entry.get("description") or None,
                parameters=parameters,
                strict=None,
            )
        )
    return tools


def reconstruct_switchyard_rollout(envelope: Any, session_id: str, converter: ResponsesConverter) -> SwitchyardTrace:
    """Validate a retrieval envelope and rebuild one token-annotated Gym rollout.

    Records are processed in the endpoint's returned order; each later prompt
    must strictly extend the already assembled history with a non-empty
    environment-message suffix. Raises `SwitchyardTraceError` on any deviation.
    """
    records = _validate_envelope(envelope, session_id)

    first = records[0]
    model, tools, tool_choice = first["model"], first["tools"], first.get("tool_choice")

    # `assembled` is the canonical history used for strict-extension comparison —
    # later prompts never carry token triples, so it must stay triple-free.
    # `annotated` is the conversion list, where each captured assistant message
    # additionally carries its token triple.
    assembled: List[Dict[str, Any]] = []
    annotated: List[Dict[str, Any]] = []
    record_uuids: List[str] = []
    # Token-level counterpart of the message-history check: trainers concatenate
    # the per-call triples into one sequence and hard-assert that each call's
    # prompt tokens extend the previous prompt+generation. Validate it here so a
    # violation (e.g. a template that re-renders history differently) masks the
    # sample instead of crashing the training step.
    token_history: List[int] = []
    for i, record in enumerate(records):
        _validate_record(record, i, session_id, record_uuids)
        if record["prompt_token_ids"][: len(token_history)] != token_history:
            raise SwitchyardTraceError(f"record {i} prompt_token_ids do not extend the prior prompt+generation tokens")
        token_history = record["prompt_token_ids"] + record["generation_token_ids"]
        if record["model"] != model:
            raise SwitchyardTraceError(f"record {i} model {record['model']!r} != session model {model!r}")
        if record["tools"] != tools or record.get("tool_choice") != tool_choice:
            raise SwitchyardTraceError(f"record {i} tools/tool_choice differ from the session's first record")

        messages = [_normalize_message(m, i) for m in record["messages"]]
        prompt, assistant = messages[:-1], messages[-1]
        if assistant["role"] != "assistant":
            raise SwitchyardTraceError(f"record {i} messages do not end with an assistant message")

        if i == 0:
            new_context = prompt
        else:
            # Compare with tool-call arguments canonicalized so a client that
            # re-serializes JSON differently across turns (compact vs. pretty)
            # does not fail the extension check. The reconstructed items keep the
            # original arguments — this normalization is comparison-only.
            if _canonicalize_tool_args(prompt[: len(assembled)]) != _canonicalize_tool_args(assembled):
                raise SwitchyardTraceError(f"record {i} prompt does not extend the reconstructed history")
            new_context = prompt[len(assembled) :]
            if not new_context:
                raise SwitchyardTraceError(f"record {i} prompt adds no environment messages")
            bad_roles = {m["role"] for m in new_context} - _ENV_ROLES
            if bad_roles:
                raise SwitchyardTraceError(f"record {i} suffix contains non-environment roles: {sorted(bad_roles)}")
        assembled.extend(new_context)
        assembled.append(assistant)

        annotated.extend(new_context)
        # The converter moves the triple onto the final generated Responses item.
        annotated.append(
            assistant
            | {
                "prompt_token_ids": record["prompt_token_ids"],
                "generation_token_ids": record["generation_token_ids"],
                "generation_log_probs": record["generation_log_probs"],
            }
        )
        record_uuids.append(record["uuid"])

    items = converter.chat_completions_messages_to_responses_items(annotated)
    input_items, output_items = split_responses_input_output_items(items)

    return SwitchyardTrace(
        input_items=input_items,
        output_items=output_items,
        tools=map_switchyard_tools(tools),
        record_uuids=record_uuids,
        model=model,
    )


def _validate_envelope(envelope: Any, session_id: str) -> List[Dict[str, Any]]:
    if not isinstance(envelope, dict):
        raise SwitchyardTraceError(f"envelope is not an object: {type(envelope).__name__}")
    if envelope.get("schema_version") != SWITCHYARD_SCHEMA_VERSION:
        raise SwitchyardTraceError(f"unsupported envelope schema_version: {envelope.get('schema_version')!r}")
    if envelope.get("session_id") != session_id:
        raise SwitchyardTraceError(f"envelope session_id {envelope.get('session_id')!r} != requested {session_id!r}")
    completions = envelope.get("completions")
    if not isinstance(completions, list) or not completions:
        raise SwitchyardTraceError("envelope has no completions")
    return completions


def _validate_record(record: Any, i: int, session_id: str, seen_uuids: List[str]) -> None:
    if not isinstance(record, dict):
        raise SwitchyardTraceError(f"record {i} is not an object")
    if record.get("schema_version") != SWITCHYARD_SCHEMA_VERSION:
        raise SwitchyardTraceError(f"record {i} has unsupported schema_version: {record.get('schema_version')!r}")
    if record.get("session_id") != session_id:
        raise SwitchyardTraceError(f"record {i} session_id {record.get('session_id')!r} != requested {session_id!r}")
    record_uuid = record.get("uuid")
    if not isinstance(record_uuid, str) or not record_uuid:
        raise SwitchyardTraceError(f"record {i} has no uuid")
    if record_uuid in seen_uuids:
        raise SwitchyardTraceError(f"record {i} duplicates uuid {record_uuid}")
    if record.get("is_valid") is not True:
        raise SwitchyardTraceError(f"record {record_uuid} is_valid is not true")
    for field in ("request_id", "model"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise SwitchyardTraceError(f"record {record_uuid} has empty {field}")
    for field in ("prompt_token_ids", "generation_token_ids"):
        if not _is_token_id_list(record.get(field)):
            raise SwitchyardTraceError(f"record {record_uuid} {field} is not a non-empty list of ints")
    log_probs = record.get("generation_log_probs")
    if not isinstance(log_probs, list) or not all(_is_finite_number(v) for v in log_probs):
        raise SwitchyardTraceError(f"record {record_uuid} generation_log_probs is not a list of finite floats")
    if len(record["generation_token_ids"]) != len(log_probs):
        raise SwitchyardTraceError(f"record {record_uuid} generation token ids and log probs have different lengths")
    if not isinstance(record.get("messages"), list) or not record["messages"]:
        raise SwitchyardTraceError(f"record {record_uuid} has no messages")


def _is_token_id_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_int(v) for v in value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalize_message(message: Any, i: int) -> Dict[str, Any]:
    """Canonicalize one chat message for history comparison and conversion.

    OpenHands serializes content as either a string or a list of text parts,
    while Switchyard's captured assistant turn always carries string content —
    the same logical message can appear in both shapes across records. Reduce
    every message to `{role, content: str}` plus tool identity so strict
    history comparison and the Responses converter see one canonical shape.
    """
    if not isinstance(message, dict):
        raise SwitchyardTraceError(f"record {i} contains a non-object message")
    role = message.get("role")
    if role not in _KNOWN_ROLES:
        raise SwitchyardTraceError(f"record {i} contains a message with unsupported role {role!r}")

    normalized: Dict[str, Any] = {"role": role, "content": _content_to_text(message.get("content"), i)}
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls:
            normalized["tool_calls"] = _normalize_tool_calls(tool_calls, i)
    elif role == "tool":
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise SwitchyardTraceError(f"record {i} contains a tool message without tool_call_id")
        normalized["tool_call_id"] = tool_call_id
    return normalized


def _content_to_text(content: Any, i: int) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise SwitchyardTraceError(f"record {i} contains an unsupported content part")
            texts.append(part["text"])
        return "".join(texts)
    raise SwitchyardTraceError(f"record {i} contains unsupported content of type {type(content).__name__}")


def _canonicalize_tool_args(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy of *messages* with assistant tool-call ``arguments`` canonicalized to
    sorted-key JSON, used only for the history-extension equality check.

    A client may re-serialize the same tool-call JSON differently across turns
    (compact vs. pretty-printed); that whitespace difference must not fail the
    extension check. The reconstructed training items keep the original arguments
    — this normalization never reaches them.
    """
    out: List[Dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            out.append(message)
            continue
        canonical_calls = []
        for call in tool_calls:
            function = call["function"]
            arguments = function["arguments"]
            try:
                arguments = json.dumps(json.loads(arguments), sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                pass  # not valid JSON — compare as-is
            canonical_calls.append({**call, "function": {**function, "arguments": arguments}})
        out.append({**message, "tool_calls": canonical_calls})
    return out


def _normalize_tool_calls(tool_calls: Any, i: int) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        raise SwitchyardTraceError(f"record {i} contains non-list tool_calls")
    normalized = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if (
            not isinstance(function, dict)
            or not isinstance(call.get("id"), str)
            or not call["id"]
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("arguments"), str)
        ):
            raise SwitchyardTraceError(f"record {i} contains a malformed tool call")
        normalized.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": function["name"], "arguments": function["arguments"]},
            }
        )
    return normalized
