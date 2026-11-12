# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify-request metadata, judge and IF result models, and input checks for the text evidence modes."""

import json
from typing import Literal

from pydantic import Field, StrictBool

from resources_servers.partner_if.task_data import StrictModel, Text
from resources_servers.partner_if.task_data import VerifierMetadata as VerifierMetadata


class JudgeItem(StrictModel):
    """Binary verdict for one rubric item."""

    id: Text
    passed: StrictBool
    reason: Text
    evidence: Text


class IFResult(StrictModel):
    task_id: Text
    rubric_version: Text
    rubric_hash: Text
    status: Literal["scored", "provisional", "unscorable"]
    passed: int = Field(ge=0)
    total: int = Field(gt=0)
    items: list[JudgeItem]
    reason: str | None = None


def validate_text_input(params: dict) -> None:
    """Reject evidence we cannot assess; never silently discard tools or multimodal inputs."""
    if any(params.get(key) for key in ("tools", "previous_response_id", "conversation", "prompt")):
        raise ValueError("Only self-contained text-response tasks are supported")
    messages = params.get("input")
    if not isinstance(messages, list) or not messages:
        raise ValueError("An explicit nonempty message list is required")
    for message in messages:
        if message.get("type", "message") != "message" or message.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
        }:
            raise ValueError("Unsupported input evidence")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if (
            not isinstance(content, list)
            or not content
            or any(
                part.get("type") not in {"input_text", "output_text"} or not isinstance(part.get("text"), str)
                for part in content
            )
        ):
            raise ValueError("Only text message content is supported")


def validate_input(params: dict, *, evidence_mode: str) -> None:
    """Validate self-contained evidence; frozen function history grants no live tool access."""
    if evidence_mode == "text_response":
        validate_text_input(params)
        return
    if evidence_mode not in {"frozen_tool_history", "tool_call_response"}:
        raise ValueError("Unsupported evidence mode")
    if evidence_mode == "tool_call_response" and not params.get("tools"):
        raise ValueError("Tool-call response tasks require advertised function definitions")
    if any(params.get(key) for key in ("previous_response_id", "conversation", "prompt")):
        raise ValueError("Frozen history cannot depend on hidden context")
    names = set()
    for tool in params.get("tools") or []:
        name = tool.get("name")
        if tool.get("type") != "function" or not isinstance(name, str) or not name or name in names:
            raise ValueError("Frozen history requires unique, named function definitions only")
        names.add(name)
    history = params.get("input")
    if not isinstance(history, list) or not history:
        raise ValueError("An explicit nonempty frozen history is required")
    pending, seen = set(), set()
    for item in history:
        kind = item.get("type", "message")
        if kind == "message":
            if pending:
                raise ValueError("Resolve recorded function calls before the next message")
            validate_text_input({"input": [item]})
        elif kind == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in seen or item.get("name") not in names:
                raise ValueError("Invalid, duplicate or undeclared recorded function call")
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                raise ValueError("Recorded function arguments must be a JSON object string")
            try:
                parsed = json.loads(arguments)
            except ValueError as error:
                raise ValueError("Recorded function arguments must be valid JSON") from error
            if not isinstance(parsed, dict):
                raise ValueError("Recorded function arguments must be a JSON object")
            pending.add(call_id)
            seen.add(call_id)
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if call_id not in pending or not isinstance(item.get("output"), str):
                raise ValueError("Orphan, duplicate or nontext recorded function output")
            pending.remove(call_id)
        else:
            raise ValueError("Unsupported frozen input evidence")
    if pending:
        raise ValueError("Frozen history has unresolved function calls")
