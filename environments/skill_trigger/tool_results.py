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
"""Extract scoring candidates from canonical Pi, Codex, and Hermes records.

Completion times come from the host collector, keyed by session-local call ID.
Missing timing evidence is an error; this module never guesses a timestamp.
Assistant statements and Codex display-only output are not scoring candidates.
"""

import json
from typing import Literal

from environments.skill_trigger.scoring import EvidenceError, ToolResult


def mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def arguments(value: object) -> dict[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return mapping(value)


def extract_tool_results(
    events: list[dict[str, object]],
    *,
    harness: Literal["pi", "codex", "hermes"],
    session_id: str,
    completed_at: dict[str, float],
    skill_paths: dict[str, str],
) -> list[ToolResult]:
    """Return only successful candidate read outputs with attributable call IDs.

    For a child session, the adapter passes its own records and timestamps and
    separately supplies copied-call IDs to the scorer; inherited history cannot
    establish a new read. This parser alone does not validate session lineage.
    """
    outputs = []
    calls = {}

    def append(call_id: object, tool: str, text: str, resolved: str | None = None) -> None:
        if not isinstance(call_id, str) or not call_id or call_id not in completed_at:
            raise EvidenceError("Successful tool output lacks a call ID or host completion timestamp")
        outputs.append(
            ToolResult(
                session_id=session_id,
                call_id=call_id,
                tool=tool,
                text=text,
                success=True,
                at=completed_at[call_id],
                resolved_skill=resolved,
            )
        )

    if harness == "pi":
        for event in events:
            call_id = event.get("toolCallId")
            if event.get("type") == "tool_execution_start":
                calls[call_id] = mapping(event.get("args"))
            if event.get("type") != "tool_execution_end" or event.get("isError"):
                continue
            tool = event.get("toolName")
            result = mapping(event.get("result"))
            exit_code = mapping(result.get("structuredContent")).get("exit_code")
            if tool not in {"read", "bash"} or exit_code not in {None, 0}:
                continue
            content = result.get("content")
            if not isinstance(content, list):
                continue
            text = "\n".join(str(mapping(c).get("text", "")) for c in content if mapping(c).get("type") == "text")
            path = calls.get(call_id, {}).get("path")
            resolved = (
                next((name for name, actual in skill_paths.items() if path == actual), None)
                if tool == "read"
                else None
            )
            append(call_id, tool, text, resolved)
        return outputs

    if harness == "codex":
        succeeded = set()
        for event in events:
            payload = mapping(event.get("payload"))
            if event.get("type") != "event_msg":
                continue
            if payload.get("type") == "exec_command_end" and payload.get("exit_code") == 0:
                succeeded.add(payload.get("call_id"))
            item = mapping(payload.get("item"))
            if (
                payload.get("type") == "item_completed"
                and item.get("type") == "CommandExecution"
                and item.get("exit_code") == 0
                and item.get("status") == "completed"
            ):
                succeeded.add(item.get("id"))
        for event in events:
            payload = mapping(event.get("payload"))
            if (
                event.get("type") == "response_item"
                and payload.get("type") == "function_call_output"
                and payload.get("call_id") in succeeded
                and isinstance(payload.get("output"), str)
            ):
                append(payload["call_id"], "terminal", payload["output"])
        return outputs

    if harness != "hermes":
        raise EvidenceError(f"Unsupported harness: {harness}")
    messages = []
    for event in events:
        embedded = event.get("messages")
        if isinstance(embedded, list):
            messages.extend(mapping(m) for m in embedded)
        else:
            messages.append(event)
    for event in messages:
        if event.get("role") == "assistant":
            tool_calls = event.get("tool_calls")
            for raw_call in tool_calls if isinstance(tool_calls, list) else []:
                call = mapping(raw_call)
                function = mapping(call.get("function"))
                name = function.get("name")
                args = arguments(function.get("arguments"))
                if name == "tool_call":
                    nested = args.get("calls", [args])
                    if isinstance(nested, list) and len(nested) == 1:
                        name = mapping(nested[0]).get("name")
                        args = arguments(mapping(nested[0]).get("arguments"))
                calls[call.get("id")] = {"name": name, "args": args}
        if event.get("kind") == "tool_start":
            calls[event.get("call_id")] = {"name": event.get("name"), "args": arguments(event.get("args"))}
        if event.get("role") != "tool":
            continue
        call_id = event.get("call_id") or event.get("tool_call_id")
        call = calls.get(call_id, {})
        name = event.get("tool_name") or event.get("name") or call.get("name")
        if name == "tool_call":
            name = call.get("name")
        value = event.get("api_content", event.get("content"))
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if not isinstance(value, dict) or value.get("error") or value.get("success") is False:
            continue
        if name not in {"terminal", "skill_view", "read_file"}:
            continue
        if name == "terminal" and value.get("exit_code") != 0:
            continue
        if name == "skill_view" and value.get("success") is not True:
            continue
        text = value.get("output") if name == "terminal" else value.get("content")
        if not isinstance(text, str):
            continue
        args = arguments(event.get("args", call.get("args", {})))
        resolved = None
        if (
            name == "skill_view"
            and args.get("name") in skill_paths
            and not args.get("file_path")
            and value.get("name") == args.get("name")
        ):
            resolved = args["name"]
        if name == "read_file":
            resolved = next((skill for skill, path in skill_paths.items() if args.get("path") == path), None)
        append(call_id, name, text, resolved)
    return outputs
