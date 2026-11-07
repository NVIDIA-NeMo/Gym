# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Code config rendering and stream-json/session-wire conversion to Responses items, usage and subagents."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.models import ChatCompletionsConverter


def _toml_str(value: Any) -> str:
    # JSON with raw Unicode is a valid TOML value. TOML rejects surrogate escapes like \ud83d and a literal DEL.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def to_toml(data: dict[str, Any], path: tuple[str, ...] = (), header: str | None = None) -> str:
    """Serialize nested dicts, arrays of tables, and scalars; JSON strings and arrays are valid TOML values."""
    lines = [header or f"[{'.'.join(path)}]"] if path else []
    tables = []
    for key, value in data.items():
        name = _toml_str(str(key))
        if value is None:
            continue
        if isinstance(value, dict):
            tables.append(to_toml(value, (*path, name)))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            array = ".".join((*path, name))
            tables.extend(to_toml(item, (*path, name), header=f"[[{array}]]") for item in value)
        else:
            if isinstance(value, list) and any(item is None for item in value):
                raise ValueError(f"TOML has no null value: {'.'.join((*path, str(key)))} holds None")
            lines.append(f"{name} = {_toml_str(value)}")
    return "\n".join(["\n".join(lines), *tables]).strip() + "\n"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _json_or_none(line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def parse_stream_json(stdout: str) -> list[Any]:
    """Convert the root agent's `--output-format stream-json` chat messages to Responses output items.

    Gym model servers return reasoning inside `<think>` tags in the content, which the converter splits off again.
    """
    converter = ChatCompletionsConverter(return_token_id_information=False)
    output = []
    for message in map(_json_or_none, stdout.splitlines()):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            output.extend(converter.postprocess_assistant_message_dict(message))
        elif message.get("role") == "tool":
            content = message.get("content")
            output.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=message.get("tool_call_id", ""),
                    output=content if isinstance(content, str) else json.dumps(content),
                )
            )
    return output


def parse_session(session_dir: Path) -> tuple[NeMoGymResponseUsage | None, list[dict[str, Any]]]:
    """Sum model usage over the root and every subagent, and list the subagents from the wire logs.

    Kimi records a call without usage as all zeros, which leaves the total unknown.
    """
    usages, subagents, missing = [], {}, False
    for wire in sorted(session_dir.glob("agents/*/wire.jsonl")):
        for event in map(_json_or_none, wire.read_text(errors="replace").splitlines()):
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "usage.record":
                usage = event.get("usage") or {}
                cached = usage.get("inputCacheRead", 0)
                input_tokens = usage.get("inputOther", 0) + cached + usage.get("inputCacheCreation", 0)
                missing |= input_tokens == 0
                usages.append(
                    NeMoGymResponseUsage(
                        input_tokens=input_tokens,
                        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached),
                        output_tokens=usage.get("output", 0),
                        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                        total_tokens=input_tokens + usage.get("output", 0),
                    )
                )
            elif kind == "subagent.spawned":
                subagents[event["subagentId"]] = {
                    "id": event["subagentId"],
                    "parent": event.get("parentAgentId"),
                    "tool_call_id": event.get("parentToolCallId"),
                    "type": event.get("subagentName"),
                    "model": event.get("model"),
                    "swarm_index": event.get("swarmIndex"),
                    "background": event.get("runInBackground", False),
                    "status": "incomplete",
                }
            elif kind in ("subagent.started", "subagent.completed", "subagent.failed", "subagent.cancelled") and (
                event.get("subagentId") in subagents
            ):
                # Other subagent.* events (progress and the like) do not end a subagent.
                child = subagents[event["subagentId"]]
                if kind == "subagent.started":
                    child["started"] = event.get("time")
                else:
                    child["ended"] = event.get("time")
                    child["status"] = {"subagent.completed": "completed", "subagent.failed": "failed"}.get(
                        kind, "incomplete"
                    )
    usage = NeMoGymResponseUsage.sum_from_list(usages) if usages and not missing else None
    return usage, list(subagents.values())


def max_concurrency(subagents: list[dict[str, Any]]) -> int:
    edges = sorted(
        (t, step)
        for child in subagents
        if child.get("started") is not None
        for t, step in ((child["started"], 1), (child.get("ended", float("inf")), -1))
    )
    running = peak = 0
    for _, step in edges:
        running += step
        peak = max(peak, running)
    return peak


def stream_session_id(stdout: str) -> str | None:
    """The session id Kimi prints in print mode; agents can start other kimi processes, so only this one counts."""
    for message in map(_json_or_none, stdout.splitlines()):
        if isinstance(message, dict) and message.get("session_id"):
            return message["session_id"]
    return None
