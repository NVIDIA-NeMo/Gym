# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen Code stream-json and session-export conversion to Responses items and subagent counts."""

import io
import json
import tarfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from nemo_gym.core import NeMoGymFunctionCallOutput, NeMoGymResponseFunctionToolCall
from nemo_gym.models import ChatCompletionsConverter


BRIDGE_TOOL = "tool_call"
_CONVERTER = ChatCompletionsConverter(return_token_id_information=False)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        kinds = {p.get("type") if isinstance(p, dict) else getattr(p, "type", None) for p in content}
        if kinds - {"input_text", "text", None}:
            raise HTTPException(422, "Qwen Code only supports text input")
        return "".join(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") for p in content)
    return "" if content is None else json.dumps(content)


def parse_stream(stdout: str) -> tuple[list[Any], dict[str, Any] | None]:
    """Convert root-session ``-o stream-json`` events into output items; also return the result event.

    Subagent events carry a ``parent_tool_use_id`` and are left to the exported session transcripts.
    """
    items: list[Any] = []
    result = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(event, dict) or event.get("parent_tool_use_id"):
            continue
        if event.get("type") == "result":
            result = event
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if event["type"] == "assistant" and kind in ("thinking", "text"):
                # Gym model servers inline reasoning as <think> tags; split it out of the graded text.
                text = f"<think>{block['thinking']}</think>" if kind == "thinking" else block.get("text", "")
                if text.strip():
                    items += _CONVERTER.postprocess_assistant_message_dict({"role": "assistant", "content": text})
            elif event["type"] == "assistant" and kind == "tool_use":
                name, arguments = block.get("name", ""), block.get("input") or {}
                # Deferred tools (MCP tools, agent, ...) are invoked through the tool_call bridge;
                # record the underlying call so verifiers see the real tool name and arguments.
                if name == BRIDGE_TOOL and isinstance(arguments, dict) and arguments.get("name"):
                    name, arguments = arguments["name"], arguments.get("arguments") or {}
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(arguments),
                        call_id=block.get("id", ""),
                        name=name,
                        id=block.get("id"),
                        status="completed",
                    )
                )
            elif event["type"] == "user" and kind == "tool_result":
                items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=block.get("tool_use_id", ""),
                        output=_text(block.get("content")),
                        status="incomplete" if block.get("is_error") else "completed",
                    )
                )
    return items, result


def _json_dict(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def summarize_subagents(sessions_archive: bytes) -> dict[str, int]:
    """Count child sessions from the exported ``projects/*/subagents/<session>/*.meta.json`` files."""
    keys = [
        "subagents",
        "subagents_background",
        "subagent_max_depth",
        "subagent_input_tokens",
        "subagent_output_tokens",
    ]
    stats = {f"qwen_{key}": 0 for key in keys}
    with tarfile.open(fileobj=io.BytesIO(sessions_archive)) as archive:
        for member in archive.getmembers():
            # Child transcripts are subagents/<session>/agent-<id>.jsonl plus agent-<id>.meta.json.
            name = Path(member.name).name
            if "/subagents/" not in member.name or not name.startswith("agent-") or not member.isfile():
                continue
            data = archive.extractfile(member).read().decode(errors="replace")
            if name.endswith(".meta.json"):
                meta = _json_dict(data)
                if meta is None:
                    # A truncated or corrupt meta file loses one child's counts, not the whole /run.
                    continue
                stats["qwen_subagents"] += 1
                stats["qwen_subagents_background"] += bool(meta.get("isBackgrounded"))
                # depth is 0 for a top-level subagent; report levels.
                stats["qwen_subagent_max_depth"] = max(stats["qwen_subagent_max_depth"], meta.get("depth", 0) + 1)
            elif name.endswith(".jsonl"):
                for line in data.splitlines():
                    # A child killed mid-write leaves a partial last line; skip unparseable lines.
                    usage = (_json_dict(line) or {}).get("usageMetadata") or {}
                    stats["qwen_subagent_input_tokens"] += usage.get("promptTokenCount", 0)
                    stats["qwen_subagent_output_tokens"] += usage.get("candidatesTokenCount", 0)
    return stats


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
