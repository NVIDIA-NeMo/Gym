# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent trajectory parsers for semantic timeline observability."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from nemo_gym.observability.events import command_attributes, stable_hash


_RETURN_CODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")
_MIN_DURATION_S = 0.001


def ingest_agent_trajectory_events(
    agent_dir: Path,
    *,
    recorder: Any,
    trajectory_id: str,
) -> int:
    """Record semantic tool spans from agent artifact trajectory JSON files.

    Idempotent — a marker file prevents double-ingestion.

    Args:
        agent_dir: Agent artifact directory (e.g. Harbor agent output).
        recorder: Active ``SandboxEventRecorder``.
        trajectory_id: Stable trajectory/trial identifier.

    Returns:
        Number of spans recorded.
    """
    marker_path = _ingest_marker_path(
        recorder.output_dir,
        agent_dir=agent_dir,
        trajectory_id=trajectory_id,
    )
    if marker_path.exists():
        return 0
    for path in _trajectory_candidates(agent_dir):
        spans = extract_agent_tool_spans(
            path,
            trajectory_id=trajectory_id,
            include_command_text=bool(recorder.include_command_text),
        )
        if not spans:
            continue
        for span in spans:
            recorder.record_event(
                "span_end",
                "trajectory.tool",
                attributes=span["attributes"],
                timestamp_unix_s=span["timestamp_unix_s"],
                monotonic_s=None,
            )
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(str(len(spans)), encoding="utf-8")
        return len(spans)
    return 0


def _ingest_marker_path(
    output_dir: Path,
    *,
    agent_dir: Path,
    trajectory_id: str,
) -> Path:
    marker_id = stable_hash(f"{trajectory_id}:{agent_dir}")
    return output_dir / ".agent_trajectory_ingested" / marker_id


def extract_agent_tool_spans(
    path: Path,
    *,
    trajectory_id: str,
    include_command_text: bool,
) -> list[dict[str, Any]]:
    """Extract semantic tool spans from a known agent trajectory file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []

    messages = data.get("messages")
    if isinstance(messages, list):
        return _spans_from_response_messages(
            messages,
            trajectory_id=trajectory_id,
            source_file=path.name,
            include_command_text=include_command_text,
        )

    steps = data.get("steps")
    if isinstance(steps, list):
        return _spans_from_atif_steps(
            steps,
            trajectory_id=trajectory_id,
            source_file=path.name,
            include_command_text=include_command_text,
        )
    return []


def _trajectory_candidates(agent_dir: Path) -> list[Path]:
    preferred = [
        agent_dir / "mini-swe-agent.trajectory.json",
        agent_dir / "openhands.trajectory.json",
        agent_dir / "trajectory.json",
    ]
    seen = set(preferred)
    for path in sorted(agent_dir.glob("*.trajectory.json")):
        if path not in seen:
            preferred.append(path)
            seen.add(path)
    return preferred


def _spans_from_response_messages(
    messages: list[Any],
    *,
    trajectory_id: str,
    source_file: str,
    include_command_text: bool,
) -> list[dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    spans: list[dict[str, Any]] = []
    tool_index = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        response_end_timestamp = _message_timestamp(message)
        for call in _response_tool_calls(message):
            call_id = call.get("call_id")
            if not call_id:
                continue
            pending[str(call_id)] = {
                **call,
                "start_timestamp": response_end_timestamp,
                "tool_index": tool_index,
            }
            tool_index += 1

        if message.get("type") != "function_call_output":
            continue
        call_id = str(message.get("call_id") or "")
        if not call_id or call_id not in pending:
            continue
        call = pending.pop(call_id)
        start_timestamp = call.get("start_timestamp")
        end_timestamp = _message_timestamp(message)
        if not isinstance(start_timestamp, (int, float)):
            start_timestamp = end_timestamp
        if not isinstance(end_timestamp, (int, float)):
            end_timestamp = start_timestamp
        spans.append(
            _tool_span(
                trajectory_id=trajectory_id,
                source_file=source_file,
                command=call.get("command"),
                tool_name=call.get("tool_name"),
                tool_call_id=call_id,
                tool_index=int(call["tool_index"]),
                start_timestamp=float(start_timestamp),
                end_timestamp=float(end_timestamp),
                return_code=_output_return_code(message),
                include_command_text=include_command_text,
            )
        )
    return spans


def _spans_from_atif_steps(
    steps: list[Any],
    *,
    trajectory_id: str,
    source_file: str,
    include_command_text: bool,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    timestamp_by_index = [
        _timestamp_value(step.get("timestamp")) if isinstance(step, dict) else None for step in steps
    ]
    tool_index = 0
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        start_timestamp = timestamp_by_index[index]
        if start_timestamp is None:
            continue
        end_timestamp = _next_timestamp(timestamp_by_index, index, start_timestamp)
        return_code = _atif_return_code(step)
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            command = _command_from_arguments(call.get("arguments"))
            tool_call_id = str(call.get("tool_call_id") or call.get("id") or tool_index)
            spans.append(
                _tool_span(
                    trajectory_id=trajectory_id,
                    source_file=source_file,
                    command=command,
                    tool_name=str(call.get("function_name") or call.get("name") or "tool"),
                    tool_call_id=tool_call_id,
                    tool_index=tool_index,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    return_code=return_code,
                    include_command_text=include_command_text,
                )
            )
            tool_index += 1
    return spans


def _response_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in message.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = str(item.get("call_id") or item.get("id") or "")
        if not call_id or call_id in seen:
            continue
        calls.append(
            {
                "call_id": call_id,
                "tool_name": str(item.get("name") or "tool"),
                "command": _command_from_arguments(item.get("arguments")),
            }
        )
        seen.add(call_id)

    extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
    for action in extra.get("actions") or []:
        if not isinstance(action, dict):
            continue
        call_id = str(action.get("tool_call_id") or action.get("call_id") or "")
        if not call_id or call_id in seen:
            continue
        calls.append(
            {
                "call_id": call_id,
                "tool_name": str(action.get("tool_name") or "bash"),
                "command": action.get("command") if isinstance(action.get("command"), str) else None,
            }
        )
        seen.add(call_id)
    return calls


def _tool_span(
    *,
    trajectory_id: str,
    source_file: str,
    command: str | None,
    tool_name: str | None,
    tool_call_id: str,
    tool_index: int,
    start_timestamp: float,
    end_timestamp: float,
    return_code: int | None,
    include_command_text: bool,
) -> dict[str, Any]:
    end_timestamp = max(end_timestamp, start_timestamp + _MIN_DURATION_S)
    attrs: dict[str, Any] = {
        "phase": "agent_tool",
        "trajectory_id": trajectory_id,
        "source": "agent_trajectory",
        "source_file": source_file,
        "tool_name": tool_name or "tool",
        "tool_call_hash": stable_hash(tool_call_id),
        "tool_index": tool_index,
        "duration_s": end_timestamp - start_timestamp,
        **command_attributes(command, include_command_text=include_command_text),
    }
    if return_code is not None:
        attrs["return_code"] = return_code
        attrs["status"] = "ok" if return_code == 0 else "error"
    else:
        attrs["status"] = "ok"
    return {
        "timestamp_unix_s": end_timestamp,
        "attributes": attrs,
    }


def _message_timestamp(message: dict[str, Any]) -> float | None:
    extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
    timestamp = _timestamp_value(extra.get("timestamp"))
    if timestamp is not None:
        return timestamp
    return _timestamp_value(message.get("created_at"))


def _timestamp_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _next_timestamp(
    timestamps: list[float | None],
    index: int,
    fallback: float,
) -> float:
    for later in timestamps[index + 1 :]:
        if later is not None and later > fallback:
            return later
    return fallback + _MIN_DURATION_S


def _command_from_arguments(arguments: Any) -> str | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    command = arguments.get("command")
    return command if isinstance(command, str) else None


def _output_return_code(message: dict[str, Any]) -> int | None:
    extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
    return_code = extra.get("returncode")
    if isinstance(return_code, int):
        return return_code
    output = message.get("output")
    if isinstance(output, str):
        return _return_code_from_text(output)
    return None


def _atif_return_code(step: dict[str, Any]) -> int | None:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return None
    for result in observation.get("results") or []:
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        if isinstance(content, str):
            return_code = _return_code_from_text(content)
            if return_code is not None:
                return return_code
    return None


def _return_code_from_text(text: str) -> int | None:
    match = _RETURN_CODE_RE.search(text)
    return int(match.group(1)) if match else None
