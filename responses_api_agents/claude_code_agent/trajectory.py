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

"""Claude Code adapter for Gym's standardized trajectory (nemo_gym.trajectory).

Claude Code writes a complete session transcript (one JSON record per event, with
timestamps, request ids, per-model-call usage, and tool execution metadata) into
``$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-id>.jsonl``. Since the agent runs each
rollout with an ephemeral ``CLAUDE_CONFIG_DIR``, those artifacts are harvested before the
directory is removed and fed through this adapter, which parses them and drives the
generic :class:`~nemo_gym.trajectory.TrajectoryBuilder`. The stream-json stdout events
are the fallback source when no transcript is available (same message structure and
token usage, but no timestamps or request ids — that telemetry stays ``None``).

Provider-specific mapping notes:

- Tool timing: an observation's ``started_at`` is the timestamp of the assistant record
  that issued the ``tool_use`` (linked via ``sourceToolAssistantUUID`` when present),
  and ``completed_at`` is its ``tool_result`` record's timestamp — each result record
  has its own, so parallel tool calls keep independent timing.
- ``toolUseResult`` execution metadata: only short scalars are kept (payloads can embed
  entire file contents; the model-visible output is already on the native
  ``function_call_output`` item).
- Subagent (sidechain) records are skipped and counted under
  ``dropped_records["sidechain"]``.
- Compaction: transcript records flagged ``isCompactSummary`` and stream-json
  ``system/compact_boundary`` events become ``context_boundary`` steps.
"""

import json
from typing import Any, Optional

from nemo_gym.trajectory import Trajectory, TrajectoryBuilder


# toolUseResult payloads can embed entire file contents; only short scalars are kept as
# execution metadata so the trajectory stays bounded.
_EXTRA_MAX_STR_LEN = 256


def decode_jsonl(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


def _thinking_text(block: dict) -> str:
    return block.get("thinking") or block.get("text") or ""


def _curate_extra(tool_use_result: Any) -> Optional[dict[str, Any]]:
    """Keep only short scalar execution metadata (durations, exit codes, flags) from toolUseResult."""
    if not isinstance(tool_use_result, dict):
        return None
    extra = {}
    for key, value in tool_use_result.items():
        if isinstance(value, bool) or isinstance(value, (int, float)):
            extra[key] = value
        elif isinstance(value, str) and len(value) <= _EXTRA_MAX_STR_LEN:
            extra[key] = value
    return extra or None


def build_trajectory(stream_events: list[dict], transcript_records: list[dict]) -> Trajectory:
    """Build a standardized trajectory from Claude Code artifacts.

    Prefers the on-disk transcript (timestamps, request ids, tool execution metadata);
    falls back to the stream-json stdout events. Run-level totals (`num_agent_steps`,
    `duration_ms`, `total_cost_usd`, provider usage) always come from the stream-json
    `result` event when present, since the transcript does not carry them.
    """
    has_transcript_messages = any(r.get("type") in ("assistant", "user") for r in transcript_records)
    if has_transcript_messages:
        builder = TrajectoryBuilder(agent="claude_code_agent", source="transcript")
        _replay_records(builder, transcript_records)
    else:
        builder = TrajectoryBuilder(agent="claude_code_agent", source="stream_json")
        _replay_records(builder, stream_events)

    for event in stream_events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            builder.set_session_id(event.get("session_id"))
        if event.get("type") == "result":
            builder.set_run_totals(
                num_agent_steps=event.get("num_turns"),
                duration_ms=event.get("duration_ms"),
                total_cost_usd=event.get("total_cost_usd"),
                provider_usage=event.get("usage") if isinstance(event.get("usage"), dict) else None,
            )
    return builder.build()


def _replay_records(builder: TrajectoryBuilder, records: list[dict]) -> None:
    """Drive the builder from transcript records or stream-json events (a transcript
    record is a stream event plus timestamps/uuids/requestId/toolUseResult)."""
    # assistant record uuid -> timestamp, for sourceToolAssistantUUID-based start times.
    assistant_record_ts: dict[str, str] = {}

    for record in records:
        rtype = record.get("type")
        if record.get("isSidechain"):
            builder.count_dropped("sidechain")
            continue
        if rtype not in ("assistant", "user", "system"):
            continue
        builder.set_session_id(record.get("sessionId") or record.get("session_id"))
        timestamp = record.get("timestamp") if isinstance(record.get("timestamp"), str) else None

        if rtype == "system":
            # stream-json emits a compaction marker as a system event.
            if record.get("subtype") == "compact_boundary":
                builder.add_context_boundary(timestamp=timestamp)
            continue

        message = record.get("message") or {}
        content = message.get("content")

        if rtype == "assistant":
            if isinstance(record.get("uuid"), str) and timestamp:
                assistant_record_ts[record["uuid"]] = timestamp
            usage = message.get("usage") or {}
            # The transcript writes one record per content block of the same API message
            # (identical message id and usage); start_agent_step dedupes on response_id.
            step = builder.start_agent_step(
                response_id=message.get("id"),
                request_id=record.get("requestId"),
                model=message.get("model"),
                timestamp=timestamp,
                stop_reason=message.get("stop_reason"),
                provider_usage=usage or None,
            )
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    builder.add_output_text(block["text"])
                elif block.get("type") in ("thinking", "reasoning") and _thinking_text(block):
                    builder.add_reasoning(_thinking_text(block))
                elif block.get("type") == "tool_use":
                    arguments = block.get("input")
                    builder.add_tool_call(
                        call_id=str(block.get("id") or f"call-{step.step_id}-{len(step.items)}"),
                        name=str(block.get("name") or ""),
                        arguments=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                    )

        elif rtype == "user":
            if record.get("isMeta"):
                continue
            if record.get("isCompactSummary"):
                builder.add_context_boundary(summary=_block_text(content), timestamp=timestamp)
                continue
            blocks = content if isinstance(content, list) else []
            tool_results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
            for block in tool_results:
                source_uuid = record.get("sourceToolAssistantUUID")
                builder.add_tool_result(
                    call_id=str(block.get("tool_use_id") or ""),
                    output=_block_text(block.get("content")),
                    completed_at=timestamp,
                    started_at=assistant_record_ts.get(source_uuid) if isinstance(source_uuid, str) else None,
                    error="tool_result flagged is_error" if block.get("is_error") else None,
                    extra=_curate_extra(record.get("toolUseResult")),
                )
            if not tool_results:
                text = _block_text(content)
                if text:
                    builder.add_user_message(text, timestamp=timestamp)
