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

"""Standardized trajectory extraction for the Claude Code agent (issue #1867).

Claude Code writes a complete session transcript (one JSON record per event, with
timestamps, request ids, per-model-call usage, and tool execution metadata) into
``$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-id>.jsonl``. Since the agent runs each
rollout with an ephemeral ``CLAUDE_CONFIG_DIR``, those artifacts are harvested before the
directory is removed and normalized into the versioned schema below. The stream-json
stdout events are used as a fallback source when no transcript is available (they carry
the same message structure and token usage, but no timestamps or request ids).

Reconstruction semantics (delta / append-only representation):

- ``steps`` is an ordered sequence. Each step holds only the *new* content it introduced:
  a user message, or one complete agent turn (one model call) with its tool calls and
  their observations.
- The model-visible input of agent turn *N* is the concatenation of all steps before it,
  starting from the most recent ``context_boundary`` step (or the beginning if none). A
  ``context_boundary`` step is emitted on compaction; its ``content`` is the summary that
  replaced the prior history, so post-compaction context = boundary content + later steps.
- Tool observations attach to the agent turn that issued the calls (matched via
  ``source_call_id``), each with independent start/end timestamps, so parallel tool calls
  keep independent timing.

Mandatory fields are non-optional in the models; optional telemetry that a source cannot
provide is ``None`` (never silently fabricated).
"""

import json
import logging
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


LOG = logging.getLogger(__name__)

TRAJECTORY_SCHEMA_VERSION = "1.0"

# toolUseResult payloads can embed entire file contents; only short scalars are kept as
# execution metadata so the trajectory stays bounded (the model-visible output is already
# recorded on the observation's `content`).
_EXTRA_MAX_STR_LEN = 256


class ClaudeCodeToolCall(BaseModel):
    call_id: str
    name: str
    arguments: str


class ClaudeCodeToolObservation(BaseModel):
    source_call_id: str
    content: str
    status: Literal["completed", "error"]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    extra: Optional[dict[str, Any]] = None


class ClaudeCodeModelCallStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    # The Anthropic Messages API does not report reasoning tokens separately.
    reasoning_tokens: Optional[int] = None


class ClaudeCodeTrajectoryStep(BaseModel):
    step_id: int
    type: Literal["user_message", "agent_turn", "context_boundary"]
    timestamp: Optional[str] = None
    content: str = ""
    reasoning_content: Optional[str] = None
    # agent_turn only: 1-based index over model calls, plus per-call identity/stats.
    turn_no: Optional[int] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    message_id: Optional[str] = None
    stop_reason: Optional[str] = None
    stats: Optional[ClaudeCodeModelCallStats] = None
    tool_calls: list[ClaudeCodeToolCall] = Field(default_factory=list)
    observations: list[ClaudeCodeToolObservation] = Field(default_factory=list)


class ClaudeCodeTrajectory(BaseModel):
    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    agent: str = "claude_code_agent"
    source: Literal["transcript", "stream_json"]
    session_id: Optional[str] = None
    model: Optional[str] = None
    steps: list[ClaudeCodeTrajectoryStep] = Field(default_factory=list)
    stats: ClaudeCodeModelCallStats = Field(default_factory=ClaudeCodeModelCallStats)
    # Claude Code's own end-of-run report (`result` event), kept verbatim for cross-checks.
    num_turns: Optional[int] = None
    duration_ms: Optional[float] = None
    total_cost_usd: Optional[float] = None
    result_usage: Optional[dict[str, Any]] = None
    # Subagent (sidechain) records are out of scope for the schema; their presence is counted
    # so consumers can tell "no subagents ran" from "subagent events were dropped".
    sidechain_records_skipped: int = 0


def validate_trajectory(data: dict[str, Any]) -> ClaudeCodeTrajectory:
    """Validate a serialized trajectory against the versioned schema."""
    return ClaudeCodeTrajectory.model_validate(data)


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


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


def _thinking_text(content: list[Any]) -> str:
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
            parts.append(block.get("thinking") or block.get("text") or "")
    return "\n".join(p for p in parts if p)


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


def _stats_from_usage(usage: dict[str, Any]) -> ClaudeCodeModelCallStats:
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    cached = usage.get("cache_read_input_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    return ClaudeCodeModelCallStats(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cached_tokens=int(cached) if cached is not None else None,
        cache_creation_tokens=int(cache_creation) if cache_creation is not None else None,
    )


def build_trajectory(
    stream_events: list[dict],
    transcript_records: list[dict],
) -> ClaudeCodeTrajectory:
    """Build a standardized trajectory from Claude Code artifacts.

    Prefers the on-disk transcript (timestamps, request ids, tool execution metadata);
    falls back to the stream-json stdout events. Run-level totals (`num_turns`,
    `duration_ms`, `total_cost_usd`, `result_usage`) always come from the stream-json
    `result` event when present, since the transcript does not carry them.
    """
    has_transcript_messages = any(r.get("type") in ("assistant", "user") for r in transcript_records)
    if has_transcript_messages:
        trajectory = _build_steps(transcript_records, source="transcript")
    else:
        trajectory = _build_steps(stream_events, source="stream_json")

    for event in stream_events:
        if event.get("type") == "system" and event.get("subtype") == "init" and trajectory.session_id is None:
            trajectory.session_id = event.get("session_id")
        if event.get("type") == "result":
            if event.get("num_turns") is not None:
                trajectory.num_turns = int(event["num_turns"])
            if event.get("duration_ms") is not None:
                trajectory.duration_ms = float(event["duration_ms"])
            if event.get("total_cost_usd") is not None:
                trajectory.total_cost_usd = float(event["total_cost_usd"])
            if isinstance(event.get("usage"), dict):
                trajectory.result_usage = event["usage"]
    return trajectory


def _build_steps(records: list[dict], source: Literal["transcript", "stream_json"]) -> ClaudeCodeTrajectory:
    trajectory = ClaudeCodeTrajectory(source=source)
    steps = trajectory.steps
    # call_id -> (step index that issued the call, timestamp of the issuing record)
    pending_calls: dict[str, tuple[int, Optional[str]]] = {}
    # assistant record uuid -> timestamp, for sourceToolAssistantUUID-based start times
    assistant_record_ts: dict[str, str] = {}
    turn_count = 0

    for record in records:
        rtype = record.get("type")
        if record.get("isSidechain"):
            trajectory.sidechain_records_skipped += 1
            continue
        if rtype not in ("assistant", "user", "system"):
            continue
        if trajectory.session_id is None:
            trajectory.session_id = record.get("sessionId") or record.get("session_id")

        timestamp = record.get("timestamp") if isinstance(record.get("timestamp"), str) else None
        if rtype == "system":
            # stream-json emits a compaction marker as a system event.
            if record.get("subtype") == "compact_boundary":
                steps.append(
                    ClaudeCodeTrajectoryStep(step_id=len(steps), type="context_boundary", timestamp=timestamp)
                )
            continue

        message = record.get("message") or {}
        content = message.get("content")

        if rtype == "assistant":
            if isinstance(record.get("uuid"), str) and timestamp:
                assistant_record_ts[record["uuid"]] = timestamp
            blocks = content if isinstance(content, list) else []
            message_id = message.get("id")
            step = _current_agent_turn(steps, message_id)
            if step is None:
                turn_count += 1
                usage = message.get("usage") or {}
                step = ClaudeCodeTrajectoryStep(
                    step_id=len(steps),
                    type="agent_turn",
                    timestamp=timestamp,
                    turn_no=turn_count,
                    model=message.get("model"),
                    request_id=record.get("requestId"),
                    message_id=message_id,
                    stats=_stats_from_usage(usage) if usage else None,
                )
                steps.append(step)
                if trajectory.model is None:
                    trajectory.model = message.get("model")
            # The transcript writes one record per content block of the same API message
            # (identical message id and usage), so blocks accumulate but stats are set once.
            step.stop_reason = message.get("stop_reason") or step.stop_reason
            text = _block_text(blocks)
            if text:
                step.content = f"{step.content}{text}" if step.content else text
            thinking = _thinking_text(blocks)
            if thinking:
                step.reasoning_content = (
                    f"{step.reasoning_content}\n{thinking}" if step.reasoning_content else thinking
                )
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    call_id = str(block.get("id") or f"call-{len(pending_calls)}")
                    arguments = block.get("input")
                    step.tool_calls.append(
                        ClaudeCodeToolCall(
                            call_id=call_id,
                            name=str(block.get("name") or ""),
                            arguments=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                        )
                    )
                    pending_calls[call_id] = (step.step_id, timestamp)

        elif rtype == "user":
            if record.get("isMeta"):
                continue
            if record.get("isCompactSummary"):
                steps.append(
                    ClaudeCodeTrajectoryStep(
                        step_id=len(steps),
                        type="context_boundary",
                        timestamp=timestamp,
                        content=_block_text(content),
                    )
                )
                continue
            blocks = content if isinstance(content, list) else []
            tool_results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
            for block in tool_results:
                _attach_observation(record, block, timestamp, steps, pending_calls, assistant_record_ts)
            if not tool_results:
                text = _block_text(content)
                if text:
                    steps.append(
                        ClaudeCodeTrajectoryStep(
                            step_id=len(steps), type="user_message", timestamp=timestamp, content=text
                        )
                    )

    trajectory.stats = _total_stats(steps)
    return trajectory


def _current_agent_turn(steps: list[ClaudeCodeTrajectoryStep], message_id: Any) -> Optional[ClaudeCodeTrajectoryStep]:
    """Return the step for `message_id` if it continues the most recent agent turn."""
    if not steps or not message_id:
        return None
    last = steps[-1]
    if last.type == "agent_turn" and last.message_id == message_id:
        return last
    return None


def _attach_observation(
    record: dict,
    block: dict,
    completed_at: Optional[str],
    steps: list[ClaudeCodeTrajectoryStep],
    pending_calls: dict[str, tuple[int, Optional[str]]],
    assistant_record_ts: dict[str, str],
) -> None:
    call_id = str(block.get("tool_use_id") or "")
    step_index, started_at = pending_calls.pop(call_id, (None, None))
    source_uuid = record.get("sourceToolAssistantUUID")
    if isinstance(source_uuid, str) and source_uuid in assistant_record_ts:
        started_at = assistant_record_ts[source_uuid]

    duration_ms = None
    started_dt, completed_dt = _parse_timestamp(started_at), _parse_timestamp(completed_at)
    if started_dt is not None and completed_dt is not None:
        duration_ms = (completed_dt - started_dt).total_seconds() * 1000.0

    observation = ClaudeCodeToolObservation(
        source_call_id=call_id,
        content=_block_text(block.get("content")),
        status="error" if block.get("is_error") else "completed",
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        extra=_curate_extra(record.get("toolUseResult")),
    )

    if step_index is None:
        # Unmatched result (e.g. truncated transcript): attach to the latest agent turn.
        step_index = next((s.step_id for s in reversed(steps) if s.type == "agent_turn"), None)
        if step_index is None:
            LOG.warning("dropping tool observation with no agent turn to attach to (call_id=%s)", call_id)
            return
    steps[step_index].observations.append(observation)


def _total_stats(steps: list[ClaudeCodeTrajectoryStep]) -> ClaudeCodeModelCallStats:
    totals = ClaudeCodeModelCallStats()
    saw_cached = saw_creation = False
    for step in steps:
        if step.stats is None:
            continue
        totals.prompt_tokens += step.stats.prompt_tokens
        totals.completion_tokens += step.stats.completion_tokens
        totals.total_tokens += step.stats.total_tokens
        if step.stats.cached_tokens is not None:
            totals.cached_tokens = (totals.cached_tokens or 0) + step.stats.cached_tokens
            saw_cached = True
        if step.stats.cache_creation_tokens is not None:
            totals.cache_creation_tokens = (totals.cache_creation_tokens or 0) + step.stats.cache_creation_tokens
            saw_creation = True
    if not saw_cached:
        totals.cached_tokens = None
    if not saw_creation:
        totals.cache_creation_tokens = None
    return totals
