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

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple, Optional
from uuid import uuid4

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)


LOG = logging.getLogger(__name__)


def _extract_text(content: list[Any]) -> str:
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


# (input, cached input, output) tokens. Input includes the prompt cache's writes and reads, which an Anthropic
# usage object reports apart from ``input_tokens``. Cached input is None when no cache count was reported.
UsageCounts = tuple[int, Optional[int], int]


def _add(a: UsageCounts, b: UsageCounts) -> UsageCounts:
    cached = None if a[1] is None and b[1] is None else (a[1] or 0) + (b[1] or 0)
    return a[0] + b[0], cached, a[2] + b[2]


def _reported_counts(
    usage: Any, *, input_key: str, output_key: str, read_key: str, write_key: str
) -> Optional[UsageCounts]:
    """The counts in one usage record, or None unless it reports input or output tokens and every count it reports
    is a non-negative integer. A count that is absent or null was not reported."""
    if not isinstance(usage, Mapping):
        return None
    reported = {key: usage[key] for key in (input_key, output_key, read_key, write_key) if usage.get(key) is not None}
    if input_key not in reported and output_key not in reported:
        return None
    if any(type(value) is not int or value < 0 for value in reported.values()):
        return None
    read, written = reported.get(read_key), reported.get(write_key)
    cached = None if read is None and written is None else read or 0
    return reported.get(input_key, 0) + (written or 0) + (read or 0), cached, reported.get(output_key, 0)


def _usage_counts(usage: Any) -> Optional[UsageCounts]:
    """An Anthropic Messages usage object's counts."""
    return _reported_counts(
        usage,
        input_key="input_tokens",
        output_key="output_tokens",
        read_key="cache_read_input_tokens",
        write_key="cache_creation_input_tokens",
    )


def _model_usage_counts(model_usage: Any) -> Optional[UsageCounts]:
    """The result event's ``modelUsage`` summed over models: every call the CLI made, its subagents' included.

    None unless every model's record is valid, so the result event's ``usage`` is used instead.
    """
    if not isinstance(model_usage, Mapping) or not model_usage:
        return None
    totals: UsageCounts = (0, None, 0)
    for usage in model_usage.values():
        counts = _reported_counts(
            usage,
            input_key="inputTokens",
            output_key="outputTokens",
            read_key="cacheReadInputTokens",
            write_key="cacheCreationInputTokens",
        )
        if counts is None:
            return None
        totals = _add(totals, counts)
    return totals


class TranscriptCall(NamedTuple):
    usage: Optional[UsageCounts]
    finished: bool


def transcript_calls(records: Iterable[Any]) -> dict[str, TranscriptCall]:
    """Each model call in Claude Code's session transcripts, by message id: its last usage and whether it finished.

    Claude Code writes an entry per content block. Once the message has finished, the entries carry its stop
    reason and its final usage, output included. An entry written before then carries neither: the usage known
    when the message started, with no output. A message whose last entry has no stop reason was cut off in flight.
    """
    calls: dict[str, TranscriptCall] = {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("id"), str) or not message["id"]:
            continue
        calls[message["id"]] = TranscriptCall(_usage_counts(message.get("usage")), bool(message.get("stop_reason")))
    return calls


def _extract_thinking(content: list[Any]) -> str:
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("thinking", "reasoning"):
            parts.append(b.get("thinking") or b.get("text") or "")
    return "\n".join(p for p in parts if p)


def parse_stream_json(
    stdout: str,
    *,
    fold_thinking: bool = True,
    transcript_records: Optional[Iterable[Mapping[str, Any]]] = None,
) -> tuple[list[Any], dict]:
    """Convert claude -p --output-format=stream-json stdout into (output_items, usage).

    Usage counts every model call the CLI made, subagents' included, with the prompt cache's reads and writes in
    the input. It comes from the result event when there is one. Without it (the CLI was killed), each call is
    counted once by message id, from the run's transcript records (``transcript_records``, None when none could
    be read): a call that finished counts its final usage, and a call cut off in flight counts the input it was
    sent with, from its transcript entry or the stream, but no output, which is reported only when a call ends.
    Context compaction, which Claude Code leaves out of its transcripts and stream, is not counted. Without
    transcripts, the stream's per-message usage is all there is, which on servers that send output only at the end
    of a message (vLLM) holds input alone. ``cached_tokens`` is None when no cache count was reported.

    ``fold_thinking=True`` (eval default) prefixes each assistant message with the model's thinking as
    ``<think>...</think>`` text. Under training-token capture that must be off: Gym attributes the finished
    transcript to a captured model call by hashing the model-authored text, and the captured call's message
    holds the bare text (reasoning travels as a separate item). Thinking is then emitted as a ``reasoning``
    item, which the fingerprint deliberately ignores. Capture output contains only the main conversation,
    with pending tool calls retained as incomplete. The last main-thread assistant message's served ``id`` is
    returned in internal metadata as ``terminal_response_id`` and becomes ``response.id`` under capture.
    It is not a declared terminal id on the ``/run`` result; other attribution witnesses still corroborate it.
    """
    raw_events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    output_items: list[Any] = []
    pending_calls: dict[str, NeMoGymResponseFunctionToolCall] = {}
    buffered_think: str | None = None
    # Token usage. The result event carries the run's cumulative usage, so it is the total when present.
    # Without one (the CLI was killed or timed out) and without transcripts, sum the stream's assistant messages
    # once each: stream-json emits one event per content block, each repeating the usage its message started with.
    result_usage: Optional[UsageCounts] = None
    message_usage: dict[str, UsageCounts] = {}
    unidentified: UsageCounts = (0, None, 0)
    num_turns: Optional[int] = None
    result_metadata: dict[str, Any] = {}
    compacting_sessions: set[str] = set()
    compaction_attempts: list[dict[str, str]] = []
    terminal_response_id: Optional[str] = None

    for event in raw_events:
        etype = event.get("type")

        if etype == "result":
            # ``usage`` covers the main conversation only; ``modelUsage`` adds its subagents' calls.
            counts = _model_usage_counts(event.get("modelUsage"))
            if counts is None:
                counts = _usage_counts(event.get("usage"))
            if counts is not None:
                result_usage = _add(result_usage or (0, None, 0), counts)
            # Claude Code's authoritative turn counter (what --max-turns bounds).
            if event.get("num_turns") is not None:
                num_turns = int(event["num_turns"])
            if isinstance(event.get("subtype"), str):
                result_metadata["subtype"] = event["subtype"]
            if isinstance(event.get("is_error"), bool):
                result_metadata["is_error"] = event["is_error"]
            duration_ms = event.get("duration_ms")
            if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool) and duration_ms >= 0:
                result_metadata["duration_ms"] = float(duration_ms)

        elif etype == "assistant":
            message = event.get("message", {})
            content = message.get("content") or []
            counts = _usage_counts(message.get("usage"))
            message_id = message.get("id")
            if counts is not None and isinstance(message_id, str) and message_id:
                message_usage[message_id] = counts
            elif counts is not None:
                unidentified = _add(unidentified, counts)

            # Subagent frames belong to a different captured chain. Keep their usage accounting,
            # but never let their text or tool ids testify to the main conversation's terminal call.
            if not fold_thinking and event.get("parent_tool_use_id"):
                continue

            if not isinstance(content, list):
                content = []

            served_id = message.get("id")
            if not event.get("parent_tool_use_id") and isinstance(served_id, str) and served_id:
                terminal_response_id = served_id

            think = _extract_thinking(content)
            if think and fold_thinking:
                buffered_think = (buffered_think + "\n" + think) if buffered_think else think
            elif think:
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[NeMoGymSummary(text=think, type="summary_text")],
                        type="reasoning",
                    )
                )

            text = _extract_text(content)
            if text:
                if buffered_think:
                    text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                    buffered_think = None
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                input_data = block.get("input") or {}
                arguments = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                call = NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=block.get("name", ""),
                    type="function_call",
                    id=call_id,
                    status="incomplete",
                )
                pending_calls[call_id] = call
                if not fold_thinking:
                    # Preserve model order even when tools finish out of order or never finish.
                    output_items.append(call)

        elif etype == "user":
            if not fold_thinking and event.get("parent_tool_use_id"):
                continue
            message = event.get("message", {})
            content = message.get("content") or []
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id", "")
                call = pending_calls.pop(tool_id, None)
                if call:
                    call.status = "completed"
                    if fold_thinking:
                        output_items.append(call)
                result_content = block.get("content") or ""
                if isinstance(result_content, list):
                    result_text = _extract_text(result_content)
                else:
                    result_text = str(result_content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_id,
                        output=result_text,
                        status="completed",
                    )
                )

        elif etype == "system" and event.get("subtype") == "status":
            session_id = event.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                continue
            if event.get("status") == "compacting":
                compacting_sessions.add(session_id)
                continue
            compact_result = event.get("compact_result")
            if compact_result in {"failed", "success"}:
                if compact_result == "failed":
                    compaction_attempts.append({"invocation_id": session_id, "outcome": "failed"})
                compacting_sessions.discard(session_id)

    compaction_attempts.extend(
        {"invocation_id": session_id, "outcome": "unknown"} for session_id in compacting_sessions
    )
    if result_usage is not None:
        totals = result_usage
    elif transcript_records is not None:
        calls = transcript_calls(transcript_records)
        totals = (0, None, 0)
        for message_id in sorted(calls.keys() | message_usage.keys()):
            call = calls.get(message_id)
            if call is not None and call.finished and call.usage is not None:
                totals = _add(totals, call.usage)
                continue
            started = call.usage if call is not None and call.usage is not None else message_usage.get(message_id)
            if started is not None:
                totals = _add(totals, (started[0], started[1], 0))
    else:
        totals = unidentified
        for counts in message_usage.values():
            totals = _add(totals, counts)
    metadata: dict = {"input_tokens": totals[0], "cached_tokens": totals[1], "output_tokens": totals[2]}
    if terminal_response_id:
        metadata["terminal_response_id"] = terminal_response_id
    if num_turns is not None:
        metadata["num_turns"] = num_turns
    if compaction_attempts:
        metadata["compaction_attempts"] = compaction_attempts
    metadata.update(result_metadata)
    return output_items, metadata


def _invocation_outcome(metadata: dict[str, Any], returncode: int | None) -> tuple[str, str | None]:
    subtype = metadata.get("subtype")
    if subtype == "error_max_turns":
        return "incomplete", subtype
    if metadata.get("is_error") is True or (isinstance(subtype, str) and subtype.startswith("error_")):
        return "failed", subtype if isinstance(subtype, str) else "agent_error"
    if returncode not in (0, None):
        return "failed", f"process_exit_{returncode}"
    if subtype == "success":
        return "completed", None
    return "incomplete", "result_missing"


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message
