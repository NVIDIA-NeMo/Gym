# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read exact tool identities and model-visible results from a Codex CLI session record."""

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)


# Compaction format compatibility: tested with Codex 0.144.4.
# This covers native compacted records and summary wrappers here, plus request
# metadata and completed SSE responses in sandbox_launcher.mjs; revalidate them together.
# Source: https://github.com/openai/codex/tree/rust-v0.144.4/codex-rs/core/src
# (compact.rs and responses_metadata.rs).
# The prefix below wraps the persisted summary. Strip it only after a structured
# compacted event; helper requests are identified separately by native metadata.
_COMPACTION_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking process. "
    "You also have access to the state of the tools that were used by that language model. "
    "Use this to build on the work that has already been done and avoid duplicating work. "
    "Here is the summary produced by the other language model, use the information in this summary "
    "to assist with your own analysis:\n"
)


_ITEM = TypeAdapter(NeMoGymResponseInputItem)
LOG = logging.getLogger(__name__)
TESTED_CODEX_VERSION = "0.144.4"
# Codex writes U+2028 and similar unescaped inside JSON strings, so str.splitlines() would cut a record in two.
_LINE_END = re.compile(r"\r\n|\n|\r")


def _jsonl_lines(text: str) -> list[str]:
    """Split session records as the sandbox snapshot does: only \\n, \\r\\n and \\r end a line."""
    lines = _LINE_END.split(text)
    if lines[-1] == "":
        lines.pop()
    return lines


def _token_usage(payload: dict) -> NeMoGymResponseUsage | None:
    # A token_count with info=null is a rate-limit update, not a new usage snapshot.
    info = payload["info"]
    if info is None:
        return None
    usage = info["total_token_usage"]
    if any(type(usage[key]) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens", "total_tokens")):
        raise ValueError("invalid Codex token totals")
    cached = usage.get("cached_input_tokens")
    reasoning = usage.get("reasoning_output_tokens")
    return NeMoGymResponseUsage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        input_tokens_details=NeMoGymResponseInputTokensDetails(
            cached_tokens=cached if type(cached) is int and cached > 0 else None
        ),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(
            reasoning_tokens=reasoning if type(reasoning) is int and reasoning > 0 else None
        ),
    )


def parse_codex_usage(transcript: str | None) -> NeMoGymResponseUsage | None:
    """Read the latest native cumulative totals when a failed turn omits CLI usage.

    These totals already include cached input and reasoning output. As in the CLI
    parser, zero detail counters cannot distinguish absent provider measurements.
    """
    for line in reversed(_jsonl_lines(transcript or "")):
        usage_record = False
        try:
            event = json.loads(line)
            if event["type"] != "event_msg" or event["payload"]["type"] != "token_count":
                continue
            usage_record = True
            usage = _token_usage(event["payload"])
            if usage is not None:
                return usage
        except (ValueError, TypeError, KeyError):
            if usage_record:
                LOG.warning("Codex token usage unavailable: unsupported token_count record")
                return None  # An older total would silently undercount the failed turn.
            continue
    return None


def read_codex_observations(
    home: Path, invocation_id: str, *, returncode: int | None, timed_out: bool
) -> AgentObservationBundle:
    """Use native call IDs; never join CLI display item IDs to calls by ordering."""
    files = sorted(home.glob("sessions/**/*.jsonl"))
    return parse_codex_observations(
        "\n".join(line for path in files for line in _jsonl_lines(path.read_text(errors="replace")))
        if files
        else None,
        invocation_id,
        returncode=returncode,
        timed_out=timed_out,
    )


def _session_transcripts(transcript: str | None) -> list[tuple[dict, str]]:
    sessions = []
    metadata = None
    lines = []
    for line in _jsonl_lines(transcript or ""):
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        if isinstance(event, dict) and event.get("type") == "session_meta":
            if metadata is not None:
                sessions.append((metadata, "\n".join(lines)))
            elif lines:
                return []  # No reliable boundary for the preceding records.
            metadata = event.get("payload")
            if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str) or not metadata["id"]:
                return []
            lines = []
        lines.append(line)
    if metadata is not None:
        sessions.append((metadata, "\n".join(lines)))
    return sessions


def _parent_thread(metadata: dict) -> str | None:
    source = metadata.get("source")
    if isinstance(source, dict):
        subagent = source.get("subagent")
        spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
        if isinstance(spawn, dict) and isinstance(spawn.get("parent_thread_id"), str):
            return spawn["parent_thread_id"]
    return None


def _child_status(transcript: str) -> Literal["completed", "incomplete", "unknown"]:
    status = "unknown"
    for line in _jsonl_lines(transcript):
        try:
            event = json.loads(line)
            if event.get("type") != "event_msg":
                continue
            kind = event["payload"].get("type")
            if kind == "task_complete":
                status = "completed"
            elif kind in {"task_started", "turn_aborted"}:
                status = "incomplete"
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return status


def parse_codex_observations(
    transcript: str | None,
    invocation_id: str,
    *,
    returncode: int | None,
    timed_out: bool,
    status: Literal["completed", "failed", "incomplete", "unknown"] | None = None,
    error_type: str | None = None,
    gaps: list[ObservationGap] | None = None,
) -> AgentObservationBundle:
    """Keep Codex thread contexts separate and link children using session metadata."""
    sessions = _session_transcripts(transcript)
    roots = [meta["id"] for meta, _ in sessions if meta.get("source") in ("exec", "cli")]
    ids = [meta["id"] for meta, _ in sessions]
    if len(roots) != 1 or len(set(ids)) != len(ids):
        fallback_gaps = list(gaps or [])
        if sessions:
            fallback_gaps.append(ObservationGap(code="subagent_hierarchy_unavailable"))
        return _parse_codex_transcript(
            transcript,
            invocation_id,
            returncode=returncode,
            timed_out=timed_out,
            status=status,
            error_type=error_type,
            gaps=fallback_gaps,
        )
    root = roots[0]
    names = {identity: invocation_id if identity == root else f"{invocation_id}.{identity}" for identity in ids}
    result = AgentObservationBundle(
        source="codex",
        records=[],
        gaps=[gap for gap in (gaps or []) if gap.code != "subagent_hierarchy_unavailable"],
    )
    parents = {}
    for metadata, text in sessions:
        identity = metadata["id"]
        is_root = identity == root
        parent = _parent_thread(metadata)
        if not is_root and (parent not in names or parent == identity):
            result.gaps.append(ObservationGap(code="subagent_hierarchy_unavailable", invocation_id=names[identity]))
        parsed = _parse_codex_transcript(
            text,
            names[identity],
            returncode=returncode if is_root else None,
            timed_out=timed_out if is_root else False,
            status=status if is_root else _child_status(text),
            error_type=error_type if is_root else None,
        )
        invocation = next(record for record in parsed.records if isinstance(record, AgentInvocation))
        invocation.parent_invocation_id = names.get(parent) if not is_root else None
        parents[identity] = (parent, invocation)
        result.records.extend(parsed.records)
        result.gaps.extend(parsed.gaps)
    for identity, (parent, invocation) in parents.items():
        ancestor, visited = identity, set()
        while ancestor != root and ancestor in parents and ancestor not in visited:
            visited.add(ancestor)
            ancestor = parents[ancestor][0]
        if ancestor != root:
            result.gaps.append(ObservationGap(code="subagent_hierarchy_unavailable", invocation_id=names[identity]))
        if parent not in parents:
            continue
        conversation = parents[parent][1].conversation or []
        spawn_calls = {
            item.call_id for item in conversation if item.type == "function_call" and item.name == "spawn_agent"
        }
        matches = []
        for item in conversation:
            if (
                item.type != "function_call_output"
                or item.call_id not in spawn_calls
                or not isinstance(item.output, str)
            ):
                continue
            try:
                output = json.loads(item.output)
            except ValueError:
                continue
            if isinstance(output, dict) and output.get("agent_id") == identity:
                matches.append(item.call_id)
        if len(matches) == 1:
            invocation.spawned_by_tool_call_id = matches[0]
        else:
            result.gaps.append(
                ObservationGap(code="subagent_spawn_reference_unavailable", invocation_id=names[identity])
            )
    return result


def _parse_codex_transcript(
    transcript: str | None,
    invocation_id: str,
    *,
    returncode: int | None,
    timed_out: bool,
    status: Literal["completed", "failed", "incomplete", "unknown"] | None = None,
    error_type: str | None = None,
    gaps: list[ObservationGap] | None = None,
) -> AgentObservationBundle:
    """Parse a transcript, using the caller's final outcome and gaps when supplied."""
    conversation = []
    compactions = []
    tools = {}
    patch_outcomes = {}
    gaps = list(gaps or [])
    # A record that could not be read may have been a tool call, unless it is a native event (token usage, a patch
    # outcome): then the tool history read is not complete, and its timing is unknown.
    tool_history_complete = True
    for line in _jsonl_lines(transcript or ""):
        native_event = False
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("expected a Codex CLI session record")
            if event.get("type") == "event_msg":
                native_event = True
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    raise ValueError("expected a native event payload")
                if payload.get("type") == "token_count":
                    _token_usage(payload)
                if payload.get("type") == "patch_apply_end" and type(payload.get("success")) is bool:
                    patch_outcomes[payload["call_id"]] = payload["success"]
                continue
            if event.get("type") == "compacted":
                native_event = True
                message = event["payload"]["message"]
                if not isinstance(message, str) or not message.startswith(_COMPACTION_PREFIX):
                    raise ValueError("unsupported Codex compaction summary")
                compactions.append(
                    ContextCompactionObservation(
                        invocation_id=invocation_id,
                        observed_at=datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp(),
                        trigger="automatic",
                        outcome="completed",
                        summary=message.removeprefix(_COMPACTION_PREFIX),
                    )
                )
                continue
            if event.get("type") != "response_item":
                if event.get("type") not in {"session_meta", "turn_context", "compacted", "world_state"}:
                    raise ValueError("unsupported Codex session record type")
                continue
            item = dict(event["payload"])
            item.pop("internal_chat_message_metadata_passthrough", None)
            parsed = _ITEM.validate_python(item)
            conversation.append(parsed)
            timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError, KeyError, ValidationError):
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable", invocation_id=invocation_id))
            tool_history_complete = tool_history_complete and native_event
            continue
        if item.get("type") in {"function_call", "custom_tool_call"}:
            tools[item["call_id"]] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=item["call_id"],
                tool_name=item["name"],
                started_at=timestamp,
                timing_source="artifact",
                status="incomplete",
            )
        elif item.get("type") == "web_search_call":
            tools[item["id"]] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=item["id"],
                tool_name="web_search",
                status=item["status"] if item["status"] in {"completed", "failed"} else "incomplete",
                completed_at=timestamp if item["status"] in {"completed", "failed"} else None,
                timing_source="artifact",
            )
        elif item.get("type") in {"function_call_output", "custom_tool_call_output"} and item.get("call_id") in tools:
            tool = tools[item["call_id"]]
            tool.completed_at = max(timestamp, tool.started_at)
            tool.duration_ms = (tool.completed_at - tool.started_at) * 1000
            # Only the native execution envelope determines the exit code;
            # command stdout can itself contain arbitrary exit-code text.
            output = item.get("output")
            envelope = output.split("\nOutput:\n", 1)[0] if isinstance(output, str) else ""
            exit_code = re.search(r"^Process exited with code (-?\d+)$", envelope, re.MULTILINE)
            if tool.tool_name == "exec_command" and exit_code:
                tool.status = "completed" if int(exit_code[1]) == 0 else "failed"
                tool.error_type = "nonzero_exit" if tool.status == "failed" else None
            else:
                tool.status = "unknown"
    # Patch success is in native execution events, not in the model-visible output text.
    for tool in tools.values():
        if tool.tool_name == "apply_patch" and tool.tool_call_id in patch_outcomes:
            tool.status = "completed" if patch_outcomes[tool.tool_call_id] else "failed"
            tool.error_type = "tool_error" if tool.status == "failed" else None
        if tool.status == "unknown":
            gaps.append(
                ObservationGap(code="tool_outcome_unavailable", invocation_id=invocation_id, detail=tool.tool_call_id)
            )
    if not tool_history_complete:
        gaps.append(ObservationGap(code="tool_timing_unavailable", invocation_id=invocation_id))
    if transcript is None:
        gaps.append(ObservationGap(code="agent_artifact_unavailable", invocation_id=invocation_id))
    if status is None:
        status = "incomplete" if timed_out else "completed" if returncode == 0 else "failed"
    return AgentObservationBundle(
        source="codex",
        records=[
            AgentInvocation(
                invocation_id=invocation_id,
                conversation=conversation,
                status=status,
                error_type=error_type,
            ),
            *tools.values(),
            *compactions,
        ],
        gaps=gaps,
    )


def associate_codex_compaction_calls(
    bundle: AgentObservationBundle,
    attempts: list[dict[str, JsonValue]],
    *,
    model_ref: ModelServerRef | None,
) -> AgentObservationBundle:
    """Attach session-local helper identities before returning the agent's bundle.

    The session transport records only completed responses to explicit compaction
    requests. The native compacted event must independently confirm the summary.
    """
    result = bundle.model_copy(deep=True)
    events = [record for record in result.records if isinstance(record, ContextCompactionObservation)]
    invocations = [record for record in result.records if isinstance(record, AgentInvocation)]
    previous_refs = [ref for event in events for ref in event.model_calls]
    for invocation in invocations:
        invocation.model_calls = [ref for ref in invocation.model_calls if ref not in previous_refs]
    result.gaps = [gap for gap in result.gaps if gap.code != "compaction_model_call_reference_unavailable"]
    helpers = []
    for attempt in attempts:
        helper = attempt.get("compaction_response")
        if (
            attempt.get("status") == 200
            and not attempt.get("transport_error")
            and isinstance(helper, dict)
            and isinstance(helper.get("response_id"), str)
            and helper["response_id"]
            and isinstance(helper.get("summary"), str)
        ):
            helpers.append(helper)
    candidates = [
        [i for i, helper in enumerate(helpers) if event.summary and helper["summary"] == event.summary]
        for event in events
    ]
    counts = Counter(index for matches in candidates for index in matches)
    identities = Counter(helper["response_id"] for helper in helpers)
    for event, matches in zip(events, candidates):
        event.model_calls = []
        owners = [inv for inv in invocations if inv.invocation_id == event.invocation_id]
        if model_ref is not None and len(matches) == 1 and counts[matches[0]] == 1 and len(owners) == 1:
            helper = helpers[matches[0]]
            if identities[helper["response_id"]] == 1:
                reference = ModelCallRef(model_ref=model_ref, response_id=helper["response_id"])
                event.model_calls = [reference]
                if reference not in owners[0].model_calls:
                    owners[0].model_calls.append(reference)
                continue
        result.gaps.append(
            ObservationGap(code="compaction_model_call_reference_unavailable", invocation_id=event.invocation_id)
        )
    return result
