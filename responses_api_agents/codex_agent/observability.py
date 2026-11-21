# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read exact tool identities and model-visible results from a Codex CLI session record."""

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
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
_OUTPUT_ITEM = TypeAdapter(NeMoGymResponseOutputItem)
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


def _session_transcripts(transcript: str | None) -> list[tuple[dict, str]] | None:
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
                return None  # No reliable boundary for the preceding records.
            metadata = event.get("payload")
            if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str) or not metadata["id"]:
                return None
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
    roots = [meta["id"] for meta, _ in (sessions or []) if meta.get("source") in ("exec", "cli")]
    ids = [meta["id"] for meta, _ in (sessions or [])]
    if len(roots) != 1 or len(set(ids)) != len(ids):
        fallback_gaps = list(gaps or [])
        if (sessions is None or sessions) and not any(
            gap.code == "subagent_hierarchy_unavailable" for gap in fallback_gaps
        ):
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
            invocation.parent_invocation_id = None
            result.gaps.append(ObservationGap(code="subagent_hierarchy_unavailable", invocation_id=names[identity]))
            continue
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


class CodexOutputUnavailable(ValueError):
    """The Codex session record cannot stand in for ``response.output``; the message says why."""


# Session records that carry no model output: Codex's runtime events, per-turn settings and compaction summary
# (the observation bundle keeps those), and the world-state snapshot.
_NON_OUTPUT_RECORDS = frozenset({"event_msg", "turn_context", "compacted", "world_state"})
_TEXT_PART_TYPES = frozenset({"input_text", "output_text", "text"})
_IMAGE_PART_TYPES = frozenset({"input_image", "image"})


def _tool_result_text(output: object, *, call_id: str) -> str:
    """A tool result as text: a string as it is, content items as their text parts, and an image as ``[image]``."""
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        raise CodexOutputUnavailable(f"the result of call {call_id} is neither text nor content items")
    parts = []
    for part in output:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in _TEXT_PART_TYPES and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif kind in _IMAGE_PART_TYPES:
            parts.append("[image]")
        else:
            raise CodexOutputUnavailable(f"the result of call {call_id} holds an unsupported content item")
    return "\n".join(parts)


def _session_output_items(payload: dict[str, Any], position: int) -> list[dict[str, Any]]:
    """The Responses output items for one session ``response_item``; none for what is not model output.

    ``position`` numbers the items that lack an ID of their own, which Codex does not always keep.
    """
    kind = payload.get("type")
    if kind == "message":
        if payload.get("role") != "assistant":
            return []  # The prompt and Codex's own instructions: Gym already holds the request.
        content = []
        for part in payload.get("content") or []:
            if not isinstance(part, dict) or part.get("type") not in ("output_text", "refusal"):
                raise CodexOutputUnavailable("an assistant message holds an unsupported content part")
            content.append({"annotations": [], **part} if part["type"] == "output_text" else part)
        message = {
            "type": "message",
            "role": "assistant",
            "id": payload.get("id") or f"msg_{position}",
            "status": "completed",
            "content": content,
        }
        if payload.get("phase") in ("commentary", "final_answer"):
            message["phase"] = payload["phase"]
        return [message]
    if kind == "reasoning":
        # Only reasoning text is kept: the encrypted payload is opaque, and the observation bundle holds it.
        summary = [
            {"type": "summary_text", "text": part["text"]}
            for part in payload.get("summary") or []
            if isinstance(part, dict) and part.get("text")
        ]
        content = [
            {"type": "reasoning_text", "text": part["text"]}
            for part in payload.get("content") or []
            if isinstance(part, dict) and part.get("text")
        ]
        if not summary and not content:
            return []
        reasoning = {"type": "reasoning", "id": payload.get("id") or f"rs_{position}", "summary": summary}
        if content:
            reasoning["content"] = content
        return [reasoning]
    if kind in ("function_call", "custom_tool_call"):
        field = "arguments" if kind == "function_call" else "input"
        # The export that reads this output rejects a blank call_id or name, and masks the whole turn for it.
        identity_given = all(isinstance(payload.get(key), str) and payload[key].strip() for key in ("call_id", "name"))
        if not identity_given or not isinstance(payload.get(field), str):
            raise CodexOutputUnavailable(f"a {kind} record lacks its call_id, name or {field}")
        call = {"type": kind, "call_id": payload["call_id"], "name": payload["name"], field: payload[field]}
        call.update({key: payload[key] for key in ("id", "namespace") if isinstance(payload.get(key), str)})
        return [call]
    if kind in ("function_call_output", "custom_tool_call_output"):
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            raise CodexOutputUnavailable(f"a {kind} record lacks its call_id")
        result = {
            "type": kind,
            "call_id": call_id,
            "output": _tool_result_text(payload.get("output"), call_id=call_id),
        }
        if kind == "custom_tool_call_output":
            result |= {"id": payload.get("id") or f"ctco_{call_id}", "status": "completed"}
        return [result]
    if kind == "web_search_call":
        # Codex runs this search on the model's side and reports no result, so it is a call with an empty result.
        # Its name and arguments are those Harbor's own Codex agent exports, which graders read to tell a search from
        # a page open (action_type).
        call_id = payload.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"ws_{position}"
        action = payload.get("action") or {}
        if not isinstance(action, dict):
            raise CodexOutputUnavailable("a web_search_call record has an action that is not an object")
        arguments = {"action_type": action.get("type", "")}
        arguments |= {key: action[key] for key in ("query", "queries", "url") if key in action}
        return [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "web_search_call",
                "arguments": json.dumps(arguments),
            },
            {"type": "function_call_output", "call_id": call_id, "output": ""},
        ]
    raise CodexOutputUnavailable(f"unsupported session item {kind!r}")


def _session_items(transcript: str | None) -> list[dict[str, Any]]:
    """The Responses output items of a Codex session record, with each result matched to its open call."""
    if transcript is None:
        raise CodexOutputUnavailable("the Codex session record is unavailable")
    items: list[dict[str, Any]] = []
    calls: dict[str, bool] = {}  # call ID -> whether its result has arrived
    sessions = 0
    for line in _jsonl_lines(transcript):
        try:
            record = json.loads(line)
        except (ValueError, RecursionError) as exc:
            raise CodexOutputUnavailable("a session record is not JSON") from exc
        if not isinstance(record, dict):
            raise CodexOutputUnavailable("a session record is not an object")
        kind = record.get("type")
        if kind == "session_meta":
            sessions += 1
            if sessions > 1:
                raise CodexOutputUnavailable("the Codex session record holds several sessions")
            continue
        if kind in _NON_OUTPUT_RECORDS:
            continue
        payload = record.get("payload")
        if kind != "response_item" or not isinstance(payload, dict):
            raise CodexOutputUnavailable(f"unsupported session record {kind!r}")
        for item in _session_output_items(payload, len(items)):
            call_id = item.get("call_id")
            if item["type"] in ("function_call", "custom_tool_call"):
                if call_id in calls:
                    raise CodexOutputUnavailable(f"call {call_id} is made more than once")
                calls[call_id] = False
            elif item["type"] in ("function_call_output", "custom_tool_call_output"):
                if calls.get(call_id) is not False:
                    raise CodexOutputUnavailable(f"the result of call {call_id} matches no open call")
                calls[call_id] = True
            items.append(item)
    if not items:
        raise CodexOutputUnavailable("the Codex session record holds no model output")
    return items


def parse_codex_output(transcript: str | None) -> list[NeMoGymResponseOutputItem]:
    """The model's own output from a Codex session record, in the order the model produced it.

    ``codex exec --json`` reports each tool as the command Codex executed: a shell-wrapped command string, without
    the model's other arguments, with results merged and ordered by completion and display IDs for call IDs. The
    session record holds what the model called: each call's name, namespace, arguments and ``call_id`` exactly, in
    model order, and each result under its call's ID. This turns that record into Responses output items, so
    ``response.output`` and the trajectory exported from it show the model's calls.

    Reasoning is kept as text only, and a content-item result as its text with ``[image]`` for each image.
    A web search, which Codex runs on the model's side and reports without a result, becomes a ``web_search_call``
    call with the arguments ``action_type`` and the action's ``query``, ``queries`` or ``url``, as Harbor's own Codex
    agent exports it, and an empty result.

    Raises :class:`CodexOutputUnavailable`, and nothing else, when the record is missing, holds no model output, cannot
    be read or has a shape this does not expect, holds a session other than the first or an item type with no mapping,
    or has a call with a blank ID or name, a result with no open call or a call ID used twice: output built from it
    would misstate the run, or fail the export that reads it. A call ID is not reused even after its result, because
    the observations key each tool by it.
    """
    try:
        return [_OUTPUT_ITEM.validate_python(item) for item in _session_items(transcript)]
    except ValidationError as exc:
        raise CodexOutputUnavailable(
            f"a session item is not a Responses output item: {exc.errors()[0]['loc']}"
        ) from exc
    except (TypeError, RecursionError) as exc:
        # A record whose JSON has a shape the mapping did not expect, such as a list where Codex writes text.
        raise CodexOutputUnavailable(f"a session record has an unexpected shape: {type(exc).__name__}: {exc}") from exc


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
