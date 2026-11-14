# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read exact tool identities and model-visible results from a Codex CLI session record."""

import json
import logging
import re
from collections import Counter, defaultdict
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
    normalize_operation_timing,
    tool_attempt_id,
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
    home: Path,
    invocation_id: str,
    *,
    returncode: int | None,
    timed_out: bool,
    agent_instance: str | None = None,
    clock_domain: str | None = None,
    observation_schema_version: Literal["1.0", "2.0"] = "1.0",
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
        agent_instance=agent_instance,
        clock_domain=clock_domain,
        observation_schema_version=observation_schema_version,
    )


_CALL_RECORDS = {"function_call", "custom_tool_call", "web_search_call"}
_RESULT_RECORDS = {"function_call_output", "custom_tool_call_output"}


def _timestamp(event: dict) -> datetime | None:
    """An absent, malformed or zone-less record time is unknown, not the reader's local time."""
    try:
        observed_at = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    except (TypeError, KeyError, AttributeError, ValueError):
        return None
    return observed_at if observed_at.tzinfo is not None else None


def _persisted_item_ids(lines: list[str]) -> set[str]:
    """Every call and item ID any persisted record uses, including records later in the stream."""
    reserved = set()
    for line in lines:
        try:
            event = json.loads(line)
            payload = event["payload"] if event["type"] == "response_item" else {}
        except (ValueError, TypeError, KeyError):
            continue
        if isinstance(payload, dict):
            reserved.update(payload[key] for key in ("id", "call_id") if isinstance(payload.get(key), str))
    return reserved


def _transcript_tools(
    attempts: list[dict], invocation_id: str, agent_instance: str | None, clock_domain: str | None
) -> tuple[list[ToolCallObservation], list[ObservationGap]]:
    """Persisted call/result records are transcript evidence, not executor or lifecycle timing.

    Codex 0.144.4 records no response ID, so the shared normalizer reports the missing model join.
    """
    tools, gaps = [], []
    producer_id = f"codex:{invocation_id}"
    for attempt in attempts:
        start, end = attempt["start"], attempt["end"]
        # Reversed endpoints are a clock discontinuity, not a negative native duration.
        duration = (end - start).total_seconds() * 1000 if start and end and end >= start else None
        timing, codes = normalize_operation_timing(
            boundary="transcript",
            timing_source="codex.response_item",
            clock_domain=clock_domain,
            started_at=start.timestamp() if start else None,
            completed_at=end.timestamp() if end else None,
            duration_ms=duration,
            duration_source="wall_clock" if duration is not None else None,
        )
        span_id = tool_attempt_id(producer_id=producer_id, invocation_id=invocation_id, operation_key=attempt["key"])
        tools.append(
            ToolCallObservation(
                invocation_id=invocation_id,
                agent_instance=agent_instance,
                tool_call_id=attempt["call_id"],
                native_tool_call_id=attempt["native"],
                span_id=span_id,
                producer_id=producer_id,
                tool_name=attempt["name"],
                timings=[timing] if timing is not None else [],
                status=attempt["status"],
                error_type=attempt["error_type"],
            )
        )
        codes += [code for code in attempt["gaps"] if code not in codes]
        if start is None:
            codes.append("tool_start_unavailable")
        if end is None:
            codes.append("tool_completion_unavailable")
        gaps.extend(
            ObservationGap(code=code, invocation_id=invocation_id, agent_instance=agent_instance, span_id=span_id)
            for code in codes
        )
        if attempt["status"] == "unknown":
            # The native ID keeps this gap identifiable after v1 projection drops span IDs.
            gaps.append(
                ObservationGap(
                    code="tool_outcome_unavailable",
                    invocation_id=invocation_id,
                    agent_instance=agent_instance,
                    span_id=span_id,
                    detail=attempt["native"],
                )
            )
    return tools, gaps


def _result_outcome(
    tool_name: str | None, output: object
) -> tuple[Literal["completed", "failed", "unknown"], str | None]:
    # Only the native execution envelope determines the exit code;
    # command stdout can itself contain arbitrary exit-code text.
    envelope = output.split("\nOutput:\n", 1)[0] if isinstance(output, str) else ""
    exit_code = re.search(r"^Process exited with code (-?\d+)$", envelope, re.MULTILINE)
    if tool_name == "exec_command" and exit_code:
        return ("completed", None) if int(exit_code[1]) == 0 else ("failed", "nonzero_exit")
    # Includes a command that returned a live session: that does not establish its process exited.
    return "unknown", None


def parse_codex_observations(
    transcript: str | None,
    invocation_id: str,
    *,
    returncode: int | None,
    timed_out: bool,
    status: Literal["completed", "failed", "incomplete", "unknown"] | None = None,
    error_type: str | None = None,
    gaps: list[ObservationGap] | None = None,
    agent_instance: str | None = None,
    clock_domain: str | None = None,
    observation_schema_version: Literal["1.0", "2.0"] = "1.0",
) -> AgentObservationBundle:
    """Parse a transcript, using the caller's final outcome and gaps when supplied.

    Schema 2.0 keeps one attempt per persisted call record. A request keeps its native call ID
    unless an earlier record used it; a reuse, or a result that no call can own, gets an ID derived
    from its record position that no persisted record or earlier allocation uses. Unlike
    OpenCode, Codex has no exact model-call join, so captured turns keep unproven native IDs that
    block the v1 projection from restoring qualified ones; qualifying only reuse keeps v1 native.
    A result closes the only open call with its native ID. A result for concurrent calls sharing
    that ID belongs to none of them; the group closes once it has had as many results as calls.

    Session files are concatenated in path order. Codex 0.144.4 spawned-agent forks copy parent
    events but not tool records into the child file, so tools are not duplicated; child tools are
    attributed to this invocation (see ``subagent_hierarchy_unavailable``).
    """
    if observation_schema_version not in {"1.0", "2.0"}:
        raise ValueError("observation_schema_version must be 1.0 or 2.0")
    qualified = observation_schema_version == "2.0"
    conversation = []
    compactions = []
    tools = {}
    patch_outcomes = {}
    gaps = list(gaps or [])
    attempts = []
    open_calls = defaultdict(lambda: {"calls": [], "results": 0})
    used_call_ids = set()
    lines = _jsonl_lines(transcript or "")
    reserved_ids = _persisted_item_ids(lines) if qualified else set()

    def record_id(native: str, index: int) -> str:
        """A replay-stable ID that no persisted record or earlier allocation uses."""
        candidate, suffix = f"{native}:record-{index}", 0
        while candidate in reserved_ids:
            suffix += 1
            candidate = f"{native}:record-{index}.{suffix}"
        reserved_ids.add(candidate)
        return candidate

    # A record that could not be read may have been a tool call, unless it is a native event (token usage, a patch
    # outcome): then the tool history read is not complete, and its timing is unknown.
    tool_history_complete = True
    for index, line in enumerate(lines):
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
                    patch_outcomes.setdefault(payload["call_id"], []).append(payload["success"])
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
            if not qualified:
                timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()
            conversation.append(parsed)
        except (ValueError, TypeError, KeyError, ValidationError):
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable", invocation_id=invocation_id))
            tool_history_complete = tool_history_complete and native_event
            continue
        kind = item.get("type")
        if qualified and kind in _CALL_RECORDS | _RESULT_RECORDS:
            # Validation requires these IDs, so every tool record here has a native identity.
            native = item["id"] if kind == "web_search_call" else item["call_id"]
            observed_at = _timestamp(event)
            group = open_calls[native]
            if kind in _RESULT_RECORDS and len(group["calls"]) > 1:
                # A fresh result ID attaches to no attempt, in the group or elsewhere.
                conversation[-1] = parsed.model_copy(update={"call_id": record_id(native, index)})
                group["results"] += 1
                for attempt in group["calls"]:
                    attempt["status"], attempt["error_type"] = "unknown", None
                    if "tool_output_ambiguous" not in attempt["gaps"]:
                        attempt["gaps"].append("tool_output_ambiguous")
                if group["results"] >= len(group["calls"]):
                    del open_calls[native]
                continue
            if kind in _RESULT_RECORDS and group["calls"]:
                attempt = open_calls.pop(native)["calls"][0]
            else:
                # Persisted record position is stable on replay; native IDs can repeat across responses.
                # A hosted search has no call_id; its item ID is its conversation identity, which the
                # shared v1 projection cannot rename, so it stays native.
                call_id = native
                if kind != "web_search_call" and native in used_call_ids:
                    call_id = record_id(native, index)
                used_call_ids.add(native)
                attempt = {
                    "key": f"record:{index}",
                    "native": native,
                    "call_id": call_id,
                    "name": "web_search" if kind == "web_search_call" else item.get("name"),
                    "start": None,
                    "end": None,
                    "status": "incomplete",
                    "error_type": None,
                    "gaps": [],
                }
                attempts.append(attempt)
            if kind == "web_search_call":
                # Hosted search persists one record after it finishes; its request time is unknown.
                if item.get("status") in {"completed", "failed"}:
                    attempt["status"], attempt["end"] = item["status"], observed_at
                continue
            if kind in _CALL_RECORDS:
                attempt["start"] = observed_at
                group["calls"].append(attempt)
            else:
                # A result without its call record still keeps its known completion endpoint.
                attempt["end"] = observed_at
                attempt["status"], attempt["error_type"] = _result_outcome(attempt["name"], item.get("output"))
            # Through a Gym model server the item ID repeats the call ID; it is renamed with it.
            rename = {"call_id": attempt["call_id"]} | ({"id": attempt["call_id"]} if item.get("id") == native else {})
            conversation[-1] = parsed.model_copy(update=rename)
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
            tool.status, tool.error_type = _result_outcome(tool.tool_name, item.get("output"))
    # Patch success is in native execution events, not in the model-visible output text.
    for tool in tools.values():
        if tool.tool_name == "apply_patch" and tool.tool_call_id in patch_outcomes:
            tool.status = "completed" if patch_outcomes[tool.tool_call_id][-1] else "failed"
            tool.error_type = "tool_error" if tool.status == "failed" else None
        if tool.status == "unknown":
            gaps.append(
                ObservationGap(code="tool_outcome_unavailable", invocation_id=invocation_id, detail=tool.tool_call_id)
            )
    native_ids = Counter(attempt["native"] for attempt in attempts)
    for attempt in attempts:
        # Events carry only the native call ID, so they classify a patch only when that ID is unique.
        # A spawned agent's fork copies parent events (not tool records) into its own session file,
        # so identical copies agree; conflicting outcomes leave the patch unknown.
        outcomes = set(patch_outcomes.get(attempt["native"], []))
        if attempt["name"] == "apply_patch" and native_ids[attempt["native"]] == 1 and len(outcomes) == 1:
            succeeded = next(iter(outcomes))
            attempt["status"], attempt["error_type"] = ("completed", None) if succeeded else ("failed", "tool_error")
    if qualified:
        transcript_tools, transcript_gaps = _transcript_tools(attempts, invocation_id, agent_instance, clock_domain)
        tools = {tool.span_id: tool for tool in transcript_tools}
        gaps.extend(transcript_gaps)
    if not tool_history_complete:
        gaps.append(ObservationGap(code="tool_timing_unavailable", invocation_id=invocation_id))
    if transcript is None:
        gaps.append(ObservationGap(code="agent_artifact_unavailable", invocation_id=invocation_id))
    if status is None:
        status = "incomplete" if timed_out else "completed" if returncode == 0 else "failed"
    return AgentObservationBundle(
        schema_version=observation_schema_version,
        source="codex",
        records=[
            AgentInvocation(
                invocation_id=invocation_id,
                agent_instance=agent_instance if qualified else None,
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
