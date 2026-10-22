# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read exact tool identities and model-visible results from a Codex CLI session record."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from nemo_gym.openai_utils import (
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap, ToolCallObservation


_ITEM = TypeAdapter(NeMoGymResponseInputItem)
LOG = logging.getLogger(__name__)
TESTED_CODEX_VERSION = "0.144.4"


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
    for line in reversed((transcript or "").splitlines()):
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
        "\n".join(line for path in files for line in path.read_text(errors="replace").splitlines()) if files else None,
        invocation_id,
        returncode=returncode,
        timed_out=timed_out,
    )


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
    """Parse a transcript, using the caller's final outcome and gaps when supplied."""
    conversation = []
    tools = {}
    patch_outcomes = {}
    gaps = list(gaps or [])
    for line in (transcript or "").splitlines():
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("expected a Codex CLI session record")
            if event.get("type") == "event_msg":
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    raise ValueError("expected a native event payload")
                if payload.get("type") == "token_count":
                    _token_usage(payload)
                if payload.get("type") == "patch_apply_end" and type(payload.get("success")) is bool:
                    patch_outcomes[payload["call_id"]] = payload["success"]
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
        ],
        gaps=gaps,
    )
