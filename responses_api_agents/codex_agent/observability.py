# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read exact tool identities and model-visible results from a Codex CLI session record."""

import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from nemo_gym.openai_utils import (
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap, ToolCallObservation


_ITEM = TypeAdapter(NeMoGymResponseInputItem)


def parse_codex_usage(transcript: str | None) -> NeMoGymResponseUsage | None:
    """Read the latest native cumulative totals when a failed turn omits CLI usage.

    These totals already include cached input and reasoning output. As in the CLI
    parser, zero detail counters cannot distinguish absent provider measurements.
    """
    for line in reversed((transcript or "").splitlines()):
        try:
            event = json.loads(line)
            if event["type"] != "event_msg" or event["payload"]["type"] != "token_count":
                continue
            usage = event["payload"]["info"]["total_token_usage"]
            if any(
                type(usage[key]) is not int or usage[key] < 0
                for key in ("input_tokens", "output_tokens", "total_tokens")
            ):
                continue
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
        except (ValueError, TypeError, KeyError):
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
    transcript: str | None, invocation_id: str, *, returncode: int | None, timed_out: bool
) -> AgentObservationBundle:
    """Parse a retained native transcript from either local or sandbox execution."""
    conversation = []
    tools = {}
    gaps = []
    for line in (transcript or "").splitlines():
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("expected a Codex CLI session record")
            if event.get("type") != "response_item":
                continue
            item = dict(event["payload"])
            item.pop("internal_chat_message_metadata_passthrough", None)
            parsed = _ITEM.validate_python(item)
            conversation.append(parsed)
            timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError, KeyError, ValidationError):
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable", invocation_id=invocation_id))
            continue
        if item.get("type") == "function_call":
            tools[item["call_id"]] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=item["call_id"],
                tool_name=item["name"],
                started_at=timestamp,
                timing_source="artifact",
                status="incomplete",
            )
        elif item.get("type") == "function_call_output" and item.get("call_id") in tools:
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
                gaps.append(
                    ObservationGap(
                        code="tool_outcome_unavailable", invocation_id=invocation_id, detail=tool.tool_call_id
                    )
                )
    if transcript is None:
        gaps.append(ObservationGap(code="agent_artifact_unavailable", invocation_id=invocation_id))
    return AgentObservationBundle(
        source="codex",
        records=[
            AgentInvocation(
                invocation_id=invocation_id,
                conversation=conversation,
                status="incomplete" if timed_out else "completed" if returncode == 0 else "failed",
            ),
            *tools.values(),
        ],
        gaps=gaps,
    )
