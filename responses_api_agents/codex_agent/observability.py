# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read exact tool identities and model-visible results from a native Codex session."""

import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from nemo_gym.openai_utils import NeMoGymResponseInputItem
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap, ToolCallObservation


_ITEM = TypeAdapter(NeMoGymResponseInputItem)


def read_codex_observations(
    home: Path, invocation_id: str, *, returncode: int | None, timed_out: bool
) -> AgentObservationBundle:
    """Use native call IDs; never join CLI display item IDs to calls by ordering."""
    conversation = []
    tools = {}
    gaps = []
    files = list(home.glob("sessions/**/*.jsonl"))
    for path in files:
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("expected a native session record")
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
    if not files:
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
