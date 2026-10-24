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
"""Convert selected canonical session traces using the pinned Harbor converters.

Conversion is for Gym interoperability, not scoring. Keep the original trace;
Harbor conversion can omit metadata. This wrapper rejects dropped tool results.
Session selection and child-lineage validation belong to the collecting adapter.
"""

import hashlib
import json
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from harbor.agents.installed.codex import Codex
from harbor.agents.installed.hermes import Hermes
from harbor.agents.installed.pi import Pi
from harbor.models.trajectories import Step, Trajectory

from environments.skill_trigger.scoring import EvidenceError


VERSIONS = {"pi": "1.3.0", "codex": "0.156.1", "hermes": "0.22.2"}


def records(text: str) -> list[dict[str, object]]:
    """Parse complete JSONL records, rejecting malformed or non-object evidence."""
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise EvidenceError("Canonical trace contains a non-object record")
        result.append(record)
    if not result:
        raise EvidenceError("Empty canonical trace")
    return result


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceError("Expected an object in the canonical trace")
    return value


def convert_trace(
    source: Path,
    *,
    harness: Literal["pi", "codex", "hermes"],
    session_id: str,
    model: str,
    system_prompt: str | None = None,
) -> Trajectory:
    """Convert one explicitly selected session; do not select a root by filename.

    Pi inputs are its SDK event stream; Codex inputs are the selected session's
    rollout JSONL; Hermes inputs are its session export. The supplied session ID
    must already be bound to the trial by the adapter. No inference is performed.
    """
    if version("harbor") != "0.24.0":
        raise EvidenceError("Trajectory conversion requires Harbor 0.24.0")
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    events = records(text)
    expected_outputs = []
    expected_calls = []
    with tempfile.TemporaryDirectory(prefix="skill-trigger-trace-") as directory:
        temp = Path(directory)
        if harness == "pi":
            # Adapt SDK envelopes to Harbor's session-log envelopes. All message
            # bodies and tool results below come from actual recorded events.
            adapted = [{"type": "session", "id": session_id}]
            for event in events:
                if event.get("type") == "message_end":
                    message = object_value(event.get("message"))
                    if message.get("role") == "toolResult":
                        continue  # represented by its tool_execution_end below
                    if message.get("role") == "assistant":
                        content = message.get("content")
                        if isinstance(content, list):
                            expected_calls.extend(
                                object_value(b).get("id") for b in content if object_value(b).get("type") == "toolCall"
                            )
                    adapted.append({"type": "message", "message": message, "timestamp": event.get("observed_at")})
                elif event.get("type") == "tool_execution_end":
                    call_id = event.get("toolCallId")
                    if not isinstance(call_id, str) or not call_id:
                        raise EvidenceError("Pi tool result has no call ID")
                    expected_outputs.append(call_id)
                    result = object_value(event.get("result"))
                    adapted.append(
                        {
                            "type": "message",
                            "timestamp": event.get("observed_at"),
                            "message": {
                                "role": "toolResult",
                                "toolCallId": call_id,
                                "toolName": event.get("toolName"),
                                "content": result.get("content"),
                                "isError": event.get("isError", False),
                            },
                        }
                    )
            agent = Pi(logs_dir=temp, model_name=model, version=VERSIONS[harness])
            trajectory = agent._convert_session_events_to_trajectory(list(enumerate(adapted, start=1)))
        elif harness == "codex":
            owner = object_value(events[0].get("payload"))
            if events[0].get("type") != "session_meta" or owner.get("id") != session_id:
                raise EvidenceError("Codex trace does not begin with the selected owning session")
            if owner.get("cli_version") != VERSIONS[harness]:
                raise EvidenceError("Codex trace version differs from the pinned benchmark")
            for event in events:
                payload = object_value(event.get("payload"))
                if event.get("type") == "turn_context" and payload.get("model") != model:
                    raise EvidenceError("Codex session model differs from the requested model")
                if event.get("type") == "response_item" and payload.get("type") in {
                    "function_call",
                    "custom_tool_call",
                }:
                    expected_calls.append(payload.get("call_id"))
                if event.get("type") == "response_item" and payload.get("type") in {
                    "function_call_output",
                    "custom_tool_call_output",
                }:
                    expected_outputs.append(payload.get("call_id"))
            # Harbor selects max(*.jsonl). Give it exactly the selected session,
            # never a directory containing a root and arbitrary child rollouts.
            (temp / "selected.jsonl").write_bytes(raw)
            agent = Codex(logs_dir=temp, model_name=model, version=VERSIONS[harness])
            trajectory = agent._convert_events_to_trajectory(temp)
        else:
            messages = []
            for event in events:
                embedded = event.get("messages")
                if embedded is None:
                    messages.append(event)
                elif isinstance(embedded, list):
                    if event.get("id") is not None and event["id"] != session_id:
                        raise EvidenceError("Hermes export belongs to a different session")
                    messages.extend(object_value(m) for m in embedded)
                else:
                    raise EvidenceError("Hermes messages must be an array")
            expected_outputs = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]
            for message in messages:
                calls = message.get("tool_calls")
                if message.get("role") == "assistant" and isinstance(calls, list):
                    expected_calls.extend(object_value(call).get("id") for call in calls)
            agent = Hermes(logs_dir=temp, model_name=model, version=VERSIONS[harness])
            trajectory = agent._convert_hermes_session_to_atif(text, session_id)
        if trajectory is None or trajectory.session_id != session_id:
            raise EvidenceError("Harbor did not produce the selected session's trajectory")
    if any(not isinstance(call, str) or not call for call in expected_outputs) or len(expected_outputs) != len(
        set(expected_outputs)
    ):
        raise EvidenceError("Missing or duplicate canonical tool-result ID")
    if any(not isinstance(call, str) or not call for call in expected_calls) or len(expected_calls) != len(
        set(expected_calls)
    ):
        raise EvidenceError("Missing or duplicate canonical tool-call ID")
    converted_calls = [call.tool_call_id for step in trajectory.steps for call in step.tool_calls or []]
    if sorted(expected_calls) != sorted(converted_calls) or not set(expected_outputs).issubset(expected_calls):
        raise EvidenceError("Harbor conversion omitted, duplicated, or invented a tool call")
    converted_outputs = [
        r.source_call_id for step in trajectory.steps if step.observation for r in step.observation.results
    ]
    if sorted(expected_outputs) != sorted(converted_outputs):
        raise EvidenceError("Harbor conversion omitted, duplicated, or invented a tool result")
    if system_prompt is not None:
        trajectory.steps.insert(0, Step(step_id=1, source="system", message=system_prompt))
        for index, step in enumerate(trajectory.steps, start=1):
            step.step_id = index
    trajectory.trajectory_id = session_id
    trajectory.extra = (trajectory.extra or {}) | {
        "canonical_trace_sha256": hashlib.sha256(raw).hexdigest(),
        "conversion_library": "harbor==0.24.0",
        "scoring_source": "canonical native evidence, not this converted trajectory",
    }
    trajectory.notes = ((trajectory.notes + "\n") if trajectory.notes else "") + (
        "Converted using Harbor's pinned harness converter. Preserve the canonical trace: "
        "metadata and content representation may be lossy. This file does not certify skill opening."
    )
    return Trajectory.model_validate(trajectory.model_dump())
