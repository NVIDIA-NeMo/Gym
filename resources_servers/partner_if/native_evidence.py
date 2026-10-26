# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allowlisted visible evidence from native Gym trajectories and artifact receipts."""

import json

from pydantic import Field, JsonValue

from resources_servers.partner_if.evidence import Artifact, EvidenceBundle, VisibleEvent
from resources_servers.partner_if.grading_contract import StrictModel


class LiveAgentCapture(StrictModel):
    """What a live agent captured outside its response; it sends this to `/verify` as `verification_input`."""

    # The input the harness gave the actor, which must be the row's frozen input.
    input: list[dict[str, JsonValue]]
    documents: dict[str, str] = Field(default_factory=dict)
    # Capture receipts by artifact name, read by captured_artifact.
    artifacts: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)


def _text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if not isinstance(part, dict) or part.get("type") not in {"input_text", "output_text", "refusal"}:
                raise ValueError("Unsupported visible message content")
            text = part.get("refusal") if part.get("type") == "refusal" else part.get("text")
            if not isinstance(text, str):
                raise ValueError("Unsupported nontext message content")
            parts.append(text)
        return "\n".join(parts)
    raise ValueError("Unsupported visible text representation")


def visible_events(output: list[dict[str, JsonValue]]) -> list[VisibleEvent]:
    """Retain ordered messages/calls/results, not reasoning, tokens or model identity.

    The caller supplies Gym Responses output (Harbor already converts ATIF). Unexpected
    visible types fail closed rather than yielding an apparently complete partial trace.
    """
    events = []
    for index, item in enumerate(output):
        kind = item.get("type", "message")
        if kind == "reasoning":
            continue
        if kind == "message":
            role = item.get("role")
            if role not in {"assistant", "system", "developer", "user", "tool"}:
                raise ValueError("Unsupported message role")
            events.append(
                VisibleEvent(
                    source_index=index,
                    kind="assistant" if role == "assistant" else "context",
                    role=role,
                    text=_text(item.get("content")),
                )
            )
        elif kind == "function_call":
            arguments = item.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass  # Malformed agent arguments are still evidence.
            events.append(
                VisibleEvent(
                    source_index=index,
                    kind="tool_call",
                    tool_name=item.get("name"),
                    call_id=item.get("call_id"),
                    arguments=arguments,
                )
            )
        elif kind == "function_call_output":
            events.append(
                VisibleEvent(
                    source_index=index, kind="tool_result", call_id=item.get("call_id"), text=_text(item.get("output"))
                )
            )
        else:
            raise ValueError(f"Unsupported visible event: {kind}")
    return events


def captured_artifact(receipt: dict[str, JsonValue]) -> Artifact:
    """Translate a capture receipt without guessing that failed capture means absence."""
    if receipt.get("error"):
        return Artifact(status="error", error=str(receipt["error"]))
    if receipt.get("exists") is False:
        return Artifact(status="absent")
    if receipt.get("exists") is True and isinstance(receipt.get("text"), str):
        return Artifact(status="present", text=receipt["text"])
    return Artifact(status="error", error="Missing or invalid artifact capture receipt")


def live_agent_evidence(
    params: dict[str, JsonValue], output: list[dict[str, JsonValue]], capture: LiveAgentCapture | None
) -> EvidenceBundle:
    """A live agent's visible trajectory, with the documents and artifacts it captured."""
    if capture is None:
        raise ValueError("A live_agent row needs the evidence its agent captured")
    if capture.input != params["input"]:
        raise ValueError("The actor's input differs from the frozen row")
    events = visible_events(output)
    return EvidenceBundle(
        context=params["input"],
        instructions=params.get("instructions"),
        tools=params.get("tools") or [],
        documents=capture.documents,
        trajectory=events,
        answer="\n".join(event.text for event in events if event.kind == "assistant"),
        artifacts={name: captured_artifact(receipt) for name, receipt in capture.artifacts.items()},
    )
