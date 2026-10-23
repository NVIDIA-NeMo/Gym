# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harness-independent grading contracts, with no runtime or catalog dependencies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


CheckKind = Literal[
    "length",
    "language",
    "element_absent",
    "element_present",
    "ending",
    "exact",
    "json",
    "calls",
    "order",
    "args",
    "after_event",
    "judge",
]
CheckStatus = Literal["pass", "fail", "not_applicable", "error"]
Place = Literal["final", "any_turn", "first_turn", "run", "after_event"]


class Contract(BaseModel):
    """Closed JSON boundaries: unknown fields cannot silently change interpretation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CheckScope(Contract):
    """When a rule takes effect, what it grades, and whether to diagnose its prefix."""

    activation: Literal["session_start", "continuation_start", "event"] = "session_start"
    event_id: str | None = Field(default=None, min_length=1)
    places: list[Place] = Field(default_factory=lambda: ["run"], min_length=1)
    check_prefix: bool = False

    @model_validator(mode="after")
    def validate_activation(self) -> "CheckScope":
        if (self.activation == "event") != (self.event_id is not None):
            raise ValueError("event activation requires event_id; other activations must omit it")
        if len(set(self.places)) != len(self.places):
            raise ValueError("grading places must be unique")
        return self


class InstructionContext(Contract):
    """Exact authored instruction and its carrier, separate from the private rubric."""

    text: str = Field(min_length=1)
    at: str = Field(min_length=1)
    carrier: str | None = None

    @model_validator(mode="after")
    def validate_text_and_placement(self) -> "InstructionContext":
        if not self.text.strip():
            raise ValueError("instruction text must not be blank")
        if self.at not in {"system_prompt", "task_message", "user_turn", "tool_output"} and not (
            self.at.startswith("tool:") and len(self.at) > 5
        ):
            raise ValueError("instruction context needs a known placement or tool:<exposed name>")
        return self


class ResolvedCheck(Contract):
    """A ready-to-evaluate criterion, independent of its author or catalog source."""

    id: str = Field(min_length=1)
    kind: CheckKind
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    rubric: str | None = None
    instruction: InstructionContext | None = None
    scope: CheckScope = Field(default_factory=CheckScope)
    taxonomy: list[str] = Field(default_factory=list)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rubric(self) -> "ResolvedCheck":
        if self.kind == "judge" and (self.rubric is None or not self.rubric.strip()):
            raise ValueError("judge check requires a nonempty literal rubric")
        return self


class ToolCall(Contract):
    """Model-visible call and normalized execution facts; neither replaces the other."""

    call_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    model_tool: str = Field(min_length=1)
    args: dict[str, JsonValue]
    model_args: JsonValue
    rejected: bool = False
    changed_repo: bool | None = None
    changed_paths: list[str] | None = None
    created_paths: list[str] | None = None
    test_output_paths: list[str] | None = None


class SessionStep(Contract):
    """One visible message; all calls emitted in that message stay grouped."""

    id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    role: Literal["system", "user", "assistant", "tool"]
    text: str = ""
    calls: list[ToolCall] = Field(default_factory=list)
    call_id: str | None = None
    invocation_id: str | None = None
    tool: str | None = None
    output: str | None = None
    error: bool = False
    planted: bool = False
    workspace_diff: str | None = None


class SessionInfo(Contract):
    """Native session identity, including real delegated child sessions."""

    id: str = Field(min_length=1)
    parent_id: str | None = None


class EventReceipt(Contract):
    """An observed activation, not merely an instruction scheduled by the input."""

    id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    call_id: str | None = None
    invocation_id: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class SessionEvidence(Contract):
    """Ordered actor evidence plus completeness, provenance, and activation receipts."""

    status: Literal["complete", "incomplete", "infrastructure_error"]
    steps: list[SessionStep]
    sessions: list[SessionInfo] = Field(default_factory=list)
    events: list[EventReceipt] = Field(default_factory=list)
    continuation_start: int | None = Field(default=None, ge=0)
    issues: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    facts: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_receipts(self) -> "SessionEvidence":
        if self.status != "complete" and not self.issues:
            raise ValueError("incomplete evidence requires an issue explaining the missing observations")
        event_ids = [event.id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event IDs must be unique")
        if any(event.step_index >= len(self.steps) for event in self.events):
            raise ValueError("event step_index must reference a captured step")
        if self.continuation_start is not None and self.continuation_start > len(self.steps):
            raise ValueError("continuation_start is outside the captured trace")
        calls: set[tuple[str, str, str]] = set()
        messages: set[tuple[str, str]] = set()
        for step in self.steps:
            identity = (step.session_id, step.id)
            if identity in messages:
                raise ValueError("message IDs must be unique within a session")
            messages.add(identity)
            for call in step.calls:
                key = (step.session_id, call.invocation_id, call.call_id)
                if key in calls:
                    raise ValueError("call IDs must be unique within a session/invocation")
                calls.add(key)
        return self


class JudgeDecision(Contract):
    """Structured judgment without conflating inapplicability and infrastructure errors."""

    status: CheckStatus
    explanation: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CheckResult(Contract):
    """One criterion's outcome and the evidence supporting it."""

    id: str
    status: CheckStatus
    explanation: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class IFResult(Contract):
    """Instruction-following scores only; task correctness belongs to the task verifier."""

    checks: list[CheckResult]
    if_all: Literal[0, 1] | None
    if_frac: float | None = Field(ge=0, le=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
