# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small shared contract for observations exposed by Agent integrations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseInputItem


if TYPE_CHECKING:
    from nemo_gym.base_responses_api_model import ModelCallRecord


class ObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelCallRef(ObservationModel):
    """Stable identifiers an Agent integration can observe for one model call."""

    model_call_id: Optional[str] = None
    model_ref: Optional[ModelServerRef] = None
    response_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_join_key(self) -> "ModelCallRef":
        if not self.model_call_id and not (self.model_ref is not None and self.response_id):
            raise ValueError("model_call_id or both model_ref and response_id are required")
        return self


class AgentInvocation(ObservationModel):
    """One root Agent or subagent conversation observed by a harness."""

    kind: Literal["agent_invocation"] = "agent_invocation"
    invocation_id: str
    parent_invocation_id: Optional[str] = None
    spawned_by_tool_call_id: Optional[str] = None
    status: Literal["completed", "failed", "incomplete", "unknown"] = Field(
        default="unknown", description="Harness-reported invocation outcome; unknown when not explicit."
    )
    duration_ms: Optional[float] = Field(default=None, ge=0)
    error_type: Optional[str] = None
    model_calls: list[ModelCallRef] = Field(default_factory=list)
    conversation: list[NeMoGymResponseInputItem] = Field(
        default_factory=list,
        description="Normalized conversation items supported by this producer; gaps describe unavailable evidence.",
    )


class ToolCallObservation(ObservationModel):
    """Timing observed for one tool call at an Agent-owned boundary."""

    kind: Literal["tool_call"] = "tool_call"
    invocation_id: str
    tool_call_id: str
    sandbox_id: Optional[str] = Field(
        default=None,
        description="Enclosing sandbox instance, shared by concurrent calls; not per-call resource attribution.",
    )
    tool_name: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = Field(default=None, ge=0)
    timing_source: Optional[Literal["executor", "artifact", "harness"]] = None
    status: Literal["completed", "failed", "timeout", "cancelled", "incomplete", "unknown"] = "unknown"
    error_type: Optional[str] = None

    @model_validator(mode="after")
    def validate_timing(self) -> "ToolCallObservation":
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self


class SandboxObservation(ObservationModel):
    """Outcome and lifetime resource usage reported by a sandbox-owning harness."""

    kind: Literal["sandbox"] = "sandbox"
    role: Literal["agent", "verifier", "environment"]
    provider: Optional[str] = None
    sandbox_id: Optional[str] = None
    outcome: Literal["completed", "failed", "timeout", "oom", "sandbox_error", "cancelled", "unknown"] = "unknown"
    exit_code: Optional[int] = None
    wall_time_s: Optional[float] = Field(default=None, ge=0)
    cpu_time_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cumulative CPU time for the sandbox, never an allocation or per-tool estimate.",
    )
    peak_memory_mib: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured sandbox high-water mark, never its configured memory limit.",
    )
    resource_usage_source: Optional[str] = None
    error_type: Optional[str] = None


class ContextCompactionObservation(ObservationModel):
    """An explicit context-compaction event reported by the Agent harness."""

    kind: Literal["context_compaction"] = "context_compaction"
    invocation_id: str
    observed_at: Optional[float] = None
    trigger: Optional[str] = None
    tokens_before: Optional[int] = None
    tokens_after: Optional[int] = None
    outcome: Literal["completed", "failed", "aborted", "unknown"] = "unknown"
    summary: Optional[str] = None
    first_kept_item_id: Optional[str] = None
    before_model_call: Optional[ModelCallRef] = None
    after_model_call: Optional[ModelCallRef] = None


class ObservationGap(ObservationModel):
    """A fact that the selected integration could not observe or join exactly."""

    code: str
    invocation_id: Optional[str] = None
    detail: Optional[str] = None


AgentObservationRecord = Annotated[
    AgentInvocation | ToolCallObservation | ContextCompactionObservation | SandboxObservation,
    Field(discriminator="kind"),
]


class AgentObservationBundle(ObservationModel):
    """Normalized observations returned by one Agent Server for one rollout."""

    source: str
    records: list[AgentObservationRecord] = Field(
        default_factory=list,
        description="Unordered typed records; list position does not imply execution order.",
    )
    gaps: list[ObservationGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> "AgentObservationBundle":
        invocation_records = [record for record in self.records if isinstance(record, AgentInvocation)]
        invocations = {record.invocation_id: record for record in invocation_records}
        if len(invocation_records) != len(invocations):
            raise ValueError("invocation_id must be unique within an observation bundle")

        resolved: set[str] = set()
        for invocation_id in invocations:
            chain: set[str] = set()
            current = invocation_id
            while current in invocations and current not in resolved:
                if current in chain:
                    raise ValueError("parent_invocation_id must not form a cycle")
                chain.add(current)
                parent = invocations[current].parent_invocation_id
                if parent is None:
                    break
                current = parent
            resolved.update(chain)
        return self


def join_model_call_observations(
    bundle: AgentObservationBundle,
    calls: Iterable[ModelCallRecord],
) -> AgentObservationBundle:
    """Resolve harness call references against captured model calls without guessing ownership."""

    result = bundle.model_copy()
    result.records = [
        record.model_copy() if isinstance(record, (AgentInvocation, ContextCompactionObservation)) else record
        for record in bundle.records
    ]
    invocations = [record for record in result.records if isinstance(record, AgentInvocation)]
    compactions = [record for record in result.records if isinstance(record, ContextCompactionObservation)]
    captured = list(calls)
    by_call_id: dict[str, list[ModelCallRecord]] = {}
    by_response: dict[tuple[str, str, str], list[ModelCallRecord]] = {}
    for call in captured:
        if call.model_call_id:
            by_call_id.setdefault(call.model_call_id, []).append(call)
        if call.model_ref is not None and call.response_id:
            key = (call.model_ref.type, call.model_ref.name, call.response_id)
            by_response.setdefault(key, []).append(call)

    def matches(ref: ModelCallRef) -> list[ModelCallRecord]:
        if ref.model_call_id:
            candidates = by_call_id.get(ref.model_call_id, [])
            return [
                call
                for call in candidates
                if (ref.model_ref is None or ref.model_ref == call.model_ref)
                and (ref.response_id is None or ref.response_id == call.response_id)
            ]
        assert ref.model_ref is not None and ref.response_id is not None
        return by_response.get((ref.model_ref.type, ref.model_ref.name, ref.response_id), [])

    def canonical(ref: ModelCallRef, call: ModelCallRecord) -> ModelCallRef:
        return ModelCallRef.model_validate(
            {
                "model_call_id": call.model_call_id,
                "model_ref": call.model_ref,
                "response_id": call.response_id,
            }
        )

    join_codes = {
        "model_call_reference_ambiguous",
        "model_call_reference_conflict",
        "model_call_reference_unmatched",
    }
    result.gaps = [
        gap
        for gap in bundle.gaps
        if gap.code not in join_codes
        and not (
            captured
            and gap.code == "model_call_ownership_unavailable"
            and gap.invocation_id is None
            and (gap.detail is None or gap.detail.startswith("capture:"))
        )
    ]

    claimed: set[int] = set()
    join_gaps: list[ObservationGap] = []
    for invocation in invocations:
        resolved: list[ModelCallRef] = []
        for ref in invocation.model_calls:
            candidates = matches(ref)
            detail = ref.model_call_id or ref.response_id
            if len(candidates) != 1:
                join_gaps.append(
                    ObservationGap(
                        code=("model_call_reference_ambiguous" if candidates else "model_call_reference_unmatched"),
                        invocation_id=invocation.invocation_id,
                        detail=detail,
                    )
                )
                resolved.append(ref)
                continue

            call = candidates[0]
            identity = id(call)
            if identity in claimed:
                join_gaps.append(
                    ObservationGap(
                        code="model_call_reference_conflict",
                        invocation_id=invocation.invocation_id,
                        detail=call.model_call_id or call.response_id,
                    )
                )
                resolved.append(ref)
                continue
            claimed.add(identity)
            resolved.append(canonical(ref, call))
        invocation.model_calls = resolved

    for compaction in compactions:
        for field_name in ("before_model_call", "after_model_call"):
            ref = getattr(compaction, field_name)
            if ref is None:
                continue
            candidates = matches(ref)
            if len(candidates) == 1:
                setattr(compaction, field_name, canonical(ref, candidates[0]))
            else:
                join_gaps.append(
                    ObservationGap(
                        code=("model_call_reference_ambiguous" if candidates else "model_call_reference_unmatched"),
                        invocation_id=compaction.invocation_id,
                        detail=f"{field_name}:{ref.model_call_id or ref.response_id}",
                    )
                )

    result.gaps.extend(join_gaps)
    for call in captured:
        if id(call) not in claimed:
            result.gaps.append(
                ObservationGap(
                    code="model_call_ownership_unavailable",
                    detail=f"capture:{call.model_call_id or call.response_id or 'unknown'}:call_index={call.call_index}",
                )
            )
    result.gaps = list({(gap.code, gap.invocation_id, gap.detail): gap for gap in result.gaps}.values())
    return result


@dataclass(frozen=True, slots=True)
class AgentEpisode:
    """An Agent response and the observations available at its execution boundary."""

    response: NeMoGymResponse
    observations: AgentObservationBundle
