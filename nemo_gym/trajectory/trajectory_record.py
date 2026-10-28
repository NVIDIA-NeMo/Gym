# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for rollout observations and trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.failure_kinds import validate_failure_kind
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseInputItem


NG_AGENT_OBSERVATIONS_KEY = "ng_agent_observations"
# The producer trajectory slot: a harness's own turns, forwarded on a result and replaced by the
# collector's canonical projection.
NG_TRAJECTORY_KEY = "ng_trajectory"
NG_VERIFIER_INVOCATIONS_KEY = "ng_verifier_invocations"


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


class TrajectoryTokenStats(ObservationModel):
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    reasoning_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    cached_tokens: Optional[int] = Field(default=None, ge=0)


class TrajectoryResponseMetadata(ObservationModel):
    response_id: Optional[str] = None
    model_ref: Optional[ModelServerRef] = None
    model: Optional[str] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None
    response_status: Optional[str] = None
    finish_reason: Optional[str] = None
    upstream_attempted: Optional[bool] = None
    response_source: Optional[Literal["upstream", "local"]] = None
    upstream_status_code: Optional[int] = None
    local_response_reason: Optional[str] = None
    error_category: Optional[str] = None
    latency_ttft_ms: Optional[float] = Field(default=None, ge=0)


class TrajectoryModelCall(ObservationModel):
    model_call_id: Optional[str] = None
    client_session_id: Optional[str] = None
    client_assistant_message_id: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = Field(default=None, ge=0)
    request: Optional[Any] = None
    response: Optional[Any] = None
    response_metadata: TrajectoryResponseMetadata = Field(default_factory=TrajectoryResponseMetadata)
    token_stats: TrajectoryTokenStats = Field(default_factory=TrajectoryTokenStats)


class TrajectoryTurn(ObservationModel):
    invocation_id: str
    activation_id: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based activation that produced this turn; omitted for producers without activation IDs.",
    )
    source_message_id: Optional[str] = Field(
        default=None, description="Persisted assistant-message ID in the harness."
    )
    source_model_ref: Optional[ModelServerRef] = Field(
        default=None, description="Configured model server for matching the source message to captured calls."
    )
    task_id: str
    rollout_id: str
    turn_no: int = Field(ge=1, description="Turn number within this invocation.")
    timestamp: float
    question: Optional[Any] = None
    answer: Optional[Any] = None
    reasoning_content: Optional[Any] = None
    resolved: Optional[bool] = None
    step_count: int = Field(ge=0, description="Producer-reported cumulative step count within this invocation.")
    model_calls: list[ModelCallRef] = Field(default_factory=list)

    # Keep the model schema; a serializer return annotation would replace it.
    @model_serializer(mode="wrap")
    def _serialize_optional_metadata(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        if self.activation_id is None:
            result.pop("activation_id", None)
        return result


class AgentInvocation(ObservationModel):
    """One agent activation or subagent invocation observed by a harness.

    A harness's saved session may span several invocations; ``harness_session_id``
    retains that context while ``invocation_id`` owns each activation's evidence.

    Harness adapters should report an iteration-limit event through an
    ``on_iteration_limit_reached(*, invocation_id: str) -> None`` observer hook.
    The hook records ``stop_reason="max_iterations"`` once on that invocation,
    leaving parent invocations, budgets, and summary generation alone. Harnesses
    with native stop events can call it directly; compatibility shims may call it
    at the point where the harness detects its limit.
    """

    kind: Literal["agent_invocation"] = "agent_invocation"
    invocation_id: str
    activation_id: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based activation owning this invocation; omitted for producers without activation IDs.",
    )
    harness_session_id: Optional[str] = Field(
        default=None,
        description="Session ID assigned by the harness, retained as context across activations; omitted when not reported.",
    )
    parent_invocation_id: Optional[str] = None
    spawned_by_tool_call_id: Optional[str] = None
    status: Literal["completed", "failed", "incomplete", "unknown"] = Field(
        default="unknown", description="Harness-reported invocation outcome; unknown when not explicit."
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="Explicit reason this invocation stopped, such as max_iterations; scoped to invocation_id.",
    )
    duration_ms: Optional[float] = Field(default=None, ge=0)
    error_type: Optional[str] = None
    model_calls: list[ModelCallRef] = Field(default_factory=list)
    conversation: list[NeMoGymResponseInputItem] = Field(
        default_factory=list,
        description="Normalized conversation items supported by this producer; gaps describe unavailable evidence.",
    )

    # Keep the model schema; a serializer return annotation would replace it.
    @model_serializer(mode="wrap")
    def _serialize_optional_metadata(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        for key in ("activation_id", "harness_session_id"):
            if getattr(self, key) is None:
                result.pop(key, None)
        return result


class VerifierInvocation(ObservationModel):
    """One verify() execution, its outcome, and all model attempts made while grading.

    Captured calls establish ownership. The shared verify boundary records the
    execution outcome, including for programmatic verifiers with no model calls.
    """

    kind: Literal["verifier_invocation"] = "verifier_invocation"
    invocation_id: str
    status: Literal["completed", "failed", "incomplete", "unknown"] = "unknown"
    failure_kind: Optional[str] = None
    model_calls: list[ModelCallRef] = Field(default_factory=list)

    @field_validator("failure_kind")
    @classmethod
    def validate_failure_kind(cls, value: Optional[str]) -> Optional[str]:
        return validate_failure_kind(value)


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


class TrajectoryToolCall(ToolCallObservation):
    """Tool observation enriched with the model-visible output in a trajectory record."""

    output: Optional[Any] = None


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
    source_message_ids: list[str] = Field(
        default_factory=list, description="Persisted assistant-message IDs explicitly associated with this compaction."
    )
    source_model_ref: Optional[ModelServerRef] = None
    observed_at: Optional[float] = None
    trigger: Optional[str] = None
    tokens_before: Optional[int] = Field(
        default=None,
        ge=0,
        description="Producer-reported token count before compaction; accounting may differ from tokens_after.",
    )
    tokens_after: Optional[int] = Field(
        default=None,
        ge=0,
        description="Producer-reported token count after compaction; accounting may differ from tokens_before.",
    )
    outcome: Literal["completed", "failed", "aborted", "unknown"] = "unknown"
    summary: Optional[str] = None
    first_kept_item_id: Optional[str] = None
    before_model_call: Optional[ModelCallRef] = Field(
        default=None,
        description="Last invocation model call observed before compaction.",
    )
    model_calls: list[ModelCallRef] = Field(
        default_factory=list,
        description=(
            "Invocation-owned model calls used for compaction, joined by explicit identifiers or a unique "
            "producer-specific exact match."
        ),
    )
    after_model_call: Optional[ModelCallRef] = Field(
        default=None,
        description="First invocation model call observed after compaction.",
    )


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
        """Require unique invocation IDs and reject cycles in the observed parent graph."""
        invocation_records = [record for record in self.records if isinstance(record, AgentInvocation)]
        invocations = {record.invocation_id: record for record in invocation_records}
        if len(invocation_records) != len(invocations):
            raise ValueError("invocation_id must be unique within an observation bundle")

        # Missing parents are valid when an opaque producer can observe only part of the invocation tree.
        checked: set[str] = set()
        for invocation_id in invocations:
            chain: set[str] = set()
            current = invocation_id
            while current in invocations and current not in checked:
                if current in chain:
                    raise ValueError("parent_invocation_id must not form a cycle")
                chain.add(current)
                parent = invocations[current].parent_invocation_id
                if parent is None:
                    break
                current = parent
            checked.update(chain)
        return self


class TrajectoryRecord(ObservationModel):
    schema_version: Literal["1.0"] = "1.0"
    task_id: str
    rollout_id: str
    invocations: list[AgentInvocation] = Field(default_factory=list)
    verifier_invocations: list[VerifierInvocation] = Field(default_factory=list)
    turns: list[TrajectoryTurn] = Field(default_factory=list)
    model_calls: list[TrajectoryModelCall] = Field(default_factory=list)
    tool_calls: list[TrajectoryToolCall] = Field(default_factory=list)
    gaps: list[ObservationGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> "TrajectoryRecord":
        invocation_ids = [invocation.invocation_id for invocation in [*self.invocations, *self.verifier_invocations]]
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError("invocation_id must be unique within a trajectory")
        model_call_ids = [call.model_call_id for call in self.model_calls if call.model_call_id]
        if len(model_call_ids) != len(set(model_call_ids)):
            raise ValueError("model_call_id must be unique within a trajectory")
        keys: set[tuple[str, int]] = set()
        for turn in self.turns:
            if turn.task_id != self.task_id or turn.rollout_id != self.rollout_id:
                raise ValueError("turn identity must match the trajectory")
            key = (turn.invocation_id, turn.turn_no)
            if key in keys:
                raise ValueError("turn number must be unique within an invocation")
            keys.add(key)
        return self


@dataclass(frozen=True, slots=True)
class AgentEpisode:
    """An Agent response and the observations available at its execution boundary."""

    response: NeMoGymResponse
    observations: AgentObservationBundle


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
