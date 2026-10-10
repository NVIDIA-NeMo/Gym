# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for rollout observations and trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseInputItem


if TYPE_CHECKING:
    from nemo_gym.base_responses_api_model import ModelCallRecord


# Request headers a harness may send with a model call so the model server's capture record can be joined to the
# harness's own record exactly. Evidence only, never authentication; absence is fail-safe.
CLIENT_CALL_ID_HEADER = "x-ng-client-call-id"  # harness-minted id of one logical call, same across its retries
CLIENT_ATTEMPT_HEADER = "x-ng-client-attempt"  # 1-based attempt number within the logical call
CLIENT_SENT_AT_HEADER = "x-ng-client-sent-at"  # epoch milliseconds on the sender's clock, just before sending
CLIENT_AGENT_HEADER = "x-ng-agent"  # harness agent or role name (OpenCode: build, plan, compaction, ...)
CLIENT_PURPOSE_HEADER = "x-ng-purpose"  # harness-declared purpose, one of MODEL_CALL_PURPOSES

MODEL_CALL_PURPOSES = (
    "agent_step",
    "subagent_step",
    "compaction_summary",
    "compaction_questions",
    "compaction_answers",
    "compaction_fallback",
)
COMPACTION_PURPOSES = frozenset(purpose for purpose in MODEL_CALL_PURPOSES if purpose.startswith("compaction"))

ModelCallPurpose = Literal[
    "agent_step",
    "subagent_step",
    "compaction_summary",
    "compaction_questions",
    "compaction_answers",
    "compaction_fallback",
]
ModelCallStatus = Literal["ok", "error", "timeout", "cancelled", "incomplete"]

_CLOCK_DOMAIN_DESCRIPTION = (
    "Clock the timestamps of this record were read from: node:<host> for a Gym process, sandbox:<id> for times "
    "copied from an artifact the sandbox wrote. Never subtract timestamps across domains."
)


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


class TrajectoryEngineMetrics(ObservationModel):
    """Serving-engine timings for one request, as the engine reports them (vLLM per-request metrics).

    Durations only: vLLM does not expose the absolute queued/scheduled timestamps, so
    model_process_start cannot be anchored exactly; model_engine_queue_time is exact.
    """

    queue_time_ms: Optional[float] = Field(default=None, ge=0, description="Admission to first scheduling.")
    time_to_first_token_ms: Optional[float] = Field(
        default=None, ge=0, description="First scheduling to first output token (prefill), engine-side."
    )
    generation_time_ms: Optional[float] = Field(default=None, ge=0, description="First to last output token.")
    mean_itl_ms: Optional[float] = Field(default=None, ge=0)
    tokens_per_second: Optional[float] = Field(default=None, ge=0)
    speculative_decoding: Optional[dict[str, Any]] = Field(
        default=None, description="Engine-reported draft/acceptance statistics; experimental shape."
    )


class TrajectoryResponseMetadata(ObservationModel):
    response_id: Optional[str] = None
    model_ref: Optional[ModelServerRef] = None
    model: Optional[str] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None
    response_status: Optional[str] = None
    finish_reason: Optional[str] = None
    error_category: Optional[str] = None
    latency_ttft_ms: Optional[float] = Field(default=None, ge=0)
    engine: Optional[TrajectoryEngineMetrics] = None


class TrajectoryModelCallAttempt(ObservationModel):
    """One HTTP attempt inside a logical model call. Retries are separate attempts, never folded.

    Durations are taken by the process that made the request. ``completed`` with a non-2xx
    ``status_code`` is an attempt the server answered but the client then retried (e.g. 429).
    """

    attempt_index: int = Field(ge=1, description="1-based position within the logical call.")
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = Field(default=None, ge=0)
    status: Literal["completed", "error", "timeout", "cancelled"]
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    clock_domain: Optional[str] = Field(default=None, description=_CLOCK_DOMAIN_DESCRIPTION)

    @model_validator(mode="after")
    def validate_timing(self) -> "TrajectoryModelCallAttempt":
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self


class TrajectoryModelCall(ObservationModel):
    model_call_id: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = Field(default=None, ge=0)
    request: Optional[Any] = None
    response: Optional[Any] = None
    response_metadata: TrajectoryResponseMetadata = Field(default_factory=TrajectoryResponseMetadata)
    token_stats: TrajectoryTokenStats = Field(default_factory=TrajectoryTokenStats)
    model_call_purpose: Optional[ModelCallPurpose] = None
    model_response_kind: Optional[Literal["tool_call", "text", "other"]] = None
    attempts: list[TrajectoryModelCallAttempt] = Field(
        default_factory=list,
        description=(
            "Every request attempt behind this logical call, in order. Empty means the producer had no "
            "attempt visibility, not that there was exactly one attempt."
        ),
    )
    # Latency fields. started_at / completed_at / duration_ms above are the logical call as the harness saw it,
    # retries included; the model server's own bounds for the attempt that answered sit beside them.
    clock_domain: Optional[str] = Field(default=None, description=_CLOCK_DOMAIN_DESCRIPTION)
    status: Optional[ModelCallStatus] = Field(
        default=None, description="Outcome of the logical call as the harness saw it; incomplete = rejected request."
    )
    client_call_id: Optional[str] = Field(
        default=None, description="The id the harness sent as x-ng-client-call-id; equals model_call_id when minted."
    )
    invocation_id: Optional[str] = Field(default=None, description="Conversation (session) this call belongs to.")
    server_started_at: Optional[float] = None
    server_completed_at: Optional[float] = None
    server_duration_ms: Optional[float] = Field(default=None, ge=0, description="Model server's own duration.")
    processed_at: Optional[float] = Field(
        default=None,
        description=(
            "When the harness finished handling the response (Terminus: decision parsed; OpenCode: step finished). "
            "The gap after completed_at is harness processing, not model time."
        ),
    )
    parsed_tool_calls: Optional[int] = Field(
        default=None, ge=0, description="Tool actions the harness parsed out of a text response (Terminus commands)."
    )
    task_complete: Optional[bool] = Field(default=None, description="The response declared the task complete.")
    parse_error: Optional[bool] = Field(
        default=None,
        description="The harness could not parse the response into actions (Terminus: Harbor's parser reported an "
        "error, so no command batch ran for this decision).",
    )
    context_overflow: Optional[bool] = Field(
        default=None, description="The request was rejected for exceeding the context window."
    )
    rejected_request_tokens: Optional[int] = Field(
        default=None, ge=0, description="Size of a rejected request as the engine reported it."
    )

    @model_validator(mode="after")
    def validate_timing(self) -> "TrajectoryModelCall":
        for start, end in (
            (self.started_at, self.completed_at),
            (self.server_started_at, self.server_completed_at),
        ):
            if start is not None and end is not None and end < start:
                raise ValueError("completed_at must not precede started_at")
        return self


class TrajectoryTurn(ObservationModel):
    invocation_id: str
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


class AgentInvocation(ObservationModel):
    """One root Agent or subagent conversation observed by a harness."""

    kind: Literal["agent_invocation"] = "agent_invocation"
    invocation_id: str
    parent_invocation_id: Optional[str] = None
    spawned_by_tool_call_id: Optional[str] = None
    status: Literal["completed", "failed", "incomplete", "unknown"] = Field(
        default="unknown", description="Harness-reported invocation outcome; unknown when not explicit."
    )
    started_at: Optional[float] = Field(
        default=None,
        description=(
            "Wall clock when the task prompt reached the initialized agent, for placing this invocation on a "
            "shared timeline. Durations should come from duration_ms, which a producer measures monotonically."
        ),
    )
    completed_at: Optional[float] = Field(
        default=None, description="Wall clock when agent execution terminated and its output was available."
    )
    duration_ms: Optional[float] = Field(default=None, ge=0)
    error_type: Optional[str] = None
    model_calls: list[ModelCallRef] = Field(default_factory=list)
    conversation: list[NeMoGymResponseInputItem] = Field(
        default_factory=list,
        description="Normalized conversation items supported by this producer; gaps describe unavailable evidence.",
    )
    clock_domain: Optional[str] = Field(default=None, description=_CLOCK_DOMAIN_DESCRIPTION)
    requested_at: Optional[float] = Field(
        default=None, description="Subagents: when the parent held the spawning tool call (its part was created)."
    )
    response_observed_at: Optional[float] = Field(
        default=None, description="Subagents: when the parent observed the spawning tool's result."
    )
    process_started_at: Optional[float] = Field(
        default=None, description="Root invocation of a process-based harness: the process was launched."
    )
    process_exited_at: Optional[float] = Field(
        default=None, description="Root invocation of a process-based harness: the process exited."
    )

    @model_validator(mode="after")
    def validate_timing(self) -> "AgentInvocation":
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self


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
    operation: Optional[str] = None
    requested_at: Optional[float] = Field(
        default=None,
        description=(
            "When the harness held the complete, parsed tool call. tool_dispatch_delay = started_at - requested_at. "
            "Same clock as started_at (see timing_source); not range-validated so odd artifact clocks report "
            "rather than fail the bundle."
        ),
    )
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    response_received_at: Optional[float] = Field(
        default=None,
        description=(
            "When the tool result was available for the next model call. "
            "tool_observation_delay = response_received_at - completed_at."
        ),
    )
    duration_ms: Optional[float] = Field(default=None, ge=0)
    timing_source: Optional[Literal["executor", "artifact", "harness"]] = None
    status: Literal["completed", "failed", "timeout", "cancelled", "incomplete", "unknown"] = "unknown"
    error_type: Optional[str] = None
    clock_domain: Optional[str] = Field(default=None, description=_CLOCK_DOMAIN_DESCRIPTION)
    batch_id: Optional[str] = Field(
        default=None, description="Harnesses that run several tool calls from one decision share this id among them."
    )
    batch_index: Optional[int] = Field(default=None, ge=1, description="1-based position within the batch.")
    requested_wait_s: Optional[float] = Field(
        default=None, ge=0, description="Wait the model asked for after sending the command (Terminus)."
    )
    output_trajectory_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Index in the rollout's response.output of the item that carried this tool's result to the model.",
    )
    output_chars: Optional[int] = Field(default=None, ge=0, description="Length of the tool result the model read.")

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
    clock_offset_s: Optional[float] = Field(
        default=None,
        description=(
            "sandbox_clock - harness_clock at sandbox start, estimated from one round trip. Add it to "
            "artifact-timed spans (timing_source=artifact) to place them on the harness timeline."
        ),
    )
    clock_offset_uncertainty_s: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Half the round-trip time of the probe. A cross-process gap smaller than this is not a measured delay."
        ),
    )


class ContextCompactionObservation(ObservationModel):
    """An explicit context-compaction event reported by the Agent harness."""

    kind: Literal["context_compaction"] = "context_compaction"
    invocation_id: str
    observed_at: Optional[float] = None
    completed_at: Optional[float] = None
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
    clock_domain: Optional[str] = Field(default=None, description=_CLOCK_DOMAIN_DESCRIPTION)
    duration_ms: Optional[float] = Field(default=None, ge=0, description="Monotonic duration when harness-timed.")
    rejected_request_tokens: Optional[int] = Field(
        default=None, ge=0, description="For an overflow compaction: the rejected request's size per the engine."
    )


class AgentPhaseObservation(ObservationModel):
    """One lifecycle phase the harness or agent server went through around the agent run.

    Names in use: seed_session, connect_sandbox, start_sandbox, agent_setup, agent_execute, agent_sandbox_exec,
    agent_export, agent_observations_snapshot, verify, sandbox_stop. Phases place pre- and post-agent time.
    """

    kind: Literal["agent_phase"] = "agent_phase"
    name: str
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = Field(default=None, ge=0)
    status: Literal["completed", "failed", "timeout", "cancelled", "incomplete", "unknown"] = "unknown"
    error_type: Optional[str] = None
    clock_domain: Optional[str] = Field(default=None, description=_CLOCK_DOMAIN_DESCRIPTION)

    @model_validator(mode="after")
    def validate_timing(self) -> "AgentPhaseObservation":
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self


class ObservationGap(ObservationModel):
    """A fact that the selected integration could not observe or join exactly."""

    code: str
    invocation_id: Optional[str] = None
    detail: Optional[str] = None


AgentObservationRecord = Annotated[
    AgentInvocation | ToolCallObservation | ContextCompactionObservation | SandboxObservation | AgentPhaseObservation,
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
    turns: list[TrajectoryTurn] = Field(default_factory=list)
    model_calls: list[TrajectoryModelCall] = Field(default_factory=list)
    tool_calls: list[TrajectoryToolCall] = Field(default_factory=list)
    compactions: list[ContextCompactionObservation] = Field(default_factory=list)
    phases: list[AgentPhaseObservation] = Field(default_factory=list)
    gaps: list[ObservationGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> "TrajectoryRecord":
        invocation_ids = [invocation.invocation_id for invocation in self.invocations]
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


def join_model_call_observations(
    bundle: AgentObservationBundle,
    calls: Iterable[ModelCallRecord],
) -> AgentObservationBundle:
    """Resolve harness call references against captured model calls without guessing ownership."""

    result = bundle.model_copy()
    result.records = [
        record.model_copy(update={"model_calls": list(record.model_calls)})
        if isinstance(record, AgentInvocation)
        else record.model_copy()
        if isinstance(record, ContextCompactionObservation)
        else record
        for record in bundle.records
    ]
    invocations = [record for record in result.records if isinstance(record, AgentInvocation)]
    compactions = [record for record in result.records if isinstance(record, ContextCompactionObservation)]
    captured = list(calls)

    invocations_by_id = {invocation.invocation_id: invocation for invocation in invocations}
    reference_keys = {
        invocation.invocation_id: {
            (
                reference.model_call_id,
                (reference.model_ref.type, reference.model_ref.name) if reference.model_ref is not None else None,
                reference.response_id,
            )
            for reference in invocation.model_calls
        }
        for invocation in invocations
    }
    for call in captured:
        if not call.client_session_id or not call.model_call_id:
            continue
        invocation = invocations_by_id.get(call.client_session_id)
        if invocation is None:
            continue
        model_ref_key = (call.model_ref.type, call.model_ref.name) if call.model_ref is not None else None
        matching_keys = {
            (call.model_call_id, None, None),
            (call.model_call_id, model_ref_key, None),
            (call.model_call_id, None, call.response_id),
            (call.model_call_id, model_ref_key, call.response_id),
        }
        if model_ref_key is not None and call.response_id is not None:
            matching_keys.add((None, model_ref_key, call.response_id))
        invocation_keys = reference_keys[invocation.invocation_id]
        if matching_keys & invocation_keys:
            continue
        reference = ModelCallRef(
            model_call_id=call.model_call_id,
            model_ref=call.model_ref,
            response_id=call.response_id,
        )
        invocation.model_calls.append(reference)
        invocation_keys.add((reference.model_call_id, model_ref_key, reference.response_id))

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

    def canonical(call: ModelCallRecord) -> ModelCallRef:
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
            gap.code == "model_call_ownership_unavailable"
            and gap.invocation_id is None
            and gap.detail is not None
            and gap.detail.startswith("capture:")
        )
    ]

    owner_by_call: dict[int, str] = {}
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
            if identity in owner_by_call:
                join_gaps.append(
                    ObservationGap(
                        code="model_call_reference_conflict",
                        invocation_id=invocation.invocation_id,
                        detail=call.model_call_id or call.response_id,
                    )
                )
                resolved.append(ref)
                continue
            owner_by_call[identity] = invocation.invocation_id
            resolved.append(canonical(call))
        invocation.model_calls = resolved

    for compaction in compactions:
        resolved: list[ModelCallRef] = []
        for ref in compaction.model_calls:
            candidates = matches(ref)
            detail = ref.model_call_id or ref.response_id
            if len(candidates) != 1:
                join_gaps.append(
                    ObservationGap(
                        code=("model_call_reference_ambiguous" if candidates else "model_call_reference_unmatched"),
                        invocation_id=compaction.invocation_id,
                        detail=f"model_calls:{detail}",
                    )
                )
                resolved.append(ref)
                continue

            call = candidates[0]
            owner = owner_by_call.get(id(call))
            if owner != compaction.invocation_id:
                join_gaps.append(
                    ObservationGap(
                        code=(
                            "model_call_reference_conflict"
                            if owner is not None
                            else "model_call_ownership_unavailable"
                        ),
                        invocation_id=compaction.invocation_id,
                        detail=f"model_calls:{call.model_call_id or call.response_id}",
                    )
                )
                resolved.append(ref)
                continue
            resolved.append(canonical(call))
        compaction.model_calls = resolved

        for field_name in ("before_model_call", "after_model_call"):
            ref = getattr(compaction, field_name)
            if ref is None:
                continue
            candidates = matches(ref)
            if len(candidates) == 1:
                call = candidates[0]
                owner = owner_by_call.get(id(call))
                if owner is not None and owner != compaction.invocation_id:
                    join_gaps.append(
                        ObservationGap(
                            code="model_call_reference_conflict",
                            invocation_id=compaction.invocation_id,
                            detail=f"{field_name}:{call.model_call_id or call.response_id}",
                        )
                    )
                    continue
                setattr(compaction, field_name, canonical(call))
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
        if id(call) not in owner_by_call:
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
