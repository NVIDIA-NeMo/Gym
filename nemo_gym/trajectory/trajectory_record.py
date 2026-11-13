# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for rollout observations and trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.failure_kinds import FailureStage, validate_failure_kind
from nemo_gym.core.failure_records import FailureReport
from nemo_gym.core.failure_transport import ModelOperationObservation
from nemo_gym.core.operation_timing import OperationTiming
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseInputItem
from nemo_gym.trajectory.model_call_record import tool_request_evidence
from nemo_gym.trajectory.trajectory_format import (
    canonical_tool_record,
    project_bundle_v1,
    project_trajectory_v1,
    reject_v2_fields,
    validate_failure_events,
)


if TYPE_CHECKING:
    from nemo_gym.trajectory.model_call_record import ModelCallRecord


NG_AGENT_OBSERVATIONS_KEY = "ng_agent_observations"
NG_AGENT_SANDBOX_OBSERVATION_KEY = "ng_agent_sandbox_observation"
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


class OutputBudgetRefit(ObservationModel):
    """A model call the server sent again with its output budget lowered, because the budget it requested exceeded
    the engine's whole context window."""

    requested_max_tokens: int = Field(ge=1)
    window: int = Field(ge=1)
    applied_max_tokens: int = Field(ge=1)


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
    failure_report: FailureReport | None = None
    model_operations: list[ModelOperationObservation] | None = None
    error_category: Optional[str] = None
    latency_ttft_ms: Optional[float] = Field(default=None, ge=0)
    # The output budget the call asked for and the one sent after it was lowered to fit the context window.
    requested_max_tokens: Optional[int] = Field(default=None, ge=1)
    applied_max_tokens: Optional[int] = Field(default=None, ge=1)
    output_budget_refit: Optional[OutputBudgetRefit] = None


class TrajectoryModelCall(ObservationModel):
    model_call_id: Optional[str] = None
    agent_instance: Optional[str] = Field(default=None, min_length=1)
    clock_domain: Optional[str] = Field(default=None, min_length=1)
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
    agent_instance: Optional[str] = Field(default=None, min_length=1)
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
    agent_instance: Optional[str] = Field(default=None, min_length=1)
    activation_id: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based activation owning this invocation; omitted for producers without activation IDs.",
    )
    harness_session_id: Optional[str] = Field(
        default=None,
        description="Session ID assigned by the harness, retained as context across activations; omitted when not reported.",
    )
    in_session: bool | None = Field(default=None, strict=True)
    verification_skipped: bool | None = Field(default=None, strict=True)
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


class GradingOperationObservation(ObservationModel):
    """A source-reviewed grading calculation actually executed inside a verifier invocation.

    ``llm_judge`` grades from judge calls made through ``call_judge``; its dependencies are those calls' model and
    response ids, and it completes only when the reviewed call site marked an explicitly parsed verdict.
    """

    operation_id: str = Field(min_length=1)
    kind: Literal["programmatic_calculation", "llm_judge"] = "programmatic_calculation"
    outcome: Literal["completed", "failed", "unknown"] = "unknown"
    result_binding_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dependencies: list[ModelCallRef] | None = None


class VerifierGradingRecord(ObservationModel):
    """The grade a verifier returned without asking any model, recorded by the shared verify path.

    It names the verifier (its resources server), the score and the verifier's verdict on it (``masked`` when the
    verifier marked the score as no valid measurement), and binds the graded response and score like a grading
    operation. A verify execution that sent a model request records none: its grading is observed only through
    its captured calls.
    """

    verifier: str = Field(min_length=1)
    score: float
    verdict: Literal["scored", "masked"]
    result_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def grading_result_binding(result: object) -> str | None:
    """Bind the actual graded response and numeric reward/components, without exporting their payload."""
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    if not isinstance(result, dict) or not isinstance(result.get("response"), dict):
        return None
    response = result["response"]
    if not isinstance(response.get("id"), str) or not response["id"] or not isinstance(response.get("output"), list):
        return None
    reward = result.get("reward")
    if type(reward) not in (int, float) or (isinstance(reward, float) and not math.isfinite(reward)):
        return None
    value = {"response": response, "reward": reward}
    if "reward_components" in result:
        components = result["reward_components"]
        if not isinstance(components, dict) or not all(
            isinstance(key, str)
            and type(number) in (int, float)
            and (not isinstance(number, float) or math.isfinite(number))
            for key, number in components.items()
        ):
            return None
        value["reward_components"] = components

    def canonical(item):
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite grading input")
            return int(item) if item.is_integer() else item
        if isinstance(item, list):
            return [canonical(child) for child in item]
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            return {key: canonical(child) for key, child in item.items()}
        raise ValueError("non-JSON grading input")

    try:
        payload = json.dumps(
            canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None


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
    grading_operations: list[GradingOperationObservation] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    grading_record: VerifierGradingRecord | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    # Owner-observed lifecycle facts; these do not certify external grading dependencies.
    sandbox_observations: list[SandboxObservation] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("failure_kind")
    @classmethod
    def validate_failure_kind(cls, value: Optional[str]) -> Optional[str]:
        return validate_failure_kind(value)


def exclude_verifier_model_calls(
    calls: Iterable[ModelCallRecord],
    verifier_invocations: Iterable[VerifierInvocation],
) -> list[ModelCallRecord]:
    """Exclude explicitly verifier-owned attempts from agent and step joins."""
    verifier_call_ids = {ref.model_call_id for inv in verifier_invocations for ref in inv.model_calls}
    return [call for call in calls if call.model_call_id not in verifier_call_ids]


class ToolCallObservation(ObservationModel):
    """Raw tool evidence; span and parent IDs are local to its producer until imported."""

    kind: Literal["tool_call"] = "tool_call"
    invocation_id: str
    tool_call_id: str
    agent_instance: Optional[str] = Field(default=None, min_length=1)
    span_id: Optional[str] = Field(default=None, min_length=1)
    native_tool_call_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    model_call_id: Optional[str] = None
    failure_id: Optional[str] = None
    timings: list[OperationTiming] = Field(default_factory=list)
    # Adapter input for exact joins and stable imported identity. Kept in observation bundles.
    producer_id: Optional[str] = Field(default=None, min_length=1)
    model_call_ref: Optional[ModelCallRef] = None
    source_message_id: Optional[str] = None
    source_model_ref: Optional[ModelServerRef] = None
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
        boundaries = [timing.boundary for timing in self.timings]
        if len(boundaries) != len(set(boundaries)):
            raise ValueError("a tool attempt has at most one measurement per boundary")
        return self


class TrajectoryToolCall(ToolCallObservation):
    """Canonical tool attempt with qualified span IDs and model-visible output.

    Unlike raw ToolCallObservation inputs, these IDs have already crossed the import
    boundary and must remain unchanged on replay.
    """

    output: Optional[Any] = None


def tool_parent_cycles(tools: Iterable[ToolCallObservation]) -> set[tuple[str | None, str]]:
    """Return only cycle members in the same-instance tool-parent relation."""
    parents = {
        (tool.agent_instance, tool.span_id): (tool.agent_instance, tool.parent_span_id)
        for tool in tools
        if tool.span_id is not None and tool.parent_span_id is not None
    }
    visited = set()
    cyclic = set()
    for start in parents:
        path = []
        positions = {}
        current = start
        while current in parents and current not in visited:
            if current in positions:
                cyclic.update(path[positions[current] :])
                break
            positions[current] = len(path)
            path.append(current)
            current = parents[current]
        visited.update(path)
    return cyclic


class SandboxUsageMeasurement(ObservationModel):
    """Measured CPU and memory of one sandbox from one accounting source, over the samples its owner took.

    ``scope`` says what the numbers cover: ``container`` is the sandbox's own cgroup; ``process_tree`` is the
    live processes in its namespaces, whose CPU is a lower bound of the container's (no CPU of processes that
    exited between samples) and whose memory is the sum of per-process RSS (no page cache or kernel memory, but a
    shared page counted once per process, so not a bound of the container's memory); ``shared_step`` is a cgroup
    the sandbox shares with Gym and sibling sandboxes, whose ``cpu_usage_usec_delta`` covers all of them while
    this sandbox lived. A ``sampled_lower_bound`` kind is the largest value of that quantity observed, so its true
    peak may be higher.
    """

    source: str = Field(
        description="Accounting read: cgroup_v2; procfs_pidns / procfs_mntns / procfs_pidns_mntns for the "
        "namespaces that selected the process tree; exec_rusage for the CPU of completed execs."
    )
    scope: Literal["container", "process_tree", "shared_step"]
    clock: str = Field(description="Clock of the sample times, such as gym_host (Unix time on the Gym host).")
    first_sampled_at: float
    last_sampled_at: float
    sample_count: int = Field(ge=1)
    cpu_usage_usec_total: Optional[int] = Field(default=None, ge=0)
    cpu_usage_usec_delta: Optional[int] = Field(default=None, ge=0)
    cpu_kind: Optional[Literal["counter", "sampled_lower_bound"]] = None
    memory_peak_bytes: Optional[int] = Field(default=None, ge=0)
    peak_kind: Optional[Literal["counter", "sampled_lower_bound"]] = None
    memory_current_bytes: Optional[int] = Field(default=None, ge=0)
    memory_anon_bytes: Optional[int] = Field(default=None, ge=0)
    memory_file_bytes: Optional[int] = Field(default=None, ge=0)
    includes_page_cache: Optional[bool] = None
    process_count: Optional[int] = Field(default=None, ge=0)


class SandboxObservation(ObservationModel):
    """Outcome and lifetime resource usage reported by a sandbox-owning harness."""

    kind: Literal["sandbox"] = "sandbox"
    stage: FailureStage | None = Field(
        default=None, description="Originating execution stage, when known to the caller."
    )
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
    resource_usage: Optional[list[SandboxUsageMeasurement]] = Field(
        default=None,
        # Left out of dumps when unmeasured, so records stay readable by schemas that predate the field.
        exclude_if=lambda value: value is None,
        description="Measured usage by scope, recorded only by the sandbox's creator; cpu_time_s and "
        "peak_memory_mib repeat its container-scope counters.",
    )
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    failure_id: Optional[str] = Field(
        default=None, description="Canonical failure event in the request failure report."
    )
    observed_client_interval_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Client-observed interval from the provider's create return to its close return; "
        "not the remote sandbox lifetime or its CPU time.",
    )
    teardown_confirmed: Optional[bool] = Field(
        default=None,
        description="Whether a provider close returned successfully; False means unconfirmed, not proof the sandbox remains alive. None if unattempted.",
    )
    cleanup_errors: list[str] = Field(
        default_factory=list, description="Every failed close or client release, in order; kept after a retry."
    )


def resource_usage_fields(summaries: Iterable[Any]) -> dict[str, Any]:
    """The ``SandboxObservation`` usage fields for a sandbox's measured usage summaries, or none.

    Each summary carries ``SandboxUsageMeasurement``'s fields (a dataclass or a mapping). ``cpu_time_s``,
    ``peak_memory_mib`` and ``resource_usage_source`` come only from a container-scope counter: a process-tree
    lower bound or a shared cgroup's total is never reported as the sandbox's own.
    """
    measurements = [
        SandboxUsageMeasurement.model_validate(asdict(summary) if is_dataclass(summary) else summary)
        for summary in summaries
    ]
    if not measurements:
        return {}
    fields: dict[str, Any] = {"resource_usage": measurements}
    container = next((item for item in measurements if item.scope == "container"), None)
    if container is not None:
        fields["resource_usage_source"] = f"{container.source}:{container.scope}"
        if container.cpu_kind == "counter" and container.cpu_usage_usec_total is not None:
            fields["cpu_time_s"] = container.cpu_usage_usec_total / 1_000_000
        if container.peak_kind == "counter" and container.memory_peak_bytes is not None:
            fields["peak_memory_mib"] = container.memory_peak_bytes / 2**20
    return fields


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
    agent_instance: Optional[str] = None
    span_id: Optional[str] = None
    detail: Optional[str] = None

    def matches_tool(self, tool: ToolCallObservation) -> bool:
        """A supplied owner constrains the span; missing ownership never selects between matches."""
        return (
            self.span_id is not None
            and self.span_id == tool.span_id
            and (self.invocation_id is None or self.invocation_id == tool.invocation_id)
            and (self.agent_instance is None or self.agent_instance == tool.agent_instance)
        )


AgentObservationRecord = Annotated[
    AgentInvocation | ToolCallObservation | ContextCompactionObservation | SandboxObservation,
    Field(discriminator="kind"),
]


class AgentObservationBundle(ObservationModel):
    """Normalized observations returned by one Agent Server for one rollout."""

    schema_version: Literal["1.0", "2.0"] = "1.0"
    source: str
    records: list[AgentObservationRecord] = Field(
        default_factory=list,
        description="Unordered typed records; list position does not imply execution order.",
    )
    gaps: list[ObservationGap] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def decode_version(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version", "1.0") == "1.0":
            reject_v2_fields(value.get("records", []))
        return value

    @model_serializer(mode="wrap")
    def serialize_version(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        if self.schema_version == "1.0":
            return project_bundle_v1(result)
        if "records" in result:
            result["records"] = [
                canonical_tool_record(record, include_join_evidence=True)
                if record.get("kind") == "tool_call"
                else record
                for record in result["records"]
            ]
        return result

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
    schema_version: Literal["1.0", "2.0"] = "1.0"
    task_id: str
    rollout_id: str
    invocations: list[AgentInvocation] = Field(default_factory=list)
    verifier_invocations: list[VerifierInvocation] = Field(default_factory=list)
    turns: list[TrajectoryTurn] = Field(default_factory=list)
    model_calls: list[TrajectoryModelCall] = Field(default_factory=list)
    tool_calls: list[TrajectoryToolCall] = Field(default_factory=list)
    failure_events: dict[str, dict[str, Any]] = Field(default_factory=dict)
    gaps: list[ObservationGap] = Field(default_factory=list)

    @property
    def capture_key(self) -> str:
        """The attempt-qualified episode identity (called rollout_id in v1)."""
        return self.rollout_id

    @model_validator(mode="before")
    @classmethod
    def decode_identity(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("schema_version", "1.0") == "1.0":
            reject_v2_fields(
                [*value.get("tool_calls", []), *value.get("invocations", []), *value.get("model_calls", [])]
            )
            if value.get("failure_events"):
                raise ValueError("failure_events requires trajectory schema 2.0")
        if "capture_key" not in value:
            return value
        if value.get("schema_version") != "2.0":
            raise ValueError("capture_key requires trajectory schema 2.0")
        if "rollout_id" in value and value["rollout_id"] != value["capture_key"]:
            raise ValueError("conflicting capture_key and rollout_id aliases")
        result = dict(value)
        result["rollout_id"] = result.pop("capture_key")
        return result

    @model_serializer(mode="wrap")
    def serialize_version(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):
        result = handler(self)
        if self.schema_version == "1.0":
            # The result writer coordinates both envelopes before either loses join evidence.
            if info.context and info.context.get("defer_v1_projection"):
                return result
            return project_trajectory_v1(result)
        if "rollout_id" in result:
            result["capture_key"] = result.pop("rollout_id")
        if "tool_calls" in result:
            result["tool_calls"] = [
                canonical_tool_record(tool, include_join_evidence=False) for tool in result["tool_calls"]
            ]
        return result

    @model_validator(mode="after")
    def validate_identity(self) -> "TrajectoryRecord":
        invocation_ids = [invocation.invocation_id for invocation in [*self.invocations, *self.verifier_invocations]]
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError("invocation_id must be unique within a trajectory")
        model_call_ids = [call.model_call_id for call in self.model_calls if call.model_call_id]
        if len(model_call_ids) != len(set(model_call_ids)):
            raise ValueError("model_call_id must be unique within a trajectory")
        if self.schema_version == "2.0":
            identities = [(tool.agent_instance, tool.span_id) for tool in self.tool_calls]
            if any(tool.span_id is None for tool in self.tool_calls) or len(identities) != len(set(identities)):
                raise ValueError("v2 tool attempts require unique span identities")
            if tool_parent_cycles(self.tool_calls):
                raise ValueError("tool parent spans must not contain cycles")
            for gap in self.gaps:
                if gap.span_id is not None and sum(gap.matches_tool(tool) for tool in self.tool_calls) != 1:
                    raise ValueError("attempt-scoped gap must resolve to one retained tool with consistent ownership")
            validate_failure_events(self.failure_events)
            calls = {call.model_call_id: call for call in self.model_calls if call.model_call_id}
            owners: dict[str, set[str | None]] = defaultdict(set)
            for invocation in self.invocations:
                owners[invocation.invocation_id].add(invocation.agent_instance)
                for reference in invocation.model_calls:
                    candidates = (
                        [calls[reference.model_call_id]]
                        if reference.model_call_id in calls
                        else []
                        if reference.model_call_id is not None
                        else self.model_calls
                    )
                    matches = [
                        candidate
                        for candidate in candidates
                        if (
                            reference.model_ref is None or candidate.response_metadata.model_ref == reference.model_ref
                        )
                        and (
                            reference.response_id is None
                            or candidate.response_metadata.response_id == reference.response_id
                        )
                    ]
                    call = matches[0] if len(matches) == 1 else None
                    if (
                        call is not None
                        and invocation.agent_instance is not None
                        and call.agent_instance is not None
                        and call.agent_instance != invocation.agent_instance
                    ):
                        raise ValueError("model call and owning invocation have different agent instances")
            for item in [*self.tool_calls, *self.turns]:
                known = owners.get(item.invocation_id, set())
                if (
                    item.agent_instance is not None
                    and known
                    and None not in known
                    and item.agent_instance not in known
                ):
                    raise ValueError("record and owning invocation have different agent instances")
            for tool in self.tool_calls:
                if tool.parent_span_id is not None and (tool.agent_instance, tool.parent_span_id) not in identities:
                    raise ValueError("parent_span_id must resolve to an attempt of the same agent instance")
                if tool.failure_id is not None and tool.failure_id not in self.failure_events:
                    raise ValueError("tool failure_id must resolve in failure_events")
                if tool.model_call_id is not None:
                    call = calls.get(tool.model_call_id)
                    if call is None:
                        raise ValueError("tool model_call_id must resolve to one captured call")
                    evidence = tool_request_evidence(
                        call.response,
                        status_code=call.response_metadata.status_code,
                        response_status=call.response_metadata.response_status,
                    )
                    if not evidence.permits_link(tool.native_tool_call_id or tool.tool_call_id):
                        raise ValueError("requesting model response does not establish this tool request")
                    if (
                        tool.agent_instance is not None
                        and call.agent_instance is not None
                        and tool.agent_instance != call.agent_instance
                    ):
                        raise ValueError("tool and requesting model call have different agent instances")
        keys: set[tuple[str, int]] = set()
        for turn in self.turns:
            if turn.task_id != self.task_id or turn.rollout_id != self.rollout_id:
                raise ValueError("turn identity must match the trajectory")
            key = (turn.invocation_id, turn.turn_no)
            if key in keys:
                raise ValueError("turn number must be unique within an invocation")
            keys.add(key)
        return self


def join_assistant_message_calls(
    bundle: AgentObservationBundle,
    trajectory: TrajectoryRecord | None,
    calls: list["ModelCallRecord"],
) -> tuple[AgentObservationBundle, TrajectoryRecord | None]:
    """Join persisted messages to HTTP attempts using session, message and model-server IDs.

    Run after invocation ownership resolution. Summary attempts belong to compactions;
    calls without a saved message (for example title generation) stay invocation-owned.
    Producers declare source_message_id(s), source_model_ref and invocation IDs.
    Source labels are not dispatch keys. No content or timing fallback is used.
    """
    result = bundle.model_copy(
        update={
            "records": [
                record.model_copy(update={"model_calls": list(record.model_calls)})
                if isinstance(record, ContextCompactionObservation)
                else record
                for record in bundle.records
            ],
            "gaps": [gap for gap in bundle.gaps if not gap.code.startswith("assistant_message_call_")],
        }
    )
    projected = (
        trajectory.model_copy(
            update={
                "turns": [turn.model_copy(update={"model_calls": list(turn.model_calls)}) for turn in trajectory.turns]
            }
        )
        if trajectory is not None
        else None
    )
    targets: dict[tuple[str, str], list[TrajectoryTurn | ContextCompactionObservation]] = defaultdict(list)
    for turn in projected.turns if projected is not None else []:
        if turn.source_message_id:
            targets[(turn.invocation_id, turn.source_message_id)].append(turn)
    for compaction in result.records:
        if isinstance(compaction, ContextCompactionObservation):
            for message_id in set(compaction.source_message_ids):
                targets[(compaction.invocation_id, message_id)].append(compaction)

    call_counts = Counter(call.model_call_id for call in calls if call.model_call_id)
    owners: dict[str, set[str]] = defaultdict(set)
    for invocation in result.records:
        if isinstance(invocation, AgentInvocation):
            for ref in invocation.model_calls:
                if ref.model_call_id:
                    owners[ref.model_call_id].add(invocation.invocation_id)

    # Recompute these links on repeated finalization, including newly ambiguous evidence.
    attributed_ids = {call.model_call_id for call in calls if call.client_assistant_message_id and call.model_call_id}
    for candidates in targets.values():
        for target in candidates:
            target.model_calls = [ref for ref in target.model_calls if ref.model_call_id not in attributed_ids]

    for call in calls:
        if not call.client_assistant_message_id:
            continue
        candidates = targets.get((call.client_session_id or "", call.client_assistant_message_id), [])
        candidates = [target for target in candidates if target.source_model_ref in (None, call.model_ref)]
        code = None
        if len(candidates) != 1:
            code = "ambiguous" if candidates else "unmatched"
        elif candidates[0].source_model_ref is None or call.model_ref is None:
            code = "scope_unavailable"
        elif not call.model_call_id:
            code = "identity_unavailable"
        elif call_counts[call.model_call_id] != 1:
            code = "ambiguous"
        elif owners[call.model_call_id] != {call.client_session_id}:
            code = "ownership_conflict"
        if code is not None:
            result.gaps.append(
                ObservationGap(
                    code=f"assistant_message_call_{code}",
                    invocation_id=call.client_session_id,
                    detail=f"{call.client_assistant_message_id}:{call.model_call_id}",
                )
            )
            continue
        candidates[0].model_calls.append(
            ModelCallRef(model_call_id=call.model_call_id, model_ref=call.model_ref, response_id=call.response_id)
        )
    result.gaps = list(
        {
            (gap.code, gap.agent_instance, gap.invocation_id, gap.span_id, gap.detail): gap for gap in result.gaps
        }.values()
    )
    return result, projected


def join_agent_model_calls(
    bundle: AgentObservationBundle,
    calls: Iterable[ModelCallRecord],
) -> AgentObservationBundle:
    """Join captured calls to agent invocations and their compactions without guessing ownership."""

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
    result.gaps = list(
        {
            (gap.code, gap.agent_instance, gap.invocation_id, gap.span_id, gap.detail): gap for gap in result.gaps
        }.values()
    )
    return result


@dataclass(frozen=True, slots=True)
class AgentEpisode:
    """An Agent response and the observations available at its execution boundary."""

    response: NeMoGymResponse
    observations: AgentObservationBundle


def failure_observations(value: Any) -> dict[str, Any]:
    """Keep the existing typed observation slots across an unsuccessful HTTP reply.

    Error strings are not searched recursively: only producer-owned top-level fields
    are accepted, and malformed observations remain unavailable.
    """
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, schema in ((NG_AGENT_OBSERVATIONS_KEY, AgentObservationBundle), (NG_TRAJECTORY_KEY, TrajectoryRecord)):
        if key in value:
            try:
                result[key] = schema.model_validate(value[key]).model_dump(mode="json")
            except (ValidationError, TypeError, ValueError):
                pass
    if isinstance(value.get(NG_VERIFIER_INVOCATIONS_KEY), list):
        try:
            result[NG_VERIFIER_INVOCATIONS_KEY] = [
                VerifierInvocation.model_validate(item).model_dump(mode="json")
                for item in value[NG_VERIFIER_INVOCATIONS_KEY]
            ]
        except (ValidationError, TypeError, ValueError):
            pass
    if isinstance(value.get("verification_skipped"), bool):
        result["verification_skipped"] = value["verification_skipped"]
    return result
