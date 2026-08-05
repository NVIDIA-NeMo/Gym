# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured semantic history and request-local context policies.

This module deliberately does not tokenize requests or construct training
rows. It owns the semantic, generation-time side of context compaction:

* a complete in-process history used as a validation shadow;
* stable media identities with occurrence-preserving request order;
* deterministic identity and visual-recency policies;
* a registry for agent-owned custom compaction protocols; and
* materialized Responses-API request views with append-compatibility metadata.

Exact generation evidence and flat-trace construction consume these contracts
at later integration boundaries. Model-serving dependencies remain unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


BUILTIN_SEMANTIC_PART_KINDS = frozenset(
    {
        "system_text",
        "task_text",
        "policy_text",
        "safety_text",
        "user_text",
        "reasoning",
        "assistant_action",
        "tool_call",
        "tool_result",
        "environment_text",
        "code",
        "file",
        "patch",
        "execution_log",
        "image",
        "video",
        "audio",
        "derived_placeholder",
        "derived_summary",
        "derived_extract",
        "derived_outline",
        "derived_diff",
    }
)

# Wire-level semantic kinds are versioned strings rather than a permanently
# closed Literal. Built-ins remain validated, and agent packages may register
# additional typed kinds before constructing authority-mode history.
SemanticPartKind = str
_REGISTERED_SEMANTIC_PART_KINDS = set(BUILTIN_SEMANTIC_PART_KINDS)


def register_semantic_part_kind(kind: str) -> None:
    if not kind or kind in _REGISTERED_SEMANTIC_PART_KINDS:
        raise ValueError(f"Invalid or duplicate semantic part kind {kind!r}")
    _REGISTERED_SEMANTIC_PART_KINDS.add(kind)


def unregister_semantic_part_kind(kind: str) -> None:
    if kind in BUILTIN_SEMANTIC_PART_KINDS:
        raise ValueError(f"Cannot unregister built-in semantic part kind {kind!r}")
    _REGISTERED_SEMANTIC_PART_KINDS.discard(kind)


LineageDisposition = Literal[
    "kept",
    "dropped",
    "replaced",
    "summarized",
    "transformed",
]


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{canonical_digest(parts)[:24]}"


_IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})
_TOKEN_METADATA_FIELDS = frozenset(
    {
        "prompt_token_ids",
        "generation_token_ids",
        "generation_log_probs",
        "routed_experts",
    }
)
_PRIVATE_ID_FIELDS = frozenset(
    {
        "_nemo_gym_event_id",
        "_nemo_gym_part_id",
        "_nemo_gym_observation_group_id",
        "_nemo_gym_media_id",
        "_nemo_gym_semantic_kind",
    }
)


class RecencyHistoryPolicyConfig(BaseModel):
    """Configuration for the initial visual-recency policy."""

    model_config = ConfigDict(extra="forbid")

    protect_initial_context: bool = True
    keep_last_image_groups: int = Field(default=3, ge=0)
    keep_all_text: bool = True
    image_omission_marker: str | None = "[Earlier image omitted]"


class HistoryPolicyConfig(BaseModel):
    """Select a built-in or agent-registered history policy."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="identity", min_length=1)
    config: RecencyHistoryPolicyConfig | dict[str, Any] = Field(default_factory=RecencyHistoryPolicyConfig)


class CompactionScheduleConfig(BaseModel):
    """Choose when an already-selected historical base may be rewritten."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rolling_recency", "turn_chunked_recency"] = "rolling_recency"
    actions_per_chunk: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_actions_per_chunk(self) -> "CompactionScheduleConfig":
        if self.type == "rolling_recency" and self.actions_per_chunk != 1:
            raise ValueError("rolling_recency requires actions_per_chunk=1")
        return self


class ContextGuardConfig(BaseModel):
    """Hard generation-admission limits evaluated only between turns."""

    model_config = ConfigDict(extra="forbid")

    max_total_tokens: int | None = Field(default=None, ge=1)
    reserved_generation_tokens: int = Field(default=0, ge=0)
    max_active_images: int | None = Field(default=None, ge=0)
    max_vision_tokens: int | None = Field(default=None, ge=0)
    projected_vision_tokens_per_image: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_vision_projection(self) -> "ContextGuardConfig":
        if self.max_vision_tokens is not None and self.projected_vision_tokens_per_image is None:
            raise ValueError("max_vision_tokens requires projected_vision_tokens_per_image")
        return self


class ContextHistoryConfig(BaseModel):
    """Capability-gated semantic context-history configuration for a Gym agent."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    shadow_only: bool = True
    policy: HistoryPolicyConfig = Field(default_factory=HistoryPolicyConfig)
    schedule: CompactionScheduleConfig = Field(default_factory=CompactionScheduleConfig)
    guards: ContextGuardConfig = Field(default_factory=ContextGuardConfig)

    @model_validator(mode="after")
    def validate_shadow_policy(self) -> "VisualHistoryConfig":
        if self.enabled and self.shadow_only and self.policy.type != "identity":
            raise ValueError("shadow_only context history requires the identity policy")
        if self.enabled and self.shadow_only and self.schedule.type != "rolling_recency":
            raise ValueError("shadow_only context history requires the rolling_recency schedule")
        return self


# Backward-compatible name used by Rohit's current agent configuration. The
# controller and policy contract are semantic and are not restricted to images.
VisualHistoryConfig = ContextHistoryConfig


def _source_media_metadata(
    source_part: Mapping[str, Any],
) -> tuple[tuple[int, int] | None, str | None, str | None]:
    source = source_part.get("image") or source_part.get("image_url") or source_part.get("url")
    if isinstance(source, Mapping):
        source = source.get("url")
    if not isinstance(source, str) or not source.startswith("data:image/"):
        return None, None, None

    media_header, _, encoded = source.partition(",")
    source_format = media_header.removeprefix("data:image/").split(";", 1)[0]
    if source_format != "png" or not encoded:
        return None, None, source_format
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError:
        return None, None, source_format
    if len(payload) < 26 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None, source_format
    width, height, bit_depth, color_type = struct.unpack(
        ">IIBB",
        payload[16:26],
    )
    color_modes = {
        0: "L",
        2: "RGB",
        3: "P",
        4: "LA",
        6: "RGBA",
    }
    color_mode = color_modes.get(color_type)
    if bit_depth != 8:
        color_mode = f"{color_mode or 'unknown'}-{bit_depth}bit"
    return (width, height), color_mode, source_format


@dataclass(frozen=True)
class MediaAsset:
    """One immutable source image stored once per logical rollout."""

    media_id: str
    content_digest: str
    source_part: Mapping[str, Any]
    original_dimensions: tuple[int, int] | None
    color_mode: str | None
    source_format: str | None


@dataclass
class MediaArena:
    """Content-addressed media ownership with occurrence-preserving lookups."""

    _assets: dict[str, MediaAsset] = field(default_factory=dict)

    @staticmethod
    def _canonical_payload(source_part: Mapping[str, Any]) -> str:
        return json.dumps(
            source_part,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=repr,
        )

    def register(self, source_part: Mapping[str, Any]) -> str:
        canonical = self._canonical_payload(source_part)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        media_id = f"media-{digest[:24]}"
        existing = self._assets.get(media_id)
        if existing is not None and existing.content_digest != digest:
            raise RuntimeError(f"Media ID collision for {media_id}")
        if existing is None:
            original_dimensions, color_mode, source_format = _source_media_metadata(source_part)
            self._assets[media_id] = MediaAsset(
                media_id=media_id,
                content_digest=digest,
                source_part=deepcopy(dict(source_part)),
                original_dimensions=original_dimensions,
                color_mode=color_mode,
                source_format=source_format,
            )
        return media_id

    def resolve(self, media_id: str) -> Mapping[str, Any]:
        try:
            return self._assets[media_id].source_part
        except KeyError as exc:
            raise ValueError(f"Unknown media ID {media_id!r}") from exc

    def export(self) -> dict[str, dict[str, Any]]:
        """Return each immutable media payload once, keyed by stable media ID."""

        return {
            media_id: {
                "media_id": asset.media_id,
                "content_digest": asset.content_digest,
                "source_part": deepcopy(dict(asset.source_part)),
                "original_dimensions": asset.original_dimensions,
                "color_mode": asset.color_mode,
                "source_format": asset.source_format,
            }
            for media_id, asset in self._assets.items()
        }

    def __len__(self) -> int:
        return len(self._assets)


@dataclass(frozen=True)
class SemanticPart:
    part_id: str
    kind: SemanticPartKind
    content_index: int | None
    observation_group_id: str | None = None
    media_id: str | None = None


@dataclass(frozen=True)
class SemanticEvent:
    event_id: str
    turn_id: int
    role: str
    item: Mapping[str, Any]
    parts: tuple[SemanticPart, ...]
    is_initial_context: bool = False
    conditions_action_turn: int | None = None


@dataclass(frozen=True)
class KeepPartRef:
    event_id: str
    part_id: str


@dataclass(frozen=True)
class OmissionArtifact:
    artifact_id: str
    source_first_part_id: str
    source_last_part_id: str
    source_part_count: int
    source_digest: str
    text: str
    anchor_part_id: str


@dataclass(frozen=True)
class UnitLineageRecord:
    source_unit_id: str
    source_digest: str
    disposition: LineageDisposition
    output_unit_ids: tuple[str, ...]
    output_digests: tuple[str, ...]


@dataclass(frozen=True)
class TransformationLineageRecord:
    transformation_id: str
    transformation_type: str
    transformation_version: str
    configuration_digest: str
    deterministic: bool
    lossy: bool
    generator_contract_id: str | None
    unit_records: tuple[UnitLineageRecord, ...]
    validator_result: Literal["passed"]


@dataclass(frozen=True)
class TransformationLineageDeltaRecord:
    transformation_id: str
    parent_transformation_id: str | None
    transformation_type: str
    transformation_version: str
    configuration_digest: str
    deterministic: bool
    lossy: bool
    generator_contract_id: str | None
    unit_upserts: tuple[UnitLineageRecord, ...]
    source_unit_count: int
    state_digest: str
    validator_result: Literal["passed"]


def lineage_state_digest(
    records: Mapping[str, UnitLineageRecord] | Sequence[UnitLineageRecord],
) -> str:
    values = records.values() if isinstance(records, Mapping) else records
    return canonical_digest(
        [
            {
                "source_unit_id": record.source_unit_id,
                "source_digest": record.source_digest,
                "disposition": record.disposition,
                "output_unit_ids": record.output_unit_ids,
                "output_digests": record.output_digests,
            }
            for record in sorted(
                values,
                key=lambda item: item.source_unit_id,
            )
        ]
    )


def build_lineage_delta(
    lineage: TransformationLineageRecord,
    *,
    previous_records: Mapping[str, UnitLineageRecord],
    parent_transformation_id: str | None,
) -> tuple[
    TransformationLineageDeltaRecord,
    dict[str, UnitLineageRecord],
]:
    current_records = {record.source_unit_id: record for record in lineage.unit_records}
    upserts = tuple(
        record for source_unit_id, record in current_records.items() if previous_records.get(source_unit_id) != record
    )
    return (
        TransformationLineageDeltaRecord(
            transformation_id=lineage.transformation_id,
            parent_transformation_id=parent_transformation_id,
            transformation_type=lineage.transformation_type,
            transformation_version=lineage.transformation_version,
            configuration_digest=lineage.configuration_digest,
            deterministic=lineage.deterministic,
            lossy=lineage.lossy,
            generator_contract_id=lineage.generator_contract_id,
            unit_upserts=upserts,
            source_unit_count=len(current_records),
            state_digest=lineage_state_digest(current_records),
            validator_result=lineage.validator_result,
        ),
        current_records,
    )


@dataclass(frozen=True)
class PolicyDecisionRecord:
    policy_name: str
    policy_version: str
    config_digest: str
    protected_part_ids: tuple[str, ...]
    changed_part_ranges: tuple[tuple[str, str], ...]
    retained_part_count: int
    omitted_part_count: int
    selection_digest: str
    inserted_artifact_ids: tuple[str, ...]
    decision_turn: int
    lineage: TransformationLineageRecord


class PolicyDecisionEvidence(BaseModel):
    """Bounded per-call reference to rollout-level transformation lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_name: str
    policy_version: str
    config_digest: str
    decision_turn: int = Field(ge=1)
    selection_digest: str
    transformation_id: str


@dataclass(frozen=True)
class HistoryViewPlan:
    keep: tuple[KeepPartRef, ...]
    artifacts: tuple[OmissionArtifact, ...]
    decision: PolicyDecisionRecord

    @property
    def retained_part_ids(self) -> frozenset[str]:
        return frozenset(ref.part_id for ref in self.keep)


@dataclass(frozen=True)
class MaterializedHistoryView:
    items: tuple[Mapping[str, Any], ...]
    media_ids: tuple[str, ...]
    descriptor: tuple[str, ...]
    decision: PolicyDecisionRecord


@dataclass(frozen=True)
class RewriteBoundaryEvent:
    event_id: str
    trigger_after_step: int
    applies_to_step: int
    reason: str
    policy_name: str
    policy_version: str
    config_digest: str
    previous_view_digest: str
    current_view_digest: str
    changed_part_ranges: tuple[tuple[str, str], ...]
    retained_part_count: int
    omitted_part_count: int
    retained_media_count: int
    removed_media_count: int
    inserted_artifact_ids: tuple[str, ...]
    schedule_name: str = "per_action"
    schedule_version: str = "1"
    schedule_config_digest: str | None = None
    chunk_id: str | None = None
    block_index: int | None = None


@dataclass(frozen=True)
class FinalizedChunkRecord:
    chunk_id: str
    block_index: int
    eligible_action_ids: tuple[str, ...]
    completion_evidence_ids: tuple[str, ...]
    first_action_turn: int
    last_action_turn: int
    configured_actions_per_chunk: int
    configured_history_groups: int | None
    actual_action_count: int
    early_close_reason: str | None
    active_observation_group_count: int
    active_raw_image_count: int


@dataclass(frozen=True)
class ContextMeasurements:
    prompt_token_count: int
    active_image_count: int
    vision_token_count: int


@dataclass(frozen=True)
class GuardEvaluation:
    guard_name: str
    measured_value: int
    configured_limit: int
    exceeded: bool
    excess: int


@dataclass(frozen=True)
class GuardOutcomeRecord:
    rollout_id: str
    chunk_id: str | None
    applies_to_step: int
    completed_action_count: int
    pending_observation_group_ids: tuple[str, ...]
    guard_name: str
    measured_value: int
    configured_limit: int
    early_chunk_close: bool
    post_compaction_value: int | None
    decision: Literal["admit", "admit_after_compaction", "reject"]


@dataclass(frozen=True)
class PreparedHistoryView:
    view: MaterializedHistoryView
    view_digest: str
    append_compatible: bool
    boundary: RewriteBoundaryEvent | None
    context_epoch: int
    segment_index: int


class GenerationContract(BaseModel):
    """Composable generation provenance carried once per rollout contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    model_contract_id: str
    tokenizer_contract_id: str
    template_contract_id: str
    sampling_contract_id: str
    processor_contract_id: str
    compaction_policy_id: str
    generation_contract_id: str
    loss_normalization: Literal["global_action_token_mean"] = "global_action_token_mean"
    training_eligible: bool = False
    incomplete_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_composition(self) -> "GenerationContract":
        component_ids = {
            "model_contract_id": self.model_contract_id,
            "tokenizer_contract_id": self.tokenizer_contract_id,
            "template_contract_id": self.template_contract_id,
            "sampling_contract_id": self.sampling_contract_id,
            "processor_contract_id": self.processor_contract_id,
            "compaction_policy_id": self.compaction_policy_id,
        }
        expected = stable_id(
            "generation-contract",
            canonical_digest(component_ids),
        )
        if self.generation_contract_id != expected:
            raise ValueError("generation_contract_id does not match its component IDs")
        if self.training_eligible and self.incomplete_reasons:
            raise ValueError("A training-eligible generation contract cannot be incomplete")
        return self


class PolicyOutputSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_output_span_id: str
    model_call_id: str
    action_ids: tuple[str, ...]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    eligible: bool
    old_logprobs_alignment: Literal["sampled_tokens"] = "sampled_tokens"

    @model_validator(mode="after")
    def validate_span(self) -> "PolicyOutputSpan":
        if self.end < self.start:
            raise ValueError("Policy output span end precedes its start")
        return self


class MediaOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    media_id: str
    occurrence_ordinal: int = Field(ge=0)
    model_call_id: str
    placeholder_span_or_position: tuple[int, int] | int | None = None
    processed_dimensions: tuple[int, int] | None = None
    model_specific_sidecars: dict[str, Any] = Field(default_factory=dict)


class ObservedCompletion(BaseModel):
    """Exact immutable evidence returned by the generation operation itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str
    completion_id: str
    action_id: str
    turn_id: int = Field(ge=1)
    prepared_request_id: str
    request_id: str
    context_epoch: int = Field(ge=0)
    segment_index: int = Field(ge=0)
    segment_id: str
    expected_append_compatible: bool
    compaction_event_id: str | None = None
    prompt_token_ids: tuple[int, ...]
    sampled_token_ids: tuple[int, ...]
    sampled_logprobs: tuple[float, ...]
    finish_reason: str | None = None
    media_ids: tuple[str, ...]
    policy_decision: PolicyDecisionEvidence
    generation_contract_id: str
    policy_output_spans: tuple[PolicyOutputSpan, ...]
    media_occurrences: tuple[MediaOccurrence, ...]
    processor_fingerprint: str | None = None
    eligible: bool = True
    evidence_source: Literal["generation_response"] = "generation_response"

    @model_validator(mode="after")
    def validate_alignment(self) -> "ObservedCompletion":
        if len(self.sampled_token_ids) != len(self.sampled_logprobs):
            raise ValueError(
                "sampled token/logprob length mismatch: "
                f"tokens={len(self.sampled_token_ids)} "
                f"logprobs={len(self.sampled_logprobs)}"
            )
        if len(self.policy_output_spans) != 1:
            raise ValueError("Initial authority contract requires one policy-output span per model call")
        span = self.policy_output_spans[0]
        if (
            span.start != 0
            or span.end != len(self.sampled_token_ids)
            or span.action_ids != (self.action_id,)
            or span.eligible != self.eligible
        ):
            raise ValueError(
                "Initial policy-output span must cover the complete sampled "
                "completion and match its action/eligibility"
            )
        if tuple(occurrence.media_id for occurrence in self.media_occurrences) != (self.media_ids):
            raise ValueError("Media occurrence order does not match completion media IDs")
        return self


class HistoryController:
    """Retry-safe per-rollout owner of request-view rewrite boundaries."""

    def __init__(self, history: "SemanticHistory", policy: "HistoryPolicy"):
        self.history = history
        self.policy = policy
        self._completed_descriptor: tuple[str, ...] | None = None
        self._completed_media_ids: tuple[str, ...] | None = None
        self._completed_view_digest: str | None = None
        self._pending_boundary: RewriteBoundaryEvent | None = None
        self._boundary_events: list[RewriteBoundaryEvent] = []
        self._context_epoch = 0
        self._segment_index = 0
        self.evaluation_count = 0

    @property
    def pending_boundary(self) -> RewriteBoundaryEvent | None:
        return self._pending_boundary

    @property
    def boundary_events(self) -> tuple[RewriteBoundaryEvent, ...]:
        return tuple(self._boundary_events)

    def prepare(self, *, applies_to_step: int) -> PreparedHistoryView:
        if applies_to_step < 1:
            raise ValueError("applies_to_step must be at least 1")
        self.evaluation_count += 1
        plan = self._plan(applies_to_step=applies_to_step)
        view = materialize_history_view(self.history, plan)
        view_digest = _view_digest(view)

        if self._pending_boundary is not None:
            if (
                self._pending_boundary.applies_to_step != applies_to_step
                or self._pending_boundary.current_view_digest != view_digest
            ):
                raise RuntimeError(
                    "Pending rewrite boundary changed before acknowledgement: "
                    f"pending_step={self._pending_boundary.applies_to_step} "
                    f"retry_step={applies_to_step}"
                )
            return PreparedHistoryView(
                view=view,
                view_digest=view_digest,
                append_compatible=False,
                boundary=self._pending_boundary,
                context_epoch=self._context_epoch,
                segment_index=self._segment_index,
            )

        append_compatible = descriptor_is_append_compatible(
            self._completed_descriptor, view.descriptor
        ) and ordered_media_is_append_compatible(self._completed_media_ids, view.media_ids)
        boundary = None
        if self._completed_descriptor is not None and not append_compatible:
            assert self._completed_view_digest is not None
            boundary = self._make_boundary(
                applies_to_step=applies_to_step,
                previous_view_digest=self._completed_view_digest,
                view=view,
                current_view_digest=view_digest,
            )
            self._pending_boundary = boundary
            self._boundary_events.append(boundary)
            self._context_epoch += 1
            self._segment_index += 1

        return PreparedHistoryView(
            view=view,
            view_digest=view_digest,
            append_compatible=append_compatible,
            boundary=boundary,
            context_epoch=self._context_epoch,
            segment_index=self._segment_index,
        )

    def _plan(self, *, applies_to_step: int) -> HistoryViewPlan:
        return self.policy.plan(self.history, decision_turn=applies_to_step)

    def acknowledge(self, prepared: PreparedHistoryView) -> None:
        if self._pending_boundary is not None:
            if prepared.boundary != self._pending_boundary:
                raise RuntimeError("Cannot acknowledge a different rewrite boundary")
            self._pending_boundary = None
        elif prepared.boundary is not None:
            raise RuntimeError("Prepared boundary is not pending")

        self._completed_descriptor = prepared.view.descriptor
        self._completed_media_ids = prepared.view.media_ids
        self._completed_view_digest = prepared.view_digest

    def _make_boundary(
        self,
        *,
        applies_to_step: int,
        previous_view_digest: str,
        view: MaterializedHistoryView,
        current_view_digest: str,
    ) -> RewriteBoundaryEvent:
        previous_media = Counter(self._completed_media_ids or ())
        current_media = Counter(view.media_ids)
        removed_media_count = sum((previous_media - current_media).values())
        identity = _config_digest(
            {
                "rollout_id": self.history.rollout_id,
                "applies_to_step": applies_to_step,
                "previous_view_digest": previous_view_digest,
                "current_view_digest": current_view_digest,
            }
        )
        decision = view.decision
        return RewriteBoundaryEvent(
            event_id=f"boundary-{applies_to_step:06d}-{identity[:12]}",
            trigger_after_step=applies_to_step - 1,
            applies_to_step=applies_to_step,
            reason="history_policy_rewrite",
            policy_name=decision.policy_name,
            policy_version=decision.policy_version,
            config_digest=decision.config_digest,
            previous_view_digest=previous_view_digest,
            current_view_digest=current_view_digest,
            changed_part_ranges=decision.changed_part_ranges,
            retained_part_count=decision.retained_part_count,
            omitted_part_count=decision.omitted_part_count,
            retained_media_count=len(view.media_ids),
            removed_media_count=removed_media_count,
            inserted_artifact_ids=decision.inserted_artifact_ids,
        )


class TurnChunkedHistoryController(HistoryController):
    """Freeze one compacted base and append a tail for up to K actions."""

    schedule_name = "turn_chunked_recency"
    schedule_version = "1"

    def __init__(
        self,
        history: "SemanticHistory",
        policy: "HistoryPolicy",
        *,
        actions_per_chunk: int,
        history_groups: int | None = None,
    ):
        if actions_per_chunk < 1:
            raise ValueError("actions_per_chunk must be at least 1")
        if history_groups is not None and history_groups < 0:
            raise ValueError("history_groups must be non-negative")
        super().__init__(history, policy)
        self.actions_per_chunk = actions_per_chunk
        self.history_groups = history_groups
        self.schedule_config_digest = _config_digest(
            {
                "schedule": self.schedule_name,
                "version": self.schedule_version,
                "actions_per_chunk": actions_per_chunk,
                "history_groups": history_groups,
            }
        )
        self._base_plan: HistoryViewPlan | None = None
        self._known_base_part_ids: frozenset[str] = frozenset()
        self._needs_new_chunk = True
        self._block_index = -1
        self._chunk_id: str | None = None
        self._action_ids: list[str] = []
        self._action_turns: list[int] = []
        self._completion_ids: list[str] = []
        self._last_acknowledged: PreparedHistoryView | None = None
        self._chunk_records: list[FinalizedChunkRecord] = []

    @property
    def current_chunk_id(self) -> str | None:
        return self._chunk_id

    @property
    def chunk_records(self) -> tuple[FinalizedChunkRecord, ...]:
        return tuple(self._chunk_records)

    @property
    def completed_actions_in_current_chunk(self) -> int:
        return len(self._action_ids)

    def _start_chunk(self, *, applies_to_step: int) -> None:
        self._block_index += 1
        identity = _config_digest(
            {
                "rollout_id": self.history.rollout_id,
                "block_index": self._block_index,
                "schedule_config_digest": self.schedule_config_digest,
            }
        )
        self._chunk_id = f"chunk-{self._block_index:06d}-{identity[:12]}"
        self._base_plan = self.policy.plan(
            self.history,
            decision_turn=applies_to_step,
        )
        self._known_base_part_ids = frozenset(part.part_id for _, part in self.history.parts)
        self._action_ids = []
        self._action_turns = []
        self._completion_ids = []
        self._last_acknowledged = None
        self._needs_new_chunk = False

    def _plan(self, *, applies_to_step: int) -> HistoryViewPlan:
        if self._needs_new_chunk:
            self._start_chunk(applies_to_step=applies_to_step)
        assert self._base_plan is not None

        tail = tuple(
            KeepPartRef(event.event_id, part.part_id)
            for event, part in self.history.parts
            if part.part_id not in self._known_base_part_ids
        )
        keep = (*self._base_plan.keep, *tail)
        base_decision = self._base_plan.decision
        retained_ids = tuple(ref.part_id for ref in keep)
        disposition_by_part_id = {
            record.source_unit_id: (
                record.disposition,
                record.output_unit_ids,
                record.output_digests,
            )
            for record in base_decision.lineage.unit_records
        }
        tail_ids = {ref.part_id for ref in tail}
        for event, part in self.history.parts:
            if part.part_id in tail_ids:
                digest = _semantic_part_digest(event, part)
                disposition_by_part_id[part.part_id] = (
                    "kept",
                    (part.part_id,),
                    (digest,),
                )
        decision = replace(
            base_decision,
            retained_part_count=len(retained_ids),
            selection_digest=_ordered_id_digest(
                (
                    *retained_ids,
                    "--base-selection--",
                    base_decision.selection_digest,
                )
            ),
            decision_turn=applies_to_step,
            lineage=_lineage_record(
                self.history,
                transformation_type=base_decision.lineage.transformation_type,
                transformation_version=(base_decision.lineage.transformation_version),
                configuration_digest=(base_decision.lineage.configuration_digest),
                disposition_by_part_id=disposition_by_part_id,
                lossy=base_decision.lineage.lossy,
            ),
        )
        return HistoryViewPlan(
            keep=keep,
            artifacts=self._base_plan.artifacts,
            decision=decision,
        )

    def acknowledge_action(
        self,
        prepared: PreparedHistoryView,
        *,
        action_id: str,
        completion_id: str,
    ) -> None:
        super().acknowledge(prepared)
        self._last_acknowledged = prepared
        self._action_ids.append(action_id)
        self._action_turns.append(prepared.view.decision.decision_turn)
        self._completion_ids.append(completion_id)
        if len(self._action_ids) == self.actions_per_chunk:
            self._finalize_chunk(early_close_reason=None)
            self._needs_new_chunk = True

    def finalize_terminal(self) -> None:
        if self._action_ids:
            self._finalize_chunk(early_close_reason="terminal")
            self._needs_new_chunk = True

    def close_for_guard(self, *, guard_name: str) -> bool:
        """Close a non-empty chunk before its next action and allow recompaction."""

        if not self._action_ids:
            return False
        if self.pending_boundary is not None:
            raise RuntimeError("Cannot close a chunk for a guard with an unacknowledged boundary")
        self._finalize_chunk(early_close_reason=f"guard:{guard_name}")
        self._needs_new_chunk = True
        return True

    def _finalize_chunk(self, *, early_close_reason: str | None) -> None:
        if self._chunk_id is None or self._last_acknowledged is None:
            raise RuntimeError("Cannot finalize a chunk without a prepared action")
        if not self._action_ids:
            raise RuntimeError("Cannot finalize an empty chunk")

        retained_part_ids = {
            value.removeprefix("part:")
            for value in self._last_acknowledged.view.descriptor
            if value.startswith("part:")
        }
        active_group_ids = {
            part.observation_group_id
            for _, part in self.history.parts
            if part.kind == "image" and part.part_id in retained_part_ids
        }
        self._chunk_records.append(
            FinalizedChunkRecord(
                chunk_id=self._chunk_id,
                block_index=self._block_index,
                eligible_action_ids=tuple(self._action_ids),
                completion_evidence_ids=tuple(self._completion_ids),
                first_action_turn=self._action_turns[0],
                last_action_turn=self._action_turns[-1],
                configured_actions_per_chunk=self.actions_per_chunk,
                configured_history_groups=self.history_groups,
                actual_action_count=len(self._action_ids),
                early_close_reason=early_close_reason,
                active_observation_group_count=len(active_group_ids),
                active_raw_image_count=len(self._last_acknowledged.view.media_ids),
            )
        )
        self._action_ids = []
        self._action_turns = []
        self._completion_ids = []
        self._last_acknowledged = None

    def _make_boundary(
        self,
        *,
        applies_to_step: int,
        previous_view_digest: str,
        view: MaterializedHistoryView,
        current_view_digest: str,
    ) -> RewriteBoundaryEvent:
        boundary = super()._make_boundary(
            applies_to_step=applies_to_step,
            previous_view_digest=previous_view_digest,
            view=view,
            current_view_digest=current_view_digest,
        )
        return replace(
            boundary,
            schedule_name=self.schedule_name,
            schedule_version=self.schedule_version,
            schedule_config_digest=self.schedule_config_digest,
            chunk_id=self._chunk_id,
            block_index=self._block_index,
        )


class HistoryPolicy(Protocol):
    name: str
    version: str

    def plan(self, history: "SemanticHistory", *, decision_turn: int) -> HistoryViewPlan: ...


HistoryPolicyFactory = Callable[[Mapping[str, Any]], HistoryPolicy]
_CUSTOM_HISTORY_POLICY_FACTORIES: dict[str, HistoryPolicyFactory] = {}


def register_history_policy(
    name: str,
    factory: HistoryPolicyFactory,
    *,
    replace_existing: bool = False,
) -> None:
    """Register an agent-owned compaction policy factory.

    Custom policies receive their JSON-compatible configuration and return an
    object implementing :class:`HistoryPolicy`. Registration is process-local,
    so an agent package can register its protocol during import without making
    NeMo-Gym depend on that package.
    """

    if not name or name in {"identity", "recency"}:
        raise ValueError(f"Invalid custom history policy name {name!r}")
    if name in _CUSTOM_HISTORY_POLICY_FACTORIES and not replace_existing:
        raise ValueError(f"History policy {name!r} is already registered")
    _CUSTOM_HISTORY_POLICY_FACTORIES[name] = factory


def unregister_history_policy(name: str) -> None:
    """Remove a custom policy registration, primarily for isolated tests."""

    _CUSTOM_HISTORY_POLICY_FACTORIES.pop(name, None)


def _as_item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        value = item.model_dump()
    elif isinstance(item, Mapping):
        value = dict(item)
    else:
        raise TypeError(f"History items must be mappings or Pydantic models, got {type(item)!r}")
    return deepcopy(value)


def strip_completion_evidence(item: Any) -> dict[str, Any]:
    """Return semantic API content with private token evidence removed."""

    value = _as_item_dict(item)
    for key in _TOKEN_METADATA_FIELDS | _PRIVATE_ID_FIELDS:
        value.pop(key, None)

    content = value.get("content")
    if isinstance(content, list):
        stripped_content = []
        for part in content:
            if not isinstance(part, Mapping):
                stripped_content.append(deepcopy(part))
                continue
            stripped_part = dict(part)
            for key in _PRIVATE_ID_FIELDS:
                stripped_part.pop(key, None)
            stripped_part.pop("logprobs", None)
            stripped_content.append(stripped_part)
        value["content"] = stripped_content
    return value


def _part_kind(*, role: str, item_type: str, part_type: str | None, initial: bool) -> SemanticPartKind:
    if part_type in _IMAGE_PART_TYPES:
        return "image"
    if item_type == "reasoning":
        return "reasoning"
    if item_type in {"function_call"} or role == "assistant":
        return "assistant_action"
    if item_type == "function_call_output":
        return "environment_text"
    if role in {"system", "developer"}:
        return "system_text"
    if initial:
        return "user_text"
    return "environment_text"


class SemanticHistory:
    """Append-only semantic history used as the implementation shadow oracle."""

    def __init__(self, rollout_id: str):
        if not rollout_id:
            raise ValueError("rollout_id must be non-empty")
        self.rollout_id = rollout_id
        self.media_arena = MediaArena()
        self._events: list[SemanticEvent] = []
        self._next_event = 0

    @property
    def events(self) -> tuple[SemanticEvent, ...]:
        return tuple(self._events)

    @property
    def parts(self) -> tuple[tuple[SemanticEvent, SemanticPart], ...]:
        return tuple((event, part) for event in self._events for part in event.parts)

    def append_items(
        self,
        items: Sequence[Any] | str,
        *,
        turn_id: int,
        is_initial_context: bool = False,
        conditions_action_turn: int | None = None,
    ) -> tuple[SemanticEvent, ...]:
        if isinstance(items, str):
            items = [{"role": "user", "type": "message", "content": items}]

        appended: list[SemanticEvent] = []
        for raw_item in items:
            semantic_item = strip_completion_evidence(raw_item)
            private_item = _as_item_dict(raw_item)
            event_number = self._next_event
            self._next_event += 1
            event_id = private_item.get("_nemo_gym_event_id", f"event-{event_number:06d}")
            if any(event.event_id == event_id for event in self._events):
                raise ValueError(f"Duplicate semantic event ID {event_id!r}")

            role = str(semantic_item.get("role") or "unknown")
            item_type = str(semantic_item.get("type") or "message")
            content = semantic_item.get("content")
            parts: list[SemanticPart] = []

            if isinstance(content, list):
                default_group_id = private_item.get(
                    "_nemo_gym_observation_group_id",
                    f"observation-{event_number:06d}",
                )
                private_content = private_item.get("content") or []
                for content_index, content_part in enumerate(content):
                    private_part = (
                        private_content[content_index]
                        if content_index < len(private_content) and isinstance(private_content[content_index], Mapping)
                        else {}
                    )
                    part_type = (
                        str(content_part.get("type"))
                        if isinstance(content_part, Mapping) and content_part.get("type") is not None
                        else None
                    )
                    part_id = private_part.get(
                        "_nemo_gym_part_id",
                        f"part-{event_number:06d}-{content_index:03d}",
                    )
                    kind = _part_kind(
                        role=role,
                        item_type=item_type,
                        part_type=part_type,
                        initial=is_initial_context,
                    )
                    declared_kind = private_part.get("_nemo_gym_semantic_kind")
                    if declared_kind is not None:
                        kind = str(declared_kind)
                    if kind not in _REGISTERED_SEMANTIC_PART_KINDS:
                        raise ValueError(f"Unregistered semantic part kind {kind!r}")
                    observation_group_id = None
                    media_id = None
                    if kind == "image":
                        observation_group_id = private_part.get("_nemo_gym_observation_group_id", default_group_id)
                        media_id = self.media_arena.register(content_part)
                        semantic_content = semantic_item.get("content")
                        assert isinstance(semantic_content, list)
                        semantic_content[content_index] = {
                            "type": part_type,
                            "_nemo_gym_media_id": media_id,
                        }
                    parts.append(
                        SemanticPart(
                            part_id=part_id,
                            kind=kind,
                            content_index=content_index,
                            observation_group_id=observation_group_id,
                            media_id=media_id,
                        )
                    )
            else:
                part_id = private_item.get("_nemo_gym_part_id", f"part-{event_number:06d}-000")
                kind = str(
                    private_item.get("_nemo_gym_semantic_kind")
                    or _part_kind(
                        role=role,
                        item_type=item_type,
                        part_type=None,
                        initial=is_initial_context,
                    )
                )
                if kind not in _REGISTERED_SEMANTIC_PART_KINDS:
                    raise ValueError(f"Unregistered semantic part kind {kind!r}")
                parts.append(
                    SemanticPart(
                        part_id=part_id,
                        kind=kind,
                        content_index=None,
                    )
                )

            event = SemanticEvent(
                event_id=event_id,
                turn_id=turn_id,
                role=role,
                item=semantic_item,
                parts=tuple(parts),
                is_initial_context=is_initial_context,
                conditions_action_turn=conditions_action_turn,
            )
            self._events.append(event)
            appended.append(event)
        return tuple(appended)


def _config_digest(value: Mapping[str, Any]) -> str:
    return canonical_digest(value)


def _semantic_part_digest(event: SemanticEvent, part: SemanticPart) -> str:
    content = event.item.get("content")
    if isinstance(content, list):
        if part.content_index is None:
            raise ValueError(f"Semantic part {part.part_id!r} has no content index")
        payload = content[part.content_index]
    else:
        payload = event.item
    return canonical_digest(
        {
            "kind": part.kind,
            "payload": payload,
            "media_id": part.media_id,
            "observation_group_id": part.observation_group_id,
        }
    )


def _lineage_record(
    history: SemanticHistory,
    *,
    transformation_type: str,
    transformation_version: str,
    configuration_digest: str,
    disposition_by_part_id: Mapping[
        str,
        tuple[LineageDisposition, tuple[str, ...], tuple[str, ...]],
    ],
    lossy: bool,
) -> TransformationLineageRecord:
    unit_records: list[UnitLineageRecord] = []
    for event, part in history.parts:
        try:
            disposition, output_ids, output_digests = disposition_by_part_id[part.part_id]
        except KeyError as exc:
            raise ValueError(f"Lineage does not account for semantic part {part.part_id!r}") from exc
        unit_records.append(
            UnitLineageRecord(
                source_unit_id=part.part_id,
                source_digest=_semantic_part_digest(event, part),
                disposition=disposition,
                output_unit_ids=output_ids,
                output_digests=output_digests,
            )
        )
    identity = canonical_digest(
        {
            "type": transformation_type,
            "version": transformation_version,
            "config": configuration_digest,
            "units": unit_records,
        }
    )
    return TransformationLineageRecord(
        transformation_id=f"transform-{identity[:24]}",
        transformation_type=transformation_type,
        transformation_version=transformation_version,
        configuration_digest=configuration_digest,
        deterministic=True,
        lossy=lossy,
        generator_contract_id=None,
        unit_records=tuple(unit_records),
        validator_result="passed",
    )


class IdentityHistoryPolicy:
    name = "identity"
    version = "1"

    def plan(self, history: SemanticHistory, *, decision_turn: int) -> HistoryViewPlan:
        keep = tuple(KeepPartRef(event.event_id, part.part_id) for event, part in history.parts)
        retained = tuple(ref.part_id for ref in keep)
        config_digest = _config_digest({"type": self.name})
        disposition_by_part_id = {
            part.part_id: (
                "kept",
                (part.part_id,),
                (_semantic_part_digest(event, part),),
            )
            for event, part in history.parts
        }
        decision = PolicyDecisionRecord(
            policy_name=self.name,
            policy_version=self.version,
            config_digest=config_digest,
            protected_part_ids=(),
            changed_part_ranges=(),
            retained_part_count=len(retained),
            omitted_part_count=0,
            selection_digest=_ordered_id_digest(retained),
            inserted_artifact_ids=(),
            decision_turn=decision_turn,
            lineage=_lineage_record(
                history,
                transformation_type="identity",
                transformation_version=self.version,
                configuration_digest=config_digest,
                disposition_by_part_id=disposition_by_part_id,
                lossy=False,
            ),
        )
        return HistoryViewPlan(keep=keep, artifacts=(), decision=decision)


class VisualRecencyHistoryPolicy:
    name = "recency"
    version = "1"

    def __init__(self, config: RecencyHistoryPolicyConfig):
        if not config.keep_all_text:
            raise NotImplementedError("The initial visual-recency policy requires keep_all_text=true")
        self.config = config
        self._config_dict = config.model_dump(mode="json")
        self._digest = _config_digest({"type": self.name, "config": self._config_dict})

    def plan(self, history: SemanticHistory, *, decision_turn: int) -> HistoryViewPlan:
        ordered_parts = history.parts
        groups: list[tuple[str, list[tuple[SemanticEvent, SemanticPart]]]] = []
        group_index: dict[str, int] = {}
        for event, part in ordered_parts:
            if part.kind != "image":
                continue
            assert part.observation_group_id is not None
            group_id = part.observation_group_id
            if group_id not in group_index:
                group_index[group_id] = len(groups)
                groups.append((group_id, []))
            groups[group_index[group_id]][1].append((event, part))

        protected_group_ids = {
            group_id
            for group_id, members in groups
            if self.config.protect_initial_context and any(event.is_initial_context for event, _ in members)
        }
        pending_group_ids = {
            group_id
            for group_id, members in groups
            if any(event.conditions_action_turn == decision_turn for event, _ in members)
        }
        later_group_ids = [
            group_id
            for group_id, _ in groups
            if group_id not in protected_group_ids and group_id not in pending_group_ids
        ]
        retained_later_group_ids = set(
            later_group_ids[-self.config.keep_last_image_groups :] if self.config.keep_last_image_groups else []
        )
        retained_group_ids = protected_group_ids | pending_group_ids | retained_later_group_ids

        keep: list[KeepPartRef] = []
        protected_part_ids: list[str] = []
        omitted_part_ids: list[str] = []
        for event, part in ordered_parts:
            retained = part.kind != "image" or (part.observation_group_id in retained_group_ids)
            if retained:
                keep.append(KeepPartRef(event.event_id, part.part_id))
                if part.kind == "image" and part.observation_group_id in (protected_group_ids | pending_group_ids):
                    protected_part_ids.append(part.part_id)
            else:
                omitted_part_ids.append(part.part_id)

        artifacts: list[OmissionArtifact] = []
        artifact_by_source_part: dict[str, OmissionArtifact] = {}
        marker = self.config.image_omission_marker
        omitted_group_runs: list[list[tuple[str, list[tuple[SemanticEvent, SemanticPart]]]]] = []
        current_run: list[tuple[str, list[tuple[SemanticEvent, SemanticPart]]]] = []
        for group in groups:
            group_id, _ = group
            if group_id not in retained_group_ids:
                current_run.append(group)
            elif current_run:
                omitted_group_runs.append(current_run)
                current_run = []
        if current_run:
            omitted_group_runs.append(current_run)

        if marker:
            for run in omitted_group_runs:
                source_parts = tuple(part.part_id for _, members in run for _, part in members)
                artifact = OmissionArtifact(
                    artifact_id=(f"omission-{self._digest[:12]}-{source_parts[0]}"),
                    source_first_part_id=source_parts[0],
                    source_last_part_id=source_parts[-1],
                    source_part_count=len(source_parts),
                    source_digest=_ordered_id_digest(source_parts),
                    text=marker,
                    anchor_part_id=source_parts[0],
                )
                artifacts.append(artifact)
                for part_id in source_parts:
                    artifact_by_source_part[part_id] = artifact

        retained_part_ids = tuple(ref.part_id for ref in keep)
        retained_part_id_set = set(retained_part_ids)
        disposition_by_part_id: dict[
            str,
            tuple[LineageDisposition, tuple[str, ...], tuple[str, ...]],
        ] = {}
        for event, part in ordered_parts:
            source_digest = _semantic_part_digest(event, part)
            if part.part_id in retained_part_id_set:
                disposition_by_part_id[part.part_id] = (
                    "kept",
                    (part.part_id,),
                    (source_digest,),
                )
                continue
            artifact = artifact_by_source_part.get(part.part_id)
            if artifact is None:
                disposition_by_part_id[part.part_id] = ("dropped", (), ())
            else:
                disposition_by_part_id[part.part_id] = (
                    "replaced",
                    (artifact.artifact_id,),
                    (
                        canonical_digest(
                            {
                                "type": "omission_marker",
                                "artifact_id": artifact.artifact_id,
                                "text": artifact.text,
                            }
                        ),
                    ),
                )
        decision = PolicyDecisionRecord(
            policy_name=self.name,
            policy_version=self.version,
            config_digest=self._digest,
            protected_part_ids=tuple(protected_part_ids),
            changed_part_ranges=tuple(
                (artifact.source_first_part_id, artifact.source_last_part_id) for artifact in artifacts
            ),
            retained_part_count=len(retained_part_ids),
            omitted_part_count=len(omitted_part_ids),
            selection_digest=_ordered_id_digest((*retained_part_ids, "--omitted--", *omitted_part_ids)),
            inserted_artifact_ids=tuple(artifact.artifact_id for artifact in artifacts),
            decision_turn=decision_turn,
            lineage=_lineage_record(
                history,
                transformation_type="visual_recency",
                transformation_version=self.version,
                configuration_digest=self._digest,
                disposition_by_part_id=disposition_by_part_id,
                lossy=bool(omitted_part_ids),
            ),
        )
        return HistoryViewPlan(keep=tuple(keep), artifacts=tuple(artifacts), decision=decision)


def build_history_policy(config: HistoryPolicyConfig) -> HistoryPolicy:
    if config.type == "identity":
        return IdentityHistoryPolicy()
    if config.type == "recency":
        recency_config = (
            config.config
            if isinstance(config.config, RecencyHistoryPolicyConfig)
            else RecencyHistoryPolicyConfig.model_validate(config.config)
        )
        return VisualRecencyHistoryPolicy(recency_config)

    factory = _CUSTOM_HISTORY_POLICY_FACTORIES.get(config.type)
    if factory is None:
        available = sorted({"identity", "recency", *_CUSTOM_HISTORY_POLICY_FACTORIES})
        raise ValueError(f"Unknown history policy {config.type!r}; available policies: {available}")
    raw_config = (
        config.config.model_dump(mode="json") if isinstance(config.config, BaseModel) else deepcopy(config.config)
    )
    policy = factory(raw_config)
    if not isinstance(getattr(policy, "name", None), str) or not isinstance(getattr(policy, "version", None), str):
        raise TypeError(f"Custom history policy {config.type!r} must expose string name and version attributes")
    if not callable(getattr(policy, "plan", None)):
        raise TypeError(f"Custom history policy {config.type!r} must implement plan()")
    return policy


def _marker_part_for(image_part: Mapping[str, Any], text: str) -> dict[str, str]:
    part_type = image_part.get("type")
    if part_type == "input_image":
        return {"type": "input_text", "text": text}
    return {"type": "text", "text": text}


def materialize_history_view(history: SemanticHistory, plan: HistoryViewPlan) -> MaterializedHistoryView:
    retained = plan.retained_part_ids
    artifacts_by_anchor = {artifact.anchor_part_id: artifact for artifact in plan.artifacts}
    items: list[Mapping[str, Any]] = []
    media_ids: list[str] = []
    descriptor: list[str] = []

    for event in history.events:
        item = deepcopy(dict(event.item))
        content = item.get("content")
        if isinstance(content, list):
            materialized_content: list[Any] = []
            for part in event.parts:
                assert part.content_index is not None
                source_part = content[part.content_index]
                artifact = artifacts_by_anchor.get(part.part_id)
                if artifact is not None:
                    if not isinstance(source_part, Mapping):
                        raise TypeError(f"Cannot anchor omission marker at non-mapping part {part.part_id}")
                    materialized_content.append(_marker_part_for(source_part, artifact.text))
                    descriptor.append(f"artifact:{artifact.artifact_id}")

                if part.part_id not in retained:
                    continue
                if part.kind == "image":
                    if part.media_id is None:
                        raise ValueError(f"Image part {part.part_id} has no media ID")
                    materialized_content.append(deepcopy(dict(history.media_arena.resolve(part.media_id))))
                    media_ids.append(part.media_id)
                else:
                    materialized_content.append(deepcopy(source_part))
                descriptor.append(f"part:{part.part_id}")

            if not materialized_content:
                continue
            item["content"] = materialized_content
            items.append(item)
            continue

        part = event.parts[0]
        artifact = artifacts_by_anchor.get(part.part_id)
        if artifact is not None:
            items.append(
                {
                    "role": event.role if event.role != "unknown" else "user",
                    "type": "message",
                    "content": artifact.text,
                }
            )
            descriptor.append(f"artifact:{artifact.artifact_id}")
        if part.part_id in retained:
            items.append(item)
            descriptor.append(f"part:{part.part_id}")

    return MaterializedHistoryView(
        items=tuple(items),
        media_ids=tuple(media_ids),
        descriptor=tuple(descriptor),
        decision=plan.decision,
    )


def descriptor_is_append_compatible(
    previous_completed_descriptor: Sequence[str] | None,
    current_descriptor: Sequence[str],
) -> bool:
    """Return whether the current semantic view only appends to the prior one."""

    if previous_completed_descriptor is None:
        return False
    prefix = tuple(previous_completed_descriptor)
    current = tuple(current_descriptor)
    return len(current) >= len(prefix) and current[: len(prefix)] == prefix


def ordered_media_is_append_compatible(
    previous_media_ids: Sequence[str] | None,
    current_media_ids: Sequence[str],
) -> bool:
    if previous_media_ids is None:
        return False
    prefix = tuple(previous_media_ids)
    current = tuple(current_media_ids)
    return len(current) >= len(prefix) and current[: len(prefix)] == prefix


def evaluate_context_guards(
    config: ContextGuardConfig,
    measurements: ContextMeasurements,
) -> tuple[GuardEvaluation, ...]:
    """Evaluate every configured hard limit without changing history state."""

    if (
        min(
            measurements.prompt_token_count,
            measurements.active_image_count,
            measurements.vision_token_count,
        )
        < 0
    ):
        raise ValueError("Context measurements must be non-negative")

    checks: list[tuple[str, int, int | None]] = [
        (
            "total_tokens",
            measurements.prompt_token_count + config.reserved_generation_tokens,
            config.max_total_tokens,
        ),
        (
            "active_images",
            measurements.active_image_count,
            config.max_active_images,
        ),
        (
            "vision_tokens",
            measurements.vision_token_count,
            config.max_vision_tokens,
        ),
    ]
    return tuple(
        GuardEvaluation(
            guard_name=name,
            measured_value=value,
            configured_limit=limit,
            exceeded=value > limit,
            excess=max(value - limit, 0),
        )
        for name, value, limit in checks
        if limit is not None
    )


def pending_observation_group_ids(
    history: SemanticHistory,
    *,
    applies_to_step: int,
) -> tuple[str, ...]:
    group_ids: list[str] = []
    for event, part in history.parts:
        if (
            part.kind == "image"
            and event.conditions_action_turn == applies_to_step
            and part.observation_group_id not in group_ids
        ):
            assert part.observation_group_id is not None
            group_ids.append(part.observation_group_id)
    return tuple(group_ids)


def build_guard_outcome_records(
    *,
    rollout_id: str,
    chunk_id: str | None,
    applies_to_step: int,
    completed_action_count: int,
    pending_group_ids: Sequence[str],
    before: Sequence[GuardEvaluation],
    after: Sequence[GuardEvaluation] | None,
    early_chunk_close: bool,
) -> tuple[GuardOutcomeRecord, ...]:
    after_by_name = {evaluation.guard_name: evaluation for evaluation in (after or ())}
    records: list[GuardOutcomeRecord] = []
    for evaluation in before:
        post = after_by_name.get(evaluation.guard_name)
        if not evaluation.exceeded:
            decision: Literal["admit", "admit_after_compaction", "reject"] = "admit"
        elif post is not None and not post.exceeded:
            decision = "admit_after_compaction"
        else:
            decision = "reject"
        records.append(
            GuardOutcomeRecord(
                rollout_id=rollout_id,
                chunk_id=chunk_id,
                applies_to_step=applies_to_step,
                completed_action_count=completed_action_count,
                pending_observation_group_ids=tuple(pending_group_ids),
                guard_name=evaluation.guard_name,
                measured_value=evaluation.measured_value,
                configured_limit=evaluation.configured_limit,
                early_chunk_close=early_chunk_close,
                post_compaction_value=(post.measured_value if post is not None else None),
                decision=decision,
            )
        )
    return tuple(records)


def normalize_semantic_items(items: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Normalize a legacy request input for comparison with a semantic view."""

    return tuple(strip_completion_evidence(item) for item in items)


def assert_identity_shadow_matches(legacy_items: Sequence[Any], view: MaterializedHistoryView) -> None:
    """Fail with bounded diagnostics if the identity shadow changes semantics."""

    normalized_legacy = normalize_semantic_items(legacy_items)
    if normalized_legacy == view.items:
        return
    raise RuntimeError(
        "Identity history shadow mismatch: "
        f"legacy_items={len(normalized_legacy)} "
        f"shadow_items={len(view.items)} "
        f"legacy_digest={_semantic_items_digest(normalized_legacy)} "
        f"shadow_digest={_semantic_items_digest(view.items)}"
    )


def capture_observed_completion(
    output_items: Sequence[Any],
    *,
    rollout_id: str,
    turn_id: int,
    media_ids: Sequence[str],
    policy_decision: PolicyDecisionRecord,
    prepared_request_id: str,
    context_epoch: int,
    segment_index: int,
    segment_id: str,
    expected_append_compatible: bool,
    compaction_event_id: str | None,
    generation_contract_id: str,
    finish_reason: str | None = None,
    processor_fingerprint: str | None = None,
    required_prefix_token_ids: Sequence[int] | None = None,
) -> ObservedCompletion:
    """Extract one exact completion record without retaining semantic payloads."""

    evidence_items: list[dict[str, Any]] = []
    required_fields = {
        "prompt_token_ids",
        "generation_token_ids",
        "generation_log_probs",
    }
    for item in output_items:
        value = _as_item_dict(item)
        if required_fields <= value.keys():
            evidence_items.append(value)

    if len(evidence_items) != 1:
        raise RuntimeError(
            f"Expected exactly one generation evidence item for a model call, found {len(evidence_items)}"
        )

    evidence = evidence_items[0]
    prompt_token_ids = tuple(evidence["prompt_token_ids"])
    sampled_token_ids = tuple(evidence["generation_token_ids"])
    sampled_logprobs = tuple(evidence["generation_log_probs"])
    required_prefix = tuple(required_prefix_token_ids or ())
    if required_prefix and prompt_token_ids[: len(required_prefix)] != required_prefix:
        raise RuntimeError(
            "Generation-observed prompt does not contain the required exact "
            "prefix: "
            f"required_count={len(required_prefix)} "
            f"prompt_count={len(prompt_token_ids)}"
        )
    identity = _config_digest(
        {
            "rollout_id": rollout_id,
            "turn_id": turn_id,
            "prompt_token_ids": prompt_token_ids,
            "sampled_token_ids": sampled_token_ids,
        }
    )
    completion_id = f"completion-{turn_id:06d}-{identity[:12]}"
    action_id = f"action-{turn_id:06d}"
    request_id = stable_id(
        "request",
        prepared_request_id,
        prompt_token_ids,
        tuple(media_ids),
    )
    model_call_id = stable_id("model-call", request_id, completion_id)
    policy_output_span = PolicyOutputSpan(
        policy_output_span_id=stable_id(
            "policy-output-span",
            model_call_id,
            action_id,
            len(sampled_token_ids),
        ),
        model_call_id=model_call_id,
        action_ids=(action_id,),
        start=0,
        end=len(sampled_token_ids),
        eligible=True,
    )
    occurrence_counts: Counter[str] = Counter()
    media_occurrences: list[MediaOccurrence] = []
    for media_id in media_ids:
        ordinal = occurrence_counts[media_id]
        occurrence_counts[media_id] += 1
        media_occurrences.append(
            MediaOccurrence(
                media_id=media_id,
                occurrence_ordinal=ordinal,
                model_call_id=model_call_id,
            )
        )
    return ObservedCompletion(
        rollout_id=rollout_id,
        completion_id=completion_id,
        action_id=action_id,
        turn_id=turn_id,
        prepared_request_id=prepared_request_id,
        request_id=request_id,
        context_epoch=context_epoch,
        segment_index=segment_index,
        segment_id=segment_id,
        expected_append_compatible=expected_append_compatible,
        compaction_event_id=compaction_event_id,
        prompt_token_ids=prompt_token_ids,
        sampled_token_ids=sampled_token_ids,
        sampled_logprobs=sampled_logprobs,
        finish_reason=finish_reason,
        media_ids=tuple(media_ids),
        policy_decision=PolicyDecisionEvidence(
            policy_name=policy_decision.policy_name,
            policy_version=policy_decision.policy_version,
            config_digest=policy_decision.config_digest,
            decision_turn=policy_decision.decision_turn,
            selection_digest=policy_decision.selection_digest,
            transformation_id=(policy_decision.lineage.transformation_id),
        ),
        generation_contract_id=generation_contract_id,
        policy_output_spans=(policy_output_span,),
        media_occurrences=tuple(media_occurrences),
        processor_fingerprint=processor_fingerprint,
    )


def _ordered_id_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(ids).encode("utf-8")).hexdigest()


def _semantic_items_digest(items: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        items,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _view_digest(view: MaterializedHistoryView) -> str:
    payload = {
        "items_digest": _semantic_items_digest(view.items),
        "descriptor": view.descriptor,
        "media_ids": view.media_ids,
    }
    return _config_digest(payload)


# Temporary compatibility alias while the prototype is integrated. The object
# is a validation representation, not a production retention commitment.
CanonicalHistory = SemanticHistory
