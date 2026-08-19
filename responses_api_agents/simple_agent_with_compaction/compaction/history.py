# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only semantic history and exact generation evidence.

This module deliberately does not tokenize requests or construct training
rows. It owns the semantic, generation-time side of context compaction:

* a complete, append-only semantic history for deriving request views;
* stable media identities with occurrence-preserving request order;
* semantic events and stable media identities;
* policy plans, lineage, and exact generation contracts; and
* exact completion evidence captured from model responses.

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
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

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
    policy_config_digest: str
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
    """Append-only semantic history used to derive model request views."""

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


def normalize_semantic_items(items: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Normalize a legacy request input for comparison with a semantic view."""

    return tuple(strip_completion_evidence(item) for item in items)


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


# Compatibility alias retained for existing callers.
CanonicalHistory = SemanticHistory
