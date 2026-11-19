# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-history policy protocols, registry, and built-in policies."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Protocol

from pydantic import BaseModel

from responses_api_agents.simple_agent_with_compaction.compaction.config import (
    HistoryPolicyConfig,
    RecencyHistoryPolicyConfig,
)
from responses_api_agents.simple_agent_with_compaction.compaction.history import (
    HistoryViewPlan,
    KeepPartRef,
    LineageDisposition,
    OmissionArtifact,
    PolicyDecisionRecord,
    SemanticEvent,
    SemanticHistory,
    SemanticPart,
    _config_digest,
    _lineage_record,
    _ordered_id_digest,
    _semantic_part_digest,
    canonical_digest,
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


class RecencyHistoryPolicy:
    name = "recency"
    version = "2"

    def __init__(self, config: RecencyHistoryPolicyConfig):
        self.config = config
        self._config_dict = config.model_dump(mode="json")
        self._digest = _config_digest({"type": self.name, "config": self._config_dict})

    def plan(self, history: SemanticHistory, *, decision_turn: int) -> HistoryViewPlan:
        ordered_parts = history.parts
        image_groups: list[tuple[str, list[tuple[SemanticEvent, SemanticPart]]]] = []
        image_group_index: dict[str, int] = {}
        for event, part in ordered_parts:
            if part.kind != "image":
                continue
            assert part.observation_group_id is not None
            group_id = part.observation_group_id
            if group_id not in image_group_index:
                image_group_index[group_id] = len(image_groups)
                image_groups.append((group_id, []))
            image_groups[image_group_index[group_id]][1].append((event, part))

        all_image_group_ids = {group_id for group_id, _ in image_groups}
        protected_image_group_ids: set[str] = set()
        pending_image_group_ids: set[str] = set()
        retained_image_group_ids = set(all_image_group_ids)
        if self.config.images.enabled:
            protected_image_group_ids = {
                group_id
                for group_id, members in image_groups
                if self.config.images.protect_initial_context and any(event.is_initial_context for event, _ in members)
            }
            pending_image_group_ids = {
                group_id
                for group_id, members in image_groups
                if any(event.conditions_action_turn == decision_turn for event, _ in members)
            }
            recency_candidates = [
                group_id for group_id, _ in image_groups if group_id not in protected_image_group_ids
            ]
            retained_recent = set(
                recency_candidates[-self.config.images.keep_last_groups :]
                if self.config.images.keep_last_groups
                else []
            )
            # Pending observations count toward the configured recency window,
            # but are unioned back for a zero-sized window because they must
            # condition the next action.
            retained_image_group_ids = protected_image_group_ids | retained_recent | pending_image_group_ids

        reasoning_turns: list[int] = []
        seen_reasoning_turns: set[int] = set()
        for event, part in ordered_parts:
            if part.kind == "reasoning" and event.turn_id not in seen_reasoning_turns:
                seen_reasoning_turns.add(event.turn_id)
                reasoning_turns.append(event.turn_id)

        protected_reasoning_turns: set[int] = set()
        retained_reasoning_turns = set(reasoning_turns)
        if self.config.reasoning.enabled:
            if self.config.reasoning.keep_first_block and reasoning_turns:
                protected_reasoning_turns.add(reasoning_turns[0])
            recency_candidates = [turn_id for turn_id in reasoning_turns if turn_id not in protected_reasoning_turns]
            retained_recent = set(
                recency_candidates[-self.config.reasoning.keep_last_blocks :]
                if self.config.reasoning.keep_last_blocks
                else []
            )
            retained_reasoning_turns = protected_reasoning_turns | retained_recent

        keep: list[KeepPartRef] = []
        protected_part_ids: list[str] = []
        omitted_part_ids: list[str] = []
        for event, part in ordered_parts:
            if part.kind == "image":
                retained = part.observation_group_id in retained_image_group_ids
                protected = part.observation_group_id in (protected_image_group_ids | pending_image_group_ids)
            elif part.kind == "reasoning":
                retained = event.turn_id in retained_reasoning_turns
                protected = event.turn_id in protected_reasoning_turns
            else:
                retained = True
                protected = False
            if retained:
                keep.append(KeepPartRef(event.event_id, part.part_id))
                if protected:
                    protected_part_ids.append(part.part_id)
            else:
                omitted_part_ids.append(part.part_id)

        artifacts: list[OmissionArtifact] = []
        artifact_by_source_part: dict[str, OmissionArtifact] = {}
        marker = self.config.images.omission_marker if self.config.images.enabled else None
        omitted_group_runs: list[list[tuple[str, list[tuple[SemanticEvent, SemanticPart]]]]] = []
        current_run: list[tuple[str, list[tuple[SemanticEvent, SemanticPart]]]] = []
        for group in image_groups:
            group_id, _ = group
            if group_id not in retained_image_group_ids:
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

        omitted_part_id_set = set(omitted_part_ids)
        changed_part_ranges: list[tuple[str, str]] = []
        current_omitted_run: list[str] = []
        for _, part in ordered_parts:
            if part.part_id in omitted_part_id_set:
                current_omitted_run.append(part.part_id)
            elif current_omitted_run:
                changed_part_ranges.append((current_omitted_run[0], current_omitted_run[-1]))
                current_omitted_run = []
        if current_omitted_run:
            changed_part_ranges.append((current_omitted_run[0], current_omitted_run[-1]))

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
            changed_part_ranges=tuple(changed_part_ranges),
            retained_part_count=len(retained_part_ids),
            omitted_part_count=len(omitted_part_ids),
            selection_digest=_ordered_id_digest((*retained_part_ids, "--omitted--", *omitted_part_ids)),
            inserted_artifact_ids=tuple(artifact.artifact_id for artifact in artifacts),
            decision_turn=decision_turn,
            lineage=_lineage_record(
                history,
                transformation_type="recency",
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
        return RecencyHistoryPolicy(recency_config)

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
