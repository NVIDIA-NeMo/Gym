# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite-boundary controllers and context-admission guards."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Literal, Sequence

from responses_api_agents.simple_agent_with_compaction.compaction.config import ContextGuardConfig
from responses_api_agents.simple_agent_with_compaction.compaction.history import (
    ContextMeasurements,
    FinalizedChunkRecord,
    GuardEvaluation,
    GuardOutcomeRecord,
    HistoryViewPlan,
    KeepPartRef,
    MaterializedHistoryView,
    PreparedHistoryView,
    RewriteBoundaryEvent,
    SemanticHistory,
    _config_digest,
    _lineage_record,
    _ordered_id_digest,
    _semantic_part_digest,
    _view_digest,
)
from responses_api_agents.simple_agent_with_compaction.compaction.materialization import (
    descriptor_is_append_compatible,
    materialize_history_view,
    ordered_media_is_append_compatible,
)
from responses_api_agents.simple_agent_with_compaction.compaction.policies import HistoryPolicy


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
    ):
        if actions_per_chunk < 1:
            raise ValueError("actions_per_chunk must be at least 1")
        super().__init__(history, policy)
        self.actions_per_chunk = actions_per_chunk
        self.schedule_config_digest = _config_digest(
            {
                "schedule": self.schedule_name,
                "version": self.schedule_version,
                "actions_per_chunk": actions_per_chunk,
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
                policy_config_digest=(self._last_acknowledged.view.decision.config_digest),
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
