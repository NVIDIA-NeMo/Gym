# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable per-rollout context-compaction orchestration for Gym agents."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInput,
)
from nemo_gym.visual_history import (
    ContextMeasurements,
    FinalizedChunkRecord,
    GenerationContract,
    GuardOutcomeRecord,
    HistoryController,
    MediaOccurrence,
    ObservedCompletion,
    PolicyDecisionEvidence,
    PolicyDecisionRecord,
    PolicyOutputSpan,
    PreparedHistoryView,
    RewriteBoundaryEvent,
    SemanticHistory,
    TransformationLineageDeltaRecord,
    TurnChunkedHistoryController,
    UnitLineageRecord,
    VisualHistoryConfig,
    assert_identity_shadow_matches,
    build_guard_outcome_records,
    build_history_policy,
    build_lineage_delta,
    canonical_digest,
    capture_observed_completion,
    evaluate_context_guards,
    pending_observation_group_ids,
    stable_id,
)


LOGGER = logging.getLogger(__name__)


class ContextCompactionContract(BaseModel):
    """Versioned marker that makes exact Gym evidence authoritative."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    mode: Literal["exact_trace_authority"] = "exact_trace_authority"
    rollout_id: str
    group_id: str | None = None
    task_id: str | None = None
    rollout_index: int | None = Field(default=None, ge=0)
    attempt_index: int | None = Field(default=None, ge=0)
    generation_contract: GenerationContract


class TransportContextCompactionContract(BaseModel):
    """Post-verification contract for the bounded Gym-to-NeMo-RL envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3] = 3
    mode: Literal["exact_trace_authority"] = "exact_trace_authority"
    rollout_id: str
    group_id: str | None = None
    task_id: str | None = None
    rollout_index: int | None = Field(default=None, ge=0)
    attempt_index: int | None = Field(default=None, ge=0)
    generation_contract: GenerationContract


class ModelCallMetadata(BaseModel):
    """Bounded model-call sidecar bound to canonical output token arrays."""

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
    finish_reason: str | None = None
    media_ids: tuple[str, ...]
    policy_decision: PolicyDecisionEvidence
    generation_contract_id: str
    policy_output_spans: tuple[PolicyOutputSpan, ...]
    media_occurrences: tuple[MediaOccurrence, ...]
    processor_fingerprint: str | None = None
    eligible: bool = True
    evidence_source: Literal["generation_response"] = "generation_response"
    generation_evidence_digest: str

    @classmethod
    def from_observed(cls, observed: ObservedCompletion) -> "ModelCallMetadata":
        payload = observed.model_dump(
            exclude={
                "prompt_token_ids",
                "sampled_token_ids",
                "sampled_logprobs",
            }
        )
        payload["generation_evidence_digest"] = canonical_digest(
            {
                "prompt_token_ids": observed.prompt_token_ids,
                "sampled_token_ids": observed.sampled_token_ids,
                "sampled_logprobs": observed.sampled_logprobs,
            }
        )
        return cls.model_validate(payload)


class ContextCompactedResponse(NeMoGymResponse):
    """Ordinary Gym response plus exact context-compaction evidence."""

    agent_input: NeMoGymResponseInput
    seed_obs: NeMoGymResponseInput = Field(default_factory=list)
    media_assets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    completion_evidence: list[ObservedCompletion] = Field(default_factory=list)
    final_policy_decision: PolicyDecisionRecord | None = None
    lineage_deltas: list[TransformationLineageDeltaRecord] = Field(default_factory=list)
    chunk_records: list[FinalizedChunkRecord] = Field(default_factory=list)
    boundary_events: list[RewriteBoundaryEvent] = Field(default_factory=list)
    guard_records: list[GuardOutcomeRecord] = Field(default_factory=list)
    context_compaction_contract: ContextCompactionContract


class ContextCompactedTransportResponse(NeMoGymResponse):
    """Bounded exact-evidence response returned after resource verification."""

    media_assets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    model_call_metadata: list[ModelCallMetadata] = Field(default_factory=list)
    final_policy_decision: PolicyDecisionRecord | None = None
    lineage_deltas: list[TransformationLineageDeltaRecord] = Field(default_factory=list)
    chunk_records: list[FinalizedChunkRecord] = Field(default_factory=list)
    boundary_events: list[RewriteBoundaryEvent] = Field(default_factory=list)
    guard_records: list[GuardOutcomeRecord] = Field(default_factory=list)
    context_compaction_contract: TransportContextCompactionContract


def build_transport_response(
    response: ContextCompactedResponse,
) -> ContextCompactedTransportResponse:
    """Drop validation-only duplication after the resource verifier succeeds."""

    result = response.model_dump(
        exclude={
            "agent_input",
            "seed_obs",
            "completion_evidence",
        }
    )
    projected_output = []
    for item in result["output"]:
        projected_item = dict(item)
        content = projected_item.get("content")
        if isinstance(content, list):
            projected_item["content"] = [
                part
                for part in content
                if not (isinstance(part, Mapping) and part.get("type") in {"input_image", "image", "image_url"})
            ]
        projected_output.append(projected_item)
    result["output"] = projected_output
    result["model_call_metadata"] = [
        ModelCallMetadata.from_observed(evidence) for evidence in response.completion_evidence
    ]
    result["context_compaction_contract"] = TransportContextCompactionContract.model_validate(
        response.context_compaction_contract.model_dump() | {"schema_version": 3}
    )
    return ContextCompactedTransportResponse.model_validate(result)


def build_generation_contract(
    *,
    body: NeMoGymResponseCreateParamsNonStreaming,
    model_server: BaseModel | Mapping[str, Any],
    visual_history: VisualHistoryConfig,
) -> GenerationContract:
    """Build Gym's immutable, server-visible generation evidence identity."""

    request_config = body.model_dump(mode="json", exclude_none=True)
    request_config.pop("input", None)
    request_config.pop("required_prefix_token_ids", None)
    model_server_config = (
        model_server.model_dump(mode="json") if isinstance(model_server, BaseModel) else dict(model_server)
    )
    component_ids = {
        "model_contract_id": stable_id(
            "model-contract",
            model_server_config,
        ),
        "tokenizer_contract_id": stable_id(
            "tokenizer-contract",
            "server-authoritative-unavailable",
        ),
        "template_contract_id": stable_id(
            "template-contract",
            "server-authoritative-unavailable",
        ),
        "sampling_contract_id": stable_id(
            "sampling-contract",
            request_config,
        ),
        "processor_contract_id": stable_id(
            "processor-contract",
            "server-authoritative-unavailable",
        ),
        "compaction_policy_id": stable_id(
            "compaction-policy",
            visual_history.model_dump(mode="json"),
        ),
    }
    return GenerationContract(
        **component_ids,
        generation_contract_id=stable_id(
            "generation-contract",
            canonical_digest(component_ids),
        ),
        training_eligible=False,
        incomplete_reasons=(
            "exact_tokenizer_identity_not_reported_by_generation_server",
            "exact_chat_template_identity_not_reported_by_generation_server",
            "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
        ),
    )


@dataclass(frozen=True)
class PreparedContextCompactionCall:
    """One finalized, guard-admitted request view."""

    turn_id: int
    request_input: tuple[Any, ...]
    prepared_history: PreparedHistoryView
    required_prefix_token_ids: tuple[int, ...] | None
    prepared_request_id: str
    segment_id: str


MeasureContext = Callable[
    [PreparedContextCompactionCall],
    Awaitable[ContextMeasurements],
]


class ContextCompactionSession:
    """Own one agent rollout's compaction state and exact evidence."""

    def __init__(
        self,
        *,
        config: VisualHistoryConfig,
        rollout_id: str,
        generation_contract: GenerationContract,
        initial_context: Sequence[Any],
        seed_observations: Sequence[Any] = (),
    ):
        if not config.enabled:
            raise ValueError("ContextCompactionSession requires visual_history.enabled")
        self.config = config
        self.rollout_id = rollout_id
        self.generation_contract = generation_contract
        self.semantic_history = SemanticHistory(rollout_id=rollout_id)
        if initial_context:
            self.semantic_history.append_items(
                initial_context,
                turn_id=0,
                is_initial_context=True,
                conditions_action_turn=(1 if not seed_observations else None),
            )
        if seed_observations:
            self.semantic_history.append_items(
                seed_observations,
                turn_id=0,
                conditions_action_turn=1,
            )

        history_policy = build_history_policy(config.policy)
        if config.schedule.type == "turn_chunked_recency":
            self.history_controller: HistoryController = TurnChunkedHistoryController(
                self.semantic_history,
                history_policy,
                actions_per_chunk=config.schedule.actions_per_chunk,
                history_groups=getattr(
                    config.policy.config,
                    "keep_last_image_groups",
                    None,
                ),
            )
        else:
            self.history_controller = HistoryController(
                self.semantic_history,
                history_policy,
            )

        self.completion_evidence: list[ObservedCompletion] = []
        self.final_policy_decision: PolicyDecisionRecord | None = None
        self.lineage_deltas: list[TransformationLineageDeltaRecord] = []
        self._lineage_state: dict[str, UnitLineageRecord] = {}
        self._parent_transformation_id: str | None = None
        self.guard_records: list[GuardOutcomeRecord] = []
        self._segment_ids: dict[int, str] = {}

    @property
    def authority_mode(self) -> bool:
        return not self.config.shadow_only

    @property
    def guards_enabled(self) -> bool:
        guards = self.config.guards
        return any(
            limit is not None
            for limit in (
                guards.max_total_tokens,
                guards.max_active_images,
                guards.max_vision_tokens,
            )
        )

    def _prepare_once(
        self,
        *,
        legacy_request_input: Sequence[Any],
        turn_id: int,
    ) -> tuple[PreparedHistoryView, tuple[Any, ...], tuple[int, ...] | None]:
        prepared = self.history_controller.prepare(applies_to_step=turn_id)
        self.final_policy_decision = prepared.view.decision
        if self.config.shadow_only:
            assert_identity_shadow_matches(legacy_request_input, prepared.view)
            request_input = tuple(legacy_request_input)
        else:
            request_input = tuple(prepared.view.items)

        required_prefix = None
        if self.authority_mode and prepared.append_compatible and self.completion_evidence:
            previous = self.completion_evidence[-1]
            required_prefix = (
                *previous.prompt_token_ids,
                *previous.sampled_token_ids,
            )
        return prepared, request_input, required_prefix

    def _call_identity(
        self,
        *,
        turn_id: int,
        prepared: PreparedHistoryView,
        request_input: tuple[Any, ...],
        required_prefix: tuple[int, ...] | None,
    ) -> PreparedContextCompactionCall:
        prepared_request_id = stable_id(
            "prepared-request",
            self.rollout_id,
            turn_id,
            prepared.view_digest,
            self.generation_contract.generation_contract_id,
        )
        segment_id = self._segment_ids.setdefault(
            prepared.segment_index,
            stable_id(
                "segment",
                self.rollout_id,
                prepared.context_epoch,
                prepared.segment_index,
                prepared.view_digest,
            ),
        )
        return PreparedContextCompactionCall(
            turn_id=turn_id,
            request_input=request_input,
            prepared_history=prepared,
            required_prefix_token_ids=required_prefix,
            prepared_request_id=prepared_request_id,
            segment_id=segment_id,
        )

    async def prepare_model_call(
        self,
        *,
        legacy_request_input: Sequence[Any],
        turn_id: int,
        measure_context: MeasureContext | None = None,
    ) -> PreparedContextCompactionCall:
        """Materialize and guard one request at a complete action boundary."""

        prepared, request_input, required_prefix = self._prepare_once(
            legacy_request_input=legacy_request_input,
            turn_id=turn_id,
        )
        call = self._call_identity(
            turn_id=turn_id,
            prepared=prepared,
            request_input=request_input,
            required_prefix=required_prefix,
        )

        if self.guards_enabled:
            if measure_context is None:
                raise RuntimeError("Context guards require a model-server measurement callback")
            before = evaluate_context_guards(
                self.config.guards,
                await measure_context(call),
            )
            exceeded = [evaluation for evaluation in before if evaluation.exceeded]
            controller = self.history_controller
            chunk_id = controller.current_chunk_id if isinstance(controller, TurnChunkedHistoryController) else None
            completed_actions = (
                controller.completed_actions_in_current_chunk
                if isinstance(controller, TurnChunkedHistoryController)
                else 0
            )
            pending_groups = pending_observation_group_ids(
                self.semantic_history,
                applies_to_step=turn_id,
            )
            early_close = False
            after = None

            if exceeded and isinstance(controller, TurnChunkedHistoryController):
                early_close = controller.close_for_guard(guard_name=exceeded[0].guard_name)
                if early_close:
                    prepared, request_input, required_prefix = self._prepare_once(
                        legacy_request_input=legacy_request_input,
                        turn_id=turn_id,
                    )
                    call = self._call_identity(
                        turn_id=turn_id,
                        prepared=prepared,
                        request_input=request_input,
                        required_prefix=required_prefix,
                    )
                    after = evaluate_context_guards(
                        self.config.guards,
                        await measure_context(call),
                    )

            outcomes = build_guard_outcome_records(
                rollout_id=self.rollout_id,
                chunk_id=chunk_id,
                applies_to_step=turn_id,
                completed_action_count=completed_actions,
                pending_group_ids=pending_groups,
                before=before,
                after=after,
                early_chunk_close=early_close,
            )
            self.guard_records.extend(outcomes)
            for outcome in outcomes:
                if outcome.decision != "admit":
                    LOGGER.warning(
                        "context_guard rollout=%s chunk=%s step=%d guard=%s "
                        "measured=%d limit=%d early_close=%s "
                        "post_compaction=%s decision=%s",
                        outcome.rollout_id,
                        outcome.chunk_id,
                        outcome.applies_to_step,
                        outcome.guard_name,
                        outcome.measured_value,
                        outcome.configured_limit,
                        outcome.early_chunk_close,
                        outcome.post_compaction_value,
                        outcome.decision,
                    )
            rejected = [outcome for outcome in outcomes if outcome.decision == "reject"]
            if rejected:
                names = ",".join(outcome.guard_name for outcome in rejected)
                raise RuntimeError(
                    f"Context guard rejected model call at complete action boundary: step={turn_id} guards={names}"
                )

        lineage_delta, self._lineage_state = build_lineage_delta(
            call.prepared_history.view.decision.lineage,
            previous_records=self._lineage_state,
            parent_transformation_id=self._parent_transformation_id,
        )
        self.lineage_deltas.append(lineage_delta)
        self._parent_transformation_id = lineage_delta.transformation_id
        return call

    def record_model_response(
        self,
        *,
        call: PreparedContextCompactionCall,
        output_items: Sequence[Any],
        finish_reason: str | None,
    ) -> ObservedCompletion | None:
        """Acknowledge one successful model call and append its semantic output."""

        observed = None
        if self.authority_mode:
            observed = capture_observed_completion(
                output_items,
                rollout_id=self.rollout_id,
                turn_id=call.turn_id,
                media_ids=call.prepared_history.view.media_ids,
                policy_decision=call.prepared_history.view.decision,
                prepared_request_id=call.prepared_request_id,
                context_epoch=call.prepared_history.context_epoch,
                segment_index=call.prepared_history.segment_index,
                segment_id=call.segment_id,
                expected_append_compatible=(call.prepared_history.append_compatible),
                compaction_event_id=(
                    call.prepared_history.boundary.event_id if call.prepared_history.boundary is not None else None
                ),
                generation_contract_id=(self.generation_contract.generation_contract_id),
                finish_reason=finish_reason,
                required_prefix_token_ids=call.required_prefix_token_ids,
            )
            self.completion_evidence.append(observed)
            if isinstance(
                self.history_controller,
                TurnChunkedHistoryController,
            ):
                self.history_controller.acknowledge_action(
                    call.prepared_history,
                    action_id=observed.action_id,
                    completion_id=observed.completion_id,
                )
            else:
                self.history_controller.acknowledge(call.prepared_history)
        else:
            self.history_controller.acknowledge(call.prepared_history)

        self.semantic_history.append_items(
            output_items,
            turn_id=call.turn_id,
        )
        return observed

    def append_observation(
        self,
        items: Sequence[Any],
        *,
        turn_id: int,
        conditions_action_turn: int,
    ) -> None:
        """Append resource/tool observations that must condition the next call."""

        self.semantic_history.append_items(
            items,
            turn_id=turn_id,
            conditions_action_turn=conditions_action_turn,
        )

    def finalize(self) -> None:
        if isinstance(self.history_controller, TurnChunkedHistoryController):
            self.history_controller.finalize_terminal()

    def build_response(
        self,
        response: NeMoGymResponse,
        *,
        output: Sequence[Any],
        agent_input: Sequence[Any],
        seed_obs: Sequence[Any] = (),
    ) -> ContextCompactedResponse:
        """Attach the complete exact-evidence envelope to an ordinary response."""

        if not self.authority_mode:
            raise RuntimeError("Shadow-only sessions do not emit authority contracts")
        result = response.model_dump()
        result.update(
            {
                "output": list(output),
                "agent_input": list(agent_input),
                "seed_obs": list(seed_obs),
                "media_assets": self.semantic_history.media_arena.export(),
                "completion_evidence": self.completion_evidence,
                "final_policy_decision": self.final_policy_decision,
                "lineage_deltas": self.lineage_deltas,
                "chunk_records": (
                    list(self.history_controller.chunk_records)
                    if isinstance(
                        self.history_controller,
                        TurnChunkedHistoryController,
                    )
                    else []
                ),
                "boundary_events": list(self.history_controller.boundary_events),
                "guard_records": self.guard_records,
                "context_compaction_contract": ContextCompactionContract(
                    rollout_id=self.rollout_id,
                    generation_contract=self.generation_contract,
                ),
            }
        )
        return ContextCompactedResponse.model_validate(result)
