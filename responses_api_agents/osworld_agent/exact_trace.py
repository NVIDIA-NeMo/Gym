# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Arash NeMo-RL exact-trace evidence for OSWorld rollouts.

The OSWorld agent owns the semantic trajectory and the generation server owns
the exact tokenization.  This module binds both views without pretending that
successive prompts are append-only.  NeMo-RL may then split one logical
rollout into prefix-contiguous physical traces while retaining one logical
reward and advantage.

The wire contract intentionally matches schema v2 from
``aroshanghias/context-compaction-v2-clean``. The semantic trajectory remains
model-independent; this module is invoked only when exact evidence is present.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from responses_api_agents.osworld_agent.trajectory import (
    canonical_digest,
    project_prompt_messages,
    stable_id,
)


_POLICY_NAME = "osworld_exact_prompt_trace"
_POLICY_VERSION = "1"
_SOURCE_UNIT_ID = "osworld-materialized-model-prompt"
_ALLOWED_IDENTITY_GAPS = (
    "exact_tokenizer_identity_not_reported_by_generation_server",
    "exact_chat_template_identity_not_reported_by_generation_server",
    "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
)


def _token_list(value: Any, *, field: str, turn_id: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has invalid {field}")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has invalid token in {field}")
    return [int(item) for item in value]


def _logprob_list(value: Any, *, turn_id: int) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has invalid generation_log_probs")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has non-finite generation_log_probs")
    return result


def _prompt_media_ids(
    prompt_messages: Sequence[Mapping[str, Any]],
    *,
    media_assets: dict[str, dict[str, Any]],
) -> list[str]:
    """Resolve every media occurrence from one materialized model prompt."""

    _, media_ids = project_prompt_messages(
        prompt_messages,
        media_assets=media_assets,
    )
    return media_ids


def _generation_contract(
    *,
    model_name: str,
    sampling_config: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> dict[str, Any]:
    component_ids = {
        "model_contract_id": stable_id(
            "model-contract",
            {"model_name": model_name, "adapter": "osworld_agent"},
        ),
        "tokenizer_contract_id": stable_id(
            "tokenizer-contract",
            "server-authoritative-unavailable",
        ),
        "template_contract_id": stable_id(
            "template-contract",
            "server-authoritative-unavailable",
        ),
        "sampling_contract_id": stable_id("sampling-contract", dict(sampling_config)),
        "processor_contract_id": stable_id(
            "processor-contract",
            "server-authoritative-unavailable",
        ),
        "compaction_policy_id": stable_id("compaction-policy", dict(policy_config)),
    }
    return {
        "schema_version": 1,
        **component_ids,
        "generation_contract_id": stable_id(
            "generation-contract",
            canonical_digest(component_ids),
        ),
        "loss_normalization": "global_action_token_mean",
        "training_eligible": False,
        "incomplete_reasons": list(_ALLOWED_IDENTITY_GAPS),
    }


def _lineage_state_digest(record: Mapping[str, Any]) -> str:
    normalized = {
        "source_unit_id": record.get("source_unit_id"),
        "source_digest": record.get("source_digest"),
        "disposition": record.get("disposition"),
        "output_unit_ids": list(record.get("output_unit_ids") or []),
        "output_digests": list(record.get("output_digests") or []),
    }
    return canonical_digest([normalized])


def _policy_lineage(
    *,
    turn_id: int,
    view_digest: str,
    policy_config_digest: str,
    generation_contract_id: str,
    rollout_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    transformation_id = stable_id(
        "transformation",
        rollout_id,
        turn_id,
        view_digest,
        policy_config_digest,
    )
    unit_record = {
        "source_unit_id": _SOURCE_UNIT_ID,
        "source_digest": view_digest,
        "disposition": "retained",
        "output_unit_ids": [_SOURCE_UNIT_ID],
        "output_digests": [view_digest],
    }
    lineage = {
        "transformation_id": transformation_id,
        "transformation_type": "osworld_prompt_materialization",
        "transformation_version": _POLICY_VERSION,
        "configuration_digest": policy_config_digest,
        "deterministic": True,
        "lossy": True,
        "generator_contract_id": generation_contract_id,
        "unit_records": [unit_record],
        "validator_result": "passed",
    }
    decision = {
        "policy_name": _POLICY_NAME,
        "policy_version": _POLICY_VERSION,
        "config_digest": policy_config_digest,
        "protected_part_ids": [],
        "changed_part_ranges": [],
        "retained_part_count": 1,
        "omitted_part_count": 0,
        "selection_digest": view_digest,
        "inserted_artifact_ids": [],
        "decision_turn": turn_id,
        "lineage": lineage,
    }
    evidence = {
        "policy_name": _POLICY_NAME,
        "policy_version": _POLICY_VERSION,
        "config_digest": policy_config_digest,
        "decision_turn": turn_id,
        "selection_digest": view_digest,
        "transformation_id": transformation_id,
    }
    return decision, evidence, unit_record


def build_exact_trace_envelope(
    *,
    model_calls: Sequence[Mapping[str, Any]],
    trajectory_contract: Mapping[str, Any],
    model_name: str,
    sampling_config: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Augment a semantic trajectory with exact, per-model-call evidence."""

    if not model_calls:
        raise ValueError("OSWorld exact trace requires at least one model call")
    capabilities = trajectory_contract.get("capabilities")
    if not isinstance(capabilities, Mapping) or capabilities.get("exact_model_call_evidence") is not True:
        raise ValueError("OSWorld exact trace requires a trajectory contract with complete model-call evidence")
    identity = {
        field: trajectory_contract.get(field)
        for field in (
            "rollout_id",
            "group_id",
            "task_id",
            "rollout_index",
            "attempt_index",
            "identity_source",
        )
    }
    identity.update(
        {
            field: trajectory_contract[field]
            for field in ("sampling_event_id", "source_group_id")
            if trajectory_contract.get(field) is not None
        }
    )
    rollout_id = identity["rollout_id"]
    if not isinstance(rollout_id, str) or not rollout_id:
        raise ValueError("OSWorld exact trace has no rollout identity")
    generation_contract = _generation_contract(
        model_name=model_name,
        sampling_config=sampling_config,
        policy_config=policy_config,
    )
    generation_contract_id = generation_contract["generation_contract_id"]
    policy_config_digest = canonical_digest(dict(policy_config))

    media_assets: dict[str, dict[str, Any]] = {}
    completion_evidence: list[dict[str, Any]] = []
    boundary_events: list[dict[str, Any]] = []
    lineage_deltas: list[dict[str, Any]] = []
    model_call_output: list[dict[str, Any]] = []

    previous_context: list[int] = []
    previous_media_ids: list[str] = []
    previous_view_digest: str | None = None
    previous_transformation_id: str | None = None
    segment_index = -1
    segment_id = ""
    final_policy_decision: dict[str, Any] | None = None

    for call_index, model_call in enumerate(model_calls):
        turn_id = call_index + 1
        if model_call.get("turn_id") != turn_id:
            raise ValueError(f"OSWorld exact trace model call {call_index} has invalid turn_id")
        model_response = model_call.get("response")
        if not isinstance(model_response, Mapping):
            raise ValueError(f"OSWorld exact trace turn {turn_id} has no model response evidence")

        prompt_token_ids = _token_list(
            model_response.get("prompt_token_ids"),
            field="prompt_token_ids",
            turn_id=turn_id,
        )
        sampled_token_ids = _token_list(
            model_response.get("generation_token_ids"),
            field="generation_token_ids",
            turn_id=turn_id,
        )
        sampled_logprobs = _logprob_list(
            model_response.get("generation_log_probs"),
            turn_id=turn_id,
        )
        if len(sampled_token_ids) != len(sampled_logprobs):
            raise ValueError(
                f"OSWorld exact trace turn {turn_id} token/logprob mismatch: "
                f"tokens={len(sampled_token_ids)} logprobs={len(sampled_logprobs)}"
            )
        eligible = model_call.get("eligible")
        if not isinstance(eligible, bool):
            raise ValueError(f"OSWorld exact trace turn {turn_id} has non-boolean eligible")

        prompt_messages = model_call.get("prompt_messages")
        if not isinstance(prompt_messages, list) or not prompt_messages:
            raise ValueError(f"OSWorld exact trace turn {turn_id} has no materialized prompt")
        media_ids = _prompt_media_ids(
            prompt_messages,
            media_assets=media_assets,
        )
        token_append_compatible = previous_context == prompt_token_ids[: len(previous_context)]
        media_append_compatible = previous_media_ids == media_ids[: len(previous_media_ids)]
        append_compatible = turn_id > 1 and token_append_compatible and media_append_compatible
        view_digest = canonical_digest(
            {
                "prompt_token_ids": prompt_token_ids,
                "media_ids": media_ids,
            }
        )

        final_policy_decision, policy_decision, unit_record = _policy_lineage(
            turn_id=turn_id,
            view_digest=view_digest,
            policy_config_digest=policy_config_digest,
            generation_contract_id=generation_contract_id,
            rollout_id=rollout_id,
        )
        transformation_id = policy_decision["transformation_id"]
        lineage_deltas.append(
            {
                "transformation_id": transformation_id,
                "parent_transformation_id": previous_transformation_id,
                "transformation_type": "osworld_prompt_materialization",
                "transformation_version": _POLICY_VERSION,
                "configuration_digest": policy_config_digest,
                "deterministic": True,
                "lossy": True,
                "generator_contract_id": generation_contract_id,
                "unit_upserts": [unit_record],
                "source_unit_count": 1,
                "state_digest": _lineage_state_digest(unit_record),
                "validator_result": "passed",
            }
        )

        boundary_event_id = None
        if not append_compatible:
            segment_index += 1
            segment_id = stable_id("segment", rollout_id, segment_index, view_digest)
            if turn_id > 1:
                boundary_event_id = stable_id(
                    "rewrite-boundary",
                    rollout_id,
                    turn_id,
                    previous_view_digest,
                    view_digest,
                )
                boundary_events.append(
                    {
                        "event_id": boundary_event_id,
                        "trigger_after_step": turn_id - 1,
                        "applies_to_step": turn_id,
                        "reason": "prompt_or_media_not_append_compatible",
                        "policy_name": _POLICY_NAME,
                        "policy_version": _POLICY_VERSION,
                        "config_digest": policy_config_digest,
                        "previous_view_digest": previous_view_digest,
                        "current_view_digest": view_digest,
                        "changed_part_ranges": [],
                        "retained_part_count": 1,
                        "omitted_part_count": 0,
                        "retained_media_count": len(media_ids),
                        "removed_media_count": sum(media_id not in media_ids for media_id in previous_media_ids),
                        "inserted_artifact_ids": [],
                        "schedule_name": "per_action",
                        "schedule_version": _POLICY_VERSION,
                        "schedule_config_digest": policy_config_digest,
                        "chunk_id": segment_id,
                        "block_index": segment_index,
                    }
                )

        completion_id = stable_id(
            "completion",
            rollout_id,
            turn_id,
            prompt_token_ids,
            sampled_token_ids,
        )
        action_id = f"policy-action-{turn_id:06d}"
        prepared_request_id = stable_id(
            "prepared-request",
            rollout_id,
            turn_id,
            view_digest,
            generation_contract_id,
        )
        request_id = stable_id(
            "request",
            prepared_request_id,
            prompt_token_ids,
            media_ids,
        )
        model_call_id = model_call.get("model_call_id")
        if not isinstance(model_call_id, str) or not model_call_id:
            raise ValueError(f"OSWorld exact trace turn {turn_id} has no model-call identity")
        occurrence_counts: Counter[str] = Counter()
        media_occurrences = []
        for media_id in media_ids:
            occurrence_ordinal = occurrence_counts[media_id]
            occurrence_counts[media_id] += 1
            media_occurrences.append(
                {
                    "media_id": media_id,
                    "occurrence_ordinal": occurrence_ordinal,
                    "model_call_id": model_call_id,
                    "placeholder_span_or_position": None,
                    "processed_dimensions": None,
                    "model_specific_sidecars": {},
                }
            )
        span = {
            "policy_output_span_id": stable_id(
                "policy-output-span",
                model_call_id,
                action_id,
                len(sampled_token_ids),
            ),
            "model_call_id": model_call_id,
            "action_ids": [action_id],
            "start": 0,
            "end": len(sampled_token_ids),
            "eligible": eligible,
            "old_logprobs_alignment": "sampled_tokens",
        }
        completion_evidence.append(
            {
                "rollout_id": rollout_id,
                "completion_id": completion_id,
                "action_id": action_id,
                "model_call_id": model_call_id,
                "turn_id": turn_id,
                "environment_step": model_call.get("environment_step"),
                "parse_attempt": model_call.get("parse_attempt"),
                "accepted": model_call.get("accepted") is True,
                "parse_error": model_call.get("parse_error"),
                "prepared_request_id": prepared_request_id,
                "request_id": request_id,
                "context_epoch": segment_index,
                "segment_index": segment_index,
                "segment_id": segment_id,
                "expected_append_compatible": append_compatible,
                "compaction_event_id": boundary_event_id,
                "prompt_token_ids": prompt_token_ids,
                "sampled_token_ids": sampled_token_ids,
                "sampled_logprobs": sampled_logprobs,
                "finish_reason": model_response.get("finish_reason"),
                "media_ids": media_ids,
                "policy_decision": policy_decision,
                "generation_contract_id": generation_contract_id,
                "policy_output_spans": [span],
                "media_occurrences": media_occurrences,
                "processor_fingerprint": model_response.get("processor_fingerprint"),
                "eligible": eligible,
                "evidence_source": "generation_response",
            }
        )
        assistant_item: dict[str, Any] = {
            "id": completion_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "annotations": [],
                    "text": str(model_response.get("raw_content") or ""),
                }
            ],
            "prompt_token_ids": prompt_token_ids,
            "generation_token_ids": sampled_token_ids,
            "generation_log_probs": sampled_logprobs,
        }
        if model_response.get("routed_experts") is not None:
            assistant_item["routed_experts"] = model_response["routed_experts"]
        model_call_output.append(assistant_item)

        previous_context = [*prompt_token_ids, *sampled_token_ids]
        previous_media_ids = media_ids
        previous_view_digest = view_digest
        previous_transformation_id = transformation_id

    assert final_policy_decision is not None
    return {
        "model_call_output": model_call_output,
        "media_assets": media_assets,
        "completion_evidence": completion_evidence,
        "final_policy_decision": final_policy_decision,
        "lineage_deltas": lineage_deltas,
        "chunk_records": [],
        "boundary_events": boundary_events,
        "guard_records": [],
        "context_compaction_contract": {
            "schema_version": 2,
            "mode": "exact_trace_authority",
            **identity,
            "trajectory_contract_id": trajectory_contract.get("trajectory_contract_id"),
            "generation_contract": generation_contract,
        },
    }
