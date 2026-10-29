# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve separately captured judge exchanges using the judge's own close evidence."""

import json
import shutil
from collections import defaultdict
from pathlib import Path

from nemo_gym.agents import AgentCloseSessionResponse
from nemo_gym.models import CaptureStore
from nemo_gym.rollouts import await_model_call_capture_quiescence, model_call_capture_dirs_from_config
from nemo_gym.trajectory import (
    AgentInvocation,
    ModelCallRecord,
    ModelCallRef,
    ObservationGap,
    TrajectoryModelCall,
    TrajectoryRecord,
    TrajectoryResponseMetadata,
    TrajectoryTokenStats,
    VerifierInvocation,
    build_model_call_record,
)


def _matches(reference: ModelCallRef, call: ModelCallRecord) -> bool:
    return (
        (reference.model_call_id is None or reference.model_call_id == call.model_call_id)
        and (reference.model_ref is None or reference.model_ref == call.model_ref)
        and (reference.response_id is None or reference.response_id == call.response_id)
    )


def _session_title(request: dict) -> bool:
    schema = (request.get("output_config") or {}).get("format", {}).get("schema", {})
    system = request.get("system") or []
    return (
        request.get("tools") == []
        and schema
        == {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        }
        and isinstance(system, list)
        and any(
            isinstance(part, dict)
            and part.get("text", "").startswith("Generate a concise, sentence-case title (3-7 words)")
            for part in system
        )
    )


def aggregate_judge_evidence(
    trajectory: TrajectoryRecord,
    close: AgentCloseSessionResponse | None,
    exchanges: list[dict],
    *,
    source_episode: str,
) -> tuple[TrajectoryRecord, dict]:
    """Reconstruct producer-declared verifier ownership; never claim original request headers."""
    result = trajectory.model_copy(deep=True)
    provenance = {
        "kind": "judge_receipt_reconstruction" if close else "judge_capture_without_receipt",
        "source_episode": source_episode,
        "judge_session_id": close.agent_session_id if close else None,
        "raw_capture_artifact": "judge-model-calls.json",
        "receipt_artifact": "judge-close.json" if close else None,
        "ownership": [],
        "excluded": [],
    }
    owners = defaultdict(set)
    statuses = {}
    captured = [build_model_call_record(exchange, call_index=index) for index, exchange in enumerate(exchanges)]
    referenced = set()
    if close is None:
        result.gaps.append(ObservationGap(code="judge_receipt_unavailable"))
    for record in close.agent_observations.records if close and close.agent_observations else []:
        if isinstance(record, AgentInvocation):
            statuses[record.invocation_id] = record.status
            for reference in record.model_calls:
                matches = [index for index, call in enumerate(captured) if _matches(reference, call)]
                # Conflicting identifiers must not turn referenced evidence into housekeeping.
                referenced.update(
                    index
                    for index, call in enumerate(captured)
                    if (reference.model_call_id and reference.model_call_id == call.model_call_id)
                    or (reference.response_id and reference.response_id == call.response_id)
                )
                if len(matches) == 1:
                    owners[matches[0]].add(record.invocation_id)
                else:
                    result.gaps.append(
                        ObservationGap(
                            code="judge_receipt_reference_ambiguous" if matches else "judge_receipt_calls_missing",
                            invocation_id=record.invocation_id,
                            detail=reference.model_call_id or reference.response_id,
                        )
                    )
    positions = {call.model_call_id: call for call in result.model_calls}
    verifier_positions = {owner.invocation_id: owner for owner in result.verifier_invocations}
    agent_ids = {owner.invocation_id for owner in result.invocations}
    for index, call in enumerate(captured):
        declared = owners.get(index, set())
        if index not in referenced and _session_title(call.request or {}):
            provenance["excluded"].append({"model_call_id": call.model_call_id, "reason": "claude_session_title"})
            continue
        metadata = {
            field: getattr(call, field) for field in TrajectoryResponseMetadata.model_fields if hasattr(call, field)
        }
        projected = TrajectoryModelCall(
            model_call_id=call.model_call_id,
            client_session_id=call.client_session_id,
            client_assistant_message_id=call.client_assistant_message_id,
            started_at=call.started_at,
            completed_at=call.completed_at,
            duration_ms=call.latency_total_ms,
            request=call.request if call.request is not None else call.request_raw,
            response=call.response if call.response is not None else call.response_raw,
            response_metadata=metadata,
            token_stats=TrajectoryTokenStats(
                prompt_tokens=call.tokens_in,
                completion_tokens=call.tokens_out,
                reasoning_tokens=call.tokens_reasoning,
                total_tokens=call.tokens_total,
                cached_tokens=call.cached_tokens,
            ),
        )
        if not call.model_call_id or (call.model_call_id in positions and positions[call.model_call_id] != projected):
            result.gaps.append(ObservationGap(code="judge_capture_identity_conflict"))
            continue
        if call.model_call_id not in positions:
            result.model_calls.append(projected)
            positions[call.model_call_id] = projected
        if len(declared) != 1 or declared & agent_ids:
            result.gaps.append(ObservationGap(code="judge_call_owner_unresolved", detail=call.model_call_id))
            continue
        owner_id = next(iter(declared))
        owner = verifier_positions.get(owner_id)
        if owner is None:
            owner = VerifierInvocation(invocation_id=owner_id, status=statuses[owner_id])
            result.verifier_invocations.append(owner)
            verifier_positions[owner_id] = owner
        reference = ModelCallRef(
            model_call_id=call.model_call_id, model_ref=call.model_ref, response_id=call.response_id
        )
        if reference not in owner.model_calls:
            owner.model_calls.append(reference)
        mapping = {
            "model_call_id": call.model_call_id,
            "verifier_invocation_id": owner_id,
            "response_id": call.response_id,
            "model_ref": call.model_ref.model_dump(mode="json") if call.model_ref else None,
        }
        if mapping not in provenance["ownership"]:
            provenance["ownership"].append(mapping)
    return result, provenance


async def collect_judge_evidence(
    trajectory: TrajectoryRecord | None, verification: dict, global_config: dict, artifact_dir: Path
) -> TrajectoryRecord | None:
    """Read the explicitly recorded judge episode; retain raw capture and collection failures."""
    directories = model_call_capture_dirs_from_config(global_config)
    episode = verification.get("judge_capture_episode")
    if not directories or not episode:
        return trajectory
    if trajectory is None:
        verification["judge_evidence_error"] = "Candidate trajectory unavailable for judge aggregation"
        return trajectory
    try:
        drained = await await_model_call_capture_quiescence(episode, directories)
        store = CaptureStore(directories[0], create=False)
        if not store.path_for(episode).is_file():
            raise FileNotFoundError("Judge capture is unavailable")
        shutil.copyfile(store.path_for(episode), artifact_dir / "judge-model-calls.capture.jsonl")
        entries, invalid_count = store.read_available(episode)
        exchanges = [exchange for _, exchange in entries]
        (artifact_dir / "judge-model-calls.json").write_text(json.dumps(exchanges, indent=2))
        raw_close = verification.get("judge_close")
        close = AgentCloseSessionResponse.model_validate(raw_close) if raw_close is not None else None
        result, provenance = aggregate_judge_evidence(trajectory, close, exchanges, source_episode=episode)
        provenance["raw_capture_jsonl_artifact"] = "judge-model-calls.capture.jsonl"
        if invalid_count:
            result.gaps.append(
                ObservationGap(code="judge_capture_records_unreadable", detail=f"count={invalid_count}")
            )
        if not drained or store.is_incomplete(episode):
            result.gaps.append(ObservationGap(code="judge_capture_incomplete"))
        verification["judge_evidence_provenance"] = provenance
        (artifact_dir / "judge-evidence-provenance.json").write_text(json.dumps(provenance, indent=2))
        return result
    except Exception as error:
        result = trajectory.model_copy(deep=True)
        result.gaps.append(ObservationGap(code="judge_capture_unavailable", detail=type(error).__name__))
        verification["judge_evidence_error"] = f"{type(error).__name__}: {error}"
        return result
