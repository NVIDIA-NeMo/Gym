# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Building a rollout's trajectory record from its model calls and the observations its agent reported."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

from nemo_gym.artifacts.rollout_rows import (
    _strip_capture_payloads,
)
from nemo_gym.config.global_config import (
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.core.episode import maybe_rollout_id_from_run_body
from nemo_gym.data.task_materialization import TASK_ID_FIELDS
from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    NG_TRAJECTORY_KEY,
    NG_VERIFIER_INVOCATIONS_KEY,
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryRecord,
    TrajectoryTokenStats,
    TrajectoryToolCall,
    TrajectoryTurn,
    VerifierInvocation,
)


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def _has_observation_gap(result: dict[str, Any], code: str) -> bool:
    for key in (NG_TRAJECTORY_KEY, NG_AGENT_OBSERVATIONS_KEY):
        observations = result.get(key)
        gaps = observations.get("gaps") if isinstance(observations, dict) else None
        if isinstance(gaps, list) and any(isinstance(gap, dict) and gap.get("code") == code for gap in gaps):
            return True
    return False


def _trajectory_identity(row: dict[str, Any]) -> tuple[str, str]:
    identity = row.get("trajectory_identity")
    if isinstance(identity, dict):
        task_id = identity.get("task_id")
        rollout_id = identity.get("rollout_id")
        if isinstance(task_id, str) and task_id and isinstance(rollout_id, str) and rollout_id:
            return task_id, rollout_id
    task_id = next(
        (str(row[key]) for key in TASK_ID_FIELDS if row.get(key) is not None),
        str(row[TASK_INDEX_KEY_NAME]),
    )
    rollout_id = maybe_rollout_id_from_run_body(row) or f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
    return task_id, rollout_id


def _turn_content(request: Any, response: Any) -> tuple[Any, Any, Any, int]:
    """Split one captured model call into the turn's question, answer, reasoning, and tool-call count.

    Handles Responses API output items and chat-completions messages, the two dialects the Model
    Server captures.
    """
    question = request.get("input", request.get("messages")) if isinstance(request, dict) else request
    if isinstance(request, dict) and isinstance(request.get("input"), list):
        # Responses API input messages may omit `type`, which defaults to "message"; turns state it.
        question = [
            {"type": "message", **item} if isinstance(item, dict) and "role" in item and "type" not in item else item
            for item in request["input"]
        ]
    if isinstance(response, dict) and isinstance(response.get("output"), list):
        output = [item for item in response["output"] if isinstance(item, dict)]
        reasoning = [item for item in output if item.get("type") == "reasoning"] or None
        answer = [item for item in output if item.get("type") != "reasoning"]
        tool_calls = sum(1 for item in answer if item.get("type") == "function_call")
        return question, answer, reasoning, tool_calls
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        answer = {key: value for key, value in message.items() if key not in ("reasoning_content", "reasoning")}
        return question, answer, reasoning, len(message.get("tool_calls") or [])
    return question, None, None, 0


def _turns_from_model_calls(
    task_id: str,
    rollout_id: str,
    invocations: list[AgentInvocation],
    model_calls: list[TrajectoryModelCall],
    resolved: Any,
    *,
    verifier_invocations: Iterable[VerifierInvocation] = (),
) -> list[TrajectoryTurn]:
    """Build one turn per captured model call that returned a response, for an agent that sent no trajectory.

    A call belongs to the invocation whose ``model_calls`` reference it. Unreferenced calls stay
    in the raw evidence but do not become turns: even a single-agent rollout may capture judge
    or auxiliary calls that do not belong to the agent. Explicit verifier ownership also excludes
    a call from turns when an agent incorrectly references it; retain both ownership claims for validation.
    """
    verifier_call_ids = {
        ref.model_call_id for invocation in verifier_invocations for ref in invocation.model_calls if ref.model_call_id
    }
    invocation_by_call_id: dict[str, str] = {}
    invocation_by_response_id: dict[str, str] = {}
    for invocation in invocations:
        for ref in invocation.model_calls:
            if ref.model_call_id:
                invocation_by_call_id[ref.model_call_id] = invocation.invocation_id
            if ref.response_id:
                invocation_by_response_id[ref.response_id] = invocation.invocation_id

    turns: list[TrajectoryTurn] = []
    turn_counts: Counter = Counter()
    tool_counts: Counter = Counter()
    for call in model_calls:
        metadata = call.response_metadata
        if call.model_call_id in verifier_call_ids:
            continue
        if call.response is None or (metadata.status_code is not None and metadata.status_code >= 400):
            # HTTP error payloads are retained attempts, not model decisions.
            continue
        invocation_id = invocation_by_call_id.get(call.model_call_id or "") or invocation_by_response_id.get(
            metadata.response_id or ""
        )
        if invocation_id is None:
            continue
        question, answer, reasoning, tool_calls = _turn_content(call.request, call.response)
        turn_counts[invocation_id] += 1
        turns.append(
            TrajectoryTurn(
                invocation_id=invocation_id,
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=turn_counts[invocation_id],
                timestamp=call.started_at or call.completed_at or 0.0,
                question=question,
                answer=answer,
                reasoning_content=reasoning,
                step_count=tool_counts[invocation_id],
                model_calls=[
                    ModelCallRef(
                        model_call_id=call.model_call_id,
                        model_ref=metadata.model_ref,
                        response_id=metadata.response_id,
                    )
                ],
            )
        )
        tool_counts[invocation_id] += tool_calls
    if turns and isinstance(resolved, bool):
        turns[-1] = turns[-1].model_copy(update={"resolved": resolved})
    return turns


def _build_trajectory_record(row: dict[str, Any], result: dict[str, Any]) -> TrajectoryRecord:
    task_id, rollout_id = _trajectory_identity(row)
    gaps: list[ObservationGap] = []
    invocations: list[AgentInvocation] = []
    verifier_invocations: list[VerifierInvocation] = []
    turns: list[TrajectoryTurn] = []
    tools: list[TrajectoryToolCall] = []
    model_calls: list[TrajectoryModelCall] = []

    producer_turns_observed = False
    raw_trajectory = result.get(NG_TRAJECTORY_KEY)
    if isinstance(raw_trajectory, dict):
        try:
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
            producer_turns_observed = True
            mismatches = [
                field
                for field, producer, canonical in (
                    ("task_id", trajectory.task_id, task_id),
                    ("rollout_id", trajectory.rollout_id, rollout_id),
                )
                if producer != canonical
            ]
            if mismatches:
                gaps.append(ObservationGap(code="producer_trajectory_identity_mismatch", detail=",".join(mismatches)))
                turns = [
                    turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id}) for turn in trajectory.turns
                ]
            else:
                turns = trajectory.turns
            gaps.extend(trajectory.gaps)
            invocations = trajectory.invocations
            verifier_invocations = trajectory.verifier_invocations
            tools = trajectory.tool_calls
            model_calls = trajectory.model_calls
        except Exception as exc:
            gaps.append(ObservationGap(code="producer_trajectory_invalid", detail=type(exc).__name__))

    observations_read = False
    raw_observations = result.get(NG_AGENT_OBSERVATIONS_KEY)
    if raw_observations is not None:
        try:
            observations = AgentObservationBundle.model_validate(raw_observations)
            gaps.extend(observations.gaps)
            observed_invocations = [record for record in observations.records if isinstance(record, AgentInvocation)]
            producer_invocation_ids = {record.invocation_id for record in invocations}
            invocations.extend(
                record for record in observed_invocations if record.invocation_id not in producer_invocation_ids
            )
            observed_tools = [record for record in observations.records if isinstance(record, ToolCallObservation)]
            if observed_tools:
                outputs = {
                    (invocation.invocation_id, item.call_id): item.output
                    for invocation in invocations
                    for item in invocation.conversation
                    if getattr(item, "type", None) in {"function_call_output", "custom_tool_call_output"}
                }
                positions = {(tool.invocation_id, tool.tool_call_id): index for index, tool in enumerate(tools)}
                for observed in observed_tools:
                    key = (observed.invocation_id, observed.tool_call_id)
                    position = positions.get(key)
                    existing = tools[position] if position is not None else None
                    merged = existing.model_dump(mode="json") if existing is not None else {}
                    update = observed.model_dump(mode="json", exclude_none=True)
                    if existing is not None and observed.status == "unknown" and existing.status != "unknown":
                        update.pop("status", None)
                    merged.update(update)
                    if key in outputs:
                        merged["output"] = outputs[key]
                    projected = TrajectoryToolCall.model_validate(merged)
                    if position is None:
                        positions[key] = len(tools)
                        tools.append(projected)
                    else:
                        tools[position] = projected
            observations_read = True
        except Exception as exc:
            gaps.append(ObservationGap(code="agent_observations_invalid", detail=type(exc).__name__))

    # Without a valid producer trajectory or valid observations, the empty tool list below is the collector's
    # default, not a report that no tool ran.
    if not producer_turns_observed and not observations_read:
        gaps.append(ObservationGap(code="tool_timing_unavailable"))

    turns.sort(key=lambda turn: (turn.timestamp, turn.invocation_id, turn.turn_no))

    capture = result.get("ng_model_call_capture")
    capture = capture if isinstance(capture, dict) else {}
    raw_calls = capture.get("calls") or []
    model_call_positions = {
        call.model_call_id: index for index, call in enumerate(model_calls) if call.model_call_id is not None
    }
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        model_call_id = raw_call.get("model_call_id")
        metadata = {
            key: raw_call[key]
            for key in (
                "response_id",
                "model_ref",
                "model",
                "dialect",
                "status_code",
                "response_status",
                "finish_reason",
                "upstream_attempted",
                "response_source",
                "upstream_status_code",
                "local_response_reason",
                "error_category",
                "latency_ttft_ms",
            )
            if raw_call.get(key) is not None
        }
        response = raw_call.get("response")
        if isinstance(response, dict) and isinstance(response.get("status"), str):
            metadata.setdefault("response_status", response["status"])
        projected = TrajectoryModelCall(
            model_call_id=model_call_id,
            client_session_id=raw_call.get("client_session_id"),
            client_assistant_message_id=raw_call.get("client_assistant_message_id"),
            started_at=raw_call.get("started_at"),
            completed_at=raw_call.get("completed_at"),
            duration_ms=raw_call.get("latency_total_ms"),
            request=raw_call.get("request") if raw_call.get("request") is not None else raw_call.get("request_raw"),
            response=raw_call.get("response")
            if raw_call.get("response") is not None
            else raw_call.get("response_raw"),
            response_metadata=metadata,
            token_stats=TrajectoryTokenStats(
                prompt_tokens=_nonnegative_int(raw_call.get("tokens_in")),
                completion_tokens=_nonnegative_int(raw_call.get("tokens_out")),
                reasoning_tokens=_nonnegative_int(raw_call.get("tokens_reasoning")),
                total_tokens=_nonnegative_int(raw_call.get("tokens_total")),
                cached_tokens=_nonnegative_int(raw_call.get("cached_tokens")),
            ),
        )
        position = model_call_positions.pop(model_call_id, None) if model_call_id is not None else None
        if position is None:
            model_calls.append(projected)
        else:
            merged = model_calls[position].model_dump(mode="json")
            update = projected.model_dump(mode="json", exclude_none=True)
            for key in ("response_metadata", "token_stats"):
                merged[key].update(update.pop(key))
            merged.update(update)
            projected = TrajectoryModelCall.model_validate(merged)
            model_calls[position] = projected

    # Join returned verifier outcomes with model-call references reconstructed from capture.
    for raw_invocation in [
        *(capture.get("verifier_invocations") or []),
        *(result.get(NG_VERIFIER_INVOCATIONS_KEY) or []),
    ]:
        try:
            invocation = VerifierInvocation.model_validate(raw_invocation)
            existing = next((i for i in verifier_invocations if i.invocation_id == invocation.invocation_id), None)
            if existing is None:
                verifier_invocations.append(invocation)
            else:
                existing.model_calls.extend(ref for ref in invocation.model_calls if ref not in existing.model_calls)
                if invocation.status != "unknown":
                    existing.status = invocation.status
                    existing.failure_kind = invocation.failure_kind
        except Exception:
            gaps.append(ObservationGap(code="verifier_invocation_invalid"))

    for raw_gap in capture.get("gaps") or []:
        if isinstance(raw_gap, dict):
            try:
                gaps.append(ObservationGap.model_validate(raw_gap))
            except Exception:
                gaps.append(ObservationGap(code="model_call_capture_gap_invalid"))
    if not isinstance(raw_trajectory, dict):
        # An agent that sends its own trajectory owns its turns, and an empty list there means no turn
        # completed. Otherwise, build turns only from calls explicitly referenced by invocations.
        turns = _turns_from_model_calls(
            task_id,
            rollout_id,
            invocations,
            model_calls,
            result.get("resolved"),
            verifier_invocations=verifier_invocations,
        )
    if not model_calls:
        gaps.append(ObservationGap(code="model_calls_unavailable"))
    if not turns and not producer_turns_observed:
        # A producer that published a trajectory and reported no turns has told
        # us something; only the absence of a producer leaves turns unavailable.
        # Without this, `rollout_missing_agent_turns` is skipped in exactly the
        # case it exists to catch.
        gaps.append(ObservationGap(code="turns_unavailable"))
    if not any(invocation.conversation for invocation in invocations):
        gaps.append(ObservationGap(code="conversation_unavailable"))

    return TrajectoryRecord(
        task_id=task_id,
        rollout_id=rollout_id,
        invocations=invocations,
        verifier_invocations=verifier_invocations,
        turns=turns,
        model_calls=model_calls,
        tool_calls=tools,
        gaps=list({(gap.code, gap.invocation_id, gap.detail): gap for gap in gaps}.values()),
    )


def _attach_trajectory_record(row: dict[str, Any], result: dict[str, Any]) -> None:
    try:
        result[NG_TRAJECTORY_KEY] = _build_trajectory_record(row, result).model_dump(mode="json")
    except Exception as exc:
        result.pop(NG_TRAJECTORY_KEY, None)
        logger.warning("Could not project standardized trajectory evidence.", exc_info=True)
        gap = ObservationGap(code="trajectory_projection_failed", detail=type(exc).__name__).model_dump(
            mode="json", exclude_none=True
        )
        target = result.get("ng_model_call_capture")
        if not isinstance(target, dict):
            target = result.get(NG_AGENT_OBSERVATIONS_KEY)
        gap_attached = False
        if isinstance(target, dict):
            gaps = target.setdefault("gaps", [])
            if isinstance(gaps, list):
                gaps.append(gap)
                gap_attached = True
        if not gap_attached:
            try:
                task_id, rollout_id = _trajectory_identity(row)
                result[NG_TRAJECTORY_KEY] = TrajectoryRecord(
                    task_id=task_id,
                    rollout_id=rollout_id,
                    gaps=[ObservationGap.model_validate(gap)],
                ).model_dump(mode="json")
            except Exception:
                logger.warning("Could not retain the trajectory projection failure gap.", exc_info=True)
    else:
        # Raw capture payloads remain as a fallback on failure. After success,
        # ng_trajectory owns them, so remove only the duplicate copies.
        _strip_capture_payloads(result)


def _nonnegative_int(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None
