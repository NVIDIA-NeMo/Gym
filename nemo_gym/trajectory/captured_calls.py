# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count every captured model call of a rollout by the role that owns it (``ng_captured_calls``)."""

from typing import Any, Callable, Optional

from nemo_gym.trajectory.trajectory_record import (
    NG_TRAJECTORY_KEY,
    CapturedCallRole,
    CapturedCallSummary,
    CapturedCallTotals,
    ModelCallRef,
    TrajectoryModelCall,
    TrajectoryRecord,
)


CAPTURE_INCOMPLETE_GAPS = frozenset(
    {
        "model_call_capture_pending",
        "model_call_capture_incomplete",
        "model_call_capture_records_unreadable",
        # The rollout's own capture was unreadable or empty while another producer's calls were found.
        "model_call_capture_unreadable",
        "model_call_capture_no_records",
        # A judge episode captured apart from the rollout (SWE-Together) that could not be read in full.
        "judge_capture_unavailable",
        "judge_capture_incomplete",
        "judge_capture_records_unreadable",
        "judge_receipt_calls_missing",
    }
)

# Token fields summed per call role, named as in ``ng_perf``.
_CAPTURED_CALL_TOKEN_FIELDS: tuple[tuple[str, str], ...] = (
    ("prompt_tokens", "prompt_tokens"),
    ("cached_prompt_tokens", "cached_tokens"),
    ("completion_tokens", "completion_tokens"),
    ("reasoning_tokens", "reasoning_tokens"),
    ("total_tokens", "total_tokens"),
)


def build_captured_calls(result: dict[str, Any]) -> Optional[CapturedCallSummary]:
    """Summarize every captured model call of a rollout by role.

    ``None`` without ``ng_trajectory`` (capture off) and whenever the trajectory holds no captured call: a missing or
    unreadable capture and a rollout that made no call look the same, so those calls are unknown, not zero. A capture
    with writes still pending or incomplete, or with some records or sources unreadable, is summarized with
    ``capture_complete=False``, so its totals read as a lower bound.
    """
    raw_trajectory = result.get(NG_TRAJECTORY_KEY)
    if not isinstance(raw_trajectory, dict):
        return None
    try:
        trajectory = TrajectoryRecord.model_validate(raw_trajectory)
    except Exception:
        return None
    calls = len(trajectory.model_calls)
    if not calls:
        return None
    roles = _call_roles(trajectory)
    by_role = _totals_by_role(trajectory.model_calls, roles)
    return {
        "calls": calls,
        "calls_without_usage": sum(totals["calls_without_usage"] for totals in by_role.values()),
        "attributed_call_coverage": len(roles) / calls,
        "capture_complete": not ({gap.code for gap in trajectory.gaps} & CAPTURE_INCOMPLETE_GAPS),
        "by_role": by_role,
    }


def model_call_matcher(trajectory: TrajectoryRecord) -> Callable[[ModelCallRef], Optional[int]]:
    """Resolve a ``ModelCallRef`` to the position of exactly one captured call in ``trajectory``.

    Index captured calls by both identities ModelCallRef supports, mirroring
    join_model_call_observations: a ref may carry model_call_id, or the exact
    (model_ref, response_id) pair. An unmatched or ambiguous ref resolves to ``None``.
    """
    calls_by_id: dict[str, list[int]] = {}
    calls_by_response: dict[tuple[str, str, str], list[int]] = {}
    for index, call in enumerate(trajectory.model_calls):
        if call.model_call_id:
            calls_by_id.setdefault(call.model_call_id, []).append(index)
        call_model_ref = call.response_metadata.model_ref
        call_response_id = call.response_metadata.response_id
        if call_model_ref is not None and call_response_id:
            calls_by_response.setdefault((call_model_ref.type, call_model_ref.name, call_response_id), []).append(
                index
            )

    def match(ref: ModelCallRef) -> Optional[int]:
        if ref.model_call_id:
            candidates = [
                index
                for index in calls_by_id.get(ref.model_call_id, [])
                if (
                    ref.model_ref is None or ref.model_ref == trajectory.model_calls[index].response_metadata.model_ref
                )
                and (
                    ref.response_id is None
                    or ref.response_id == trajectory.model_calls[index].response_metadata.response_id
                )
            ]
        elif ref.model_ref is not None and ref.response_id:
            candidates = calls_by_response.get((ref.model_ref.type, ref.model_ref.name, ref.response_id), [])
        else:
            candidates = []
        return candidates[0] if len(candidates) == 1 else None

    return match


def _call_roles(trajectory: TrajectoryRecord) -> dict[int, CapturedCallRole]:
    """Map the position of each referenced captured call to its owner.

    Agent references are applied before verifier references and the first owner wins, so a call referenced by
    both is an agent call. Positions, not ids, identify a call, so two references to one call count it once.
    A reference that matches no single captured call attributes nothing.
    """
    verifier_refs = [ref for invocation in trajectory.verifier_invocations for ref in invocation.model_calls]
    agent_refs = [ref for invocation in trajectory.invocations for ref in invocation.model_calls]
    match = model_call_matcher(trajectory)
    roles: dict[int, CapturedCallRole] = {}
    for role, refs in (("agent", agent_refs), ("verifier", verifier_refs)):
        for ref in refs:
            index = match(ref)
            if index is not None:
                roles.setdefault(index, role)
    return roles


def _totals_by_role(
    calls: list[TrajectoryModelCall], roles: dict[int, CapturedCallRole]
) -> dict[CapturedCallRole, CapturedCallTotals]:
    """Add up calls, reported tokens and call durations per role; calls without a role are ``unattributed``."""
    by_role: dict[CapturedCallRole, CapturedCallTotals] = {}
    for index, call in enumerate(calls):
        totals = by_role.setdefault(roles.get(index, "unattributed"), {"calls": 0, "calls_without_usage": 0})
        totals["calls"] += 1
        stats = call.token_stats
        if stats.prompt_tokens is None and stats.completion_tokens is None and stats.total_tokens is None:
            totals["calls_without_usage"] += 1
        for key, attr in _CAPTURED_CALL_TOKEN_FIELDS:
            value = getattr(stats, attr)
            if value is not None:
                totals[key] = totals.get(key, 0) + value
        if call.duration_ms is not None:
            totals["call_latency_ms"] = totals.get("call_latency_ms", 0.0) + call.duration_ms
    return by_role
