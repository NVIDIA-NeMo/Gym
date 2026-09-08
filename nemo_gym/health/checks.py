# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact normalization and single-rollout health checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from nemo_gym.health.types import (
    ROLLOUT_INDEX_KEY,
    TASK_INDEX_KEY,
    CheckInput,
    CheckScope,
    CheckSpec,
    CheckSubject,
    Finding,
    _AgentStep,
    _CallBindings,
)
from nemo_gym.rollout_observability import TrajectoryRecord


CHECK_REGISTRY: tuple[CheckSpec, ...] = (
    CheckSpec(
        id="check_execution_error",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.CHECK_EXECUTION,
        reads=frozenset({CheckInput.RECORD}),
    ),
    CheckSpec(
        id="record_unreadable",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.RECORD,
        reads=frozenset({CheckInput.RECORD}),
    ),
    CheckSpec(
        id="rollout_duplicate_identity",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.ROLLOUT,
        reads=frozenset({CheckInput.RECORD}),
    ),
    CheckSpec(
        id="rollout_missing_agent_turns",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.ROLLOUT,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.AGENT_TURNS}),
    ),
    CheckSpec(
        id="agent_turn_hollow",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.AGENT_TURN,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.AGENT_TURNS}),
    ),
    CheckSpec(
        id="model_call_zero_completion_tokens",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.BOUND_CALLS}),
    ),
    CheckSpec(
        id="model_call_missing_token_counts",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.BOUND_CALLS}),
    ),
    CheckSpec(
        id="trajectory_capture_mismatch",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.TRAJECTORY_CAPTURE,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.BOUND_CALLS}),
    ),
    CheckSpec(
        id="model_call_failed",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.BOUND_CALLS}),
    ),
    CheckSpec(
        id="rollout_token_count_mismatch",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.ROLLOUT,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.BOUND_CALLS}),
    ),
    CheckSpec(
        id="model_call_runaway_generation",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.BOUND_CALLS}),
    ),
    CheckSpec(
        id="task_consistently_unhealthy",
        evaluation_scope=CheckScope.TASK,
        subject=CheckSubject.TASK,
        reads=frozenset({CheckInput.REPEAT_VERDICTS}),
    ),
    CheckSpec(
        id="task_no_successful_model_calls",
        evaluation_scope=CheckScope.TASK,
        subject=CheckSubject.TASK,
        reads=frozenset({CheckInput.REPEAT_DIGESTS}),
    ),
)

_ROLLOUT_SPECS = tuple(spec for spec in CHECK_REGISTRY if spec.evaluation_scope == CheckScope.ROLLOUT)
_TASK_SPECS = tuple(spec for spec in CHECK_REGISTRY if spec.evaluation_scope == CheckScope.TASK)


def normalize_ignored_checks(checks: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize and validate check IDs supplied by library, CLI, or Hydra config."""
    if checks is None:
        return ()
    raw_checks = checks.split(",") if isinstance(checks, str) else checks
    normalized = tuple(dict.fromkeys(check.strip() for check in raw_checks if check.strip()))
    known_checks = {spec.id for spec in CHECK_REGISTRY}
    unknown_checks = sorted(set(normalized) - known_checks)
    if unknown_checks:
        raise ValueError(f"Unknown rollout health check(s): {', '.join(unknown_checks)}")
    return normalized


def _subject(task_index: int | str, rollout_index: int | str | None = None) -> dict[str, int | str]:
    subject: dict[str, int | str] = {TASK_INDEX_KEY: task_index}
    if rollout_index is not None:
        subject[ROLLOUT_INDEX_KEY] = rollout_index
    return subject


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_nonempty(item) for item in value)
    if isinstance(value, dict):
        return any(
            _nonempty(value.get(key))
            for key in (
                "text",
                "content",
                "output_text",
                "answer",
                "refusal",
                "encrypted_content",
                "reasoning",
                "reasoning_content",
                "summary",
            )
        )
    return False


def _call_ref_key(ref: Any) -> str | None:
    if not isinstance(ref, dict):
        return None
    if ref.get("model_call_id"):
        return f"call:{ref['model_call_id']}"
    model_ref = ref.get("model_ref")
    response_id = ref.get("response_id")
    if isinstance(model_ref, dict) and response_id:
        return f"response:{model_ref.get('type')}:{model_ref.get('name')}:{response_id}"
    return None


_AGENT_TOOL_CALL_TYPES = frozenset(
    {
        "function_call",
        "tool_call",
        "tool_use",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "file_search_call",
        "web_search_call",
        "computer_call",
        "image_generation_call",
        "code_interpreter_call",
        "local_shell_call",
        "custom_tool_call",
    }
)

_INCOMPLETE_MODEL_CALL_GAPS = frozenset(
    {
        "model_calls_unavailable",
        "model_call_capture_incomplete",
        "model_call_capture_records_unreadable",
        "model_call_capture_unreadable",
    }
)

_REFERENCE_CONTRADICTION_GAPS = {
    "model_call_reference_unmatched": "missing_captured_call",
    "model_call_reference_ambiguous": "duplicated_captured_call",
    "model_call_reference_conflict": "conflicting_call_ownership",
}

_LENGTH_LIMIT_FINISH_REASONS = frozenset({"length", "max_output_tokens", "max_tokens"})


def _item_has_tool_call(item: Any) -> bool:
    if isinstance(item, (list, tuple)):
        return any(_item_has_tool_call(value) for value in item)
    if not isinstance(item, dict):
        return False
    if item.get("type") in _AGENT_TOOL_CALL_TYPES:
        return True
    return bool(item.get("tool_calls"))


def _canonical_trajectory(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    raw = record.get("ng_trajectory")
    if raw is None:
        return None, None
    try:
        return TrajectoryRecord.model_validate(raw).model_dump(mode="json"), None
    except Exception as exc:
        return None, type(exc).__name__


def _trajectory_has_gap(trajectory: dict[str, Any], code: str) -> bool:
    return any(isinstance(gap, dict) and gap.get("code") == code for gap in trajectory.get("gaps") or [])


def _trajectory_has_any_gap(trajectory: dict[str, Any], codes: frozenset[str]) -> bool:
    return any(isinstance(gap, dict) and gap.get("code") in codes for gap in trajectory.get("gaps") or [])


def _trajectory_reference_contradictions(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gap
        for gap in trajectory.get("gaps") or []
        if isinstance(gap, dict) and gap.get("code") in _REFERENCE_CONTRADICTION_GAPS
    ]


def _agent_steps(trajectory: dict[str, Any]) -> list[_AgentStep]:
    """Normalize canonical TrajectoryTurn records for structural checks."""
    steps = []
    for position, turn in enumerate(trajectory.get("turns") or []):
        refs = tuple(filter(None, (_call_ref_key(ref) for ref in turn.get("model_calls") or [])))
        steps.append(
            _AgentStep(
                locator={"turn": turn.get("turn_no", position)},
                has_message=_nonempty(turn.get("answer")) or _nonempty(turn.get("reasoning_content")),
                has_tool_calls=_item_has_tool_call(turn.get("answer")),
                model_call_refs=refs,
            )
        )
    return steps


def _normalized_trajectory_calls(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for position, raw in enumerate(trajectory.get("model_calls") or []):
        metadata = raw.get("response_metadata") or {}
        tokens = raw.get("token_stats") or {}
        calls.append(
            {
                "call_index": position,
                "model_call_id": raw.get("model_call_id"),
                "response_id": metadata.get("response_id"),
                "model_ref": metadata.get("model_ref"),
                "status_code": metadata.get("status_code"),
                "response_status": metadata.get("response_status"),
                "finish_reason": metadata.get("finish_reason"),
                "error_category": metadata.get("error_category"),
                "tokens_in": tokens.get("prompt_tokens"),
                "tokens_out": tokens.get("completion_tokens"),
                "request": raw.get("request"),
                "response": raw.get("response"),
            }
        )
    return calls


def _is_failed(call: dict[str, Any]) -> bool:
    status = call.get("status_code")
    response_status = call.get("response_status")
    return (
        (isinstance(status, int) and status >= 400)
        or bool(call.get("error_category"))
        or (isinstance(response_status, str) and response_status in {"failed", "error", "cancelled"})
    )


def _is_successful(call: dict[str, Any]) -> bool:
    status = call.get("status_code")
    return not _is_failed(call) and (status is None or (isinstance(status, int) and 200 <= status < 400))


def _call_identity(call: dict[str, Any]) -> str | None:
    if call.get("model_call_id"):
        return f"call:{call['model_call_id']}"
    model_ref = call.get("model_ref")
    response_id = call.get("response_id")
    if isinstance(model_ref, dict) and response_id:
        return f"response:{model_ref.get('type')}:{model_ref.get('name')}:{response_id}"
    if response_id:
        return f"response::{response_id}"
    return None


def _canonical_model_call_references(trajectory: dict[str, Any]) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return explicit model-call references from canonical TrajectoryTurn records only."""
    return tuple(
        (reference, raw_reference)
        for turn in trajectory.get("turns") or []
        for raw_reference in turn.get("model_calls") or []
        if isinstance(raw_reference, dict) and (reference := _call_ref_key(raw_reference)) is not None
    )


def _bind_policy_calls(trajectory: dict[str, Any], calls: list[dict[str, Any]]) -> _CallBindings:
    reference_items = _canonical_model_call_references(trajectory)
    references = tuple(reference for reference, _ in reference_items)
    calls_by_call_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls_by_response: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        if call.get("model_call_id"):
            calls_by_call_id[str(call["model_call_id"])].append(call)
        model_ref = call.get("model_ref")
        response_id = call.get("response_id")
        if isinstance(model_ref, dict) and response_id:
            calls_by_response[(str(model_ref.get("type")), str(model_ref.get("name")), str(response_id))].append(call)

    matched_calls: list[dict[str, Any]] = []
    missing_references: list[str] = []
    duplicated_references: list[tuple[str, int]] = []
    unique_references = dict(reference_items)
    for reference, raw_reference in unique_references.items():
        model_call_id = raw_reference.get("model_call_id")
        model_ref = raw_reference.get("model_ref")
        response_id = raw_reference.get("response_id")
        if model_call_id:
            matches = [
                call
                for call in calls_by_call_id.get(str(model_call_id), [])
                if (model_ref is None or model_ref == call.get("model_ref"))
                and (response_id is None or response_id == call.get("response_id"))
            ]
        else:
            assert isinstance(model_ref, dict) and response_id
            response_key = (str(model_ref.get("type")), str(model_ref.get("name")), str(response_id))
            matches = calls_by_response.get(response_key, [])
        if not matches:
            missing_references.append(reference)
        elif len(matches) > 1:
            duplicated_references.append((reference, len(matches)))
        else:
            matched_calls.append(matches[0])
    return _CallBindings(
        references=references,
        matched_calls=tuple(matched_calls),
        missing_references=tuple(missing_references),
        duplicated_references=tuple(duplicated_references),
    )


def _replay_identity(call: dict[str, Any]) -> str | None:
    # Gym assigns model_call_id per invocation. Provider response IDs are only a
    # fallback: some backends reuse a placeholder response ID for distinct calls.
    return _call_identity(call)


def _call_locator(call: dict[str, Any], fallback: int) -> dict[str, int | str]:
    return {"call_id": str(call.get("model_call_id") or call.get("response_id") or fallback)}


def _response_has_content(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    if _nonempty(response.get("output_text")) or _nonempty(response.get("content")):
        return True
    output = response.get("output")
    choices = response.get("choices")
    chat_content = any(
        _nonempty(choice.get("text")) or _nonempty((choice.get("message") or {}).get("content"))
        for choice in choices or []
        if isinstance(choice, dict)
    )
    return _nonempty(output) or chat_content


def _usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion = usage.get("output_tokens", usage.get("completion_tokens"))
    return (
        prompt if type(prompt) is int and prompt >= 0 else None,
        completion if type(completion) is int and completion >= 0 else None,
    )


def _token_count(call: dict[str, Any], key: str) -> int:
    value = call.get(key)
    return value if type(value) is int and value >= 0 else 0


def _transcript_tokens(record: dict[str, Any]) -> tuple[int, int, bool]:
    response = record.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    prompt, completion = _usage_tokens(usage)
    return prompt or 0, completion or 0, prompt is not None and completion is not None


def _rollout_missing_agent_turns(trajectory: dict[str, Any], subject: dict[str, int | str]) -> list[Finding]:
    steps = _agent_steps(trajectory)
    if any(step.has_model_activity for step in steps):
        return []
    return [
        Finding(
            check="rollout_missing_agent_turns",
            subject=subject,
            detail={"reason": "no agent turn with model activity"},
        )
    ]


def _agent_turn_hollow(trajectory: dict[str, Any], subject: dict[str, int | str]) -> list[Finding]:
    return [
        Finding(
            check="agent_turn_hollow",
            subject=subject,
            locator=step.locator,
            detail={"reason": "agent turn has no message or tool calls"},
        )
        for step in _agent_steps(trajectory)
        if not step.has_message and not step.has_tool_calls
    ]


def _model_call_zero_completion_tokens(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        Finding(
            check="model_call_zero_completion_tokens",
            subject=subject,
            locator=_call_locator(call, position),
            detail={"completion_tokens": 0},
        )
        for position, call in enumerate(bindings.matched_calls)
        if call.get("tokens_out") == 0
    ]


def _model_call_missing_token_counts(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        Finding(
            check="model_call_missing_token_counts",
            subject=subject,
            locator=_call_locator(call, position),
            detail={
                "missing": [
                    field
                    for field, key in (("prompt_tokens", "tokens_in"), ("completion_tokens", "tokens_out"))
                    if call.get(key) is None
                ]
            },
        )
        for position, call in enumerate(bindings.matched_calls)
        if call.get("tokens_in") is None or call.get("tokens_out") is None
    ]


def _trajectory_capture_mismatch(
    trajectory: dict[str, Any],
    bindings: _CallBindings,
    subject: dict[str, int | str],
) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for reference in bindings.missing_references:
        locator = reference.split(":")[-1]
        seen.add(("missing_captured_call", locator))
        findings.append(
            Finding(
                check="trajectory_capture_mismatch",
                subject=subject,
                locator={"call_id": locator},
                detail={"kind": "missing_captured_call"},
            )
        )
    for reference, count in bindings.duplicated_references:
        locator = reference.split(":")[-1]
        seen.add(("duplicated_captured_call", locator))
        findings.append(
            Finding(
                check="trajectory_capture_mismatch",
                subject=subject,
                locator={"call_id": locator},
                detail={"kind": "duplicated_captured_call", "count": count},
            )
        )
    for gap in _trajectory_reference_contradictions(trajectory):
        kind = _REFERENCE_CONTRADICTION_GAPS[gap["code"]]
        detail = gap.get("detail") or "unknown"
        locator = str(detail).split(":")[-1]
        if (kind, locator) in seen:
            continue
        findings.append(
            Finding(
                check="trajectory_capture_mismatch",
                subject=subject,
                locator={"call_id": locator},
                detail={
                    "kind": kind,
                    "observation_gap": gap["code"],
                    "invocation_id": gap.get("invocation_id"),
                },
            )
        )
    return findings


def _model_call_failed(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        Finding(
            check="model_call_failed",
            subject=subject,
            locator=_call_locator(call, position),
            detail={
                "status": call.get("status_code"),
                "error_category": call.get("error_category"),
                "terminal": bindings.complete and position == len(bindings.matched_calls) - 1,
            },
        )
        for position, call in enumerate(bindings.matched_calls)
        if _is_failed(call)
    ]


def _rollout_token_count_mismatch(
    record: dict[str, Any], bindings: _CallBindings, subject: dict[str, int | str]
) -> list[Finding]:
    transcript_prompt, transcript_completion, transcript_usage_present = _transcript_tokens(record)
    capture_prompt = sum(_token_count(call, "tokens_in") for call in bindings.matched_calls)
    capture_completion = sum(_token_count(call, "tokens_out") for call in bindings.matched_calls)
    if transcript_usage_present and (
        transcript_prompt != capture_prompt or transcript_completion != capture_completion
    ):
        return [
            Finding(
                check="rollout_token_count_mismatch",
                subject=subject,
                detail={
                    "transcript_prompt": transcript_prompt,
                    "transcript_completion": transcript_completion,
                    "capture_prompt": capture_prompt,
                    "capture_completion": capture_completion,
                },
            )
        ]
    return []


def _model_call_runaway_generation(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        Finding(
            check="model_call_runaway_generation",
            subject=subject,
            locator=_call_locator(call, position),
            detail={"finish_reason": call.get("finish_reason")},
        )
        for position, call in enumerate(bindings.matched_calls)
        if call.get("finish_reason") in _LENGTH_LIMIT_FINISH_REASONS
        and not _response_has_content(call.get("response"))
    ]


_ROLLOUT_CHECKS: dict[
    str,
    Callable[
        [dict[str, Any], dict[str, Any], _CallBindings, dict[str, int | str]],
        list[Finding],
    ],
] = {
    "check_execution_error": lambda record, trajectory, bindings, subject: [],
    "record_unreadable": lambda record, trajectory, bindings, subject: [],
    "rollout_duplicate_identity": lambda record, trajectory, bindings, subject: [],
    "rollout_missing_agent_turns": lambda record, trajectory, bindings, subject: _rollout_missing_agent_turns(
        trajectory, subject
    ),
    "agent_turn_hollow": lambda record, trajectory, bindings, subject: _agent_turn_hollow(trajectory, subject),
    "model_call_zero_completion_tokens": lambda record, trajectory, bindings, subject: (
        _model_call_zero_completion_tokens(bindings, subject)
    ),
    "model_call_missing_token_counts": lambda record, trajectory, bindings, subject: (
        _model_call_missing_token_counts(bindings, subject)
    ),
    "trajectory_capture_mismatch": lambda record, trajectory, bindings, subject: (
        _trajectory_capture_mismatch(trajectory, bindings, subject)
    ),
    "model_call_failed": lambda record, trajectory, bindings, subject: _model_call_failed(bindings, subject),
    "rollout_token_count_mismatch": lambda record, trajectory, bindings, subject: (
        _rollout_token_count_mismatch(record, bindings, subject)
    ),
    "model_call_runaway_generation": lambda record, trajectory, bindings, subject: (
        _model_call_runaway_generation(bindings, subject)
    ),
}
