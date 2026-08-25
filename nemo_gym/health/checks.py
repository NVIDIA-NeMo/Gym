# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact normalization and single-rollout health checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import orjson

from nemo_gym.base_responses_api_model import build_model_call_record
from nemo_gym.health.types import (
    ROLLOUT_INDEX_KEY,
    TASK_INDEX_KEY,
    CheckReads,
    CheckScope,
    CheckSpec,
    CheckSubject,
    Finding,
    _AgentStep,
    _AgentStepSource,
    _CallBindings,
)


CHECK_REGISTRY: tuple[CheckSpec, ...] = (
    CheckSpec(
        id="record_unreadable",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.RECORD,
        reads=CheckReads.RECORD,
    ),
    CheckSpec(
        id="rollout_missing_agent_turns",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.ROLLOUT,
        reads=CheckReads.RECORD,
    ),
    CheckSpec(
        id="agent_turn_hollow",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.AGENT_TURN,
        reads=CheckReads.RECORD,
    ),
    CheckSpec(
        id="model_call_zero_completion_tokens",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=CheckReads.BOUND_CALLS,
    ),
    CheckSpec(
        id="model_call_missing_token_counts",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=CheckReads.BOUND_CALLS,
    ),
    CheckSpec(
        id="trajectory_capture_mismatch",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.TRAJECTORY_CAPTURE,
        reads=CheckReads.BOTH,
    ),
    CheckSpec(
        id="model_call_failed",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=CheckReads.BOUND_CALLS,
    ),
    CheckSpec(
        id="rollout_token_count_mismatch",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.ROLLOUT,
        reads=CheckReads.BOTH,
    ),
    CheckSpec(
        id="model_call_runaway_generation",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=CheckReads.BOUND_CALLS,
    ),
    CheckSpec(
        id="task_consistently_unhealthy",
        evaluation_scope=CheckScope.TASK,
        subject=CheckSubject.TASK,
        reads=CheckReads.REPEAT_VERDICTS,
    ),
    CheckSpec(
        id="task_no_healthy_model_calls",
        evaluation_scope=CheckScope.TASK,
        subject=CheckSubject.TASK,
        reads=CheckReads.REPEAT_DIGESTS,
    ),
)

_ROLLOUT_SPECS = tuple(spec for spec in CHECK_REGISTRY if spec.evaluation_scope == CheckScope.ROLLOUT)
_TASK_SPECS = tuple(spec for spec in CHECK_REGISTRY if spec.evaluation_scope == CheckScope.TASK)
_FALLBACK_TRANSCRIPT_CHECK_IDS = frozenset({"rollout_missing_agent_turns", "agent_turn_hollow"})


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


def _finding(
    check: str,
    subject: dict[str, int | str],
    *,
    locator: dict[str, int | str] | None = None,
    **detail: Any,
) -> Finding:
    return Finding(check=check, subject=subject, locator=locator, detail=detail)


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
_AGENT_TURN_BOUNDARY_TYPES = frozenset(
    {
        "function_call_output",
        "tool_call_output",
        "tool_result",
        "computer_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
        "mcp_approval_response",
    }
)


def _item_has_tool_call(item: Any) -> bool:
    if isinstance(item, (list, tuple)):
        return any(_item_has_tool_call(value) for value in item)
    if not isinstance(item, dict):
        return False
    if item.get("type") in _AGENT_TOOL_CALL_TYPES:
        return True
    return bool(item.get("tool_calls"))


def _item_is_agent_content(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return item.get("role") in {"assistant", "agent"} or item.get("type") == "reasoning" or _item_has_tool_call(item)


def _item_ends_agent_turn(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return item.get("role") == "user" or item.get("type") in _AGENT_TURN_BOUNDARY_TYPES


def _response_output_steps(output: list[Any]) -> list[_AgentStep]:
    """Group adjacent agent-side Responses items into transcript turns."""
    steps: list[_AgentStep] = []
    has_message = False
    has_tool_calls = False
    has_agent_content = False

    def flush() -> None:
        nonlocal has_message, has_tool_calls, has_agent_content
        if has_agent_content:
            steps.append(
                _AgentStep(
                    locator={"turn": len(steps)},
                    has_message=has_message,
                    has_tool_calls=has_tool_calls,
                    model_call_refs=(),
                )
            )
        has_message = False
        has_tool_calls = False
        has_agent_content = False

    for item in output:
        if _item_ends_agent_turn(item):
            flush()
            continue
        if not _item_is_agent_content(item):
            continue
        has_agent_content = True
        has_tool_calls = has_tool_calls or _item_has_tool_call(item)
        has_message = has_message or _nonempty(item)
    flush()
    return steps


def _agent_steps_with_source(record: dict[str, Any]) -> tuple[list[_AgentStep], _AgentStepSource]:
    """Return agent-step evidence and the persisted source used to derive it."""
    trajectory = record.get("ng_trajectory")
    if isinstance(trajectory, dict):
        turns = trajectory.get("turns")
        if isinstance(turns, list) and turns:
            normalized = []
            for position, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                refs = tuple(filter(None, (_call_ref_key(ref) for ref in turn.get("model_calls") or [])))
                normalized.append(
                    _AgentStep(
                        locator={"turn": turn.get("turn_no", position)},
                        has_message=_nonempty(turn.get("answer")) or _nonempty(turn.get("reasoning_content")),
                        has_tool_calls=_item_has_tool_call(turn.get("answer"))
                        or _item_has_tool_call(turn.get("tool_calls")),
                        model_call_refs=refs,
                    )
                )
            if normalized:
                return normalized, "trajectory_turns"

        invocations = trajectory.get("invocations")
        if isinstance(invocations, list):
            normalized = []
            for invocation_position, invocation in enumerate(invocations):
                if not isinstance(invocation, dict):
                    continue
                invocation_refs = tuple(
                    filter(None, (_call_ref_key(ref) for ref in invocation.get("model_calls") or []))
                )
                conversation = invocation.get("conversation")
                agent_items = [item for item in conversation or [] if _item_is_agent_content(item)]
                if agent_items:
                    normalized.append(
                        _AgentStep(
                            locator={"invocation": str(invocation.get("invocation_id", invocation_position))},
                            has_message=any(_nonempty(item) for item in agent_items),
                            has_tool_calls=any(_item_has_tool_call(item) for item in agent_items),
                            model_call_refs=invocation_refs,
                        )
                    )
            if normalized:
                return normalized, "trajectory_invocations"

    response = record.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(output, list):
        return [], "none"
    return _response_output_steps(output), "response_output"


def _agent_steps(record: dict[str, Any]) -> list[_AgentStep]:
    """Return the best available agent-step evidence for health checks."""
    return _agent_steps_with_source(record)[0]


def _normalized_embedded_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = record.get("ng_trajectory")
    raw_calls = trajectory.get("model_calls") if isinstance(trajectory, dict) else None
    if isinstance(raw_calls, list) and raw_calls:
        calls: list[dict[str, Any]] = []
        for position, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                continue
            metadata = raw.get("response_metadata") if isinstance(raw.get("response_metadata"), dict) else {}
            tokens = raw.get("token_stats") if isinstance(raw.get("token_stats"), dict) else {}
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

    capture = record.get("ng_model_call_capture")
    raw_calls = capture.get("calls") if isinstance(capture, dict) else None
    return [dict(call) for call in raw_calls or [] if isinstance(call, dict)]


def _parse_capture(path: str | None, record: dict[str, Any]) -> tuple[list[dict[str, Any]], int, bool]:
    """Read a sidecar once, or use the persisted projection when no sidecar is available."""
    if path is None:
        embedded = _normalized_embedded_calls(record)
        return embedded, 0, bool(embedded)

    calls: list[dict[str, Any]] = []
    invalid = 0
    with open(path, "rb") as handle:
        for position, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                raw = orjson.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("capture line is not an object")
                if "call_index" in raw or "tokens_in" in raw or "token_stats" in raw:
                    normalized = dict(raw)
                    normalized.setdefault("call_index", position)
                else:
                    normalized = build_model_call_record(raw, call_index=position).model_dump(mode="json")
                calls.append(normalized)
            except Exception:
                invalid += 1
    return calls, invalid, True


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


def _canonical_model_call_references(record: dict[str, Any]) -> tuple[str, ...]:
    """Return explicit model-call references from canonical TrajectoryTurn records only."""
    trajectory = record.get("ng_trajectory")
    turns = trajectory.get("turns") if isinstance(trajectory, dict) else None
    if not isinstance(turns, list) or not turns:
        return ()
    return tuple(
        reference
        for turn in turns
        if isinstance(turn, dict)
        for raw_reference in turn.get("model_calls") or []
        if (reference := _call_ref_key(raw_reference)) is not None
    )


def _bind_policy_calls(record: dict[str, Any], calls: list[dict[str, Any]]) -> _CallBindings:
    references = _canonical_model_call_references(record)
    calls_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        if identity := _call_identity(call):
            calls_by_identity[identity].append(call)

    matched_calls: list[dict[str, Any]] = []
    missing_references: list[str] = []
    duplicated_references: list[tuple[str, int]] = []
    for reference in dict.fromkeys(references):
        matches = calls_by_identity.get(reference, [])
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


def _rollout_missing_agent_turns(record: dict[str, Any], subject: dict[str, int | str]) -> list[Finding]:
    steps = _agent_steps(record)
    if any(step.has_model_activity for step in steps):
        return []
    return [_finding("rollout_missing_agent_turns", subject, reason="no agent turn with model activity")]


def _agent_turn_hollow(record: dict[str, Any], subject: dict[str, int | str]) -> list[Finding]:
    return [
        _finding("agent_turn_hollow", subject, locator=step.locator, reason="agent turn has no message or tool calls")
        for step in _agent_steps(record)
        if not step.has_message and not step.has_tool_calls
    ]


def _model_call_zero_completion_tokens(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        _finding(
            "model_call_zero_completion_tokens",
            subject,
            locator=_call_locator(call, position),
            completion_tokens=0,
        )
        for position, call in enumerate(bindings.matched_calls)
        if call.get("tokens_out") == 0
    ]


def _model_call_missing_token_counts(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        _finding(
            "model_call_missing_token_counts",
            subject,
            locator=_call_locator(call, position),
            missing=[
                field
                for field, key in (("prompt_tokens", "tokens_in"), ("completion_tokens", "tokens_out"))
                if call.get(key) is None
            ],
        )
        for position, call in enumerate(bindings.matched_calls)
        if call.get("tokens_in") is None or call.get("tokens_out") is None
    ]


def _trajectory_capture_mismatch(
    bindings: _CallBindings,
    invalid_capture_lines: int,
    subject: dict[str, int | str],
) -> list[Finding]:
    findings: list[Finding] = []
    if invalid_capture_lines:
        findings.append(
            _finding(
                "trajectory_capture_mismatch",
                subject,
                kind="unreadable_capture_records",
                count=invalid_capture_lines,
            )
        )
    for reference in bindings.missing_references:
        findings.append(
            _finding(
                "trajectory_capture_mismatch",
                subject,
                locator={"call_id": reference.split(":")[-1]},
                kind="missing_captured_call",
            )
        )
    for reference, count in bindings.duplicated_references:
        findings.append(
            _finding(
                "trajectory_capture_mismatch",
                subject,
                locator={"call_id": reference.split(":")[-1]},
                kind="duplicated_captured_call",
                count=count,
            )
        )
    return findings


def _model_call_failed(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        _finding(
            "model_call_failed",
            subject,
            locator=_call_locator(call, position),
            status=call.get("status_code"),
            error_category=call.get("error_category"),
            terminal=bindings.complete and position == len(bindings.matched_calls) - 1,
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
            _finding(
                "rollout_token_count_mismatch",
                subject,
                transcript_prompt=transcript_prompt,
                transcript_completion=transcript_completion,
                capture_prompt=capture_prompt,
                capture_completion=capture_completion,
            )
        ]
    return []


def _model_call_runaway_generation(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        _finding(
            "model_call_runaway_generation",
            subject,
            locator=_call_locator(call, position),
            finish_reason="length",
        )
        for position, call in enumerate(bindings.matched_calls)
        if call.get("finish_reason") == "length" and not _response_has_content(call.get("response"))
    ]


_ROLLOUT_CHECKS: dict[
    str,
    Callable[
        [dict[str, Any], list[dict[str, Any]], _CallBindings, int, dict[str, int | str]],
        list[Finding],
    ],
] = {
    "record_unreadable": lambda record, calls, bindings, invalid, subject: [],
    "rollout_missing_agent_turns": lambda record, calls, bindings, invalid, subject: _rollout_missing_agent_turns(
        record, subject
    ),
    "agent_turn_hollow": lambda record, calls, bindings, invalid, subject: _agent_turn_hollow(record, subject),
    "model_call_zero_completion_tokens": lambda record, calls, bindings, invalid, subject: (
        _model_call_zero_completion_tokens(bindings, subject)
    ),
    "model_call_missing_token_counts": lambda record, calls, bindings, invalid, subject: (
        _model_call_missing_token_counts(bindings, subject)
    ),
    "trajectory_capture_mismatch": lambda record, calls, bindings, invalid, subject: (
        _trajectory_capture_mismatch(bindings, invalid, subject)
    ),
    "model_call_failed": lambda record, calls, bindings, invalid, subject: _model_call_failed(bindings, subject),
    "rollout_token_count_mismatch": lambda record, calls, bindings, invalid, subject: (
        _rollout_token_count_mismatch(record, bindings, subject)
    ),
    "model_call_runaway_generation": lambda record, calls, bindings, invalid, subject: (
        _model_call_runaway_generation(bindings, subject)
    ),
}
