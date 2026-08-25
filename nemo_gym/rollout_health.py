# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic post-run rollout quality verification.

The module deliberately operates only on persisted rollout and model-call capture
artifacts.  Checks return evidence; verdicts are derived centrally by the runner.
"""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_responses_api_model import build_model_call_record


TASK_INDEX_KEY = "_ng_task_index"
ROLLOUT_INDEX_KEY = "_ng_rollout_index"
ROLLOUT_ID_KEY = "_ng_rollout_id"
QUALITY_SUMMARY_FILENAME = "quality_summary.json"
ROLLOUT_VERDICTS_FILENAME = "rollout_verdicts.jsonl"

Verdict = Literal["healthy", "unhealthy", "unobserved"]
_AgentStepSource = Literal["trajectory_turns", "trajectory_invocations", "response_output", "none"]


class CheckScope(str, Enum):
    ROLLOUT = "rollout"
    TASK = "task"
    RUN = "run"


class CheckSubject(str, Enum):
    RECORD = "record"
    ROLLOUT = "rollout"
    AGENT_TURN = "agent_turn"
    MODEL_CALL = "model_call"
    TRAJECTORY_CAPTURE = "trajectory_capture"
    TASK = "task"


class CheckReads(str, Enum):
    RECORD = "record"
    CAPTURE = "capture"
    BOTH = "both"
    BOUND_CALLS = "bound_calls"
    REPEAT_VERDICTS = "repeat_verdicts"
    REPEAT_DIGESTS = "repeat_digests"


class CheckSpec(BaseModel):
    """Stable, self-describing health-check contract."""

    model_config = ConfigDict(frozen=True)

    id: str
    evaluation_scope: CheckScope
    subject: CheckSubject
    reads: CheckReads


class Finding(BaseModel):
    """Evidence emitted by a check. Checks never emit verdicts."""

    check: str
    subject: dict[str, int | str]
    locator: dict[str, int | str] | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


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


class RolloutDigest(BaseModel):
    task_index: int | str
    rollout_index: int | str
    rollout_id: str
    verdict: Verdict
    findings: list[Finding]
    unobserved: list[str]
    capture_observed: bool
    policy_calls_observed: bool = False
    model_calls: int = 0
    successful_model_calls: int = 0
    model_call_errors: int = 0
    errors_by_status: dict[str, int] = Field(default_factory=dict)
    ended_on_error: bool = False
    duplicated_calls: int = 0
    transcript_prompt_tokens: int = 0
    transcript_completion_tokens: int = 0
    capture_prompt_tokens: int = 0
    capture_completion_tokens: int = 0


class HealthCheckResult(BaseModel):
    summary: dict[str, Any]
    rollouts: list[RolloutDigest]
    summary_path: Path
    verdicts_path: Path


@dataclass(frozen=True, slots=True)
class _LineSlice:
    path: str
    offset: int
    length: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class _WorkerInput:
    line: _LineSlice
    capture_dirs: tuple[str, ...]
    captures_exist: bool
    capture_enabled: bool | None
    driver_bypass: bool
    ignored_checks: frozenset[str]


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    digest: RolloutDigest
    agent_step_source: _AgentStepSource


@dataclass(frozen=True, slots=True)
class _AgentStep:
    locator: dict[str, int | str]
    has_message: bool
    has_tool_calls: bool
    model_call_refs: tuple[str, ...]

    @property
    def has_model_activity(self) -> bool:
        return self.has_message or self.has_tool_calls or bool(self.model_call_refs)


@dataclass(frozen=True, slots=True)
class _CallBindings:
    references: tuple[str, ...]
    matched_calls: tuple[dict[str, Any], ...]
    missing_references: tuple[str, ...]
    duplicated_references: tuple[tuple[str, int], ...]

    @property
    def observed(self) -> bool:
        return bool(self.references)

    @property
    def complete(self) -> bool:
        return self.observed and not self.missing_references and not self.duplicated_references


@dataclass(frozen=True, slots=True)
class _TaskRepeat:
    rollout_index: int | str
    verdict: Verdict
    policy_calls_observed: bool
    successful_model_calls: int


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


def _read_record(line: _LineSlice) -> tuple[dict[str, Any], str | None]:
    with open(line.path, "rb") as handle:
        handle.seek(line.offset)
        raw = handle.read(line.length).strip()
    try:
        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("rollout line is not an object")
        return parsed, None
    except Exception as exc:
        return {}, type(exc).__name__


def _worker(payload: _WorkerInput) -> _WorkerResult:
    record, parse_error = _read_record(payload.line)
    agent_step_source: _AgentStepSource = "none" if parse_error else _agent_steps_with_source(record)[1]
    task_index = record.get(TASK_INDEX_KEY, payload.line.ordinal)
    rollout_index = record.get(ROLLOUT_INDEX_KEY, 0)
    trajectory = record.get("ng_trajectory")
    trajectory_rollout_id = trajectory.get("rollout_id") if isinstance(trajectory, dict) else None
    rollout_id = str(record.get(ROLLOUT_ID_KEY) or trajectory_rollout_id or f"{task_index}-{rollout_index}")
    subject = _subject(task_index, rollout_index)

    capture_path = next(
        (
            str(candidate)
            for directory in payload.capture_dirs
            if (candidate := Path(directory) / f"{rollout_id}.capture.jsonl").is_file()
        ),
        None,
    )
    if parse_error:
        capture_path = None
        calls, invalid_capture_lines, embedded_capture = [], 0, False
    elif payload.driver_bypass:
        capture_path = None
        calls, invalid_capture_lines, embedded_capture = [], 0, False
    else:
        calls, invalid_capture_lines, embedded_capture = _parse_capture(capture_path, record)
    if payload.driver_bypass:
        capture_observed = False
    elif capture_path is not None or embedded_capture:
        capture_observed = True
    elif payload.capture_enabled is False or payload.captures_exist or payload.capture_enabled is True:
        capture_observed = False
    else:
        capture_observed = embedded_capture
    bindings = _bind_policy_calls(record, calls)
    findings: list[Finding] = []
    unobserved: list[str] = []

    for spec in _ROLLOUT_SPECS:
        if spec.id in payload.ignored_checks:
            continue
        if parse_error:
            if spec.id == "record_unreadable":
                findings.append(
                    _finding("record_unreadable", subject, reason="rollout record is unreadable", error=parse_error)
                )
            else:
                unobserved.append(spec.id)
            continue
        needs_capture = spec.reads in {CheckReads.CAPTURE, CheckReads.BOTH, CheckReads.BOUND_CALLS}
        if needs_capture and not capture_observed:
            unobserved.append(spec.id)
            continue
        if spec.reads == CheckReads.BOUND_CALLS and not bindings.matched_calls:
            unobserved.append(spec.id)
            continue
        if spec.id == "trajectory_capture_mismatch" and not bindings.observed and not invalid_capture_lines:
            unobserved.append(spec.id)
            continue
        if spec.id == "rollout_token_count_mismatch" and (
            not bindings.complete
            or not bindings.matched_calls
            or not _transcript_tokens(record)[2]
            or any(call.get("tokens_in") is None or call.get("tokens_out") is None for call in bindings.matched_calls)
        ):
            unobserved.append(spec.id)
            continue
        try:
            findings.extend(_ROLLOUT_CHECKS[spec.id](record, calls, bindings, invalid_capture_lines, subject))
        except Exception as exc:
            unobserved.append(spec.id)
            findings.append(
                _finding(
                    "record_unreadable",
                    subject,
                    reason="check input is unreadable",
                    failed_check=spec.id,
                    error=type(exc).__name__,
                )
            )

    verdict: Verdict = "unhealthy" if findings else "unobserved" if unobserved else "healthy"
    failed = [call for call in calls if _is_failed(call)]
    errors_by_status = Counter(
        str(call.get("status_code") if call.get("status_code") is not None else "unknown") for call in failed
    )
    identities = [identity for call in calls if (identity := _replay_identity(call)) is not None]
    duplicated = sum(count - 1 for count in Counter(identities).values() if count > 1)
    transcript_prompt, transcript_completion, _ = _transcript_tokens(record)

    return _WorkerResult(
        digest=RolloutDigest(
            task_index=task_index,
            rollout_index=rollout_index,
            rollout_id=rollout_id,
            verdict=verdict,
            findings=findings,
            unobserved=unobserved,
            capture_observed=capture_observed,
            policy_calls_observed=bindings.complete and not invalid_capture_lines,
            model_calls=len(calls),
            successful_model_calls=sum(_is_successful(call) for call in bindings.matched_calls),
            model_call_errors=len(failed),
            errors_by_status=dict(errors_by_status),
            ended_on_error=bool(calls and _is_failed(calls[-1])),
            duplicated_calls=duplicated,
            transcript_prompt_tokens=transcript_prompt,
            transcript_completion_tokens=transcript_completion,
            capture_prompt_tokens=sum(_token_count(call, "tokens_in") for call in calls),
            capture_completion_tokens=sum(_token_count(call, "tokens_out") for call in calls),
        ),
        agent_step_source=agent_step_source,
    )


def _index_jsonl(paths: Sequence[Path]) -> list[_LineSlice]:
    slices: list[_LineSlice] = []
    ordinal = 0
    for path in paths:
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                slices.append(_LineSlice(str(path), offset, len(line), ordinal))
                ordinal += 1
    return slices


def _unique_task_repeats(digests: list[RolloutDigest]) -> list[_TaskRepeat]:
    """Collapse duplicate persisted records for task-level repeat semantics."""
    grouped: dict[int | str, list[RolloutDigest]] = defaultdict(list)
    for digest in digests:
        grouped[digest.rollout_index].append(digest)

    repeats: list[_TaskRepeat] = []
    for rollout_index, copies in grouped.items():
        verdicts = {copy.verdict for copy in copies}
        repeats.append(
            _TaskRepeat(
                rollout_index=rollout_index,
                verdict=verdicts.pop() if len(verdicts) == 1 else "unobserved",
                policy_calls_observed=all(copy.policy_calls_observed for copy in copies),
                successful_model_calls=max(copy.successful_model_calls for copy in copies),
            )
        )
    return repeats


def _task_findings(
    grouped: dict[int | str, list[_TaskRepeat]],
    ignored_checks: frozenset[str],
) -> tuple[dict[int | str, list[Finding]], dict[str, dict[str, int]]]:
    findings: dict[int | str, list[Finding]] = defaultdict(list)
    coverage = {spec.id: {"evaluated": 0, "unobserved": 0, "ignored": 0} for spec in _TASK_SPECS}
    for task_index, repeats in grouped.items():
        subject = _subject(task_index)

        if "task_consistently_unhealthy" in ignored_checks:
            coverage["task_consistently_unhealthy"]["ignored"] += 1
        else:
            computable = [repeat for repeat in repeats if repeat.verdict != "unobserved"]
            if len(computable) >= 2:
                coverage["task_consistently_unhealthy"]["evaluated"] += 1
                if all(repeat.verdict == "unhealthy" for repeat in computable):
                    findings[task_index].append(
                        _finding(
                            "task_consistently_unhealthy",
                            subject,
                            computable_repeats=len(computable),
                        )
                    )
            else:
                coverage["task_consistently_unhealthy"]["unobserved"] += 1

        if "task_no_healthy_model_calls" in ignored_checks:
            coverage["task_no_healthy_model_calls"]["ignored"] += 1
        else:
            if repeats and all(repeat.policy_calls_observed for repeat in repeats):
                coverage["task_no_healthy_model_calls"]["evaluated"] += 1
                if not any(repeat.successful_model_calls for repeat in repeats):
                    findings[task_index].append(_finding("task_no_healthy_model_calls", subject, repeats=len(repeats)))
            else:
                coverage["task_no_healthy_model_calls"]["unobserved"] += 1
    return findings, coverage


def _reduce(digests: list[RolloutDigest], ignored_checks: frozenset[str]) -> dict[str, Any]:
    records_by_task: dict[int | str, list[RolloutDigest]] = defaultdict(list)
    for digest in digests:
        records_by_task[digest.task_index].append(digest)
    grouped = {task_index: _unique_task_repeats(records) for task_index, records in records_by_task.items()}
    task_findings, task_coverage = _task_findings(grouped, ignored_checks)

    coverage = {spec.id: {"evaluated": 0, "unobserved": 0, "ignored": 0} for spec in CHECK_REGISTRY}
    for digest in digests:
        unobserved = set(digest.unobserved)
        for spec in _ROLLOUT_SPECS:
            if spec.id in ignored_checks:
                coverage[spec.id]["ignored"] += 1
            else:
                coverage[spec.id]["unobserved" if spec.id in unobserved else "evaluated"] += 1
    coverage.update(task_coverage)

    issues = Counter(finding.check for digest in digests for finding in digest.findings)
    issues.update(finding.check for findings in task_findings.values() for finding in findings)
    verdicts = Counter(digest.verdict for digest in digests)
    error_statuses: Counter[str] = Counter()
    for digest in digests:
        error_statuses.update(digest.errors_by_status)

    tasks: dict[str, Any] = {}
    for task_index in sorted(grouped, key=lambda value: (isinstance(value, str), str(value))):
        repeats = grouped[task_index]
        repeat_verdicts = Counter(repeat.verdict for repeat in repeats)
        tasks[str(task_index)] = {
            "repeats": len(repeats),
            "healthy": repeat_verdicts["healthy"],
            "unhealthy": repeat_verdicts["unhealthy"],
            "unobserved": repeat_verdicts["unobserved"],
            "flags": [finding.check for finding in task_findings[task_index]],
        }

    return {
        "run": {
            "ignored_checks": sorted(ignored_checks),
            "artifacts": {
                "records": len(digests),
                "captures": sum(digest.capture_observed for digest in digests),
                "coverage": coverage,
            },
            "verdicts": {
                "healthy": verdicts["healthy"],
                "unhealthy": verdicts["unhealthy"],
                "unobserved": verdicts["unobserved"],
            },
            "issues": {spec.id: issues[spec.id] for spec in CHECK_REGISTRY},
            "stats": {
                "model_call_errors": {
                    "total": sum(digest.model_call_errors for digest in digests),
                    "by_status": dict(sorted(error_statuses.items())),
                    "rollouts_affected": sum(bool(digest.model_call_errors) for digest in digests),
                    "ended_on_error": sum(digest.ended_on_error for digest in digests),
                },
                "duplicated_calls": {
                    "replayed": sum(digest.duplicated_calls for digest in digests),
                    "rollouts": sum(bool(digest.duplicated_calls) for digest in digests),
                },
                "tokens": {
                    "prompt": sum(digest.transcript_prompt_tokens for digest in digests),
                    "completion": sum(digest.transcript_completion_tokens for digest in digests),
                    "capture_prompt": sum(digest.capture_prompt_tokens for digest in digests),
                    "capture_completion": sum(digest.capture_completion_tokens for digest in digests),
                },
            },
        },
        "tasks": tasks,
    }


def _sort_key(digest: RolloutDigest) -> tuple[tuple[int, Any], tuple[int, Any]]:
    def part(value: int | str) -> tuple[int, Any]:
        return (0, value) if isinstance(value, int) else (1, str(value))

    return part(digest.task_index), part(digest.rollout_index)


def _write_reports(summary: dict[str, Any], digests: list[RolloutDigest], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / QUALITY_SUMMARY_FILENAME
    verdicts_path = output_dir / ROLLOUT_VERDICTS_FILENAME
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    with verdicts_path.open("wb") as handle:
        for digest in sorted(digests, key=_sort_key):
            findings = [
                finding.model_dump(mode="json", exclude={"subject"}, exclude_none=True) for finding in digest.findings
            ]
            row = {
                TASK_INDEX_KEY: digest.task_index,
                ROLLOUT_INDEX_KEY: digest.rollout_index,
                "rollout_id": digest.rollout_id,
                "verdict": digest.verdict,
                "findings": findings,
                "unobserved": digest.unobserved,
            }
            handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
    return summary_path, verdicts_path


def _warn_noncanonical_agent_steps(worker_results: Sequence[_WorkerResult], ignored_checks: frozenset[str]) -> None:
    enabled_transcript_checks = _FALLBACK_TRANSCRIPT_CHECK_IDS - ignored_checks
    if not enabled_transcript_checks:
        return
    counts = Counter(
        result.agent_step_source
        for result in worker_results
        if result.agent_step_source in {"trajectory_invocations", "response_output"}
        and any(check_id not in result.digest.unobserved for check_id in enabled_transcript_checks)
    )
    if not counts:
        return
    details = []
    if counts["trajectory_invocations"]:
        details.append(f"{counts['trajectory_invocations']} used coarse ng_trajectory.invocations evidence")
    if counts["response_output"]:
        details.append(f"{counts['response_output']} used heuristic response.output grouping")
    warnings.warn(
        f"ng_trajectory.turns was unavailable for {sum(counts.values())} rollout record(s); "
        f"{'; '.join(details)}. Turn-based health results for these records are best-effort. "
        "Current producers should emit TrajectoryRecord.turns.",
        RuntimeWarning,
        stacklevel=3,
    )


def run_health_checks(
    rollout_paths: Path | Sequence[Path],
    *,
    output_dir: Path | None = None,
    capture_dirs: Sequence[Path] = (),
    workers: int | None = None,
    capture_enabled: bool | None = None,
    driver_bypass: bool = False,
    ignored_checks: Sequence[str] = (),
) -> HealthCheckResult:
    """Run the RFC's map/group/reduce pipeline and write both reports."""
    ignored = frozenset(normalize_ignored_checks(ignored_checks))
    paths = [rollout_paths] if isinstance(rollout_paths, Path) else list(rollout_paths)
    if not paths:
        raise ValueError("at least one rollout JSONL path is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Rollout JSONL not found: {path}")

    lines = _index_jsonl(paths)
    capture_dir_strings = tuple(str(path) for path in capture_dirs)
    captures_exist = any(directory.rglob("*.capture.jsonl") for directory in capture_dirs if directory.exists())
    worker_inputs = [
        _WorkerInput(
            line=line,
            capture_dirs=capture_dir_strings,
            captures_exist=captures_exist,
            capture_enabled=capture_enabled,
            driver_bypass=driver_bypass,
            ignored_checks=ignored,
        )
        for line in lines
    ]

    max_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)
    if max_workers < 1:
        raise ValueError("workers must be at least 1")
    if len(worker_inputs) <= 1 or max_workers == 1:
        worker_results = [_worker(item) for item in worker_inputs]
    else:
        try:
            pool = ProcessPoolExecutor(max_workers=max_workers)
        except (NotImplementedError, OSError) as exc:
            warnings.warn(
                f"Process pool unavailable ({exc}); running rollout health checks serially.",
                RuntimeWarning,
                stacklevel=2,
            )
            worker_results = [_worker(item) for item in worker_inputs]
        else:
            try:
                with pool:
                    worker_results = list(pool.map(_worker, worker_inputs))
            except (BrokenProcessPool, OSError) as exc:
                warnings.warn(
                    f"Process pool failed ({exc}); running rollout health checks serially.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                worker_results = [_worker(item) for item in worker_inputs]

    _warn_noncanonical_agent_steps(worker_results, ignored)
    digests = [result.digest for result in worker_results]
    summary = _reduce(digests, ignored)
    report_dir = output_dir or paths[0].parent
    summary_path, verdicts_path = _write_reports(summary, digests, report_dir)
    return HealthCheckResult(
        summary=summary,
        rollouts=digests,
        summary_path=summary_path,
        verdicts_path=verdicts_path,
    )


def _discover_rollouts(run_dir: Path) -> list[Path]:
    if run_dir.is_file():
        return [run_dir]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    conventional = run_dir / "rollouts.jsonl"
    if conventional.is_file():
        return [conventional]
    excluded = ("rollout_verdicts", "failures", "materialized")
    candidates = [
        path
        for path in sorted(run_dir.glob("*.jsonl"))
        if not any(marker in path.stem for marker in excluded) and not path.name.endswith(".capture.jsonl")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected {conventional} or exactly one rollout JSONL in {run_dir}; found {len(candidates)}")
    return candidates


def _discover_capture_dirs(run_dir: Path) -> list[Path]:
    root = run_dir if run_dir.is_dir() else run_dir.parent
    captures = list(root.rglob("*.capture.jsonl"))
    return sorted({path.parent for path in captures})


def format_health_report(result: HealthCheckResult) -> str:
    verdicts = result.summary["run"]["verdicts"]
    checked = sum(verdicts.values())
    ignored = result.summary["run"].get("ignored_checks", [])
    ignored_note = f" (ignored: {', '.join(ignored)})" if ignored else ""
    return (
        f"Rollout health: {checked} checked, {verdicts['healthy']} healthy, "
        f"{verdicts['unhealthy']} unhealthy, {verdicts['unobserved']} unobserved{ignored_note}\n"
        f"Quality summary: {result.summary_path}"
    )


def health_check_run_dir(
    run_dir: str | Path,
    *,
    workers: int | None = None,
    ignored_checks: Sequence[str] = (),
) -> HealthCheckResult:
    path = Path(run_dir)
    rollout_paths = _discover_rollouts(path)
    capture_dirs = _discover_capture_dirs(path)
    result = run_health_checks(
        rollout_paths,
        output_dir=path if path.is_dir() else path.parent,
        capture_dirs=capture_dirs,
        workers=workers,
        capture_enabled=True if capture_dirs else None,
        ignored_checks=ignored_checks,
    )
    print(format_health_report(result))
    return result
