# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact normalization and single-rollout health checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from nemo_gym.rollouts.health.types import (
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
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord


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
        id="rollout_ended_on_failed_model_call",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.ROLLOUT,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.OBSERVED_MODEL_CALLS}),
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
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.OWNED_MODEL_CALLS}),
    ),
    CheckSpec(
        id="model_call_missing_token_counts",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.OWNED_MODEL_CALLS}),
    ),
    CheckSpec(
        id="trajectory_capture_mismatch",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.TRAJECTORY_CAPTURE,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.OWNED_MODEL_CALLS}),
    ),
    CheckSpec(
        id="rollout_token_count_mismatch",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.ROLLOUT,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.OWNED_MODEL_CALLS}),
    ),
    CheckSpec(
        id="model_call_runaway_generation",
        evaluation_scope=CheckScope.ROLLOUT,
        subject=CheckSubject.MODEL_CALL,
        reads=frozenset({CheckInput.RECORD, CheckInput.TRAJECTORY, CheckInput.OWNED_MODEL_CALLS}),
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
    CheckSpec(
        id="run_excessive_failures",
        evaluation_scope=CheckScope.RUN,
        subject=CheckSubject.RUN,
        reads=frozenset({CheckInput.RUN_ACCOUNTING}),
    ),
)

_ROLLOUT_SPECS = tuple(spec for spec in CHECK_REGISTRY if spec.evaluation_scope == CheckScope.ROLLOUT)
_TASK_SPECS = tuple(spec for spec in CHECK_REGISTRY if spec.evaluation_scope == CheckScope.TASK)


def normalize_ignored_checks(checks: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize and validate check IDs supplied by library, CLI, or Hydra config."""
    if checks is None:
        return ()
    raw_checks = checks.split(",") if isinstance(checks, str) else checks
    normalized = tuple(
        dict.fromkeys(
            "rollout_ended_on_failed_model_call" if check.strip() == "model_call_failed" else check.strip()
            for check in raw_checks
            if check.strip()
        )
    )
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
    # The session each invocation belongs to: activations of one harness session share it.
    sessions = {
        invocation["invocation_id"]: invocation.get("harness_session_id") or invocation["invocation_id"]
        for invocation in trajectory.get("invocations") or []
        if isinstance(invocation, dict) and invocation.get("invocation_id")
    }
    calls: list[dict[str, Any]] = []
    for position, raw in enumerate(trajectory.get("model_calls") or []):
        metadata = raw.get("response_metadata") or {}
        tokens = raw.get("token_stats") or {}
        calls.append(
            {
                "call_index": position,
                # Ordering signal. `call_index` is the position in the merged
                # trajectory list, which is not chronological.
                "started_at": raw.get("started_at"),
                "model_call_id": raw.get("model_call_id"),
                "response_id": metadata.get("response_id"),
                "model_ref": metadata.get("model_ref"),
                "status_code": metadata.get("status_code"),
                "response_status": metadata.get("response_status"),
                "finish_reason": metadata.get("finish_reason"),
                "error_category": metadata.get("error_category"),
                "upstream_attempted": metadata.get("upstream_attempted"),
                "upstream_status_code": metadata.get("upstream_status_code"),
                "local_response_reason": metadata.get("local_response_reason"),
                "tokens_in": tokens.get("prompt_tokens"),
                "tokens_out": tokens.get("completion_tokens"),
                "request": raw.get("request"),
                "response": raw.get("response"),
                "session": sessions.get(raw.get("client_session_id"), raw.get("client_session_id")),
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


# The categories vllm_model records for an upstream refusal on length: the conversation outgrew the window, or the
# request's own output budget exceeds all of it.
_LENGTH_REFUSAL_CATEGORIES = frozenset({"context_length_exceeded", "invalid_output_token_parameter"})


def _is_length_refusal(call: dict[str, Any]) -> bool:
    """An upstream refusal on length, of either kind: the engine generated nothing, so the token checks have nothing
    to read. Only an overflow (`_is_context_overflow_rejection`) can be the model's own end."""
    return (
        call.get("upstream_attempted") is True
        and call.get("upstream_status_code") == 400
        and not _LENGTH_REFUSAL_CATEGORIES.isdisjoint({call.get("local_response_reason"), call.get("error_category")})
    )


def _is_context_overflow_rejection(call: dict[str, Any]) -> bool:
    """Exclude only explicit upstream context rejections, never inferred empty generations."""
    return (
        call.get("upstream_attempted") is True
        and call.get("upstream_status_code") == 400
        and (
            call.get("local_response_reason") == "context_length_exceeded"
            or call.get("error_category") == "context_length_exceeded"
        )
    )


def _is_failed_attempt(call: dict[str, Any]) -> bool:
    """A failed model call that returned no finished response, so it has no usage to account for.

    Covers HTTP error replies and calls cut by a cancellation or timeout. This does not mean
    nothing was generated: the server may have spent tokens on a cut call, and a cut stream may
    have delivered part of its output first. Gym cannot tell which from the capture (the first
    streamed event can arrive before any token), so these calls are skipped rather than guessed
    at; `run.stats.model_call_errors.usage_unknown` counts them. Whether the rollout recovered is
    `rollout_ended_on_failed_model_call`'s question.

    A failed call that did finish (it has a finish reason) keeps the token checks, as does a
    body capture could not parse, which is a capture defect rather than a failed call.
    """
    return (
        _is_failed(call) and call.get("finish_reason") is None and call.get("error_category") != "capture_parse_error"
    )


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


def _item_model_call_references(items: Any) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return usable model-call references from turns or invocations."""
    return tuple(
        (reference, raw_reference)
        for item in items or []
        if isinstance(item, dict)
        for raw_reference in item.get("model_calls") or []
        if isinstance(raw_reference, dict) and (reference := _call_ref_key(raw_reference)) is not None
    )


def _call_reference_signature(reference: dict[str, Any]) -> tuple[str, str, str, str]:
    model_ref = reference.get("model_ref")
    return (
        str(reference.get("model_call_id") or ""),
        str(model_ref.get("type") or "") if isinstance(model_ref, dict) else "",
        str(model_ref.get("name") or "") if isinstance(model_ref, dict) else "",
        str(reference.get("response_id") or ""),
    )


def _deduplicate_reference_items(
    reference_items: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for reference, raw_reference in reference_items:
        signature = _call_reference_signature(raw_reference)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append((reference, raw_reference))
    return tuple(unique)


def _bind_policy_call_views(
    trajectory: dict[str, Any], calls: list[dict[str, Any]]
) -> tuple[_CallBindings, _CallBindings]:
    """Bind turn-only and turn-or-invocation references with shared indexes and resolutions."""
    turn_reference_items = _deduplicate_reference_items(_item_model_call_references(trajectory.get("turns")))
    invocation_reference_items = _item_model_call_references(trajectory.get("invocations"))
    owned_reference_items = _deduplicate_reference_items((*turn_reference_items, *invocation_reference_items))

    calls_by_call_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls_by_response: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        if call.get("model_call_id"):
            calls_by_call_id[str(call["model_call_id"])].append(call)
        model_ref = call.get("model_ref")
        response_id = call.get("response_id")
        if isinstance(model_ref, dict) and response_id:
            calls_by_response[(str(model_ref.get("type")), str(model_ref.get("name")), str(response_id))].append(call)

    resolved: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for _, raw_reference in owned_reference_items:
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
        resolved[_call_reference_signature(raw_reference)] = matches

    call_positions = {id(call): position for position, call in enumerate(calls)}

    def build_bindings(reference_items: tuple[tuple[str, dict[str, Any]], ...]) -> _CallBindings:
        matched_calls: list[dict[str, Any]] = []
        matched_positions: set[int] = set()
        missing_references: list[str] = []
        duplicated_references: list[tuple[str, int]] = []
        for reference, raw_reference in reference_items:
            matches = resolved[_call_reference_signature(raw_reference)]
            if not matches:
                missing_references.append(reference)
            elif len(matches) > 1:
                duplicated_references.append((reference, len(matches)))
            else:
                call = matches[0]
                call_position = call_positions[id(call)]
                if call_position not in matched_positions:
                    matched_positions.add(call_position)
                    matched_calls.append(call)
        matched_calls.sort(key=lambda call: call_positions[id(call)])
        return _CallBindings(
            references=tuple(reference for reference, _ in reference_items),
            matched_calls=tuple(matched_calls),
            missing_references=tuple(missing_references),
            duplicated_references=tuple(duplicated_references),
            # What each reference resolved to, for the hollow-turn exemption. A missing reference resolves to (),
            # so its empty turn stays flagged.
            resolutions={
                _call_reference_signature(raw_reference): tuple(resolved[_call_reference_signature(raw_reference)])
                for _, raw_reference in reference_items
            },
        )

    return build_bindings(turn_reference_items), build_bindings(owned_reference_items)


def _replay_identity(call: dict[str, Any]) -> str | None:
    # Gym assigns model_call_id per invocation. Provider response IDs are only a
    # fallback: some backends reuse a placeholder response ID for distinct calls.
    return _call_identity(call)


def _call_locator(call: dict[str, Any], fallback: int) -> dict[str, int | str]:
    return {"call_id": str(call.get("model_call_id") or call.get("response_id") or fallback)}


def _response_has_usable_output(response: Any) -> bool:
    """Whether a saved response holds something the agent can act on: answer text, a refusal or a tool call, in the
    Responses, Chat Completions or Messages shape. Reasoning alone, a summary of it included, is not usable."""
    if not isinstance(response, dict):
        return False
    if _nonempty(response.get("output_text")):
        return True
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") == "reasoning":
            continue
        # Any other output item is a tool call of some kind.
        if item.get("type") != "message" or _nonempty(item.get("content")):
            return True
    content = response.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and (
            block.get("type") == "tool_use" or (block.get("type") == "text" and _nonempty(block.get("text")))
        ):
            return True
    for choice in response.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        if _nonempty(choice.get("text")) or _nonempty(message.get("content")) or message.get("tool_calls"):
            return True
    return False


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


def _agent_turn_hollow(
    trajectory: dict[str, Any], bindings: _CallBindings, subject: dict[str, int | str]
) -> list[Finding]:
    # An empty turn that ended on its own model call being rejected for a full context is the model filling its
    # context, when another check accounts for that overflow (`_turn_ended_on_context_overflow`). The exemption is not
    # applied when an incomplete capture leaves those checks unobserved. The calls are optional evidence here: without
    # captured calls, every empty turn is still flagged.
    incomplete = _trajectory_has_any_gap(trajectory, _INCOMPLETE_MODEL_CALL_GAPS)
    chains = _model_chains(_normalized_trajectory_calls(trajectory))
    return [
        Finding(
            check="agent_turn_hollow",
            subject=subject,
            locator=step.locator,
            detail={"reason": "agent turn has no message or tool calls"},
        )
        for step, turn in zip(_agent_steps(trajectory), trajectory.get("turns") or [], strict=True)
        if not step.has_message
        and not step.has_tool_calls
        and (incomplete or not _turn_ended_on_context_overflow(turn, bindings, chains))
    ]


def _turn_ended_on_context_overflow(
    turn: dict[str, Any], bindings: _CallBindings, chains: dict[Any, list[dict[str, Any]]]
) -> bool:
    """Whether a turn's own model calls end on a context-overflow rejection that another check accounts for.

    Each of the turn's references must resolve to exactly one call: a call that another turn's reference
    resolved says nothing about this turn. Of the turn's calls, the last in time decides, as in
    `_chain_ended_on_failure`, so an overflow followed by an empty retry is still hollow. When the calls
    cannot be ordered, every one must be an overflow.

    Each such overflow must be accounted for: by a success of the same model in the same conversation, or by
    `rollout_ended_on_failed_model_call` flagging that model's chain. An unordered chain that mixes in another
    conversation's success is flagged by neither, so its empty turn stays hollow.
    """
    calls = []
    for raw_reference in turn.get("model_calls") or []:
        if not isinstance(raw_reference, dict) or _call_ref_key(raw_reference) is None:
            continue
        matches = bindings.resolutions.get(_call_reference_signature(raw_reference), ())
        if len(matches) != 1:
            return False
        calls.append(matches[0])
    if not calls:
        return False
    ends = calls
    if all(call.get("started_at") is not None for call in calls):
        ends = sorted(calls, key=lambda call: (call.get("started_at") or 0.0, call.get("call_index") or 0))[-1:]
    for end in ends:
        if not _is_context_overflow_rejection(end):
            return False
        ref = end.get("model_ref")
        chain = chains.get((ref.get("type"), ref.get("name")) if isinstance(ref, dict) else None, [])
        answered = any(_is_successful(call) and _same_conversation(call, end) for call in chain)
        if not answered and not _chain_ended_on_failure(chain):
            return False
    return True


def _model_call_zero_completion_tokens(bindings: _CallBindings, subject: dict[str, int | str]) -> list[Finding]:
    return [
        Finding(
            check="model_call_zero_completion_tokens",
            subject=subject,
            locator=_call_locator(call, position),
            detail={"completion_tokens": 0},
        )
        for position, call in enumerate(bindings.matched_calls)
        if not _is_length_refusal(call) and not _is_failed_attempt(call) and call.get("tokens_out") == 0
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
        if not _is_length_refusal(call)
        and not _is_failed_attempt(call)
        and (call.get("tokens_in") is None or call.get("tokens_out") is None)
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
        finding_key = ("missing_captured_call", locator)
        if finding_key in seen:
            continue
        seen.add(finding_key)
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
        finding_key = ("duplicated_captured_call", locator)
        if finding_key in seen:
            continue
        seen.add(finding_key)
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


def _model_chains(calls: Sequence[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    """Group calls by the model they went to."""
    chains: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        ref = call.get("model_ref")
        key = (ref.get("type"), ref.get("name")) if isinstance(ref, dict) else None
        chains[key].append(call)
    return chains


def _conversation_head(call: dict[str, Any]) -> Any:
    """How the conversation a call continues opens: its instructions, system prompt, or first message.

    Calls of one conversation share it; another conversation on the same model server, such as a user simulator's,
    opens differently. A call whose request was not captured has none.
    """
    request = call.get("request")
    if not isinstance(request, dict):
        return None
    for key in ("instructions", "system"):
        if request.get(key):
            return request[key]
    items = request.get("messages", request.get("input"))
    if isinstance(items, list):
        return items[0] if items else None
    return items or None


def _same_conversation(call: dict[str, Any], other: dict[str, Any]) -> bool:
    """Whether two calls continue one conversation: their requests open the same way, and no captured session
    tells them apart. Two agents can share instructions, so a session only ever separates calls."""
    head = _conversation_head(call)
    sessions = {call.get("session"), other.get("session")} - {None}
    return head is not None and head == _conversation_head(other) and len(sessions) <= 1


def _chain_ended_on_failure(chain: Sequence[dict[str, Any]]) -> bool:
    """Whether this model's own last call, in time, failed.

    Per model rather than across every captured call: a judge or auxiliary
    request landing after a failed policy call would otherwise read as the
    rollout recovering, and hide the failure.

    Without timestamps a multi-call chain cannot be ordered -- `call_index` is
    the position in the merged trajectory list, so a producer-supplied retry
    can precede the attempt it retried. Say nothing rather than call a
    recovered rollout unhealthy.

    A last call the upstream rejected because the context was full is the
    model's own end, scored like any other answer, once the model has answered
    earlier in the same conversation (`_same_conversation`). A success in another
    conversation on the same server, such as a user simulator's or another
    session's, says nothing of whether this request fit. Before any answer the request never fit the deployment (a
    prompt or output budget too large for it), which is a failure like any
    other. An unordered chain of only failures has no answer and ended on one of
    them, so it is flagged.
    """
    if not chain:
        return False
    failed = [call for call in chain if _is_failed(call)]
    if not failed:
        return False
    if all(call.get("started_at") is not None for call in chain):
        last = sorted(chain, key=lambda call: (call.get("started_at") or 0.0, call.get("call_index") or 0))[-1]
        answered = any(_is_successful(call) and _same_conversation(call, last) for call in chain)
        return _is_failed(last) and not (answered and _is_context_overflow_rejection(last))
    return len(failed) == len(chain)


def _ended_on_failed_call(calls: Sequence[dict[str, Any]]) -> bool:
    """Whether any model's own last call failed."""
    return any(_chain_ended_on_failure(chain) for chain in _model_chains(calls).values())


def _rollout_ended_on_failed_model_call(trajectory: dict[str, Any], subject: dict[str, int | str]) -> list[Finding]:
    """Flag a model whose own last call in the rollout failed.

    The bound-call checks cannot see this: binding resolves a reference by
    `(model_ref, response_id)` or `model_call_id`, and a call that failed came
    back with none of them, so it is absent from `matched_calls` no matter what
    the producer claims. Reading the captured calls directly also covers the
    agents that publish no trajectory at all.

    Judged per model and in time order -- see `_chain_ended_on_failure`. Only
    the chain's last call counts, so a failure the client retried successfully
    stays healthy; the signal is that the rollout ENDED on a failure, which is
    what makes its reward indistinguishable from a genuine zero. A last call
    the upstream rejected as a context-length overflow is not flagged once the
    model has answered earlier in the same conversation: that zero is the
    model's own result.
    """
    findings = []
    for key, chain in _model_chains(_normalized_trajectory_calls(trajectory)).items():
        if not _chain_ended_on_failure(chain):
            continue
        ordered = sorted(chain, key=lambda call: (call.get("started_at") or 0.0, call.get("call_index") or 0))
        last = ordered[-1]
        findings.append(
            Finding(
                check="rollout_ended_on_failed_model_call",
                subject=subject,
                locator=_call_locator(last, last.get("call_index")),
                detail={
                    "model_ref": "/".join(str(part) for part in key) if key else None,
                    "status": last.get("status_code"),
                    "error_category": last.get("error_category"),
                    "observed_calls": len(chain),
                    "successful_calls": sum(1 for call in chain if _is_successful(call)),
                },
            )
        )
    return findings


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
        if not _is_length_refusal(call)
        and call.get("finish_reason") in _LENGTH_LIMIT_FINISH_REASONS
        and not _response_has_usable_output(call.get("response"))
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
    "rollout_ended_on_failed_model_call": lambda record, trajectory, bindings, subject: (
        _rollout_ended_on_failed_model_call(trajectory, subject)
    ),
    "agent_turn_hollow": lambda record, trajectory, bindings, subject: _agent_turn_hollow(
        trajectory, bindings, subject
    ),
    "model_call_zero_completion_tokens": lambda record, trajectory, bindings, subject: (
        _model_call_zero_completion_tokens(bindings, subject)
    ),
    "model_call_missing_token_counts": lambda record, trajectory, bindings, subject: _model_call_missing_token_counts(
        bindings, subject
    ),
    "trajectory_capture_mismatch": lambda record, trajectory, bindings, subject: _trajectory_capture_mismatch(
        trajectory, bindings, subject
    ),
    "rollout_token_count_mismatch": lambda record, trajectory, bindings, subject: _rollout_token_count_mismatch(
        record, bindings, subject
    ),
    "model_call_runaway_generation": lambda record, trajectory, bindings, subject: _model_call_runaway_generation(
        bindings, subject
    ),
}
