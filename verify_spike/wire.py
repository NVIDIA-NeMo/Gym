"""Sidecar wire-capture loading and policy-call filtering.

Ports the evalpipeline Gym backend's capture_record path (backends/
gym.py): the per-rollout ``model_calls/<rollout_id>.capture.jsonl``
sidecars (NeMo Gym PR #1715) are the ground-truth record of every model
call that went over the wire. Loading selects the attempt file the
rollout row reconciles against, discriminates policy from non-policy
(user-simulator) calls by model_ref grouping, and returns a WireRecord;
success filtering, dedup, binding, and counting live in checks.py.

SENSITIVITY: ``model_ref.name`` values may identify unreleased models.
They are compared by equality only and MUST never appear in exception
messages or reports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from verify_spike.gym_records import _int_or_none, _is_int
from verify_spike.models import Trajectory, WireCall, WireRecord, read_jsonl

MODEL_CALLS_DIRNAME = "model_calls"
_CAPTURE_SUFFIX = ".capture.jsonl"


def find_model_calls_dir(artifacts_dir: Path) -> Path | None:
    """The sidecar directory for an artifacts root.

    Native Gym runs keep ``model_calls/`` next to the rollouts file;
    some local layouts keep the rollouts file in a ``results/`` subdir
    with ``model_calls/`` one level up.
    """
    for candidate in (
        artifacts_dir / MODEL_CALLS_DIRNAME,
        artifacts_dir.parent / MODEL_CALLS_DIRNAME,
    ):
        if candidate.is_dir():
            return candidate
    return None


def load_wire_record(
    capture_dir: Path | None,
    trajectory: Trajectory,
) -> WireRecord:
    """Selected ground-truth wire calls for one rollout.

    A row that names its own attempt gets that attempt's file, so a
    per-attempt rollout row is never compared to a later retry's
    capture; otherwise the last (highest) attempt is the successful
    rerun. Policy discrimination runs over the sidecar records BEFORE
    anything counts them.
    """
    identity = trajectory.identity
    if identity is None:
        raise ValueError("load_wire_record requires trajectory.identity")
    if capture_dir is None or not capture_dir.is_dir():
        return WireRecord(found=False)
    rollout_id = f"{identity.task}-{identity.trial}"
    attempts = _attempt_files(capture_dir, rollout_id)
    if not attempts:
        return WireRecord(found=False)
    selected = _select_attempt(attempts, identity.attempt)
    dropped = [path for _, path in attempts if path != selected]
    records = [record for _, record in read_jsonl(selected)]
    policy_names, disagreement = _policy_names(records, trajectory)
    policy_records = [
        record
        for record in records
        if _record_model_name(record) in policy_names
    ]
    # The capture-integrity check (in-row aggregate == sum of in-row
    # calls) is not computed by either producer yet (reserved).
    return WireRecord(
        found=True,
        calls=tuple(_wire_call(record) for record in policy_records),
        retry_sessions=len(dropped),
        dropped_calls=sum(_line_count(path) for path in dropped),
        non_policy_calls=len(records) - len(policy_records),
        discriminator_disagreement=disagreement,
        capture_integrity_mismatch=False,
    )


def _attempt_files(
    capture_dir: Path,
    rollout_id: str,
) -> list[tuple[int, Path]]:
    attempts = []
    for path in sorted(capture_dir.glob(f"{rollout_id}*{_CAPTURE_SUFFIX}")):
        attempt = _attempt_of(path.name, rollout_id)
        if attempt is not None:
            attempts.append((attempt, path))
    attempts.sort(key=lambda pair: pair[0])
    return attempts


def _select_attempt(
    attempts: list[tuple[int, Path]],
    identity_attempt: Any,
) -> Path:
    """The capture file this rollout row reconciles against."""
    if _is_int(identity_attempt) and identity_attempt > 0:
        exact = [
            path for attempt, path in attempts if attempt == identity_attempt
        ]
        if exact:
            return exact[0]
    return attempts[-1][1]


def _attempt_of(filename: str, rollout_id: str) -> int | None:
    if filename == f"{rollout_id}{_CAPTURE_SUFFIX}":
        return 0
    prefix = f"{rollout_id}-a"
    if filename.startswith(prefix) and filename.endswith(_CAPTURE_SUFFIX):
        digits = filename[len(prefix) : -len(_CAPTURE_SUFFIX)]
        if digits.isdigit():
            return int(digits)
    return None


def _line_count(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def _policy_names(
    records: list[dict[str, Any]],
    trajectory: Trajectory,
) -> tuple[set, bool]:
    """Policy model_ref.name group(s) plus the disagreement flag.

    Names are compared by equality only and never printed. A single
    name group is all policy; with several, the seed-system-prompt
    discriminator is primary, call_id threading the cross-check, and
    the largest group the flagged last resort.
    """
    groups: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_record_model_name(record), []).append(record)
    if len(groups) <= 1:
        return set(groups), False
    seed_candidates = _seed_system_candidates(groups, trajectory)
    step_ids = {
        tool_call.tool_call_id
        for step in trajectory.steps
        for tool_call in step.tool_calls
    }
    thread_candidates = {
        name
        for name, group in groups.items()
        if _group_record_tool_ids(group) & step_ids
    }
    return _decide_policy(groups, seed_candidates, thread_candidates)


def _decide_policy(
    groups: dict[Any, list[dict[str, Any]]],
    seed_candidates: set,
    thread_candidates: set,
) -> tuple[set, bool]:
    seed_decided = len(seed_candidates) == 1
    thread_decided = len(thread_candidates) == 1
    if seed_decided and thread_decided:
        # Both discriminators decided: prefer the seed-system-prompt
        # equality (the more reliable witness on real data) and flag
        # when threading disagrees.
        return seed_candidates, seed_candidates != thread_candidates
    if thread_decided:
        return thread_candidates, False
    if seed_decided:
        return seed_candidates, False
    largest = max(groups, key=lambda name: len(groups[name]))
    return {largest}, True


def _seed_system_candidates(
    groups: dict[Any, list[dict[str, Any]]],
    trajectory: Trajectory,
) -> set:
    seed_system = _seed_system_message(trajectory)
    if seed_system is None:
        return set()
    return {
        name
        for name, group in groups.items()
        if _first_message_content(group[0]) == seed_system
    }


def _seed_system_message(trajectory: Trajectory) -> Any:
    matches = [
        step.message
        for step in trajectory.steps
        if step.source == "system" and (step.extra or {}).get("from_seed")
    ]
    return matches[0] if matches else None


def _first_message_content(record: dict[str, Any]) -> Any:
    request = record.get("request")
    if not isinstance(request, dict):
        return None
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0]
    return first.get("content") if isinstance(first, dict) else None


def _record_model_name(record: dict[str, Any]) -> Any:
    model_ref = record.get("model_ref")
    return model_ref.get("name") if isinstance(model_ref, dict) else None


def _group_record_tool_ids(group: list[dict[str, Any]]) -> set:
    return {
        call_id
        for record in group
        for call_id in _record_tool_call_ids(record)
    }


def _record_tool_call_ids(record: dict[str, Any]) -> tuple[str, ...]:
    response = record.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    ids: list[str] = []
    for choice in choices if isinstance(choices, list) else []:
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = (
            message.get("tool_calls") if isinstance(message, dict) else None
        )
        for item in calls if isinstance(calls, list) else []:
            call_id = item.get("id") if isinstance(item, dict) else None
            if isinstance(call_id, str):
                ids.append(call_id)
    return tuple(ids)


def _wire_call(record: dict[str, Any]) -> WireCall:
    call_id = record.get("model_call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("sidecar capture record is missing model_call_id")
    request = record.get("request")
    usage = _record_usage(record)
    return WireCall(
        call_id=call_id,
        status_code=_int_or_none(record.get("status_code")),
        request_hash=_request_hash(request) if request is not None else None,
        prompt_tokens=_usage_tokens(usage, "prompt_tokens", "tokens_in"),
        completion_tokens=_usage_tokens(
            usage, "completion_tokens", "tokens_out"
        ),
        tool_call_ids=_record_tool_call_ids(record),
    )


def _record_usage(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    return usage if isinstance(usage, dict) else {}


def _usage_tokens(
    usage: dict[str, Any],
    primary: str,
    fallback: str,
) -> int | None:
    value = usage.get(primary)
    if _is_int(value):
        return value
    value = usage.get(fallback)
    return value if _is_int(value) else None


def _request_hash(request: Any) -> str:
    payload = json.dumps(request, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
