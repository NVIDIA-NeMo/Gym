# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic post-run rollout quality verification workflow.

Checks operate only on persisted rollout records and their canonical
``ng_trajectory`` evidence. They return evidence; this module derives verdicts
and writes reports.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ValidationError

from nemo_gym.artifacts.run_files import (
    QUALITY_SUMMARY_FILENAME,
    QUALITY_SUMMARY_SOURCES_KEY,
    ROLLOUT_VERDICTS_FILENAME,
)
from nemo_gym.config.config_types import ConfigError
from nemo_gym.health.checks import (
    _INCOMPLETE_MODEL_CALL_GAPS,
    _LENGTH_LIMIT_FINISH_REASONS,
    _ROLLOUT_CHECKS,
    _ROLLOUT_SPECS,
    _TASK_SPECS,
    CHECK_REGISTRY,
    _bind_policy_call_views,
    _canonical_trajectory,
    _ended_on_failed_call,
    _is_failed,
    _is_failed_attempt,
    _is_length_refusal,
    _is_successful,
    _normalized_trajectory_calls,
    _replay_identity,
    _response_has_usable_output,
    _subject,
    _token_count,
    _trajectory_has_any_gap,
    _trajectory_has_gap,
    _trajectory_reference_contradictions,
    _transcript_tokens,
    normalize_ignored_checks,
)
from nemo_gym.health.timeouts import read_rollout_timing, summarize_timeouts
from nemo_gym.health.types import (
    CALL_BINDING_INPUTS,
    ROLLOUT_ID_KEY,
    ROLLOUT_INDEX_KEY,
    TASK_INDEX_KEY,
    CheckInput,
    CheckScope,
    CheckSpec,
    CheckSubject,
    Finding,
    HealthCheckResult,
    LimitsSource,
    RolloutDigest,
    RunHealthLimits,
    Verdict,
    _LineSlice,
    _TaskRepeat,
    _WorkerInput,
)
from nemo_gym.rollout_records import RUN_ID_KEY, resolve_rollout_path
from nemo_gym.rollout_recovery import manifest_path_for
from nemo_gym.rollout_store import RolloutStore, raw_outcomes_are_selected
from nemo_gym.utils.path_utils import (
    aggregation_sources_path_for,
    failures_path_for,
    materialized_path_for,
    resolved_config_path_for,
)


logger = logging.getLogger(__name__)


class JournalHealthUnavailable(ConfigError):
    """This history contains superseded outcomes and requires an exported projection."""


_PROCESS_POOL_CHUNKS_PER_WORKER = 4
_PROCESS_POOL_MAX_CHUNKSIZE = 128


__all__ = [
    "CHECK_REGISTRY",
    "CheckInput",
    "CheckScope",
    "CheckSpec",
    "CheckSubject",
    "Finding",
    "HealthCheckResult",
    "RolloutDigest",
    "Verdict",
    "evaluate_record",
    "format_health_report",
    "health_check_run_dir",
    "normalize_ignored_checks",
    "run_health_checks",
]


def _process_pool_chunksize(item_count: int, workers: int) -> int:
    """Keep several schedulable chunks per worker without unbounded IPC batches."""
    target_chunks = workers * _PROCESS_POOL_CHUNKS_PER_WORKER
    adaptive_size = (item_count + target_chunks - 1) // target_chunks
    return max(1, min(adaptive_size, _PROCESS_POOL_MAX_CHUNKSIZE))


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


def _worker(payload: _WorkerInput) -> RolloutDigest:
    record, parse_error = _read_record(payload.line)
    return evaluate_record(record, payload.ignored_checks, line=payload.line, parse_error=parse_error)


def evaluate_record(
    record: dict[str, Any],
    ignored_checks: frozenset[str],
    *,
    line: _LineSlice | None = None,
    parse_error: str | None = None,
) -> RolloutDigest:
    """Evaluate every rollout-level check on one rollout record.

    The one code path behind both the post-run report (which reads ``record`` from a JSONL ``line``)
    and live emission (which passes the in-memory result, so ``line`` is ``None``).
    """
    unreadable_identity = (
        f"__unreadable_record__:input-{line.source_index}:line-{line.line_number}" if line is not None else None
    )
    task_index = record.get(
        TASK_INDEX_KEY, unreadable_identity if parse_error else (line.ordinal if line is not None else 0)
    )
    rollout_index = record.get(ROLLOUT_INDEX_KEY, 0)
    trajectory = record.get("ng_trajectory")
    trajectory_rollout_id = trajectory.get("rollout_id") if isinstance(trajectory, dict) else None
    rollout_id = str(record.get(ROLLOUT_ID_KEY) or trajectory_rollout_id or f"{task_index}-{rollout_index}")
    subject = _subject(task_index, rollout_index)

    trajectory, trajectory_error = (None, None) if parse_error else _canonical_trajectory(record)
    trajectory_observed = trajectory is not None and not _trajectory_has_gap(
        trajectory, "trajectory_projection_failed"
    )
    turns_observed = trajectory_observed and not _trajectory_has_gap(trajectory, "turns_unavailable")
    turn_call_scope_complete = trajectory_observed and not _trajectory_has_gap(
        trajectory, "turn_model_call_scope_incomplete"
    )
    model_calls_observed = trajectory_observed and not _trajectory_has_any_gap(trajectory, _INCOMPLETE_MODEL_CALL_GAPS)
    calls = _normalized_trajectory_calls(trajectory) if trajectory_observed else []
    turn_bindings, owned_bindings = _bind_policy_call_views(trajectory if trajectory_observed else {}, calls)
    findings: list[Finding] = []
    unobserved: list[str] = []

    for spec in _ROLLOUT_SPECS:
        if spec.id in ignored_checks:
            continue
        if parse_error:
            if spec.id == "record_unreadable":
                findings.append(
                    Finding(
                        check="record_unreadable",
                        subject=subject,
                        locator=({"source_file": line.path, "line": line.line_number} if line is not None else None),
                        detail={"reason": "rollout record is unreadable", "error": parse_error},
                    )
                )
            else:
                unobserved.append(spec.id)
            continue
        if trajectory_error and spec.id == "record_unreadable":
            findings.append(
                Finding(
                    check="record_unreadable",
                    subject=subject,
                    detail={"reason": "canonical trajectory is unreadable", "error": trajectory_error},
                )
            )
            continue
        if CheckInput.TRAJECTORY in spec.reads and not trajectory_observed:
            unobserved.append(spec.id)
            continue
        if CheckInput.AGENT_TURNS in spec.reads and not turns_observed:
            unobserved.append(spec.id)
            continue
        if CheckInput.OBSERVED_MODEL_CALLS in spec.reads and not (model_calls_observed and calls):
            unobserved.append(spec.id)
            continue
        binding_input = next(iter(spec.reads & CALL_BINDING_INPUTS), None)
        bindings = owned_bindings if binding_input == CheckInput.OWNED_MODEL_CALLS else turn_bindings
        if binding_input is not None:
            if not model_calls_observed:
                unobserved.append(spec.id)
                continue
            has_required_bindings = (
                bindings.observed or bool(_trajectory_reference_contradictions(trajectory))
                if spec.id == "trajectory_capture_mismatch"
                else bindings.matched_calls
            )
            if not has_required_bindings:
                unobserved.append(spec.id)
                continue
        if spec.id == "rollout_token_count_mismatch" and (
            not turn_call_scope_complete
            or not bindings.complete
            or not bindings.matched_calls
            or not _transcript_tokens(record)[2]
            or any(
                not _is_failed_attempt(call) and (call.get("tokens_in") is None or call.get("tokens_out") is None)
                for call in bindings.matched_calls
            )
        ):
            unobserved.append(spec.id)
            continue
        if spec.id == "model_call_runaway_generation" and any(
            not _is_length_refusal(call)
            and call.get("finish_reason") in _LENGTH_LIMIT_FINISH_REASONS
            and call.get("response") is None
            for call in bindings.matched_calls
        ):
            unobserved.append(spec.id)
            continue
        try:
            findings.extend(_ROLLOUT_CHECKS[spec.id](record, trajectory, bindings, subject))
        except Exception as exc:
            unobserved.append(spec.id)
            if "check_execution_error" not in ignored_checks:
                findings.append(
                    Finding(
                        check="check_execution_error",
                        subject=subject,
                        detail={
                            "reason": "check raised an unexpected exception",
                            "failed_check": spec.id,
                            "error": type(exc).__name__,
                        },
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

    return RolloutDigest(
        task_index=task_index,
        rollout_index=rollout_index,
        stage_index=record.get("stage_index"),
        rollout_id=rollout_id,
        verdict=verdict,
        findings=findings,
        unobserved=unobserved,
        capture_observed=bool(calls),
        policy_calls_observed=turn_call_scope_complete and turn_bindings.complete,
        model_calls=len(calls),
        successful_model_calls=sum(_is_successful(call) for call in turn_bindings.matched_calls),
        responses=_response_population(
            turn_call_scope_complete
            and model_calls_observed
            and not _trajectory_has_any_gap(
                trajectory or {}, frozenset({"model_call_capture_pending", "model_call_capture_no_records"})
            ),
            turn_bindings,
        ),
        model_call_errors=len(failed),
        upstream_recovery_observations=_observed_upstream_recovery(calls),
        model_call_errors_usage_unknown=sum(
            _is_failed_attempt(call) and (call.get("tokens_in") is None or call.get("tokens_out") is None)
            for call in failed
        ),
        errors_by_status=dict(errors_by_status),
        ended_on_error=model_calls_observed and _ended_on_failed_call(calls),
        context_window_exceeded=_context_window_exceeded(record),
        verification_skipped=(
            record.get("verification_skipped") if isinstance(record.get("verification_skipped"), bool) else None
        ),
        duplicated_calls=duplicated,
        transcript_prompt_tokens=transcript_prompt,
        transcript_completion_tokens=transcript_completion,
        capture_prompt_tokens=sum(_token_count(call, "tokens_in") for call in calls),
        capture_completion_tokens=sum(_token_count(call, "tokens_out") for call in calls),
        timing=read_rollout_timing(record),
    )


def _observed_upstream_recovery(calls: list[dict[str, Any]]) -> dict[str, int] | None:
    """Count only observed failed attempts and explicitly completed operations; never certify completeness."""
    attempts = {}
    outcomes = {}
    for call in calls:
        report = call.get("failure_report")
        operations = call.get("model_operations")
        if isinstance(report, dict):
            for event_id, event in (report.get("events") or {}).items():
                if (
                    isinstance(event, dict)
                    and event.get("boundary") == "model"
                    and event.get("operation") == "model_request"
                    and event.get("operation_id")
                    and event.get("attempt_id")
                ):
                    attempts[event_id] = event["operation_id"]
        for observation in operations if isinstance(operations, list) else []:
            if not isinstance(observation, dict) or not observation.get("operation_id"):
                continue
            if observation.get("outcome") in {"completed", "failed"}:
                outcome = observation["outcome"]
                # A 2xx header alone cannot recover a call whose body subsequently failed.
                if outcome == "completed" and (
                    type(call.get("status_code")) is not int
                    or not 200 <= call["status_code"] < 400
                    or _is_failed(call)
                ):
                    continue
                outcomes.setdefault(observation["operation_id"], set()).add(outcome)
    if not attempts:
        return None  # No failed-attempt observation is not proof that no earlier failure occurred.
    failed_operations = set(attempts.values())
    completed = {operation for operation, values in outcomes.items() if values == {"completed"}}
    ended = {operation for operation, values in outcomes.items() if len(values) == 1}
    return {
        "observed_attempt_errors": len(attempts),
        "observed_failed_operations": len(failed_operations),
        "recovered_attempt_errors": sum(operation in completed for operation in attempts.values()),
        "recovered_operations": len(failed_operations & completed),
        "operations_without_terminal_observation": len(failed_operations - ended),
    }


def _index_jsonl(paths: Sequence[Path], *, source_indices: Mapping[str, int] | None = None) -> list[_LineSlice]:
    slices: list[_LineSlice] = []
    ordinal = 0
    for source_index, path in enumerate(paths):
        with path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                if source_indices is not None:
                    try:
                        record = orjson.loads(line)
                        source_index = source_indices[record[RUN_ID_KEY]]
                    except (ValueError, KeyError, TypeError) as error:
                        raise ConfigError(
                            f"Selected projection {path}:{line_number} must identify one of its source runs."
                        ) from error
                slices.append(
                    _LineSlice(
                        path=str(path),
                        offset=offset,
                        length=len(line),
                        ordinal=ordinal,
                        source_index=source_index,
                        line_number=line_number,
                    )
                )
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


def _mark_duplicate_identities(
    digests: list[RolloutDigest], lines: list[_LineSlice], ignored_checks: frozenset[str]
) -> None:
    """Flag physical records of one input that claim the same logical rollout identity.

    Each input numbers its rollouts on its own, as the run accounting counts them: runs merged from different
    environment servers can reuse task indices.
    """
    if "rollout_duplicate_identity" in ignored_checks:
        return

    grouped: dict[tuple[int, int | str | None, int | str, int | str], list[RolloutDigest]] = defaultdict(list)
    for digest, line in zip(digests, lines, strict=True):
        grouped[(line.source_index, digest.stage_index, digest.task_index, digest.rollout_index)].append(digest)

    for copies in grouped.values():
        if len(copies) < 2:
            continue
        for digest in copies:
            digest.findings.append(
                Finding(
                    check="rollout_duplicate_identity",
                    subject=_subject(digest.task_index, digest.rollout_index),
                    detail={"duplicate_count": len(copies)},
                )
            )
            digest.verdict = "unhealthy"


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
                        Finding(
                            check="task_consistently_unhealthy",
                            subject=subject,
                            detail={"computable_repeats": len(computable)},
                        )
                    )
            else:
                coverage["task_consistently_unhealthy"]["unobserved"] += 1

        if "task_no_successful_model_calls" in ignored_checks:
            coverage["task_no_successful_model_calls"]["ignored"] += 1
        else:
            if repeats and all(repeat.policy_calls_observed for repeat in repeats):
                coverage["task_no_successful_model_calls"]["evaluated"] += 1
                if not any(repeat.successful_model_calls for repeat in repeats):
                    findings[task_index].append(
                        Finding(
                            check="task_no_successful_model_calls",
                            subject=subject,
                            detail={"repeats": len(repeats)},
                        )
                    )
            else:
                coverage["task_no_successful_model_calls"]["unobserved"] += 1
    return findings, coverage


def _response_population(scope_complete: bool, bindings: Any) -> dict[str, list[str]] | None:
    """The call ids of the rollout's verified successful raw model responses selected for evaluation, and of the
    faulted ones among them. IDs only: no response text is kept.

    Selected means bound to an agent turn of a complete call scope. A bound call that failed is a trial matter and
    is left out. Every other bound call must carry an explicit 2xx/3xx status, a raw response and a call id, or the
    population is unknown. A fault is judged from the raw output itself: a length-limited finish that is not a
    length refusal (truncated), or an output with nothing usable (`_response_has_usable_output`: empty, reasoning
    only or malformed). Token counts alone are never a fault.
    """
    if not scope_complete or not bindings.complete:
        return None
    selected: list[str] = []
    faulted: list[str] = []
    for call in bindings.matched_calls:
        status = call.get("status_code")
        # The SSE capture marks a stream that stopped on its output budget as failed; the model still answered.
        length_incomplete = (
            type(status) is int
            and 200 <= status < 400
            and isinstance(call.get("response"), dict)
            and call.get("response_status") == "incomplete"
            and call.get("error_category") == "incomplete"
            and call.get("finish_reason") in _LENGTH_LIMIT_FINISH_REASONS
            and not _is_length_refusal(call)
        )
        if _is_failed(call) and not length_incomplete:
            continue
        identity = str(call.get("model_call_id") or call.get("response_id") or "")
        if type(status) is not int or not 200 <= status < 400 or call.get("response") is None or not identity:
            return None
        selected.append(identity)
        truncated = call.get("finish_reason") in _LENGTH_LIMIT_FINISH_REASONS and not _is_length_refusal(call)
        if truncated or not _response_has_usable_output(call["response"]):
            faulted.append(identity)
    return {"selected": sorted(selected), "faulted": sorted(faulted)}


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
    selected_ids = Counter(i for digest in digests if digest.responses for i in digest.responses["selected"])
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
                "observed_upstream_recovery": (
                    {
                        key: sum((d.upstream_recovery_observations or {}).get(key, 0) for d in digests)
                        for key in (
                            "observed_attempt_errors",
                            "observed_failed_operations",
                            "recovered_attempt_errors",
                            "recovered_operations",
                            "operations_without_terminal_observation",
                        )
                    }
                    if any(d.upstream_recovery_observations is not None for d in digests)
                    else None
                ),
                "model_call_errors": {
                    "total": sum(digest.model_call_errors for digest in digests),
                    "by_status": dict(sorted(error_statuses.items())),
                    "rollouts_affected": sum(bool(digest.model_call_errors) for digest in digests),
                    "ended_on_error": sum(digest.ended_on_error for digest in digests),
                    # Rows by their own step_usage: a sub-step recorded context_window_exceeded, none, or no step_usage.
                    "context_window_exceeded": {
                        "true": sum(digest.context_window_exceeded is True for digest in digests),
                        "false": sum(digest.context_window_exceeded is False for digest in digests),
                        "absent": sum(digest.context_window_exceeded is None for digest in digests),
                    },
                    # Rows by their own verification_skipped: written True, written False, or not written.
                    "verification_skipped": {
                        "true": sum(digest.verification_skipped is True for digest in digests),
                        "false": sum(digest.verification_skipped is False for digest in digests),
                        "absent": sum(digest.verification_skipped is None for digest in digests),
                    },
                    # Failed calls with no finished response and no usage: the token checks skip them,
                    # and any tokens the server spent on them are counted nowhere else.
                    "usage_unknown": sum(digest.model_call_errors_usage_unknown for digest in digests),
                },
                # Counted by call id across the whole run, so a response selected twice is a duplicate, never two.
                "responses": {
                    "rollouts": len(digests),
                    "unknown": sum(digest.responses is None for digest in digests),
                    "selected": len(selected_ids),
                    "faulted": len({i for d in digests if d.responses for i in d.responses["faulted"]}),
                    "duplicated": sum(count > 1 for count in selected_ids.values()),
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


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file's objects; an inventory can be hundreds of MB."""
    with path.open("rb") as handle:
        for line in handle:
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue  # Blank lines, and the truncated last line a killed writer can leave.
            if isinstance(row, dict):
                yield row


@dataclass(frozen=True)
class _Failures:
    """The rollouts that failed and never landed, and the rollouts the run owed."""

    # Each such rollout's last failed attempt, by (input index, task index, rollout index).
    rows: dict[tuple[Any, ...], dict[str, Any]]
    # None when an input has no `_materialized_inputs.jsonl` inventory.
    owed: set[tuple[Any, ...]] | None
    any_sidecar: bool


def _read_failures(landed: set[tuple[Any, ...]], paths: Sequence[Path]) -> _Failures:
    """Read every input's failures sidecar and inventory.

    The last attempt of a rollout stands, and a rollout that later landed is not failed. With an inventory, only
    owed rollouts count: an unreadable record or a sidecar row outside the inventory is not one.
    """
    # Lazy: rollout_collection imports this module.
    from nemo_gym.rollout_collection import is_non_superseding_rollout_row, is_omitted_rollout_row

    failed: dict[tuple[Any, ...], dict[str, Any]] = {}
    any_sidecar = False
    owed: set[tuple[Any, ...]] | None = set()
    for source_index, path in enumerate(paths):
        if failures_path_for(path).is_file():
            any_sidecar = True
            for row in _jsonl_rows(failures_path_for(path)):
                # Main's rule: a row the dispatch budget never started is no attempt and selects nothing; nor does
                # the evidence of a failed /run with routing off.
                if is_non_superseding_rollout_row(row):
                    continue
                failed[(source_index, row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY))] = row
        if not materialized_path_for(path).is_file():
            owed = None
        elif owed is not None:
            owed.update(
                (source_index, row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY))
                for row in _jsonl_rows(materialized_path_for(path))
            )
    # Main's rule: a rollout whose latest attempt recorded no result is missing, not failed.
    failed = {key: row for key, row in failed.items() if not is_omitted_rollout_row(row)}
    for key in landed:
        failed.pop(key, None)
    if owed is not None:
        failed = {key: row for key, row in failed.items() if key in owed}
    return _Failures(rows=failed, owed=owed, any_sidecar=any_sidecar)


def _landed_in(landed: set[tuple[Any, ...]], source_index: int) -> set[tuple[Any, ...]]:
    """``landed`` keys of one input, keyed as ``_read_failures`` keys that input read alone."""
    return {(0, *key[1:]) for key in landed if key[0] == source_index}


def _last_failed_attempts(landed: set[tuple[Any, ...]], paths: Sequence[Path]) -> list[dict[str, Any]]:
    """The selected failed attempt of every rollout that never landed.

    A run with a manifest selects it from its run history, as its accounting does; a legacy output's sidecar row
    that stands last is it.
    """
    rows: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths):
        if manifest_path_for(path).exists():
            rows += RolloutStore.read(path).selected("failure")
        else:
            rows += _read_failures(_landed_in(landed, source_index), [path]).rows.values()
    return rows


def _rollout_accounting(
    landed: set[tuple[Any, ...]],
    paths: Sequence[Path],
    *,
    limits: RunHealthLimits,
    limits_source: LimitsSource | Mapping[str, LimitsSource],
    duplicated: int,
    run_coverage: Sequence[Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Project managed runs from selected history; preserve raw accounting for legacy runs.

    Managed ``missing`` means reserved attempts with no outcome. Never-dispatched
    work and intentional omissions have their own categories. Final failures include
    exhausted budgets as well as terminal outcomes. A stopped collector supplies its
    in-memory coverage so this path never reloads large rollout records on SIGTERM.
    In a legacy sidecar, a ``record_omitted_rollouts`` row records no result: an interrupted attempt is the
    rollout's latest execution and leaves it missing, and a row never started is skipped.
    """
    # Lazy: rollout_collection imports this module, and health workers need none of its dependencies.
    from nemo_gym.artifacts.rollout_rows import NG_FAILURE_CLASS_KEY
    from nemo_gym.rollout_collection import is_terminal_failure

    expected = missing = not_started = omitted = final = not_final = 0
    by_class: Counter[str] = Counter()
    inventory_known = True
    any_sidecar = False
    for source_index, path in enumerate(paths):
        coverage = run_coverage[source_index] if run_coverage is not None else None
        if coverage is None and manifest_path_for(path).exists():
            coverage = RolloutStore.read(path).coverage()
        if coverage is not None:
            expected += coverage["expected"]
            missing += coverage["missing"]
            not_started += coverage["not_started"]
            omitted += coverage["intentionally_omitted"]
            final += coverage["failed_final"]
            not_final += coverage["failed_not_final"]
            by_class.update(coverage["failed_by_class"])
            any_sidecar = True
            continue

        # Legacy outputs have no manifest from which to recover attempt selection.
        landed_here = _landed_in(landed, source_index)
        legacy = _read_failures(landed_here, [path])
        any_sidecar = any_sidecar or legacy.any_sidecar
        if legacy.owed is None:
            inventory_known = False
        else:
            expected += len(legacy.owed)
            missing += len(legacy.owed - landed_here - legacy.rows.keys())
        final_here = sum(is_terminal_failure(row) for row in legacy.rows.values())
        final += final_here
        not_final += len(legacy.rows) - final_here
        by_class.update(row.get(NG_FAILURE_CLASS_KEY) or "unknown" for row in legacy.rows.values())

    reasons = []
    if inventory_known:
        if expected == 0:
            reasons.append("no rollouts expected")
        else:
            if missing or not_started:
                reasons.append(f"missing {missing + not_started} of {expected}")
            if duplicated:
                reasons.append(f"{duplicated} records share a rollout identity")
    return {
        "expected": expected if inventory_known else None,
        "failed": {
            "total": final + not_final,
            "final": final,
            "not_final": not_final,
            "by_class": dict(sorted(by_class.items())),
        }
        if any_sidecar
        else None,
        "missing": missing if inventory_known else None,
        "not_started": not_started if inventory_known else None,
        "omitted": omitted,
        "healthy": not reasons if inventory_known else None,
        "unhealthy_reasons": reasons,
        "limits": limits.summary(limits_source),
    }


_RUN_FAILURES_CHECK = "run_excessive_failures"


def _run_excessive_failures(
    artifacts: dict[str, Any], limits: RunHealthLimits, ignored_checks: frozenset[str]
) -> tuple[list[Finding], dict[str, int]]:
    """The ``run_excessive_failures`` check: one finding per group of failed rollouts that reaches its limit.

    It reads only the run accounting, so a stopped run can evaluate it without the rollouts. With no limit
    configured, or no inventory to count expected rollouts against, it is unobserved.
    """
    coverage = {"evaluated": 0, "unobserved": 0, "ignored": 0}
    if _RUN_FAILURES_CHECK in ignored_checks:
        coverage["ignored"] = 1
        return [], coverage
    expected, failed = artifacts["expected"], artifacts["failed"]
    groups = (
        ("not_final", limits.max_not_final_failure_fraction, "not_final"),
        ("final", limits.max_final_failure_fraction, "final"),
    )
    if not expected or failed is None or all(limit is None for _, limit, _ in groups):
        coverage["unobserved"] = 1
        return [], coverage
    coverage["evaluated"] = 1
    findings = [
        Finding(
            check=_RUN_FAILURES_CHECK,
            subject={},
            detail={"group": group, "count": failed[field], "expected": expected, "limit": limit},
        )
        for group, limit, field in groups
        if limit is not None and failed[field] and failed[field] / expected >= limit
    ]
    return findings, coverage


def _judge_run(artifacts: dict[str, Any], findings: list[Finding], coverage: dict[str, int]) -> None:
    """Add the run-level check to the accounting: its coverage, and a reason and an unhealthy verdict per finding."""
    artifacts.setdefault("coverage", {})[_RUN_FAILURES_CHECK] = coverage
    for finding in findings:
        detail = finding.detail
        share = detail["count"] / detail["expected"]
        artifacts["unhealthy_reasons"].append(
            f"{detail['group'].replace('_', ' ')} failures {detail['count']} of {detail['expected']} ({share:.1%}), "
            f"limit {detail['limit'] * 100:g}%"
        )
    if findings:
        artifacts["healthy"] = False


def _sort_key(digest: RolloutDigest) -> tuple[tuple[int, Any], ...]:
    def part(value: int | str | None) -> tuple[int, Any]:
        return (0, value) if isinstance(value, int) else (1, str(value))

    return part(digest.task_index), part(digest.rollout_index), part(digest.stage_index)


def _report_sources(paths: Sequence[Path], report_dir: Path) -> list[str]:
    """The rollouts files a report was computed from, relative to the folder it is written in
    (``run.<QUALITY_SUMMARY_SOURCES_KEY>``): the reports keep one name per folder whatever the rollouts file is
    called, so a reader with several runs' rollouts in one folder can tell whose they are. Only the directories are
    resolved, as :func:`_aggregated_shards` does, so a shard keeps its own name."""
    base = report_dir.resolve()
    return [Path(os.path.relpath(path.parent.resolve() / path.name, base)).as_posix() for path in paths]


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
                "responses": digest.responses,
            }
            if digest.stage_index is not None:
                row["stage_index"] = digest.stage_index
            handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
    return summary_path, verdicts_path


def capture_dirs_for_health(global_config_dict: Any) -> list[Path]:
    """The run's model-call capture directories for ``run_health_checks``; none when capture is off or misconfigured,
    so a capture setting never stops the health check."""
    from nemo_gym.observability.model_calls import model_call_capture_dirs_from_config

    try:
        return model_call_capture_dirs_from_config(global_config_dict)
    except ValueError:
        return []


def _with_captured_calls(rows: Iterable[dict[str, Any]], capture_dirs: Sequence[Path]) -> Iterator[dict[str, Any]]:
    """Failed rollouts' rows, each with the model calls captured under its rollout id, read with the collector's
    key and reader.

    Collection joins captures only into a rollout that produced a result, so a row that failed without one (an
    episode cut by its deadline, say) has none, although the model server captured its calls. Each capture dir is
    listed once, never created. A row comes back as it is, its time unmeasured, when it carries its own capture
    (so no call counts twice), when its capture is missing or partial (a write still claimed, a failed write, an
    unreadable record), or when its identity or its capture cannot be read.
    """
    if not capture_dirs:
        yield from rows
        return
    # Lazy: the capture store pulls in the model-server stack, which health workers do not need.
    from nemo_gym.base_responses_api_model import _PENDING_SUFFIX, CaptureStore
    from nemo_gym.core.episode import maybe_rollout_id_from_run_body
    from nemo_gym.observability.model_calls import read_available_model_call_records

    listings = []
    for directory in capture_dirs:
        try:
            names = set(os.listdir(directory))
        except OSError as exc:
            warnings.warn(
                f"Cannot list the model-call capture dir {directory} ({exc}); failed rollouts stay unmeasured.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        # A claimed write is named `<rollout id>.<call token>.capture.pending`; the token has no dot.
        pending = {
            name.removesuffix(_PENDING_SUFFIX).rsplit(".", 1)[0] for name in names if name.endswith(_PENDING_SUFFIX)
        }
        listings.append((CaptureStore(directory, create=False), names, pending))
    for row in rows:
        calls = None
        if "ng_model_call_capture" not in row:
            try:
                rollout_id = maybe_rollout_id_from_run_body(row)
                for store, names, pending in listings:
                    if rollout_id in pending or store.incomplete_path_for(rollout_id).name in names:
                        break
                    if store.path_for(rollout_id).name in names:
                        records, unreadable = read_available_model_call_records(store, rollout_id)
                        if not unreadable:
                            calls = [
                                record.model_dump(exclude={"request", "response", "request_raw", "response_raw"})
                                for record in records
                            ]
                        break
            except (OSError, TypeError, ValueError):
                pass  # A malformed identity or an unreadable capture names no calls; the record checks report the row.
        yield row if calls is None else {**row, "ng_model_call_capture": {"calls": calls}}


def _health_artifact_digests(paths: Sequence[Path]) -> list[dict[str, str | None]]:
    """Bind gate facts to saved rollout inputs, not to external captures or observation completeness."""
    return [
        {
            name: _sha256(source) if source.is_file() else None
            for name, source in {
                "rollouts_jsonl_fpath": path,
                "failures_jsonl_fpath": failures_path_for(path),
                "materialized_inputs_jsonl_fpath": materialized_path_for(path),
                "resolved_config_yaml_fpath": resolved_config_path_for(path),
            }.items()
        }
        for path in paths
    ]


def run_health_checks(
    rollout_paths: Path | Sequence[Path],
    *,
    output_dir: Path | None = None,
    workers: int | None = None,
    ignored_checks: Sequence[str] = (),
    limits: RunHealthLimits = RunHealthLimits(),
    limits_source: LimitsSource | Mapping[str, LimitsSource] = "config",
    run_coverage: Sequence[Mapping[str, Any] | None] | None = None,
    selected_projection: Path | None = None,
    model_call_capture_dirs: Sequence[Path] = (),
) -> HealthCheckResult:
    """Run the RFC's map/group/reduce pipeline and write both reports.

    ``model_call_capture_dirs`` are the run's model-call capture directories. A failed rollout's row carries no
    capture, so its timing in ``run.stats.timeouts`` is read from these instead.
    ``selected_projection`` is the merged output of these managed runs, used when
    their raw files contain superseded outcomes. Source histories still own failure
    accounting; each projected row's run ID retains its source identity.
    """
    ignored = frozenset(normalize_ignored_checks(ignored_checks))
    paths = list([rollout_paths] if isinstance(rollout_paths, Path) else rollout_paths)
    if not paths:
        raise ValueError("at least one rollout JSONL path is required")
    needs_projection = any(not raw_outcomes_are_selected(path, artifacts=(path,)) for path in paths)
    if needs_projection and selected_projection is None:
        raise JournalHealthUnavailable(
            "Health reports for histories with superseded outcomes are a follow-up to evaluation resume. "
            "Run health checks on a merged selected-result projection from `gym eval aggregate`."
        )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Rollout JSONL not found: {path}")

    # A legacy run reached through a link keeps the link's name when its companions sit beside the link, as the
    # aggregation that read it did; otherwise its companions are the ones beside the file the link resolves to.
    paths = [resolve_rollout_path(path, read_only=True) for path in paths]
    source_digests = _health_artifact_digests(paths)
    if needs_projection:
        source_indices = {}
        for index, path in enumerate(paths):
            history = RolloutStore.read(path, import_legacy=False)
            if history is None or history.manifest.run_id in source_indices:
                raise JournalHealthUnavailable("A selected projection requires distinct manifest-backed source runs.")
            source_indices[history.manifest.run_id] = index
        lines = _index_jsonl([selected_projection], source_indices=source_indices)
    else:
        lines = _index_jsonl(paths)
    worker_inputs = [
        _WorkerInput(
            line=line,
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
                    worker_results = list(
                        pool.map(
                            _worker,
                            worker_inputs,
                            chunksize=_process_pool_chunksize(len(worker_inputs), max_workers),
                        )
                    )
            except (BrokenProcessPool, OSError) as exc:
                warnings.warn(
                    f"Process pool failed ({exc}); running rollout health checks serially.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                worker_results = [_worker(item) for item in worker_inputs]

    digests = worker_results
    _mark_duplicate_identities(digests, lines, ignored)
    summary = _reduce(digests, ignored)
    artifacts = summary["run"]["artifacts"]
    landed = {(line.source_index, d.task_index, d.rollout_index) for d, line in zip(digests, lines, strict=True)}
    artifacts.update(
        _rollout_accounting(
            landed,
            paths,
            limits=limits,
            limits_source=limits_source,
            duplicated=summary["run"]["issues"]["rollout_duplicate_identity"],
            run_coverage=run_coverage,
        )
    )
    # Every rollout's final record: its saved row, or the last failed attempt of one that never landed.
    timings = [digest.timing for digest in digests if digest.timing is not None]
    timings += map(
        read_rollout_timing, _with_captured_calls(_last_failed_attempts(landed, paths), model_call_capture_dirs)
    )
    # Every failed attempt in the sidecars, so a limit or enforcer that only an earlier attempt named is sent as 0.
    attempts = (
        read_rollout_timing(row)
        for path in paths
        if failures_path_for(path).is_file()
        for row in _jsonl_rows(failures_path_for(path))
    )
    summary["run"]["stats"]["timeouts"] = summarize_timeouts(timings, attempts).model_dump(mode="json")
    # Preserve observations from unsuccessful writer rows as well as scored rows.
    # These are known attempt/runtime facts; they do not establish a complete trial population.
    observed_records = (
        row
        for path in paths
        for source in (path, failures_path_for(path))
        if source.is_file()
        for row in _jsonl_rows(source)
    )
    observed_calls = []
    runtime = {
        "invocation_observations": 0,
        "in_session": {"true": 0, "false": 0, "absent": 0},
        "verification_skipped": {"true": 0, "false": 0, "absent": 0},
        "row_elapsed_s": {"sum_observed": 0.0, "absent": 0},
    }
    for record in observed_records:
        trajectory, _ = _canonical_trajectory(record)
        if trajectory is not None:
            # Keep only the bounded metadata needed below, rather than retaining all request/response payloads.
            observed_calls.extend(
                {
                    key: call.get(key)
                    for key in (
                        "status_code",
                        "response_status",
                        "error_category",
                        "failure_report",
                        "model_operations",
                    )
                }
                for call in _normalized_trajectory_calls(trajectory)
            )
            for invocation in trajectory.get("invocations") or []:
                runtime["invocation_observations"] += 1
                for key in ("in_session", "verification_skipped"):
                    value = invocation.get(key)
                    runtime[key][str(value).lower() if isinstance(value, bool) else "absent"] += 1
        elapsed = record.get("_ng_elapsed_s")
        if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
            runtime["row_elapsed_s"]["sum_observed"] += elapsed
        else:
            runtime["row_elapsed_s"]["absent"] += 1
    summary["run"]["stats"]["observed_upstream_recovery"] = _observed_upstream_recovery(observed_calls)
    summary["run"]["stats"]["observed_runtime"] = runtime
    operations = {}
    for call in observed_calls:
        for operation in call.get("model_operations") or []:
            # The producer's opaque ID is the join key, not capture position or retry-label parsing.
            operations.setdefault(operation["operation_id"], []).append(operation)
    effort = {}
    for key in ("http_attempts", "transport_failures", "retry_wait_s", "request_elapsed_ms"):
        values = []
        for observations in operations.values():
            known = {
                o[key]
                for o in observations
                if type(o.get(key)) in (int, float) and math.isfinite(o[key]) and o[key] >= 0
            }
            if len(known) == 1:
                values.append(known.pop())
        effort[key] = {
            "sum_observed": sum(values) if values else None,
            "operations_observed": len(values),
            "operations_unknown": len(operations) - len(values),
        }
    summary["run"]["stats"]["observed_request_effort"] = effort
    run_findings, run_coverage = _run_excessive_failures(artifacts, limits, ignored)
    _judge_run(artifacts, run_findings, run_coverage)
    summary["run"]["issues"][_RUN_FAILURES_CHECK] = len(run_findings)
    summary["run"]["findings"] = [finding.model_dump(mode="json", exclude={"subject"}) for finding in run_findings]
    summary["run"]["provenance"] = run_provenance(paths[0])
    if source_digests == _health_artifact_digests(paths):
        summary["run"]["provenance"]["rollout_artifact_digests"] = source_digests
    report_dir = output_dir or paths[0].parent
    summary["run"][QUALITY_SUMMARY_SOURCES_KEY] = _report_sources(paths, report_dir)
    summary_path, verdicts_path = _write_reports(summary, digests, report_dir)
    return HealthCheckResult(
        summary=summary,
        rollouts=digests,
        summary_path=summary_path,
        verdicts_path=verdicts_path,
    )


def write_provisional_summary(
    rollout_path: Path,
    landed: set[tuple[Any, Any]],
    *,
    limits: RunHealthLimits = RunHealthLimits(),
    ignored_checks: Sequence[str] = (),
    run_coverage: Mapping[str, Any] | None = None,
) -> Path:
    """Write the run accounting of a stopped run fast: ``rollouts.jsonl`` is not read, and the per-record checks
    are skipped.

    ``landed`` is the ``(task index, rollout index)`` of every persisted record, as rollout collection holds it.
    Duplicates come from the per-record pass, so ``healthy`` is false when the accounting already has a reason
    and null otherwise. ``gym eval health-check`` on the run dir writes the full summary.
    """
    landed_keys = {(0, task, rollout) for task, rollout in landed}
    artifacts = _rollout_accounting(
        landed_keys,
        [rollout_path],
        limits=limits,
        limits_source="config",
        duplicated=0,
        run_coverage=[run_coverage],
    )
    ignored = frozenset(normalize_ignored_checks(ignored_checks))
    run_findings, run_coverage = _run_excessive_failures(artifacts, limits, ignored)
    _judge_run(artifacts, run_findings, run_coverage)
    artifacts["partial"] = True
    # Only run_excessive_failures runs here; every other check needs the rollouts and is not evaluated.
    artifacts["not_evaluated"] = sorted(spec.id for spec in CHECK_REGISTRY if spec.id != _RUN_FAILURES_CHECK)
    if artifacts["healthy"]:
        artifacts["healthy"] = None
    summary_path = rollout_path.parent / QUALITY_SUMMARY_FILENAME
    run = {
        "ignored_checks": sorted(ignored),
        "artifacts": artifacts,
        "issues": {_RUN_FAILURES_CHECK: len(run_findings)},
        "findings": [finding.model_dump(mode="json", exclude={"subject"}) for finding in run_findings],
        QUALITY_SUMMARY_SOURCES_KEY: _report_sources([rollout_path], rollout_path.parent),
    }
    summary_path.write_bytes(orjson.dumps({"run": run}, option=orjson.OPT_INDENT_2))
    (rollout_path.parent / ROLLOUT_VERDICTS_FILENAME).unlink(missing_ok=True)
    return summary_path


def _flag(value: Any) -> bool | None:
    """A config flag as written: a bool is known, anything else (absent, interpolated, malformed) is unknown."""
    return value if isinstance(value, bool) else None


def _context_window_exceeded(record: Mapping[str, Any]) -> bool | None:
    """From the row's own `step_usage` (written by scicode_agent): a sub-step recorded as context_window_exceeded.

    scicode_agent sets that status from a best-effort error-text match or model metadata, so it is a signal of a
    context-window failure, not proof that the input was clipped.
    """
    steps = record.get("step_usage")
    if not isinstance(steps, list) or not all(isinstance(s, dict) and isinstance(s.get("status"), str) for s in steps):
        return None
    return any(step["status"] == "context_window_exceeded" for step in steps)


# The three MLflow name fields `MLFlowConfig.is_available` reads (its token check is not judged here).
_MLFLOW_REQUIRED_NAMES = ("mlflow_tracking_uri", "mlflow_experiment_name", "mlflow_run_name")


def _mlflow_configured(config: Mapping[str, Any]) -> bool | None:
    """A names-only fact about the saved config snapshot: whether its three MLflow name fields are concrete.

    True when all three are concrete strings, False when any is absent, null or empty, None when a value is
    interpolated or not a string. It is not `MLFlowConfig.is_available` (the token, which the snapshot may mask, is
    not judged) and says nothing about whether an exporter was activated, whether tracking was required or
    disabled, or whether anything was delivered: overrides, resumes and post-run exports can differ from the
    snapshot. No value is copied.
    """
    states = []
    for key in _MLFLOW_REQUIRED_NAMES:
        value = config.get(key)
        if value is None or value == "":
            states.append("missing")
        elif isinstance(value, str) and value.strip() and "${" not in value:
            states.append("set")
        else:
            states.append("unknown")
    if "unknown" in states:
        return None
    return all(state == "set" for state in states)


def run_provenance(rollout_path: Path) -> dict[str, Any]:
    """Facts about how the run was configured, read from the resolved config saved beside the rollouts.

    Only values the config states are reported: ``route_failures_to_sidecar`` and each agent's ``skip_verification``
    are booleans when written as booleans and ``None`` otherwise, never a default. Each top-level server entry
    reports its kind (``responses_api_agents``, ``resources_servers`` or ``environment_servers``), its implementation
    key and, for an agent, the resources server it names. No values beyond these names and flags are copied.
    """
    import yaml

    config_path = resolved_config_path_for(rollout_path)
    try:
        config = yaml.safe_load(config_path.read_text())
    except FileNotFoundError:
        return {"resolved_config": "absent"}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {"resolved_config": "unreadable"}
    if not isinstance(config, dict):
        return {"resolved_config": "unreadable"}
    servers: dict[str, Any] = {}
    for name, entry in config.items():
        if not isinstance(entry, dict):
            continue
        for kind in ("responses_api_agents", "resources_servers", "environment_servers"):
            block = entry.get(kind)
            if not isinstance(block, dict) or len(block) != 1:
                continue
            [(implementation, settings)] = block.items()
            settings = settings if isinstance(settings, dict) else {}
            fact: dict[str, Any] = {"kind": kind, "implementation": str(implementation)}
            if kind == "responses_api_agents":
                resources = settings.get("resources_server")
                fact["resources_server"] = resources.get("name") if isinstance(resources, dict) else None
                fact["skip_verification"] = _flag(settings.get("skip_verification"))
            servers[str(name)] = fact
    return {
        "resolved_config": "read",
        "route_failures_to_sidecar": _flag(config.get("route_failures_to_sidecar")),
        "servers": servers,
        "mlflow": {"configured": _mlflow_configured(config)},
    }


def _persisted_limits(run_dir: Path) -> RunHealthLimits:
    """The limits the run's earlier summary was judged by; none (no limit) when it has no readable summary."""
    try:
        persisted = orjson.loads((run_dir / QUALITY_SUMMARY_FILENAME).read_bytes())["run"]["artifacts"]["limits"]
        return RunHealthLimits.model_validate({k: v for k, v in persisted.items() if k != "source"})
    except (OSError, orjson.JSONDecodeError, KeyError, TypeError, ValueError):
        return RunHealthLimits()


class _AggregationSources(BaseModel):
    """The shards `gym eval aggregate` merged into a rollout file, in its order, each as its path, size and SHA-256,
    and that file's size and SHA-256."""

    shards: list[tuple[str, int, str]]
    merged_bytes: int
    merged_sha256: str


def _sha256(fpath: Path) -> str:
    with fpath.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_aggregation_sources(
    merged_fpath: Path, shards: Sequence[tuple[Path, int, str]], merged_contents_fpath: Path
) -> None:
    """Record the shards ``merged_fpath`` is merged from, for a recheck of its directory to read.

    The merged file holds only the shards' rows: their failure sidecars and inventories stay next to them.
    ``shards`` holds each shard's path with the size and SHA-256 of what the merge read from it, and
    ``merged_contents_fpath`` the rows about to take ``merged_fpath``'s place; the record names both by size and
    SHA-256, so it never vouches for other contents. Each shard is recorded relative to the record's directory,
    so the shards and the merged file can move together, and under its own file name: a shard reached through a
    file link keeps the sidecars beside the link, as the aggregation read them.
    """
    sources_fpath = aggregation_sources_path_for(merged_fpath)
    base = sources_fpath.parent.resolve()
    sources = _AggregationSources(
        shards=[
            (os.path.relpath(path.parent.resolve() / path.name, base), size, sha256) for path, size, sha256 in shards
        ],
        merged_bytes=merged_contents_fpath.stat().st_size,
        merged_sha256=_sha256(merged_contents_fpath),
    )
    sources_fpath.write_text(sources.model_dump_json(indent=2) + "\n")


def _aggregated_shards(run_dir: Path, *, rollout_name: str = "rollouts.jsonl") -> list[Path] | None:
    """The shards `gym eval aggregate` merged into ``run_dir/rollout_name``, in its order; None if it merged none.

    The merged rows have none of the shards' failures or inventories, so a recheck reads the shards, as the
    aggregation did. A merged file that is gone, or not byte for byte the one the list was written for, is not
    theirs; a shard that is gone cannot be read, and one not byte for byte the one merged holds other rows. Each
    raises rather than check rows other than the merged ones, or without their failures.
    """
    merged_fpath = run_dir / rollout_name
    sources_fpath = aggregation_sources_path_for(merged_fpath)
    if not sources_fpath.is_file():
        return None
    try:
        sources = _AggregationSources.model_validate_json(sources_fpath.read_bytes())
    except ValidationError as exc:
        raise ValueError(f"Unreadable shard list {sources_fpath}: {exc}") from exc
    # The size first, as it is cheap; equal sizes alone do not make equal contents.
    if (
        not merged_fpath.is_file()
        or merged_fpath.stat().st_size != sources.merged_bytes
        or _sha256(merged_fpath) != sources.merged_sha256
    ):
        raise ValueError(
            f"{merged_fpath} is not the file `gym eval aggregate` merged from the shards in {sources_fpath}. Rerun "
            "the aggregation, or pass --rollouts-file rollouts.jsonl to check that file alone."
        )
    # Only the directories are resolved: each shard keeps its own name, and with it the sidecars beside it.
    shards = [path.parent.resolve() / path.name for path in (run_dir / shard for shard, _, _ in sources.shards)]
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{merged_fpath} was merged from shards that are gone: {', '.join(missing)}. Recheck the shards, or pass "
            "--rollouts-file rollouts.jsonl to check the merged rows alone, without the shards' failures."
        )
    changed = [
        str(shard)
        for shard, (_, size, sha256) in zip(shards, sources.shards)
        if shard.stat().st_size != size or _sha256(shard) != sha256
    ]
    if changed:
        raise ValueError(
            f"{merged_fpath} was merged from shards that changed since: {', '.join(changed)}. Rerun "
            "`gym eval aggregate` to merge and check them as they are now."
        )
    return shards


def _resolve_rollout_path(run_dir: Path, rollout_file: str | Path | None) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    selected = Path(rollout_file) if rollout_file is not None else Path("rollouts.jsonl")
    rollout_path = selected if selected.is_absolute() else run_dir / selected
    if not rollout_path.is_file():
        raise FileNotFoundError(f"Rollout JSONL not found: {rollout_path}")
    return rollout_path


def format_health_report(result: HealthCheckResult) -> str:
    verdicts = result.summary["run"]["verdicts"]
    checked = sum(verdicts.values())
    ignored = result.summary["run"].get("ignored_checks", [])
    ignored_note = f" (ignored: {', '.join(ignored)})" if ignored else ""
    responses = result.summary["run"].get("stats", {}).get("responses")
    response_note = ""
    if responses is not None:
        response_note = (
            f"Model responses: {responses['faulted']}/{responses['selected']} faulted; "
            f"{responses['unknown']} rollout scopes unobserved; "
            f"{responses['duplicated']} duplicate call IDs\n"
        )
    return (
        f"Rollout checks: {checked} checked, {verdicts['healthy']} passed, "
        f"{verdicts['unhealthy']} flagged, {verdicts['unobserved']} unobserved{ignored_note}\n"
        f"{response_note}"
        f"Quality summary: {result.summary_path}"
    )


def health_check_run_dir(
    run_dir: str | Path,
    *,
    rollout_file: str | Path | None = None,
    workers: int | None = None,
    ignored_checks: Sequence[str] = (),
    limits: RunHealthLimits | None = None,
    json_output: bool = False,
    model_call_capture_dir: str | Path | None = None,
) -> HealthCheckResult:
    """Recheck a run directory. The run's own persisted limits judge it, so the verdict of the same artifacts
    does not change. Each limit set explicitly on ``limits`` overrides only that persisted limit and is recorded
    as an override; set to ``None`` it removes the limit. A limit not set keeps its persisted value.

    A directory ``gym eval aggregate`` merged shards into is rechecked from those shards, with the failures and
    inventories beside them, as the aggregation checked them. An explicit ``rollout_file`` is checked alone.

    ``model_call_capture_dir`` is the run's ``model_call_capture_dir``; with it, failed rollouts' captured calls
    count in the timeouts section's time split."""
    path = Path(run_dir)
    shards = _aggregated_shards(path) if rollout_file is None else None
    rollout_paths = shards if shards is not None else _resolve_rollout_path(path, rollout_file)
    overrides = {name: getattr(limits, name) for name in limits.model_fields_set} if limits is not None else {}
    limits = _persisted_limits(path).model_copy(update=overrides)
    limits_source = {name: "override" if name in overrides else "persisted" for name in RunHealthLimits.model_fields}
    result = run_health_checks(
        rollout_paths,
        selected_projection=path / "rollouts.jsonl" if shards is not None else None,
        output_dir=path,
        workers=workers,
        ignored_checks=ignored_checks,
        limits=limits,
        limits_source=limits_source,
        model_call_capture_dirs=[Path(model_call_capture_dir)] if model_call_capture_dir is not None else [],
    )
    # The run dir does not record its capture dir, so a recheck without one cannot time a failed rollout.
    timeouts, failed = result.summary["run"]["stats"]["timeouts"], result.summary["run"]["artifacts"]["failed"]
    unsplit = timeouts["timed_out"] - timeouts["time_split"]["rollouts_with_captures"]
    if model_call_capture_dir is None and failed and failed["total"] and unsplit:
        warnings.warn(
            "run.stats.timeouts.time_split times a rollout that failed without a row only from the run's model-call "
            "captures: pass --model-call-capture-dir <the run's model_call_capture_dir>.",
            RuntimeWarning,
            stacklevel=2,
        )
    if json_output:
        print(orjson.dumps(result.summary).decode())
    else:
        print(format_health_report(result))
    return result


def _remove_health_reports(output_fpath: Path) -> None:
    # A fresh run or a resume reports its own health; a report left by an earlier job must not stand for it.
    for report in (QUALITY_SUMMARY_FILENAME, ROLLOUT_VERDICTS_FILENAME):
        (output_fpath.parent / report).unlink(missing_ok=True)


def _health_check_collected_rollouts(
    output_fpath: Path,
    config: Any,
    landed_keys: set | None = None,
    *,
    coverage: dict | None = None,
    terminated: bool = False,
    model_call_capture_dirs: Sequence[Path] = (),
) -> None:
    """Write the quality summary for whatever a run collected; a run that never started collection has none.

    After SIGTERM, Slurm kills the job once KillWait (120 s on some clusters) is over, and the full check can take
    minutes on a large run. So a stopped run with ``landed_keys`` gets the provisional summary: the run
    accounting only, without reading the rollouts.
    """
    if not output_fpath.exists():
        return
    from nemo_gym.health.timeouts import export_run_health_timeouts

    if terminated and landed_keys is not None:
        if aggregation_sources_path_for(output_fpath).is_file():
            # A run drops the shard list with the first row it writes, so this run added none: the summary the
            # aggregation wrote from its shards still holds, and the run accounting alone would replace it wrongly.
            print("Shard-backed health was not recomputed after stopping; run `gym eval health-check` to check it.")
            return
        try:
            summary_path = write_provisional_summary(
                output_fpath,
                landed_keys,
                limits=config.run_health_limits,
                ignored_checks=config.health_check_ignored_checks,
                run_coverage=coverage,
            )
        except JournalHealthUnavailable as error:
            print(f"Rollout health checks skipped: {error}")
        except Exception:
            logger.exception("Writing the provisional quality summary failed; rollout artifacts are still available.")
        else:
            print(f"Provisional quality summary (run accounting only): {summary_path}")
        return
    try:
        # An output `gym eval aggregate` merged shards into is checked from those shards, as the collector and a
        # recheck check it: its own rows have none of their failures or inventories.
        shards = _aggregated_shards(output_fpath.parent, rollout_name=output_fpath.name)
        health_result = run_health_checks(
            shards if shards is not None else output_fpath,
            selected_projection=output_fpath if shards is not None else None,
            output_dir=output_fpath.parent,
            workers=config.health_check_workers,
            ignored_checks=config.health_check_ignored_checks,
            limits=config.run_health_limits,
            # This run's coverage describes its own rows; listed shards are checked with their own histories.
            run_coverage=None if shards is not None else [coverage],
            model_call_capture_dirs=model_call_capture_dirs,
        )
    except JournalHealthUnavailable as error:
        print(f"Rollout health checks skipped: {error}")
    except Exception:
        logger.exception("Rollout health checks failed after collection; rollout artifacts are still available.")
    else:
        print(format_health_report(health_result))
        export_run_health_timeouts(health_result)
