# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic post-run rollout quality verification workflow.

Checks operate only on persisted rollout records and their canonical
``ng_trajectory`` evidence. They return evidence; this module derives verdicts
and writes reports.
"""

from __future__ import annotations

import hashlib
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
from nemo_gym.utils.path_utils import aggregation_sources_path_for, failures_path_for, materialized_path_for


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
        model_call_errors=len(failed),
        model_call_errors_usage_unknown=sum(
            _is_failed_attempt(call) and (call.get("tokens_in") is None or call.get("tokens_out") is None)
            for call in failed
        ),
        errors_by_status=dict(errors_by_status),
        ended_on_error=model_calls_observed and _ended_on_failed_call(calls),
        duplicated_calls=duplicated,
        transcript_prompt_tokens=transcript_prompt,
        transcript_completion_tokens=transcript_completion,
        capture_prompt_tokens=sum(_token_count(call, "tokens_in") for call in calls),
        capture_completion_tokens=sum(_token_count(call, "tokens_out") for call in calls),
        timing=read_rollout_timing(record),
    )


def _index_jsonl(paths: Sequence[Path]) -> list[_LineSlice]:
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
                    # Failed calls with no finished response and no usage: the token checks skip them,
                    # and any tokens the server spent on them are counted nowhere else.
                    "usage_unknown": sum(digest.model_call_errors_usage_unknown for digest in digests),
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
    from nemo_gym.rollout_collection import is_omitted_rollout_row, is_unstarted_rollout_row

    failed: dict[tuple[Any, ...], dict[str, Any]] = {}
    any_sidecar = False
    owed: set[tuple[Any, ...]] | None = set()
    for source_index, path in enumerate(paths):
        if failures_path_for(path).is_file():
            any_sidecar = True
            for row in _jsonl_rows(failures_path_for(path)):
                # Main's rule: a row the dispatch budget never started is no attempt and selects nothing.
                if is_unstarted_rollout_row(row):
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


def _rollout_accounting(
    landed: set[tuple[Any, ...]],
    failures: _Failures,
    *,
    limits: RunHealthLimits,
    limits_source: LimitsSource | Mapping[str, LimitsSource],
    duplicated: int,
) -> dict[str, Any]:
    """Count the rollouts each input owed against those that landed, failed or left no row, and judge the run.

    ``landed`` holds ``(input index, task index, rollout index)`` of every persisted record. Failures come
    from each input's ``_failures.jsonl`` sidecar, where the last attempt of a rollout stands; one that
    later landed is not failed. A ``record_omitted_rollouts`` row records no result: an interrupted attempt is the
    rollout's latest execution and leaves it missing, and a row never started is skipped.
    ``is_terminal_failure`` splits them into
    final ones, which are not retried, and not final ones, which are; the split says nothing of the cause. ``failed`` is null when no input has a sidecar.
    ``expected``, ``missing`` and ``healthy`` need the ``_materialized_inputs.jsonl`` inventory and
    are null when an input has none. A run is healthy when nothing is missing, no rollout is
    recorded twice, and each kind of failure with a configured limit stays below it as a share of expected.
    The limits are off (None) by default, so failures alone never make a run unhealthy unless configured.
    """
    # Lazy: rollout_collection imports this module, and health workers need none of its dependencies.
    from nemo_gym.artifacts.rollout_rows import NG_FAILURE_CLASS_KEY
    from nemo_gym.rollout_collection import is_terminal_failure

    failed, owed, any_sidecar = failures.rows, failures.owed, failures.any_sidecar
    expected = None if owed is None else len(owed)
    missing = None if owed is None else len(owed - landed - failed.keys())
    final = sum(is_terminal_failure(row) for row in failed.values())
    not_final = len(failed) - final
    reasons = []
    if expected == 0:
        reasons.append("no rollouts expected")
    elif expected is not None:
        if missing:
            reasons.append(f"missing {missing} of {expected}")
        if duplicated:
            reasons.append(f"{duplicated} records share a rollout identity")
    return {
        "expected": expected,
        "failed": {
            "total": len(failed),
            "final": final,
            "not_final": not_final,
            "by_class": dict(
                sorted(Counter(row.get(NG_FAILURE_CLASS_KEY) or "unknown" for row in failed.values()).items())
            ),
        }
        if any_sidecar
        else None,
        "missing": missing,
        "healthy": None if expected is None else not reasons,
        "unhealthy_reasons": reasons,
        # The limits the verdict was judged by, so a recheck reuses them and the decision can be reproduced.
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


def run_health_checks(
    rollout_paths: Path | Sequence[Path],
    *,
    output_dir: Path | None = None,
    workers: int | None = None,
    ignored_checks: Sequence[str] = (),
    limits: RunHealthLimits = RunHealthLimits(),
    limits_source: LimitsSource | Mapping[str, LimitsSource] = "config",
    model_call_capture_dirs: Sequence[Path] = (),
) -> HealthCheckResult:
    """Run the RFC's map/group/reduce pipeline and write both reports.

    ``model_call_capture_dirs`` are the run's model-call capture directories. A failed rollout's row carries no
    capture, so its timing in ``run.stats.timeouts`` is read from these instead.
    """
    ignored = frozenset(normalize_ignored_checks(ignored_checks))
    paths = [rollout_paths] if isinstance(rollout_paths, Path) else list(rollout_paths)
    if not paths:
        raise ValueError("at least one rollout JSONL path is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Rollout JSONL not found: {path}")

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
    failures = _read_failures(landed, paths)
    artifacts.update(
        _rollout_accounting(
            landed,
            failures,
            limits=limits,
            limits_source=limits_source,
            duplicated=summary["run"]["issues"]["rollout_duplicate_identity"],
        )
    )
    # Every rollout's final record: its saved row, or the last failed attempt of one that never landed.
    timings = [digest.timing for digest in digests if digest.timing is not None]
    timings += map(read_rollout_timing, _with_captured_calls(failures.rows.values(), model_call_capture_dirs))
    # Every failed attempt in the sidecars, so a limit or enforcer that only an earlier attempt named is sent as 0.
    attempts = (
        read_rollout_timing(row)
        for path in paths
        if failures_path_for(path).is_file()
        for row in _jsonl_rows(failures_path_for(path))
    )
    summary["run"]["stats"]["timeouts"] = summarize_timeouts(timings, attempts).model_dump(mode="json")
    run_findings, run_coverage = _run_excessive_failures(artifacts, limits, ignored)
    _judge_run(artifacts, run_findings, run_coverage)
    summary["run"]["issues"][_RUN_FAILURES_CHECK] = len(run_findings)
    summary["run"]["findings"] = [finding.model_dump(mode="json", exclude={"subject"}) for finding in run_findings]
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
        _read_failures(landed_keys, [rollout_path]),
        limits=limits,
        limits_source="config",
        duplicated=0,
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
    return (
        f"Rollout health: {checked} checked, {verdicts['healthy']} healthy, "
        f"{verdicts['unhealthy']} unhealthy, {verdicts['unobserved']} unobserved{ignored_note}\n"
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
