# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic post-run rollout quality verification workflow.

Checks operate only on persisted rollout records and their canonical
``ng_trajectory`` evidence. They return evidence; this module derives verdicts
and writes reports.
"""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import orjson

from nemo_gym.health.checks import (
    _INCOMPLETE_MODEL_CALL_GAPS,
    _ROLLOUT_CHECKS,
    _ROLLOUT_SPECS,
    _TASK_SPECS,
    CHECK_REGISTRY,
    _bind_policy_calls,
    _canonical_trajectory,
    _is_failed,
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
from nemo_gym.health.types import (
    QUALITY_SUMMARY_FILENAME,
    ROLLOUT_ID_KEY,
    ROLLOUT_INDEX_KEY,
    ROLLOUT_VERDICTS_FILENAME,
    TASK_INDEX_KEY,
    CheckInput,
    CheckScope,
    CheckSpec,
    CheckSubject,
    Finding,
    HealthCheckResult,
    RolloutDigest,
    Verdict,
    _LineSlice,
    _TaskRepeat,
    _WorkerInput,
)


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
    unreadable_identity = f"__unreadable_record__:input-{payload.line.source_index}:line-{payload.line.line_number}"
    task_index = record.get(TASK_INDEX_KEY, unreadable_identity if parse_error else payload.line.ordinal)
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
    model_calls_observed = trajectory_observed and not _trajectory_has_any_gap(trajectory, _INCOMPLETE_MODEL_CALL_GAPS)
    calls = _normalized_trajectory_calls(trajectory) if trajectory_observed else []
    bindings = _bind_policy_calls(trajectory, calls) if trajectory_observed else _bind_policy_calls({}, [])
    findings: list[Finding] = []
    unobserved: list[str] = []

    for spec in _ROLLOUT_SPECS:
        if spec.id in payload.ignored_checks:
            continue
        if parse_error:
            if spec.id == "record_unreadable":
                findings.append(
                    Finding(
                        check="record_unreadable",
                        subject=subject,
                        locator={"source_file": payload.line.path, "line": payload.line.line_number},
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
        if CheckInput.BOUND_CALLS in spec.reads:
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
            not bindings.complete
            or not bindings.matched_calls
            or not _transcript_tokens(record)[2]
            or any(call.get("tokens_in") is None or call.get("tokens_out") is None for call in bindings.matched_calls)
        ):
            unobserved.append(spec.id)
            continue
        try:
            findings.extend(_ROLLOUT_CHECKS[spec.id](record, trajectory, bindings, subject))
        except Exception as exc:
            unobserved.append(spec.id)
            if "check_execution_error" not in payload.ignored_checks:
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
        rollout_id=rollout_id,
        verdict=verdict,
        findings=findings,
        unobserved=unobserved,
        capture_observed=bool(calls),
        policy_calls_observed=bindings.complete,
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


def _mark_duplicate_identities(digests: list[RolloutDigest], ignored_checks: frozenset[str]) -> None:
    """Flag physical records that claim the same logical rollout identity."""
    if "rollout_duplicate_identity" in ignored_checks:
        return

    grouped: dict[tuple[int | str, int | str], list[RolloutDigest]] = defaultdict(list)
    for digest in digests:
        grouped[(digest.task_index, digest.rollout_index)].append(digest)

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


def run_health_checks(
    rollout_paths: Path | Sequence[Path],
    *,
    output_dir: Path | None = None,
    workers: int | None = None,
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
    _mark_duplicate_identities(digests, ignored)
    summary = _reduce(digests, ignored)
    report_dir = output_dir or paths[0].parent
    summary_path, verdicts_path = _write_reports(summary, digests, report_dir)
    return HealthCheckResult(
        summary=summary,
        rollouts=digests,
        summary_path=summary_path,
        verdicts_path=verdicts_path,
    )


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
    json_output: bool = False,
) -> HealthCheckResult:
    path = Path(run_dir)
    rollout_path = _resolve_rollout_path(path, rollout_file)
    result = run_health_checks(
        rollout_path,
        output_dir=path,
        workers=workers,
        ignored_checks=ignored_checks,
    )
    if json_output:
        print(orjson.dumps(result.summary).decode())
    else:
        print(format_health_report(result))
    return result
