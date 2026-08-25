# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic post-run rollout quality verification workflow.

Checks operate only on persisted rollout and model-call capture artifacts.
They return evidence; this module derives verdicts and writes reports.
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
    _FALLBACK_TRANSCRIPT_CHECK_IDS,
    _ROLLOUT_CHECKS,
    _ROLLOUT_SPECS,
    _TASK_SPECS,
    CHECK_REGISTRY,
    _agent_steps_with_source,
    _bind_policy_calls,
    _finding,
    _is_failed,
    _is_successful,
    _parse_capture,
    _replay_identity,
    _subject,
    _token_count,
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
    _AgentStepSource,
    _LineSlice,
    _TaskRepeat,
    _WorkerInput,
    _WorkerResult,
)


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
        needs_capture = CheckInput.CAPTURE in spec.reads
        if needs_capture and not capture_observed:
            unobserved.append(spec.id)
            continue
        if CheckInput.BOUND_CALLS in spec.reads and not bindings.matched_calls:
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

        if "task_no_successful_model_calls" in ignored_checks:
            coverage["task_no_successful_model_calls"]["ignored"] += 1
        else:
            if repeats and all(repeat.policy_calls_observed for repeat in repeats):
                coverage["task_no_successful_model_calls"]["evaluated"] += 1
                if not any(repeat.successful_model_calls for repeat in repeats):
                    findings[task_index].append(
                        _finding("task_no_successful_model_calls", subject, repeats=len(repeats))
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
    captures_exist = any(any(directory.glob("*.capture.jsonl")) for directory in capture_dirs if directory.exists())
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


def _standalone_capture_dirs(run_dir: Path, capture_dirs: Sequence[str | Path] | None) -> list[Path]:
    if capture_dirs is None:
        root = run_dir if run_dir.is_dir() else run_dir.parent
        return [root / "model_calls"]

    resolved = [Path(directory) for directory in capture_dirs]
    missing = [directory for directory in resolved if not directory.is_dir()]
    if missing:
        formatted = ", ".join(str(directory) for directory in missing)
        raise ValueError(f"Capture directory not found: {formatted}")
    return resolved


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
    capture_dirs: Sequence[str | Path] | None = None,
) -> HealthCheckResult:
    path = Path(run_dir)
    rollout_paths = _discover_rollouts(path)
    selected_capture_dirs = _standalone_capture_dirs(path, capture_dirs)
    result = run_health_checks(
        rollout_paths,
        output_dir=path if path.is_dir() else path.parent,
        capture_dirs=selected_capture_dirs,
        workers=workers,
        capture_enabled=True if any(directory.is_dir() for directory in selected_capture_dirs) else None,
        ignored_checks=ignored_checks,
    )
    print(format_health_report(result))
    return result
