#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit and transactionally quarantine poisoned GDPVal judge caches.

The pinned GDPVal runtime resumes a failed judgement only when the corresponding
main-output row is absent, a retryable row is present in the failures sidecar,
and Stirrup cannot replay the bad cached ``/verify`` response.  This helper
changes those three artifacts as one recoverable transaction.  It never removes
the cached policy deliverable.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "gdpval-checkpoint-e2e-judge-state-v1"
FILE_MODE = 0o400
PRIVATE_MODE = 0o600
DIRECTORY_MODE = 0o700
DEFAULT_MAX_ATTEMPTS = 3
TRIALS_PER_MATCHUP = 4
EXPECTED_STAGE_ROW_COUNTS = {0: 45, 1: 220}
EXPECTED_FINAL_STAGE_INDEX = 1
EXPECTED_JUDGE_MODELS = {
    "gpt-5.5": "openai/openai/gpt-5.5",
    "gemini-3.1-pro": "gcp/google/gemini-3.1-pro-preview",
    "claude-opus-4.8": "aws/anthropic/bedrock-claude-opus-4-8",
}
TASK_INDEX_KEY = "_ng_task_index"
ROLLOUT_INDEX_KEY = "_ng_rollout_index"
ATTEMPT_INDEX_KEY = "_ng_attempt_index"
SANITIZER_KEY = "_ng_judge_state_sanitizer"
CACHE_NAME = re.compile(r"^repeat_(?P<repeat>[0-9]+)_verify_response(?:_(?P<key>[0-9a-f]{12,16}))?\.json$")


class JudgeStateError(ValueError):
    """Raised when judge state cannot be changed without guessing."""


@dataclass(frozen=True, order=True)
class RowIdentity:
    stage_index: int
    task_index: int
    rollout_index: int

    def as_dict(self) -> dict[str, int]:
        return {
            "stage_index": self.stage_index,
            TASK_INDEX_KEY: self.task_index,
            ROLLOUT_INDEX_KEY: self.rollout_index,
        }


@dataclass(frozen=True)
class CacheFinding:
    path: Path
    relative_path: Path
    sha256: str
    identity: RowIdentity
    task_id: str
    deliverables_dir: Path
    reasons: tuple[str, ...]
    row: dict[str, Any]


def _fail(message: str) -> None:
    raise JudgeStateError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, *, mode: int = PRIVATE_MODE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _publish_immutable(path: Path, payload: bytes, *, mode: int = FILE_MODE) -> None:
    if path.exists() or path.is_symlink():
        _require_regular(path)
        if path.read_bytes() != payload:
            _fail(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _require_regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"not a regular non-symlink file: {path}")


def _resolved_regular(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        _fail(f"{label} must be absolute: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a resolved regular non-symlink file: {path}")
    return resolved


def _resolved_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        _fail(f"{label} must be absolute: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_dir():
        _fail(f"{label} must be a resolved non-symlink directory: {path}")
    return resolved


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JudgeStateError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} is not a JSON object")
    return value


def _jsonl_rows(path: Path, *, label: str) -> list[tuple[bytes, dict[str, Any]]]:
    rows: list[tuple[bytes, dict[str, Any]]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            rows.append((raw, _json_object(raw, label=f"{label} {path}:{line_number}")))
    return rows


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} is not an integer >= {minimum}")
    return value


def _row_identity(row: Mapping[str, Any], *, label: str) -> RowIdentity:
    if "stage_index" not in row or TASK_INDEX_KEY not in row or ROLLOUT_INDEX_KEY not in row:
        _fail(f"{label} has no complete stage/task/rollout identity")
    return RowIdentity(
        _exact_int(row["stage_index"], label=f"{label} stage_index"),
        _exact_int(row[TASK_INDEX_KEY], label=f"{label} {TASK_INDEX_KEY}"),
        _exact_int(row[ROLLOUT_INDEX_KEY], label=f"{label} {ROLLOUT_INDEX_KEY}"),
    )


def _journal_fingerprint(path: Path) -> str:
    fingerprints: set[str] = set()
    for _, row in _jsonl_rows(path, label="multistage journal"):
        fingerprint = row.get("fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            _fail(f"journal row has an invalid fingerprint: {path}")
        fingerprints.add(fingerprint)
    if len(fingerprints) != 1:
        _fail(f"journal must contain exactly one fingerprint, found {sorted(fingerprints)}")
    return next(iter(fingerprints))


def _completed_stage_assignments(path: Path) -> dict[tuple[int, str], str]:
    """Return frozen task/reference pairs for journal stages with an outcome.

    Early v1.2.x runtime overlays namespace-keyed Stirrup's cache filename but
    accidentally let GDPVal's request model drop the namespace from the cached
    payload and tagged output. Such rows are safe to retain only when their
    task/reference pair belongs to an already-completed frozen journal plan.
    """

    plans: dict[int, dict[str, Any]] = {}
    completed: set[int] = set()
    for _, row in _jsonl_rows(path, label="multistage journal"):
        stage = row.get("stage_index")
        status = row.get("status")
        if type(stage) is not int:
            continue
        if status == "planned":
            plans[stage] = row
        elif status in {"complete", "partial_complete"}:
            completed.add(stage)

    result: dict[tuple[int, str], str] = {}
    for stage in completed:
        plan = plans.get(stage)
        if plan is None:
            _fail(f"completed Stage{stage} has no frozen plan")
        assignments = plan.get("task_reference_ids")
        if not isinstance(assignments, dict) or not all(
            isinstance(task_id, str) and isinstance(reference_id, str) for task_id, reference_id in assignments.items()
        ):
            _fail(f"completed Stage{stage} has malformed frozen assignments")
        for task_id, reference_id in assignments.items():
            result[(stage, task_id)] = reference_id
    return result


def _verify_cache_key(reference_ids: Any, namespace: str) -> str:
    if reference_ids is not None:
        if not isinstance(reference_ids, list) or not all(isinstance(value, str) for value in reference_ids):
            _fail("active verify cache has malformed reference_ids")
        normalized_references: list[str] | None = sorted(set(reference_ids))
    else:
        normalized_references = None
    normalized = json.dumps(
        {"namespace": namespace, "reference_ids": normalized_references},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _comparison_defects(row: Mapping[str, Any], *, label: str, require_output_receipts: bool) -> tuple[str, ...]:
    """Return strict comparison defects for a main row or raw verify cache.

    This deliberately duplicates the row-local contract in ``campaign.py``.
    Keeping the repair scanner independent lets it run beside a pinned runtime
    checkout while still ensuring final validation cannot discover poisoned
    trial evidence only after every retry epoch has finished.  Main rows also
    require the two receipts added by the multistage output wrapper; raw verify
    caches are validated before that wrapper-only enrichment.
    """

    del label  # Retained for useful call-site context and future diagnostics.
    reasons: set[str] = set()

    def exact_nonnegative_int(value: Any) -> bool:
        return type(value) is int and value >= 0

    def exact_int(value: Any, expected: int) -> bool:
        return type(value) is int and value == expected

    def finite_number(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))

    def error_is_present(value: Any) -> bool:
        # Match campaign.py's identity checks: 0 and {} are errors, despite
        # both being falsey and 0 comparing equal to False.
        return value is not None and value is not False

    if row.get("verify_mode") != "comparison":
        reasons.add("verify_mode_mismatch")
    stage = row.get("stage_index")
    if type(stage) is not int or stage not in EXPECTED_STAGE_ROW_COUNTS:
        reasons.add("invalid_stage_index")
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        reasons.add("invalid_task_id")
    invalid = row.get("invalid_judge_response")
    if invalid is not None and invalid is not False:
        reasons.add("invalid_judge_response")
    if row.get("_ng_failure_class") is not None or row.get("_ng_no_persist"):
        reasons.add("failure_marker")
    if error_is_present(row.get("error")):
        reasons.add("top_level_error")
    response = row.get("response")
    if isinstance(response, dict) and response.get("error") is not None:
        reasons.add("response_error")

    # These receipts are attached by the multistage output wrapper after the
    # resources server returns.  They are required on the persisted main row,
    # but are intentionally absent from Stirrup's cached raw /verify response.
    if require_output_receipts:
        if not exact_int(row.get("expected_final_stage_index"), EXPECTED_FINAL_STAGE_INDEX):
            reasons.add("final_stage_receipt_mismatch")
        if stage in EXPECTED_STAGE_ROW_COUNTS and not exact_int(
            row.get("expected_stage_row_count"), EXPECTED_STAGE_ROW_COUNTS[stage]
        ):
            reasons.add("stage_row_count_receipt_mismatch")

    references = row.get("reference_ids")
    valid_reference = isinstance(references, list) and len(references) == 1 and isinstance(references[0], str)
    if not valid_reference:
        reasons.add("reference_assignment_malformed")
        reference_id: str | None = None
    else:
        reference_id = references[0]

    reference_counts: Mapping[str, Any] | None = None
    per_reference = row.get("per_reference")
    if (
        reference_id is None
        or not isinstance(per_reference, dict)
        or set(per_reference) != {reference_id}
        or not isinstance(per_reference.get(reference_id), dict)
    ):
        reasons.add("top_level_per_reference_malformed")
    else:
        reference_counts = per_reference[reference_id]
        reference_votes = 0
        for field in ("wins", "losses", "ties"):
            value = reference_counts.get(field)
            if not exact_nonnegative_int(value):
                reasons.add("top_level_per_reference_tally_malformed")
            else:
                reference_votes += value
        if reference_votes != TRIALS_PER_MATCHUP:
            reasons.add("top_level_per_reference_non_four_trials")
        if not finite_number(reference_counts.get("reference_elo")):
            reasons.add("reference_elo_malformed")

    judge = row.get("judge_response")
    if not isinstance(judge, dict):
        return tuple(sorted(reasons | {"judge_response_missing"}))
    if error_is_present(judge.get("error")) or error_is_present(judge.get("scoring_error")):
        reasons.add("judge_error")
    if judge.get("ref_errors") != {}:
        reasons.add("reference_errors")
    if not exact_int(judge.get("total_judged"), TRIALS_PER_MATCHUP):
        reasons.add("non_four_total_judged")
    if not exact_int(judge.get("total_invalid"), 0):
        reasons.add("invalid_total")

    av_routed = judge.get("av_routed")
    if type(av_routed) is not bool:
        reasons.add("av_routed_receipt_missing")
        av_routed = False
    expected_panel_names = {"gemini-3.1-pro"} if av_routed else set(EXPECTED_JUDGE_MODELS)
    panel = judge.get("judge_panel")
    if not isinstance(panel, list) or not panel:
        reasons.add("judge_panel_malformed")
    else:
        panel_names: set[str] = set()
        for member in panel:
            if not isinstance(member, dict):
                reasons.add("judge_panel_malformed")
                continue
            name = member.get("name")
            if name not in EXPECTED_JUDGE_MODELS:
                reasons.add("judge_panel_unexpected_member")
                continue
            weight = member.get("weight")
            if member.get("model") != EXPECTED_JUDGE_MODELS[name] or not finite_number(weight) or float(weight) != 1.0:
                reasons.add("judge_panel_model_or_weight_mismatch")
            if name in panel_names:
                reasons.add("judge_panel_duplicate_member")
            panel_names.add(name)
        if panel_names != expected_panel_names:
            reasons.add("judge_panel_route_mismatch")

    top_per_judge = judge.get("per_judge")
    top_judge_trials = 0
    if not isinstance(top_per_judge, dict):
        reasons.add("top_level_per_judge_malformed")
    else:
        if av_routed and set(top_per_judge) != {"gemini-3.1-pro"}:
            reasons.add("av_route_non_gemini_tally")
        for name, counts_by_judge in top_per_judge.items():
            if name not in EXPECTED_JUDGE_MODELS or not isinstance(counts_by_judge, dict):
                reasons.add("top_level_per_judge_malformed")
                continue
            trials = counts_by_judge.get("trials")
            invalid_count = counts_by_judge.get("invalid_count")
            if not exact_nonnegative_int(trials):
                reasons.add("top_level_per_judge_malformed")
            else:
                top_judge_trials += trials
            if not exact_int(invalid_count, 0):
                reasons.add("top_level_per_judge_invalid_trials")
        if top_judge_trials != TRIALS_PER_MATCHUP:
            reasons.add("top_level_per_judge_non_four_trials")

    matchups = judge.get("per_ref_repeat")
    if not isinstance(matchups, list) or len(matchups) != 1 or not isinstance(matchups[0], dict):
        return tuple(sorted(reasons | {"no_single_matchup_trial_evidence"}))

    matchup = matchups[0]
    if matchup.get("ref_id") != reference_id or matchup.get("ref_repeat") != "repeat_0":
        reasons.add("matchup_reference_or_repeat_mismatch")

    trial_judges = matchup.get("trial_judges")
    if (
        not isinstance(trial_judges, list)
        or len(trial_judges) != TRIALS_PER_MATCHUP
        or any(name not in EXPECTED_JUDGE_MODELS for name in trial_judges)
    ):
        reasons.add("invalid_four_trial_judge_schedule")
    elif av_routed and trial_judges != ["gemini-3.1-pro"] * TRIALS_PER_MATCHUP:
        reasons.add("av_route_non_gemini_schedule")
    if not exact_int(matchup.get("invalid_count"), 0):
        reasons.add("matchup_invalid_trials")

    matchup_counts: list[int] = []
    for field in ("win_count_a", "win_count_b", "tie_count"):
        value = matchup.get(field)
        if not exact_nonnegative_int(value):
            reasons.add("matchup_tally_malformed")
        else:
            matchup_counts.append(value)
    matchup_tally_valid = len(matchup_counts) == 3
    if not matchup_tally_valid or sum(matchup_counts) != TRIALS_PER_MATCHUP:
        reasons.add("matchup_non_four_trials")
    if not exact_int(matchup.get("task_count"), TRIALS_PER_MATCHUP):
        reasons.add("matchup_task_count_mismatch")

    raw_responses = matchup.get("raw_responses")
    if raw_responses is not None and (not isinstance(raw_responses, list) or len(raw_responses) != TRIALS_PER_MATCHUP):
        reasons.add("matchup_non_four_raw_responses")

    matchup_per_judge = matchup.get("per_judge")
    matchup_judge_trials = matchup_judge_invalid = 0
    if not isinstance(matchup_per_judge, dict) or not matchup_per_judge:
        reasons.add("matchup_per_judge_missing")
    else:
        for name, counts_by_judge in matchup_per_judge.items():
            if name not in EXPECTED_JUDGE_MODELS or not isinstance(counts_by_judge, dict):
                reasons.add("matchup_per_judge_malformed")
                continue
            trials = counts_by_judge.get("trials")
            invalid_count = counts_by_judge.get("invalid_count")
            if not exact_nonnegative_int(trials) or not exact_nonnegative_int(invalid_count):
                reasons.add("matchup_per_judge_malformed")
                continue
            matchup_judge_trials += trials
            matchup_judge_invalid += invalid_count
        if matchup_judge_trials != TRIALS_PER_MATCHUP:
            reasons.add("matchup_per_judge_non_four_trials")
        if matchup_judge_invalid != 0:
            reasons.add("matchup_per_judge_invalid_trials")

    if judge.get("ref_repeat_count") != 1:
        reasons.add("matchup_count_mismatch")
    if matchup_tally_valid:
        matchup_losses, matchup_wins, matchup_ties = matchup_counts
        if (
            judge.get("total_wins") != matchup_wins
            or judge.get("total_losses") != matchup_losses
            or judge.get("total_ties") != matchup_ties
        ):
            reasons.add("matchup_tally_mismatch")

        outcome_values: list[int] = []
        for field in ("total_wins", "total_losses", "total_ties"):
            value = row.get(field, judge.get(field, 0))
            if not exact_nonnegative_int(value):
                reasons.add("top_level_tally_malformed")
            else:
                outcome_values.append(value)
        if len(outcome_values) != 3 or sum(outcome_values) != TRIALS_PER_MATCHUP:
            reasons.add("top_level_tally_non_four_trials")
        if len(outcome_values) == 3 and outcome_values != [matchup_wins, matchup_losses, matchup_ties]:
            reasons.add("top_level_tally_mismatch")
        if reference_counts is not None and len(outcome_values) == 3:
            expected_outcomes = [
                reference_counts.get("wins"),
                reference_counts.get("losses"),
                reference_counts.get("ties"),
            ]
            if outcome_values != expected_outcomes:
                reasons.add("top_level_tally_differs_from_per_reference")
    return tuple(sorted(reasons))


def _cache_findings(deliverables: Path, fingerprint: str) -> tuple[list[CacheFinding], int]:
    findings: list[CacheFinding] = []
    active_count = 0
    by_identity: dict[RowIdentity, Path] = {}
    for path in sorted(deliverables.rglob("repeat_*_verify_response*.json")):
        match = CACHE_NAME.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            _fail(f"verify cache is not a regular non-symlink file: {path}")
        payload = path.read_bytes()
        row = _json_object(payload, label=f"verify cache {path}")
        if row.get("verify_cache_namespace") != fingerprint:
            continue
        active_count += 1
        identity = _row_identity(row, label=f"verify cache {path}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            _fail(f"active verify cache has no task_id: {path}")
        relative = path.relative_to(deliverables)
        expected_parent = deliverables / f"task_{task_id}"
        if path.parent != expected_parent:
            _fail(f"active verify cache task path does not match task_id: {path}")
        if int(match.group("repeat")) != identity.rollout_index:
            _fail(f"active verify cache repeat does not match rollout identity: {path}")
        expected_name = f"repeat_{identity.rollout_index}_verify_response_"
        expected_name += _verify_cache_key(row.get("reference_ids"), fingerprint) + ".json"
        if path.name != expected_name:
            _fail(f"active verify cache filename does not match its namespace/reference set: {path}")
        raw_deliverables_dir = row.get("deliverables_dir")
        if not isinstance(raw_deliverables_dir, str):
            _fail(f"active verify cache has no deliverables_dir: {path}")
        cache_deliverables = Path(raw_deliverables_dir)
        expected_deliverables = expected_parent / f"repeat_{identity.rollout_index}"
        try:
            resolved_cache_deliverables = cache_deliverables.resolve(strict=True)
        except FileNotFoundError as exc:
            raise JudgeStateError(f"cached policy deliverable is missing for {path}: {cache_deliverables}") from exc
        if (
            not cache_deliverables.is_absolute()
            or cache_deliverables != resolved_cache_deliverables
            or cache_deliverables.is_symlink()
            or not cache_deliverables.is_dir()
            or cache_deliverables != expected_deliverables
        ):
            _fail(f"active verify cache points at an unexpected deliverables directory: {path}")
        if identity in by_identity:
            _fail(f"multiple active verify caches share {identity}: {by_identity[identity]} and {path}")
        by_identity[identity] = path
        reasons = _comparison_defects(
            row,
            label=f"verify cache {path}",
            require_output_receipts=False,
        )
        if reasons:
            findings.append(
                CacheFinding(
                    path=path,
                    relative_path=relative,
                    sha256=_sha256_bytes(payload),
                    identity=identity,
                    task_id=task_id,
                    deliverables_dir=cache_deliverables,
                    reasons=reasons,
                    row=row,
                )
            )
    return findings, active_count


def _failures_path(output: Path) -> Path:
    return output.with_name(output.stem + "_failures.jsonl")


def _analyze(
    *, output: Path, journal: Path, deliverables: Path, max_attempts: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = _resolved_regular(output, label="judge output")
    journal = _resolved_regular(journal, label="multistage journal")
    deliverables = _resolved_directory(deliverables, label="deliverables")
    fingerprint = _journal_fingerprint(journal)
    completed_assignments = _completed_stage_assignments(journal)
    findings, active_cache_count = _cache_findings(deliverables, fingerprint)
    finding_by_identity = {finding.identity: finding for finding in findings}

    output_rows = _jsonl_rows(output, label="judge output")
    output_by_identity: dict[RowIdentity, tuple[bytes, dict[str, Any]]] = {}
    invalid_output: set[RowIdentity] = set()
    kept_lines: list[bytes] = []
    removed_lines: list[bytes] = []
    for raw, row in output_rows:
        identity = _row_identity(row, label="judge output row")
        if identity in output_by_identity:
            _fail(f"judge output contains duplicate identity {identity}")
        output_by_identity[identity] = (raw, row)
        defects = _comparison_defects(
            row,
            label=f"judge output row {identity}",
            require_output_receipts=True,
        )
        namespace = row.get("verify_cache_namespace")
        if namespace != fingerprint:
            if namespace is not None:
                _fail("judge output contains a row from another verify-cache namespace")
            task_id = row.get("task_id")
            references = row.get("reference_ids")
            expected_reference = completed_assignments.get((identity.stage_index, task_id))
            if defects or expected_reference is None or references != [expected_reference]:
                _fail(
                    "missing verify-cache namespace is allowed only for valid evidence from a completed frozen stage"
                )
        if defects:
            invalid_output.add(identity)
        if identity in finding_by_identity:
            if not defects:
                _fail(f"invalid cache disagrees with a valid main-output row for {identity}")
            cache_task_id = finding_by_identity[identity].task_id
            if row.get("task_id") != cache_task_id:
                _fail(f"cache/main task_id mismatch for {identity}")
            removed_lines.append(raw)
        else:
            kept_lines.append(raw)

    missing_caches = invalid_output - set(finding_by_identity)
    if missing_caches:
        _fail(f"invalid main-output rows have no matching invalid active cache: {sorted(missing_caches)}")

    failures = _failures_path(output)
    failure_rows: list[tuple[bytes, dict[str, Any]]] = []
    if failures.exists() or failures.is_symlink():
        _require_regular(failures)
        failure_rows = _jsonl_rows(failures, label="judge failures sidecar")
    attempts: Counter[RowIdentity] = Counter()
    terminal: set[RowIdentity] = set()
    for _, row in failure_rows:
        if "stage_index" not in row or TASK_INDEX_KEY not in row or ROLLOUT_INDEX_KEY not in row:
            continue
        identity = _row_identity(row, label="judge failures sidecar row")
        attempts[identity] += 1
        failure_class = row.get("_ng_failure_class")
        if failure_class == "skipped" or (
            failure_class != "timeout_exceeded" and bool(row.get("_ng_failure_terminal"))
        ):
            terminal.add(identity)

    for finding in findings:
        if finding.identity in terminal:
            _fail(f"cannot make terminal sidecar identity retryable: {finding.identity}")
        if attempts[finding.identity] + 1 >= max_attempts:
            _fail(
                f"sanitizer sidecar record would max out {finding.identity}: "
                f"existing={attempts[finding.identity]} max={max_attempts}"
            )

    report = {
        "schema": SCHEMA,
        "status": "SANITIZATION_REQUIRED" if findings else "CLEAN",
        "fingerprint": fingerprint,
        "output": str(output),
        "journal": str(journal),
        "deliverables": str(deliverables),
        "active_cache_count": active_cache_count,
        "quarantine_count": len(findings),
        "remove_output_rows": len(removed_lines),
        "identities": [finding.identity.as_dict() for finding in findings],
        "reasons": {str(finding.relative_path): list(finding.reasons) for finding in findings},
    }
    internal = {
        "output_rows": output_rows,
        "output_after": b"".join(kept_lines),
        "removed_lines": removed_lines,
        "failure_rows": failure_rows,
        "failures": failures,
        "findings": findings,
        "fingerprint": fingerprint,
        "output": output,
        "journal": journal,
        "deliverables": deliverables,
    }
    return report, internal


def _sidecar_record(finding: CacheFinding, *, transaction_id: str, prior_attempts: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        **finding.identity.as_dict(),
        "task_id": finding.task_id,
        "reference_ids": finding.row.get("reference_ids"),
        "verify_cache_namespace": finding.row.get("verify_cache_namespace"),
        "deliverables_dir": str(finding.deliverables_dir),
        ATTEMPT_INDEX_KEY: prior_attempts,
        "_ng_failure_class": "judge_invalid",
        "reuse_cached_deliverable": True,
        "invalid_judge_response": True,
        "failure_reason": "quarantined invalid cached GDPVal judge evidence; retry judgement",
        SANITIZER_KEY: {
            "schema": SCHEMA,
            "transaction_id": transaction_id,
            "cache_sha256": finding.sha256,
            "reasons": list(finding.reasons),
        },
    }
    return record


def _with_appended_jsonl(existing: bytes, records: Iterable[Mapping[str, Any]]) -> bytes:
    payload = existing
    if payload and not payload.endswith(b"\n"):
        payload += b"\n"
    return payload + b"".join(_canonical_bytes(record) for record in records)


def _status_path(transaction: Path) -> Path:
    return transaction / "status"


def _set_status(transaction: Path, status: str) -> None:
    _atomic_write(_status_path(transaction), (status + "\n").encode())


def _read_status(transaction: Path) -> str:
    status_path = _status_path(transaction)
    _require_regular(status_path)
    return status_path.read_text(encoding="utf-8").strip()


def _load_plan(transaction: Path) -> dict[str, Any]:
    plan_path = transaction / "plan.json"
    _require_regular(plan_path)
    plan = _json_object(plan_path.read_bytes(), label=f"transaction plan {plan_path}")
    if plan.get("schema") != SCHEMA or plan.get("transaction_id") != transaction.name:
        _fail(f"invalid transaction plan identity: {plan_path}")
    return plan


def _write_prepared_transaction(*, state_root: Path, internal: Mapping[str, Any], max_attempts: int) -> Path:
    output: Path = internal["output"]
    failures: Path = internal["failures"]
    deliverables: Path = internal["deliverables"]
    findings: list[CacheFinding] = internal["findings"]
    output_before = output.read_bytes()
    sidecar_before = failures.read_bytes() if failures.exists() else b""
    output_before_sha256 = _sha256_bytes(output_before)
    sidecar_before_sha256 = _sha256_bytes(sidecar_before)
    attempts: Counter[RowIdentity] = Counter()
    for _, row in internal["failure_rows"]:
        if "stage_index" in row and TASK_INDEX_KEY in row and ROLLOUT_INDEX_KEY in row:
            attempts[_row_identity(row, label="judge failures sidecar row")] += 1

    transaction_seed = {
        "schema": SCHEMA,
        "fingerprint": internal["fingerprint"],
        "output_sha256": output_before_sha256,
        # Distinguish a later retry that reproduced byte-identical bad judge
        # evidence but consumed another sidecar attempt.
        "failures_sha256": sidecar_before_sha256,
        "caches": [(str(f.relative_path), f.sha256) for f in findings],
    }
    transaction_id = _sha256_bytes(_canonical_bytes(transaction_seed))[:24]
    records = [
        _sidecar_record(finding, transaction_id=transaction_id, prior_attempts=attempts[finding.identity])
        for finding in findings
    ]
    output_after: bytes = internal["output_after"]
    sidecar_after = _with_appended_jsonl(sidecar_before, records)
    transactions = state_root / "transactions"
    transactions.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)
    transaction = transactions / transaction_id
    if transaction.exists() or transaction.is_symlink():
        _fail(f"transaction path already exists unexpectedly: {transaction}")
    temporary = transactions / f".{transaction_id}.prepare.{os.getpid()}"
    temporary.mkdir(mode=DIRECTORY_MODE)
    try:
        quarantine = temporary / "quarantine"
        quarantine.mkdir(mode=DIRECTORY_MODE)
        for finding in findings:
            (quarantine / finding.relative_path).parent.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)

        output_backup = temporary / "output.before.jsonl"
        if _sha256_file(output) != output_before_sha256:
            _fail("judge output changed while the sanitizer transaction was being prepared")
        os.link(output, output_backup, follow_symlinks=False)
        if _sha256_file(output_backup) != output_before_sha256:
            _fail("judge output backup changed while the sanitizer transaction was being prepared")
        if failures.exists():
            if _sha256_file(failures) != sidecar_before_sha256:
                _fail("failures sidecar changed while the sanitizer transaction was being prepared")
            os.link(failures, temporary / "failures.before.jsonl", follow_symlinks=False)
            if _sha256_file(temporary / "failures.before.jsonl") != sidecar_before_sha256:
                _fail("failures backup changed while the sanitizer transaction was being prepared")
        _atomic_write(temporary / "output.after.jsonl", output_after)
        _atomic_write(temporary / "failures.after.jsonl", sidecar_after)
        plan = {
            "schema": SCHEMA,
            "transaction_id": transaction_id,
            "fingerprint": internal["fingerprint"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "max_attempts": max_attempts,
            "output": {
                "path": str(output),
                "before_sha256": output_before_sha256,
                "after_sha256": _sha256_bytes(output_after),
                "removed_rows": len(internal["removed_lines"]),
            },
            "failures": {
                "path": str(failures),
                "before_exists": failures.exists(),
                "before_sha256": sidecar_before_sha256,
                "after_sha256": _sha256_bytes(sidecar_after),
                "appended_rows": len(records),
                "records": records,
            },
            "deliverables": str(deliverables),
            "caches": [
                {
                    "original": str(finding.path),
                    "quarantine_relative": str(finding.relative_path),
                    "sha256": finding.sha256,
                    "identity": finding.identity.as_dict(),
                    "reasons": list(finding.reasons),
                }
                for finding in findings
            ],
        }
        _publish_immutable(temporary / "plan.json", _canonical_bytes(plan))
        _set_status(temporary, "PREPARED")
        _fsync_directory(temporary)
        os.replace(temporary, transaction)
        _fsync_directory(transactions)
    finally:
        if temporary.exists():
            # A failure before publication has not changed any live judge state.
            for path in sorted(temporary.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_dir() and not path.is_symlink():
                    path.rmdir()
                else:
                    path.unlink()
            temporary.rmdir()
    return transaction


def _current_digest(path: Path, *, absent_digest: str | None = None) -> str | None:
    if not path.exists() and not path.is_symlink():
        return absent_digest
    _require_regular(path)
    return _sha256_file(path)


def _ensure_output_publishable(transaction: Path, plan: Mapping[str, Any]) -> None:
    destination = Path(plan["output"]["path"])
    before = plan["output"]["before_sha256"]
    after = plan["output"]["after_sha256"]
    current = _current_digest(destination)
    if current == after:
        return
    if current != before:
        _fail(f"output drift while recovering transaction {transaction.name}: {destination}")
    prepared = transaction / "output.after.jsonl"
    _require_regular(prepared)
    if _sha256_file(prepared) != after:
        _fail(f"prepared output digest mismatch in transaction {transaction.name}")
    os.replace(prepared, destination)
    _fsync_directory(destination.parent)


def _apply_transaction(transaction: Path) -> Path:
    plan = _load_plan(transaction)
    failures = Path(plan["failures"]["path"])

    before_failures = plan["failures"]["before_sha256"]
    if not plan["failures"]["before_exists"] and not failures.exists():
        # The absent sidecar's logical digest is SHA-256(empty).
        current = before_failures
    else:
        current = _current_digest(failures)
    if current not in {before_failures, plan["failures"]["after_sha256"]}:
        _fail(f"failures sidecar drift while recovering transaction {transaction.name}: {failures}")
    if current != plan["failures"]["after_sha256"]:
        prepared = transaction / "failures.after.jsonl"
        _require_regular(prepared)
        if _sha256_file(prepared) != plan["failures"]["after_sha256"]:
            _fail(f"prepared failures digest mismatch in transaction {transaction.name}")
        os.replace(prepared, failures)
        _fsync_directory(failures.parent)
    _set_status(transaction, "SIDECAR_PUBLISHED")

    quarantine_root = transaction / "quarantine"
    for cache in plan["caches"]:
        original = Path(cache["original"])
        quarantine = quarantine_root / cache["quarantine_relative"]
        original_exists = original.exists() or original.is_symlink()
        quarantine_exists = quarantine.exists() or quarantine.is_symlink()
        if original_exists and quarantine_exists:
            _fail(f"both live and quarantined cache exist during recovery: {original}")
        if not original_exists and not quarantine_exists:
            _fail(f"cache disappeared during recovery: {original}")
        if original_exists:
            _require_regular(original)
            if _sha256_file(original) != cache["sha256"]:
                _fail(f"verify cache drift before quarantine: {original}")
            quarantine.parent.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)
            os.replace(original, quarantine)
            _fsync_directory(original.parent)
            _fsync_directory(quarantine.parent)
        else:
            _require_regular(quarantine)
            if _sha256_file(quarantine) != cache["sha256"]:
                _fail(f"quarantined cache digest mismatch: {quarantine}")
    _set_status(transaction, "CACHES_QUARANTINED")

    _ensure_output_publishable(transaction, plan)
    _set_status(transaction, "OUTPUT_PUBLISHED")

    receipt = {
        "schema": SCHEMA,
        "status": "COMMITTED",
        "transaction_id": transaction.name,
        "fingerprint": plan["fingerprint"],
        # Frozen in the immutable plan so a crash after receipt publication but
        # before the COMPLETE status can replay without receipt drift.
        "completed_at": plan["created_at"],
        "output": {**plan["output"], "backup": str(transaction / "output.before.jsonl")},
        "failures": {key: value for key, value in plan["failures"].items() if key != "records"}
        | {"backup": str(transaction / "failures.before.jsonl") if plan["failures"]["before_exists"] else None},
        "deliverables": plan["deliverables"],
        "quarantined_caches": [
            {
                **cache,
                "quarantine": str(quarantine_root / cache["quarantine_relative"]),
            }
            for cache in plan["caches"]
        ],
    }
    receipt_path = transaction / "receipt.json"
    receipt_payload = _canonical_bytes(receipt)
    _publish_immutable(receipt_path, receipt_payload)
    digest_payload = _canonical_bytes(
        {
            "schema": f"{SCHEMA}-sha256",
            "path": str(receipt_path),
            "bytes": len(receipt_payload),
            "sha256": _sha256_bytes(receipt_payload),
        }
    )
    _publish_immutable(transaction / "receipt.json.sha256.json", digest_payload)
    _set_status(transaction, "COMPLETE")
    return receipt_path


def _state_root_for(output: Path, requested: Path | None) -> Path:
    state_root = requested or (output.parent / ".checkpoint_e2e_judge_state")
    if not state_root.is_absolute():
        _fail(f"state root must be absolute: {state_root}")
    if state_root.exists() or state_root.is_symlink():
        resolved = state_root.resolve(strict=True)
        if resolved != state_root or state_root.is_symlink() or not state_root.is_dir():
            _fail(f"state root must be a resolved non-symlink directory: {state_root}")
    else:
        state_root.mkdir(parents=True, mode=DIRECTORY_MODE)
        resolved = state_root.resolve(strict=True)
        if resolved != state_root:
            _fail(f"state root must already be resolved: {state_root} -> {resolved}")
    if state_root.stat().st_dev != output.stat().st_dev:
        _fail("state root and judge output must be on the same filesystem for atomic publication")
    return state_root


def _pending_transactions(state_root: Path) -> list[Path]:
    transactions = state_root / "transactions"
    if not transactions.exists():
        return []
    if transactions.is_symlink() or not transactions.is_dir():
        _fail(f"invalid transactions directory: {transactions}")
    pending: list[Path] = []
    for transaction in sorted(transactions.iterdir()):
        if transaction.name.startswith("."):
            continue
        if transaction.is_symlink() or not transaction.is_dir():
            _fail(f"invalid transaction entry: {transaction}")
        if _read_status(transaction) != "COMPLETE":
            pending.append(transaction)
    return pending


def audit(*, output: Path, journal: Path, deliverables: Path, max_attempts: int) -> dict[str, Any]:
    report, _ = _analyze(
        output=output,
        journal=journal,
        deliverables=deliverables,
        max_attempts=max_attempts,
    )
    return report


def sanitize(
    *,
    output: Path,
    journal: Path,
    deliverables: Path,
    state_root: Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    if max_attempts < 2:
        _fail("max_attempts must be >= 2")
    output = _resolved_regular(output, label="judge output")
    journal = _resolved_regular(journal, label="multistage journal")
    deliverables = _resolved_directory(deliverables, label="deliverables")
    root = _state_root_for(output, state_root)
    lock_path = root / "sanitize.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        PRIVATE_MODE,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JudgeStateError(f"another judge-state sanitizer owns {root}") from exc
        pending = _pending_transactions(root)
        if len(pending) > 1:
            _fail(f"multiple incomplete judge-state transactions: {pending}")
        if pending:
            plan = _load_plan(pending[0])
            if (
                Path(plan["output"]["path"]) != output
                or Path(plan["deliverables"]) != deliverables
                or plan["fingerprint"] != _journal_fingerprint(journal)
            ):
                _fail(f"pending transaction does not belong to the requested judge run: {pending[0]}")
            receipt = _apply_transaction(pending[0])
            return {
                "schema": SCHEMA,
                "status": "RECOVERED",
                "transaction_id": pending[0].name,
                "receipt": str(receipt),
            }

        report, internal = _analyze(
            output=output,
            journal=journal,
            deliverables=deliverables,
            max_attempts=max_attempts,
        )
        if not internal["findings"]:
            return report
        # Keep quarantine outside the tree being scanned for live caches.
        try:
            root.relative_to(deliverables)
        except ValueError:
            pass
        else:
            _fail("state root must not be inside the live deliverables tree")
        for finding in internal["findings"]:
            if finding.path.stat().st_dev != root.stat().st_dev:
                _fail(f"verify cache is on another filesystem and cannot be atomically quarantined: {finding.path}")
        transaction = _write_prepared_transaction(
            state_root=root,
            internal=internal,
            max_attempts=max_attempts,
        )
        receipt = _apply_transaction(transaction)
        return {
            **report,
            "status": "SANITIZED",
            "transaction_id": transaction.name,
            "receipt": str(receipt),
        }
    finally:
        os.close(lock_descriptor)


def _default_journal(output: Path) -> Path:
    return output.with_name(output.stem + "_multistage_state.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "sanitize"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--journal", type=Path)
        command.add_argument("--deliverables", type=Path, required=True)
        command.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
        if name == "sanitize":
            command.add_argument("--state-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().absolute()
    journal = (args.journal or _default_journal(output)).expanduser().absolute()
    deliverables = args.deliverables.expanduser().absolute()
    try:
        if args.command == "audit":
            report = audit(
                output=output,
                journal=journal,
                deliverables=deliverables,
                max_attempts=args.max_attempts,
            )
        else:
            state_root = args.state_root.expanduser().absolute() if args.state_root else None
            report = sanitize(
                output=output,
                journal=journal,
                deliverables=deliverables,
                state_root=state_root,
                max_attempts=args.max_attempts,
            )
    except (JudgeStateError, OSError) as exc:
        print(f"CHECKPOINT_E2E_JUDGE_STATE_FAIL: {exc}", file=sys.stderr)
        return 64
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
