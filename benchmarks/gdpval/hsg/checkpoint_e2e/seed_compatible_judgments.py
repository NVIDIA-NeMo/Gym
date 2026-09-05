#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed exact task/reference judgment rows from a compatible completed campaign.

The target campaign keeps its own calibration rows and frozen Stage-1 plan. Only
successful Stage-1 rows whose task/reference assignment exactly matches that
plan are eligible. Operational receipt fields are rebound to the target run;
the underlying judge response and vote evidence remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_JUDGE_MODELS = {
    "gpt-5.5": "openai/openai/gpt-5.5",
    "gemini-3.1-pro": "gcp/google/gemini-3.1-pro-preview",
    "claude-opus-4.8": "aws/anthropic/bedrock-claude-opus-4-8",
}


class SeedError(RuntimeError):
    """Raised when exact row compatibility cannot be proven."""


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise SeedError(f"missing file: {path}")
        return []
    if not path.is_file() or path.is_symlink():
        raise SeedError(f"expected regular non-symlink file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SeedError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SeedError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)


def _latest_stage1_plan(journal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    planned: dict[str, Any] | None = None
    completed = False
    for row in journal_rows:
        if row.get("stage_index") != 1:
            continue
        if row.get("status") == "planned":
            planned = row
        elif row.get("status") == "complete":
            completed = True
    if planned is None:
        raise SeedError("target journal has no Stage-1 plan")
    if completed:
        raise SeedError("target Stage 1 is already complete")
    task_ids = planned.get("task_ids")
    assignments = planned.get("task_reference_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 220 or len(set(task_ids)) != 220:
        raise SeedError("target Stage-1 plan must contain exactly 220 unique tasks")
    if not isinstance(assignments, dict) or set(assignments) != set(task_ids):
        raise SeedError("target Stage-1 assignment map does not match its task list")
    if not all(isinstance(task_id, str) and isinstance(assignments[task_id], str) for task_id in task_ids):
        raise SeedError("target Stage-1 plan contains malformed task/reference IDs")
    fingerprint = planned.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise SeedError("target Stage-1 plan has no valid fingerprint")
    return planned


def _target_task_indices(
    preprocessed_rows: list[dict[str, Any]],
    planned_task_ids: list[str],
) -> dict[str, int]:
    """Bind task IDs to Gym's input-row indices, not multistage plan order."""

    if len(preprocessed_rows) != 220:
        raise SeedError("target preprocessed dataset must contain exactly 220 rows")
    indices: dict[str, int] = {}
    for index, row in enumerate(preprocessed_rows):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise SeedError(f"target preprocessed row {index} has no valid task ID")
        if task_id in indices:
            raise SeedError(f"target preprocessed dataset contains duplicate task {task_id}")
        indices[task_id] = index
    planned = set(planned_task_ids)
    if set(indices) != planned:
        missing = sorted(planned - set(indices))
        extra = sorted(set(indices) - planned)
        raise SeedError(
            "target preprocessed task identities do not match the frozen Stage-1 plan: "
            f"missing={missing} extra={extra}"
        )
    return indices


def _is_valid_source_row(row: dict[str, Any]) -> bool:
    if row.get("stage_index") != 1:
        return False
    if row.get("_ng_failure_class") is not None or row.get("_ng_no_persist"):
        return False
    if row.get("invalid_judge_response") not in (None, False):
        return False
    if row.get("error") not in (None, False):
        return False
    response = row.get("response")
    if isinstance(response, dict) and response.get("error") is not None:
        return False
    judge = row.get("judge_response")
    return bool(
        isinstance(judge, dict)
        and judge.get("error") in (None, False)
        and judge.get("scoring_error") in (None, False)
        and judge.get("ref_errors") == {}
        and type(judge.get("total_judged")) is int
        and judge.get("total_judged") == 4
        and type(judge.get("total_invalid")) is int
        and judge.get("total_invalid") == 0
    )


def _rebind_row(
    row: dict[str, Any],
    *,
    target_task_index: int,
    target_reference: str,
    target_namespace: str,
    target_deliverables: Path,
) -> dict[str, Any]:
    rebound = dict(row)
    task_id = str(rebound["task_id"])
    rollout_index = rebound.get("_ng_rollout_index")
    if type(rollout_index) is not int or rollout_index != 0:
        raise SeedError(f"source task {task_id} does not have rollout index 0")
    references = rebound.get("reference_ids")
    if references != [target_reference]:
        raise SeedError(f"source task {task_id} reference changed during rebind")
    deliverables_dir = target_deliverables / f"task_{task_id}" / "repeat_0"
    if not deliverables_dir.is_dir() or deliverables_dir.is_symlink():
        raise SeedError(f"target deliverables directory is unusable: {deliverables_dir}")
    rebound["deliverables_dir"] = str(deliverables_dir)
    rebound["_ng_task_index"] = target_task_index
    rebound["expected_final_stage_index"] = 1
    rebound["expected_stage_row_count"] = 220
    rebound["verify_cache_namespace"] = target_namespace
    rebound["_ng_attempt_index"] = None
    rebound["invalid_judge_retryable"] = None
    judge = rebound.get("judge_response")
    if not isinstance(judge, dict):
        raise SeedError(f"source task {task_id} has no judge response")
    panel = judge.get("judge_panel")
    if not isinstance(panel, list) or not panel:
        raise SeedError(f"source task {task_id} has no judge-panel receipt")
    rebound_panel: list[dict[str, Any]] = []
    for raw_member in panel:
        if not isinstance(raw_member, dict):
            raise SeedError(f"source task {task_id} has a malformed judge-panel member")
        name = raw_member.get("name")
        if name not in EXPECTED_JUDGE_MODELS or raw_member.get("weight") != 1.0:
            raise SeedError(f"source task {task_id} has an unexpected judge-panel member")
        source_model = raw_member.get("model")
        if source_model not in ("judge", EXPECTED_JUDGE_MODELS[name]):
            raise SeedError(f"source task {task_id} has an unexpected judge model alias")
        member = dict(raw_member)
        member["model"] = EXPECTED_JUDGE_MODELS[name]
        rebound_panel.append(member)
    judge = dict(judge)
    judge["judge_panel"] = rebound_panel
    rebound["judge_response"] = judge
    return rebound


def _reindex_target_row(
    row: dict[str, Any],
    *,
    target_task_index: int,
    target_reference: str,
    target_namespace: str,
    target_deliverables: Path,
) -> dict[str, Any]:
    """Repair only the dispatcher index on already target-bound evidence."""

    task_id = str(row["task_id"])
    rollout_index = row.get("_ng_rollout_index")
    if type(rollout_index) is not int or rollout_index != 0:
        raise SeedError(f"target task {task_id} does not have rollout index 0")
    if row.get("reference_ids") != [target_reference]:
        raise SeedError(f"target task {task_id} conflicts with its frozen reference assignment")
    if row.get("verify_cache_namespace") != target_namespace:
        raise SeedError(f"target task {task_id} is bound to a different fingerprint namespace")
    expected_deliverables = target_deliverables / f"task_{task_id}" / "repeat_0"
    if not expected_deliverables.is_dir() or expected_deliverables.is_symlink():
        raise SeedError(f"target deliverables directory is unusable: {expected_deliverables}")
    try:
        actual_deliverables = Path(str(row.get("deliverables_dir"))).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SeedError(f"target task {task_id} has an unusable deliverables receipt") from exc
    if actual_deliverables != expected_deliverables.resolve(strict=True):
        raise SeedError(f"target task {task_id} points outside its target deliverables directory")
    rebound = dict(row)
    rebound["_ng_task_index"] = target_task_index
    return rebound


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.seed-{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-judge", type=Path, required=True)
    parser.add_argument("--source-judge", type=Path, required=True)
    parser.add_argument("--target-deliverables", type=Path, required=True)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    target = args.target_judge.expanduser().resolve(strict=True)
    source = args.source_judge.expanduser().resolve(strict=True)
    deliverables = args.target_deliverables.expanduser().resolve(strict=True)
    if target == source:
        raise SeedError("source and target judge directories must differ")

    target_output = target / "gdpval_aav2.jsonl"
    target_failures = target / "gdpval_aav2_failures.jsonl"
    target_journal = target / "gdpval_aav2_multistage_state.jsonl"
    target_preprocessed = target / "preprocessed_datasets" / "benchmark.jsonl"
    source_output = source / "gdpval_aav2.jsonl"

    target_rows = _read_jsonl(target_output)
    source_rows = _read_jsonl(source_output)
    target_failures_rows = _read_jsonl(target_failures, required=False)
    journal_rows = _read_jsonl(target_journal)
    preprocessed_rows = _read_jsonl(target_preprocessed)
    plan = _latest_stage1_plan(journal_rows)
    task_ids = list(plan["task_ids"])
    assignments = dict(plan["task_reference_ids"])
    namespace = str(plan["fingerprint"])
    task_indices = _target_task_indices(preprocessed_rows, task_ids)

    target_stage0 = [row for row in target_rows if row.get("stage_index") == 0]
    for row in target_stage0:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in task_indices:
            raise SeedError("target contains a Stage-0 row outside its preprocessed dataset")
        if row.get("_ng_task_index") != task_indices[task_id]:
            raise SeedError(f"target Stage-0 task {task_id} has a stale dispatcher index")
    target_stage1: dict[str, dict[str, Any]] = {}
    reindexed: list[str] = []
    for row in target_rows:
        if row.get("stage_index") != 1 or not _is_valid_source_row(row):
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in assignments:
            raise SeedError("target contains a Stage-1 row outside its frozen plan")
        if row.get("reference_ids") != [assignments[task_id]]:
            raise SeedError(f"target task {task_id} conflicts with its frozen reference assignment")
        if task_id in target_stage1:
            raise SeedError(f"target contains duplicate valid Stage-1 task {task_id}")
        normalized = _reindex_target_row(
            row,
            target_task_index=task_indices[task_id],
            target_reference=assignments[task_id],
            target_namespace=namespace,
            target_deliverables=deliverables,
        )
        target_stage1[task_id] = normalized
        if normalized["_ng_task_index"] != row.get("_ng_task_index"):
            reindexed.append(task_id)
    if len(target_stage1) != len(
        [row for row in target_rows if row.get("stage_index") == 1 and _is_valid_source_row(row)]
    ):
        raise SeedError("target contains duplicate valid Stage-1 task rows")

    source_candidates: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        if not _is_valid_source_row(row):
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in assignments:
            continue
        if row.get("reference_ids") != [assignments[task_id]]:
            continue
        if task_id in source_candidates:
            raise SeedError(f"source contains duplicate compatible Stage-1 task {task_id}")
        source_candidates[task_id] = row

    merged_stage1 = dict(target_stage1)
    imported: list[str] = []
    for task_id in task_ids:
        if task_id in merged_stage1 or task_id not in source_candidates:
            continue
        merged_stage1[task_id] = _rebind_row(
            source_candidates[task_id],
            target_task_index=task_indices[task_id],
            target_reference=assignments[task_id],
            target_namespace=namespace,
            target_deliverables=deliverables,
        )
        imported.append(task_id)

    for task_id, row in merged_stage1.items():
        if row.get("_ng_task_index") != task_indices[task_id]:
            raise SeedError(f"merged target task {task_id} has a stale dispatcher index")
    merged_indices = [row["_ng_task_index"] for row in merged_stage1.values()]
    if len(merged_indices) != len(set(merged_indices)):
        raise SeedError("merged Stage-1 rows collide on target dispatcher indices")

    merged_rows = target_stage0 + [merged_stage1[task_id] for task_id in task_ids if task_id in merged_stage1]
    preserved_failures = [row for row in target_failures_rows if row.get("stage_index") == 0]
    has_changes = bool(imported or reindexed)
    summary: dict[str, Any] = {
        "status": "READY" if has_changes else "NO_CHANGES",
        "target_judge": str(target),
        "source_judge": str(source),
        "target_preprocessed": str(target_preprocessed),
        "target_preprocessed_sha256": _sha256(target_preprocessed),
        "target_preprocessed_rows": len(preprocessed_rows),
        "target_stage0_rows": len(target_stage0),
        "target_stage1_rows_before": len(target_stage1),
        "source_compatible_rows": len(source_candidates),
        "imported_stage1_rows": len(imported),
        "reindexed_existing_stage1_rows": len(reindexed),
        "stage1_rows_after": len(merged_stage1),
        "stage1_rows_remaining": 220 - len(merged_stage1),
        "preserved_stage0_failures": len(preserved_failures),
        "plan_fingerprint": namespace,
        "imported_task_ids": imported,
        "reindexed_task_ids": reindexed,
        "applied": False,
    }
    if not args.apply:
        if args.preview_output is not None:
            preview = args.preview_output.expanduser().resolve(strict=False)
            if not preview.parent.is_dir():
                raise SeedError(f"preview parent directory does not exist: {preview.parent}")
            _atomic_write(preview, _canonical_bytes(merged_rows))
            summary["preview_output"] = str(preview)
            summary["preview_output_sha256"] = _sha256(preview)
        print(json.dumps(summary, sort_keys=True))
        return 0
    if not has_changes:
        raise SeedError("no compatible rows need importing or dispatcher-index repair")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = target / f"seed_backup_{stamp}"
    backup.mkdir(mode=0o700)
    for path in (target_output, target_failures, target_journal):
        if path.exists():
            shutil.copy2(path, backup / path.name)
            os.chmod(backup / path.name, 0o400)

    pre_output_sha = _sha256(target_output)
    source_output_sha = _sha256(source_output)
    _atomic_write(target_output, _canonical_bytes(merged_rows))
    _atomic_write(target_failures, _canonical_bytes(preserved_failures))
    summary.update(
        {
            "applied": True,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "backup_dir": str(backup),
            "target_output_sha256_before": pre_output_sha,
            "target_output_sha256_after": _sha256(target_output),
            "source_output_sha256": source_output_sha,
            "target_journal_sha256": _sha256(target_journal),
        }
    )
    receipt = target / f"seed_receipt_{stamp}.json"
    _atomic_write(receipt, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    os.chmod(receipt, 0o400)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SeedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
