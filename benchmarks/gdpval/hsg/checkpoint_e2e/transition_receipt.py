#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a provenance receipt for a three-row GDPVal runtime transition.

This tool records the mixed-runtime boundary used by a fast tail repair.  It
does not run Gym or Slurm.  When a pre-tail snapshot is supplied, it also proves
that the final JSONL preserves every old row and adds exactly the named three
Stage-1 task rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "gdpval.fast-tail-transition.v1"
OLD_RUNTIME_SCHEMAS = frozenset(("gdpval.transport-runtime.v2", "gdpval.transport-runtime.v3"))
NEW_RUNTIME_SCHEMA = "gdpval.transport-runtime.v3"
PACKAGE_VERSION = "1.4.11"
PACKAGE_INVENTORY_SCHEMA = "gdpval.existing-judge-package-inventory.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
JOB_ID_RE = re.compile(r"[1-9][0-9]*")


class TransitionError(RuntimeError):
    """A transition input cannot be bound without ambiguity."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise TransitionError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise TransitionError(f"{label} is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise TransitionError(f"{label} is not a regular file: {resolved}")
    return resolved


def _directory(path: Path, *, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise TransitionError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise TransitionError(f"{label} is unavailable: {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise TransitionError(f"{label} is not a directory: {resolved}")
    return resolved


def _binding(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransitionError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TransitionError(f"{label} must be a JSON object: {path}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise TransitionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _runtime_binding(path: Path, *, label: str, allowed_schemas: frozenset[str]) -> dict[str, Any]:
    document = _json_object(path, label=label)
    schema = document.get("schema")
    if schema not in allowed_schemas:
        raise TransitionError(f"{label} schema {schema!r} is not one of {sorted(allowed_schemas)}")
    revision = document.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise TransitionError(f"{label} has no exact Gym revision")
    outputs = document.get("output_sha256")
    if (
        not isinstance(outputs, dict)
        or not outputs
        or any(not isinstance(name, str) or not name for name in outputs)
        or any(not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None for digest in outputs.values())
    ):
        raise TransitionError(f"{label} output inventory is malformed")
    return {
        "manifest": _binding(path),
        "schema": schema,
        "gym_revision": revision,
        "output_inventory_sha256": _sha256_bytes(_canonical_bytes(outputs)),
        "output_files": len(outputs),
    }


def _scan_package(root: Path) -> tuple[list[dict[str, Any]], int, int]:
    entries: list[dict[str, Any]] = []
    files = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if "__pycache__" in relative_parts:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise TransitionError(f"package inventory contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            entries.append(
                {"path": relative, "type": "directory", "mode": f"{stat.S_IMODE(metadata.st_mode):04o}"}
            )
        elif stat.S_ISREG(metadata.st_mode):
            size = metadata.st_size
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                    "bytes": size,
                    "sha256": _sha256(path),
                }
            )
            files += 1
            total_bytes += size
        else:
            raise TransitionError(f"package inventory contains a special file: {path}")
    if not entries:
        raise TransitionError(f"package inventory is empty: {root}")
    return sorted(entries, key=lambda row: (row["path"], row["type"])), files, total_bytes


def _package_binding(root: Path) -> dict[str, Any]:
    version_path = _regular(root / "VERSION", label="package VERSION")
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise TransitionError("package VERSION is not UTF-8") from exc
    if version != PACKAGE_VERSION:
        raise TransitionError(f"package VERSION must be {PACKAGE_VERSION}, got {version!r}")
    entries, files, total_bytes = _scan_package(root)
    return {
        "schema": PACKAGE_INVENTORY_SCHEMA,
        "root": str(root),
        "version": version,
        "version_file_sha256": _sha256(version_path),
        "inventory_sha256": _sha256_bytes(_canonical_bytes(entries)),
        "files": files,
        "bytes": total_bytes,
    }


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise TransitionError(f"{label} contains a blank row at line {line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TransitionError(f"{label} contains invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TransitionError(f"{label} row {line_number} is not an object")
            rows.append(row)
    return rows


def _row_key(row: dict[str, Any], *, label: str) -> tuple[int, str, int]:
    stage = row.get("stage_index")
    task_id = row.get("task_id")
    rollout = row.get("_ng_rollout_index")
    if type(stage) is not int or stage not in (0, 1):
        raise TransitionError(f"{label} has an invalid stage_index")
    if not isinstance(task_id, str) or not task_id:
        raise TransitionError(f"{label} has an invalid task_id")
    if type(rollout) is not int or rollout != 0:
        raise TransitionError(f"{label} has an invalid rollout index")
    return stage, task_id, rollout


def _row_map(rows: list[dict[str, Any]], *, label: str) -> dict[tuple[int, str, int], dict[str, Any]]:
    result: dict[tuple[int, str, int], dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        key = _row_key(row, label=f"{label} row {index}")
        if key in result:
            raise TransitionError(f"{label} contains duplicate row key {key}")
        result[key] = row
    return result


def _validate_seed(path: Path, *, fingerprint: str) -> tuple[dict[str, Any], str]:
    document = _json_object(path, label="seed receipt")
    if document.get("status") != "READY" or document.get("applied") is not True:
        raise TransitionError("seed receipt is not an applied READY receipt")
    if document.get("plan_fingerprint") != fingerprint:
        raise TransitionError("seed receipt fingerprint differs from the frozen fingerprint")
    if document.get("stage1_rows_after") != 217 or document.get("stage1_rows_remaining") != 3:
        raise TransitionError("seed receipt does not describe an exact 217 + 3 Stage-1 transition")
    before = document.get("target_stage1_rows_before")
    imported = document.get("imported_stage1_rows")
    if type(before) is not int or type(imported) is not int or before < 0 or imported < 0 or before + imported != 217:
        raise TransitionError("seed receipt Stage-1 row accounting is inconsistent")
    pre_tail_sha = _require_sha256(
        document.get("target_output_sha256_after"), label="seed target_output_sha256_after"
    )
    _require_sha256(document.get("target_output_sha256_before"), label="seed target_output_sha256_before")
    return document, pre_tail_sha


def _validate_result(path: Path, *, final_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _json_object(path, label="strict campaign result")
    required = {
        "status": "PASS",
        "stage1_tasks": 220,
        "stage1_trials": 880,
        "invalid": 0,
    }
    for field, expected in required.items():
        if document.get(field) != expected or type(document.get(field)) is not type(expected):
            raise TransitionError(f"strict campaign result {field} is not {expected!r}")
    for field in ("eval_elo", "normalized_elo", "stage0_elo", "stage0_normalized_elo"):
        value = document.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TransitionError(f"strict campaign result {field} is not numeric")
    result_output = document.get("output")
    if not isinstance(result_output, str):
        raise TransitionError("strict campaign result has no output path")
    try:
        resolved_result_output = Path(result_output).resolve(strict=True)
    except OSError as exc:
        raise TransitionError(f"strict campaign result output is unavailable: {result_output}") from exc
    if resolved_result_output != final_output:
        raise TransitionError("strict campaign result points at a different final output")
    fields = {
        key: document[key]
        for key in (
            "status",
            "rows",
            "stage0_tasks",
            "stage1_tasks",
            "stage0_trials",
            "stage1_trials",
            "invalid",
            "stage0_partial",
            "stage0_elo",
            "stage0_normalized_elo",
            "eval_elo",
            "normalized_elo",
            "top4",
        )
        if key in document
    }
    return document, fields


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    fingerprint = _require_sha256(args.frozen_fingerprint, label="frozen fingerprint")
    slurm_job_ids = list(args.slurm_job_id)
    if any(JOB_ID_RE.fullmatch(job_id) is None for job_id in slurm_job_ids):
        raise TransitionError("every Slurm job ID must be a positive numeric job ID")
    if len(set(slurm_job_ids)) != len(slurm_job_ids):
        raise TransitionError("Slurm job IDs must be unique and ordered by attempt")
    tail_task_ids = list(args.tail_task_id)
    if len(tail_task_ids) != 3 or len(set(tail_task_ids)) != 3 or any(not task_id for task_id in tail_task_ids):
        raise TransitionError("exactly three unique non-empty tail task IDs are required")
    tail_task_ids = sorted(tail_task_ids)

    old_manifest = _regular(args.old_runtime_manifest, label="old runtime manifest")
    new_manifest = _regular(args.new_runtime_manifest, label="new runtime manifest")
    old_runtime = _runtime_binding(
        old_manifest,
        label="old runtime manifest",
        allowed_schemas=OLD_RUNTIME_SCHEMAS,
    )
    new_runtime = _runtime_binding(
        new_manifest,
        label="new runtime manifest",
        allowed_schemas=frozenset((NEW_RUNTIME_SCHEMA,)),
    )
    if old_runtime["manifest"]["sha256"] == new_runtime["manifest"]["sha256"]:
        raise TransitionError("old and new runtime manifests are byte-identical")

    package_root = _directory(args.package_root, label="v1.4.11 package root")
    seed_path = _regular(args.seed_receipt, label="corrected seed receipt")
    seed_document, pre_tail_sha = _validate_seed(seed_path, fingerprint=fingerprint)

    final_output = _regular(args.final_output, label="final output")
    final_rows = _read_jsonl(final_output, label="final output")
    final_map = _row_map(final_rows, label="final output")
    final_stage1 = {task_id: row for (stage, task_id, _), row in final_map.items() if stage == 1}
    if len(final_stage1) != 220:
        raise TransitionError(f"final output has {len(final_stage1)} Stage-1 tasks, expected 220")
    if not set(tail_task_ids).issubset(final_stage1):
        raise TransitionError("final output is missing one or more named tail tasks")
    for task_id, row in final_stage1.items():
        if row.get("verify_cache_namespace") != fingerprint:
            raise TransitionError(f"final Stage-1 task {task_id} is bound to another fingerprint")

    pre_tail_binding: dict[str, Any] = {"sha256": pre_tail_sha}
    if args.pre_tail_output is not None:
        pre_tail_output = _regular(args.pre_tail_output, label="pre-tail output snapshot")
        pre_tail_binding = _binding(pre_tail_output)
        if pre_tail_binding["sha256"] != pre_tail_sha:
            raise TransitionError("pre-tail snapshot SHA differs from the corrected seed receipt")
        pre_tail_rows = _read_jsonl(pre_tail_output, label="pre-tail output snapshot")
        pre_tail_map = _row_map(pre_tail_rows, label="pre-tail output snapshot")
        if any(key not in final_map or final_map[key] != row for key, row in pre_tail_map.items()):
            raise TransitionError("final output changed or removed a pre-tail row")
        added = set(final_map) - set(pre_tail_map)
        expected_added = {(1, task_id, 0) for task_id in tail_task_ids}
        if added != expected_added:
            raise TransitionError(f"final output additions are not the exact tail task set: {sorted(added)}")

    result_path = _regular(args.final_result, label="strict campaign result")
    _, result_fields = _validate_result(result_path, final_output=final_output)
    imported_task_ids = seed_document.get("imported_task_ids")
    if isinstance(imported_task_ids, list) and set(tail_task_ids) & set(imported_task_ids):
        raise TransitionError("a named tail task appears in the seed receipt imported task set")

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "runtime_transition": {"old": old_runtime, "new": new_runtime},
        "package": _package_binding(package_root),
        "seed": {
            "receipt": _binding(seed_path),
            "pre_tail_output": pre_tail_binding,
            "stage1_rows": 217,
            "stage1_rows_remaining": 3,
        },
        "tail": {
            "task_ids": tail_task_ids,
            "frozen_fingerprint": fingerprint,
            "slurm_job_ids": slurm_job_ids,
        },
        "final": {
            "output": _binding(final_output),
            "result": _binding(result_path),
            "result_fields": result_fields,
        },
    }


def _publish(path: Path, payload: bytes) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TransitionError(f"receipt output must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise TransitionError(f"existing transition receipt differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-runtime-manifest", type=Path, required=True)
    parser.add_argument("--new-runtime-manifest", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--seed-receipt", type=Path, required=True)
    parser.add_argument("--pre-tail-output", type=Path)
    parser.add_argument("--tail-task-id", action="append", required=True)
    parser.add_argument("--frozen-fingerprint", required=True)
    parser.add_argument("--slurm-job-id", action="append", required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_receipt(args)
        payload = _canonical_bytes(receipt)
        _publish(args.output, payload)
        sidecar = args.output.expanduser().absolute().with_suffix(args.output.suffix + ".sha256")
        _publish(sidecar, f"{_sha256_bytes(payload)}  {args.output.expanduser().absolute()}\n".encode())
    except (TransitionError, OSError) as exc:
        print(f"TRANSITION_RECEIPT_FAIL: {exc}", file=sys.stderr)
        return 64
    print(json.dumps({"status": "PASS", "receipt": str(args.output.expanduser().absolute()), "sha256": _sha256_bytes(payload)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
