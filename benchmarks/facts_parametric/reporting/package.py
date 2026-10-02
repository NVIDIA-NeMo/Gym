# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic run-package builder and validator.

Layout (fixed):

    <benchmark>-<model>-<run-id>/
      README.md
      manifest/run-manifest.json
      manifest/normalized-run.json
      raw/rollouts.jsonl
      raw/failures.jsonl
      raw/materialized-inputs.jsonl
      raw/aggregate-metrics.json
      raw/judge-or-verifier-receipts.jsonl
      calibration/upstream-vs-gym.jsonl
      calibration/summary.json
      blade/metrics.json  blade/d1-metrics.json  blade/d2-anchor-facts.json  blade/d3-shallow-baseline.md  blade/BLADE-report.md
      paper/<pdf>                      (when the pinned paper was fetched)
      prose/report-input.json          (the grounding context handed to the report writer)
      prose/report.md  prose/report.html  prose/report.pdf  prose/report.json  prose/report-generation.json
      checksums.sha256

Given the same inputs, run id and timestamps, the builder writes byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .blade import write_blade_files
from .schema import NormalizedRun


CHECKSUMS = "checksums.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_checksums(package_dir: Path) -> Path:
    lines = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.name != CHECKSUMS:
            lines.append(f"{sha256_file(path)}  ./{path.relative_to(package_dir).as_posix()}")
    target = package_dir / CHECKSUMS
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def verify_checksums(package_dir: Path) -> list[str]:
    problems: list[str] = []
    listed: dict[str, str] = {}
    for line in (package_dir / CHECKSUMS).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ./")
        listed[rel] = digest
        path = package_dir / rel
        if not path.exists():
            problems.append(f"missing file listed in checksums: {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"checksum mismatch: {rel}")
    for path in package_dir.rglob("*"):
        if path.is_file() and path.name != CHECKSUMS and path.relative_to(package_dir).as_posix() not in listed:
            problems.append(f"file not covered by checksums: {path.relative_to(package_dir).as_posix()}")
    return problems


def _copy(source: Path | None, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source is None or not Path(source).exists():
        target.write_text("", encoding="utf-8")
        return
    shutil.copyfile(source, target)


def extract_receipts(rollouts_path: Path, receipts_path: Path, extract: Any) -> int:
    """Write one JSONL row per judge/verifier receipt using the benchmark's ``extract(rollout) -> list[dict]``."""
    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with rollouts_path.open("rb") as source, receipts_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            rollout = json.loads(line)
            for receipt in extract(rollout):
                target.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
                count += 1
    return count


def build_package(
    normalized: NormalizedRun,
    *,
    package_dir: Path,
    manifest: dict[str, Any],
    rollouts_path: Path,
    failures_path: Path | None,
    materialized_inputs_path: Path | None,
    aggregate_metrics_path: Path | None,
    receipts_extractor: Any,
    calibration_jsonl: Path | None,
    calibration_summary: dict[str, Any] | None,
    paper_pdf: Path | None,
    readme_text: str,
    notes_path: Path | None = None,
) -> Path:
    package_dir = Path(package_dir)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    write_json(package_dir / "manifest" / "run-manifest.json", manifest)
    write_json(package_dir / "manifest" / "normalized-run.json", normalized.model_dump(mode="json"))
    _copy(rollouts_path, package_dir / "raw" / "rollouts.jsonl")
    _copy(failures_path, package_dir / "raw" / "failures.jsonl")
    _copy(materialized_inputs_path, package_dir / "raw" / "materialized-inputs.jsonl")
    _copy(aggregate_metrics_path, package_dir / "raw" / "aggregate-metrics.json")
    extract_receipts(rollouts_path, package_dir / "raw" / "judge-or-verifier-receipts.jsonl", receipts_extractor)
    _copy(calibration_jsonl, package_dir / "calibration" / "upstream-vs-gym.jsonl")
    write_json(package_dir / "calibration" / "summary.json", calibration_summary or {"status": "not_run"})
    write_blade_files(normalized, package_dir / "blade")
    if notes_path is not None and Path(notes_path).exists():
        _copy(Path(notes_path), package_dir / "calibration" / "inspection-notes.md")
    if paper_pdf is not None and Path(paper_pdf).exists():
        _copy(paper_pdf, package_dir / "paper" / Path(paper_pdf).name)
    (package_dir / "prose").mkdir(exist_ok=True)
    (package_dir / "README.md").write_text(readme_text, encoding="utf-8")
    write_checksums(package_dir)
    return package_dir


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def validate_package(package_dir: Path) -> list[str]:
    """Recompute checksums and reconcile the BLADE sidecars and denominators against the normalized run."""
    package_dir = Path(package_dir)
    problems = verify_checksums(package_dir)
    normalized = NormalizedRun.model_validate(
        json.loads((package_dir / "manifest" / "normalized-run.json").read_text(encoding="utf-8"))
    )
    metrics = json.loads((package_dir / "blade" / "metrics.json").read_text(encoding="utf-8"))
    d1 = json.loads((package_dir / "blade" / "d1-metrics.json").read_text(encoding="utf-8"))
    anchors = json.loads((package_dir / "blade" / "d2-anchor-facts.json").read_text(encoding="utf-8"))
    primary = normalized.primary_metric()
    if metrics.get("primary_metric", {}).get("value") != primary.value:
        problems.append("blade/metrics.json primary metric does not match the normalized run")
    if d1.get("total_rollouts") != normalized.outcomes.expected_rollouts:
        problems.append("blade/d1-metrics.json total_rollouts does not match the expected rollouts")
    if d1.get("scored_rollouts") != normalized.outcomes.scored_rollouts:
        problems.append("blade/d1-metrics.json scored_rollouts does not match the normalized run")
    scored_rows = _count_lines(package_dir / "raw" / "rollouts.jsonl")
    if scored_rows != normalized.outcomes.scored_rollouts:
        problems.append(
            f"raw/rollouts.jsonl has {scored_rows} rows but the run reports {normalized.outcomes.scored_rollouts} scored rollouts"
        )
    known_ids = {rollout.task_id for rollout in normalized.rollouts} | {
        rollout.rollout_id for rollout in normalized.rollouts
    }
    for rollout in normalized.rollouts:
        known_ids.update(rollout.receipt_ids)
    if normalized.calibration is not None:
        known_ids.update(str(key) for key in normalized.calibration.details.get("case_ids", []))
    for fact in anchors.get("anchor_facts", []):
        for evidence in fact.get("evidence", []):
            if evidence not in known_ids:
                problems.append(f"anchor fact {fact.get('id')} cites unknown evidence {evidence!r}")
    if len(anchors.get("anchor_facts", [])) != len(normalized.anchor_facts):
        problems.append("blade/d2-anchor-facts.json count differs from the normalized run")
    for metric in normalized.metrics:
        if metric.numerator is not None and metric.denominator and metric.unit == "rate":
            if abs(metric.numerator / metric.denominator - (metric.value or 0.0)) > 1e-9:
                problems.append(f"metric {metric.id} value does not equal numerator/denominator")
    return problems
