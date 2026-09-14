# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic run-package builder and validator.

Layout::

    <package>/
      README.md
      normalized-run.json
      manifest/run-manifest.json
      raw/rollouts.jsonl  raw/failures.jsonl  raw/materialized-inputs.jsonl  raw/aggregate-metrics.json
      raw/verifier-or-judge-receipts.jsonl
      calibration/upstream-vs-gym.jsonl  calibration/summary.json
      blade/metrics.json  blade/d1-metrics.json  blade/d2-anchor-facts.json  blade/d3-shallow-baseline.md
      blade/BLADE-report.md
      prose/  (filled by generate_model_card_report)
      checksums.sha256

Same inputs and the same ``--generated-at`` produce byte-identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .blade import build_blade_files
from .schema import NormalizedRun


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy(source: Path | None, target: Path) -> str | None:
    if source is None or not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return sha256_file(target)


def _receipts(run: NormalizedRun) -> str:
    lines = []
    for rollout in run.rollouts:
        lines.append(
            json.dumps(
                {
                    "receipt_id": rollout.receipt_id,
                    "task_id": rollout.task_id,
                    "rollout_id": rollout.rollout_id,
                    "outcome_class": rollout.outcome_class,
                    "reward": rollout.reward,
                    "slices": rollout.slices,
                    "facts": rollout.facts,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
    return "".join(line + "\n" for line in lines)


def _readme(run: NormalizedRun, package_name: str) -> str:
    primary = [metric for metric in run.metrics if metric.kind == "primary"]
    lines = [
        f"# {package_name}",
        "",
        f"{run.benchmark['display_name']} run of `{run.run['model']}` collected with NeMo Gym "
        f"(`gym eval run`, harness `{run.run['harness']}`, Gym revision `{run.run['gym_revision']}`).",
        "",
        f"Protocol: {run.benchmark['protocol']}",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value | Numerator / denominator | Direction |",
        "|---|---:|---:|---|",
    ]
    for metric in primary:
        value = (
            "n/a" if metric.value is None else (f"{metric.value:.1%}" if metric.unit == "rate" else f"{metric.value}")
        )
        lines.append(f"| {metric.name} | {value} | {metric.numerator} / {metric.denominator} | {metric.direction} |")
    lines += [
        "",
        "## Outcome accounting",
        "",
        f"- expected tasks: {run.outcomes.expected_tasks}; expected rollouts: {run.outcomes.expected_rollouts}; "
        f"scored rollouts: {run.outcomes.scored_rollouts}",
        f"- policy-scored: {run.outcomes.policy_scored}; invalid model output: {run.outcomes.invalid_model_output}; "
        f"infrastructure failures: {run.outcomes.infrastructure_failure}; judge missing: {run.outcomes.judge_missing}; "
        f"pending annotation: {run.outcomes.pending_annotation}",
        f"- reward semantics: {run.reward_semantics}",
        "",
        "## Contents",
        "",
        "- `manifest/run-manifest.json` - resolved configuration, revisions, dataset hashes, limits, input checksums",
        "- `raw/` - verbatim Gym rollouts, failures sidecar, materialized inputs, aggregate metrics, per-rollout receipts",
        "- `calibration/` - upstream-vs-Gym case comparison and summary",
        "- `blade/` - deterministic D1 metrics, D2 evidence-anchored facts, D3 shallow baseline, BLADE report",
        "- `prose/` - model-card report (JSON, Markdown, HTML, PDF) written by the configured report model",
        "- `paper/` - the locally fetched benchmark paper PDF (hash recorded in the manifest) when supplied",
        "- `checksums.sha256` - SHA-256 of every file in the package",
        "",
    ]
    return "\n".join(lines)


def build_package(
    run: NormalizedRun,
    *,
    output_dir: Path,
    package_name: str,
    generated_at: str,
    rollouts: Path,
    failures: Path | None,
    materialized_inputs: Path | None,
    aggregate_metrics: Path | None,
    calibration_dir: Path | None,
    paper_pdf: Path | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    raw = output_dir / "raw"
    input_hashes = {
        "rollouts.jsonl": _copy(rollouts, raw / "rollouts.jsonl"),
        "failures.jsonl": _copy(failures, raw / "failures.jsonl"),
        "materialized-inputs.jsonl": _copy(materialized_inputs, raw / "materialized-inputs.jsonl"),
        "aggregate-metrics.json": _copy(aggregate_metrics, raw / "aggregate-metrics.json"),
    }
    _write(raw / "verifier-or-judge-receipts.jsonl", _receipts(run))
    if paper_pdf is not None and paper_pdf.exists():
        input_hashes[f"paper/{paper_pdf.name}"] = _copy(paper_pdf, output_dir / "paper" / paper_pdf.name)
    if calibration_dir is not None:
        _copy(calibration_dir / "upstream-vs-gym.jsonl", output_dir / "calibration" / "upstream-vs-gym.jsonl")
        _copy(calibration_dir / "summary.json", output_dir / "calibration" / "summary.json")
    _write(output_dir / "normalized-run.json", dump_json(run.model_dump(mode="json")))
    manifest = {
        "package_name": package_name,
        "generated_at": generated_at,
        "schema_version": run.schema_version,
        "benchmark": run.benchmark,
        "run": run.run,
        "outcomes": run.outcomes.model_dump(mode="json"),
        "reward_is_semantic": run.reward_is_semantic,
        "reward_semantics": run.reward_semantics,
        "input_sha256": {key: value for key, value in input_hashes.items() if value},
        "calibration": run.calibration.model_dump(mode="json") if run.calibration else None,
        **(extra_manifest or {}),
    }
    _write(output_dir / "manifest" / "run-manifest.json", dump_json(manifest))
    build_blade_files(run, output_dir / "blade", generated_at=generated_at)
    _write(output_dir / "README.md", _readme(run, package_name))
    (output_dir / "prose").mkdir(exist_ok=True)
    write_checksums(output_dir)
    return output_dir


def write_checksums(package_dir: Path) -> Path:
    entries = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            entries.append(f"{sha256_file(path)}  ./{path.relative_to(package_dir).as_posix()}")
    target = package_dir / "checksums.sha256"
    target.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return target


def validate_package(package_dir: Path) -> list[str]:
    """Return a list of problems; an empty list means the package reconciles."""
    problems: list[str] = []
    required = [
        "README.md",
        "normalized-run.json",
        "manifest/run-manifest.json",
        "raw/rollouts.jsonl",
        "raw/verifier-or-judge-receipts.jsonl",
        "blade/metrics.json",
        "blade/d1-metrics.json",
        "blade/d2-anchor-facts.json",
        "blade/d3-shallow-baseline.md",
        "blade/BLADE-report.md",
        "checksums.sha256",
    ]
    for relative in required:
        if not (package_dir / relative).is_file():
            problems.append(f"missing {relative}")
    if problems:
        return problems
    for line in (package_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, _, relative = line.partition("  ./")
        path = package_dir / relative
        if not path.is_file():
            problems.append(f"checksum entry without file: {relative}")
        elif sha256_file(path) != digest:
            problems.append(f"checksum mismatch: {relative}")
    listed = {line.partition("  ./")[2] for line in (package_dir / "checksums.sha256").read_text().splitlines()}
    for path in sorted(package_dir.rglob("*")):
        relative = path.relative_to(package_dir).as_posix()
        if path.is_file() and path.name != "checksums.sha256" and relative not in listed:
            problems.append(f"file not covered by checksums: {relative}")

    run = NormalizedRun.model_validate(json.loads((package_dir / "normalized-run.json").read_text(encoding="utf-8")))
    outcomes = run.outcomes
    classes = (
        outcomes.policy_scored + outcomes.invalid_model_output + outcomes.judge_missing + outcomes.pending_annotation
    )
    if classes != outcomes.scored_rollouts:
        problems.append(f"outcome classes {classes} != scored rollouts {outcomes.scored_rollouts}")
    if (
        outcomes.scored_rollouts + outcomes.infrastructure_failure + len(outcomes.missing_task_ids)
        != outcomes.expected_rollouts
    ):
        problems.append("scored + infrastructure failures + missing != expected rollouts")
    if len(run.rollouts) != outcomes.scored_rollouts:
        problems.append(f"{len(run.rollouts)} rollout summaries != {outcomes.scored_rollouts} scored rollouts")
    receipts = (package_dir / "raw/verifier-or-judge-receipts.jsonl").read_text(encoding="utf-8").splitlines()
    if len(receipts) != len(run.rollouts):
        problems.append("receipt count differs from rollout summaries")
    raw_rows = sum(
        1 for line in (package_dir / "raw/rollouts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if raw_rows != outcomes.scored_rollouts:
        problems.append(f"raw rollouts.jsonl has {raw_rows} rows, expected {outcomes.scored_rollouts}")
    for metric in run.metrics:
        if metric.numerator is not None and metric.denominator:
            if abs(metric.numerator / metric.denominator - (metric.value or 0.0)) > 1e-9:
                problems.append(f"metric {metric.id} value does not equal numerator/denominator")
        if metric.denominator == 0 and metric.value is not None:
            problems.append(f"metric {metric.id} has zero denominator but a value")
    anchor_ids = {fact.id for fact in run.anchor_facts}
    known_ids = (
        anchor_ids
        | {r.task_id for r in run.rollouts}
        | {r.rollout_id for r in run.rollouts}
        | {r.receipt_id for r in run.rollouts}
    )
    for fact in run.anchor_facts:
        for evidence in fact.evidence:
            if evidence not in known_ids:
                problems.append(f"anchor {fact.id} cites unknown evidence {evidence}")
    return problems


def main() -> None:  # pragma: no cover - thin CLI over validate_package
    parser = argparse.ArgumentParser(description="Validate a run package (checksums, denominators, evidence ids).")
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    problems = validate_package(args.package)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("package OK" if not problems else f"{len(problems)} problem(s)")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":  # pragma: no cover
    main()
