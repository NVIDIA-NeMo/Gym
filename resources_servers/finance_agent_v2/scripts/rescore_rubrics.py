#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rescore finished finance_agent_v2 rollouts under severity weighting, offline.

Upstream added `modifiers` (severity, must_pass) to the public rubrics in Aug 2026
without touching any criterion's text, so verdicts collected before then are still
valid — only the aggregation changed. This joins the stored per-criterion scores to
the new weights by exact criterion text. No model is called and no judging repeats.

Scoring goes through `app.aggregate_rubric_scores`, the same function the server uses
live, so a rescored number cannot drift from a freshly collected one. Criteria the
weights do not cover are reported, and abort under `--strict`: a silent fallback to
weight 1.0 would look like a successful rescore while reverting part of the question
to the old unweighted scheme.

Weights default to the prepared benchmark dataset; a raw upstream CSV works too.

Usage:
    python rescore_rubrics.py RUN.jsonl [RUN2.jsonl ...] [--write OUT.jsonl] [--strict]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import RubricJudgement, aggregate_rubric_scores  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WEIGHTS = _REPO_ROOT / "benchmarks" / "finance_agent_v2" / "data" / "vals_v2_public_27q.jsonl"

# The rubric column name in a raw upstream CSV export.
_CSV_RUBRIC_COLUMN = "Rubric"


def _iter_rubrics(path: Path) -> Iterable[list]:
    """Yield each row's parsed rubric list from a prepared JSONL or an upstream CSV."""
    if path.suffix.lower() == ".csv":
        # Upstream rubrics are a single JSON blob per row and blow past the default
        # 128 KiB field cap.
        csv.field_size_limit(10_000_000)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = row.get(_CSV_RUBRIC_COLUMN)
                if raw:
                    yield json.loads(raw)
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line).get("rubric")
            if raw:
                yield json.loads(raw) if isinstance(raw, str) else raw


def load_weights(path: Path) -> Dict[str, Tuple[float, bool]]:
    """Map criterion text -> (severity, must_pass)."""
    weights: Dict[str, Tuple[float, bool]] = {}
    for rubric in _iter_rubrics(path):
        for check in rubric:
            text = (check.get("criteria") or "").strip()
            if not text:
                continue
            modifiers = check.get("modifiers") or {}
            try:
                severity = float(modifiers.get("severity", 1.0))
            except (TypeError, ValueError):
                severity = 1.0
            weights[text] = (severity, modifiers.get("category") == "must_pass")
    return weights


def rescore_row(row: Dict[str, Any], weights: Dict[str, Tuple[float, bool]]) -> Tuple[Dict[str, Any], List[str]]:
    """Return this row's new score fields and any criteria the weights did not cover.

    `rubric_judgements` is rewritten with the weights applied, so a rescored row can
    explain its own reward and be rescored again. A row with no judgements (no
    submission) keeps its 0.0 and is not counted as unmatched.
    """
    judgements = row.get("rubric_judgements") or []
    if not judgements:
        return {}, []

    missing: List[str] = []
    records = []
    enriched: List[Dict[str, Any]] = []
    for j in judgements:
        text = (j.get("criteria") or "").strip()
        if text in weights:
            severity, must_pass = weights[text]
        else:
            missing.append(text)
            severity, must_pass = 1.0, False
        records.append(
            RubricJudgement(
                criteria=text,
                operator=j.get("operator"),
                severity=severity,
                must_pass=must_pass,
                score=j.get("score"),
                votes=j.get("votes") or [],
                unanimous=j.get("unanimous"),
            )
        )
        enriched.append({**j, "severity": severity, "must_pass": must_pass})

    scores = aggregate_rubric_scores(records)
    scores["rubric_judgements"] = enriched
    return scores, missing


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="rollout JSONL file(s)")
    parser.add_argument("--weights", type=Path, default=_DEFAULT_WEIGHTS, help="prepared JSONL or upstream CSV")
    parser.add_argument("--write", type=Path, help="write a rescored copy (single input only)")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any criterion is unmatched")
    args = parser.parse_args(argv)

    if args.write and len(args.runs) > 1:
        parser.error("--write takes a single input run")
    if not args.weights.exists():
        parser.error(f"weights not found: {args.weights}\nRun `gym eval prepare` first, or pass --weights.")

    weights = load_weights(args.weights)
    weighted = sum(1 for severity, must_pass in weights.values() if severity != 1.0 or must_pass)
    print(f"weights: {len(weights)} criteria from {args.weights.name} ({weighted} carry a non-default modifier)\n")
    if not weighted:
        print("WARNING: every criterion is severity 1.0 and non-gating, so rescoring is a no-op.")
        print("         The weights source probably predates the Aug 2026 modifiers.\n")

    unmatched_total = 0
    for run in args.runs:
        rows = [json.loads(line) for line in open(run, encoding="utf-8") if line.strip()]
        rescored: List[Dict[str, Any]] = []
        unmatched: List[str] = []
        for row in rows:
            scores, missing = rescore_row(row, weights)
            unmatched.extend(missing)
            if scores:
                row = {**row, **scores, "reward": scores["rubric_partial_credit"]}
            rescored.append(row)

        scored = [r for r in rescored if r.get("rubric_judgements")]
        n = len(rescored)
        if not n:
            print(f"{run.name}: empty")
            continue

        # Unscorable rollouts (no submission) stay in the denominator: they are
        # model failures, and dropping them would flatter a run that often gave up.
        def mean(field: str) -> float:
            return sum(float(r.get(field) or 0.0) for r in rescored) / n

        tripped = sum(1 for r in scored if r.get("rubric_dealbreakers_failed"))
        gate_cost = sum(
            1 for r in scored if r.get("rubric_dealbreakers_failed") and (r.get("rubric_weighted_fraction") or 0) > 0
        )
        print(
            f"{run.name}: n={n} "
            f"partial_credit={mean('rubric_partial_credit'):.3f} "
            f"weighted_ungated={mean('rubric_weighted_fraction'):.3f} "
            f"all_pass={sum(1 for r in rescored if r.get('rubric_all_pass')) / n:.3f} "
            f"unweighted={mean('rubric_fraction'):.3f}"
        )
        print(
            f"{' ' * len(run.name)}  dealbreaker tripped in {tripped}/{len(scored)} scored rollouts "
            f"({gate_cost} of them forfeited credit they had otherwise earned); "
            f"{n - len(scored)} rollouts had no submission"
        )
        if unmatched:
            unmatched_total += len(unmatched)
            distinct = sorted(set(unmatched))
            print(f"  WARNING: {len(unmatched)} criteria ({len(distinct)} distinct) not found in the weights source.")
            print("           They were scored at severity 1.0 and non-gating. Sample:")
            for text in distinct[:3]:
                print(f"             - {text[:100]}")
        print()

        if args.write:
            with open(args.write, "w", encoding="utf-8") as f:
                for row in rescored:
                    f.write(json.dumps(row) + "\n")
            print(f"wrote {args.write}")

    if unmatched_total and args.strict:
        print(f"FAILED: {unmatched_total} unmatched criteria (--strict)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
