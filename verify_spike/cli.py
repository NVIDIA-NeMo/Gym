"""Command-line entry point.

``python3 -m verify_spike <artifacts_dir> [--out DIR]`` reads the
NeMo-Gym-native artifacts (``evaluator_rollouts.jsonl`` — or
``rollouts.jsonl`` for local layouts — plus the ``model_calls/``
sidecars next to the rollouts file or one level up), computes the
per-rollout health verdicts, and writes into --out (default
``<artifacts_dir>/verify_spike_out``):

- ``rollout_health.jsonl`` — one verdict per rollout, input order;
- ``verification_summary.json`` — run-level summary;
- ``health_report.md`` — human-readable report.

Input records are never mutated; the tool only reads the artifacts.

``--compare`` diffs the summary field-by-field against an existing
evalpipeline ``verification_summary.json``; ``--compare-atif`` diffs
per-rollout verdicts (and the dirty-rollout sets) against an existing
evalpipeline ``trajectories_atif.jsonl`` (``extra.health``). Any
difference makes the exit status non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from verify_spike.checks import verify
from verify_spike.gym_records import TOKEN_SEMANTICS, map_rollout
from verify_spike.models import Health, Trajectory, read_jsonl
from verify_spike.report import health_report, summarize
from verify_spike.wire import find_model_calls_dir, load_wire_record

_ROLLOUTS_FILENAMES = ("evaluator_rollouts.jsonl", "rollouts.jsonl")
HEALTH_FILENAME = "rollout_health.jsonl"
SUMMARY_FILENAME = "verification_summary.json"
REPORT_FILENAME = "health_report.md"


def main(argv: "list[str] | None" = None) -> int:
    args = _parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()
    out_dir = (
        args.out if args.out is not None else artifacts_dir / "verify_spike_out"
    )
    rollouts_path = _rollouts_path(artifacts_dir)
    capture_dir = find_model_calls_dir(rollouts_path.parent)
    results = _verify_all(rollouts_path, capture_dir, artifacts_dir.name)
    summary = summarize(tuple(health for _, health in results))
    _write_outputs(out_dir, results, summary)
    print(f"verified {summary['total_trajectories']} rollouts")
    print(f"  rollouts: {rollouts_path}")
    print(f"  sidecars: {capture_dir if capture_dir else '(none found)'}")
    print(f"  output:   {out_dir}")
    exit_code = 0
    if args.compare is not None:
        exit_code |= _compare_summary(summary, args.compare)
    if args.compare_atif is not None:
        exit_code |= _compare_atif(results, args.compare_atif)
    return exit_code


def _parse_args(argv: "list[str] | None") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m verify_spike",
        description="Trajectory health verification over NeMo-Gym-native "
        "artifacts (standalone port of the evalpipeline checks).",
    )
    parser.add_argument(
        "artifacts_dir",
        type=Path,
        help="directory holding evaluator_rollouts.jsonl (or rollouts.jsonl) "
        "and the model_calls/ sidecars (next to the rollouts file or one "
        "level up)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <artifacts_dir>/verify_spike_out)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="evalpipeline verification_summary.json to diff field-by-field",
    )
    parser.add_argument(
        "--compare-atif",
        type=Path,
        default=None,
        help="evalpipeline trajectories_atif.jsonl to diff per-rollout "
        "health verdicts (extra.health)",
    )
    return parser.parse_args(argv)


def _rollouts_path(artifacts_dir: Path) -> Path:
    for name in _ROLLOUTS_FILENAMES:
        path = artifacts_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{artifacts_dir}: none of {', '.join(_ROLLOUTS_FILENAMES)} found"
    )


def _verify_all(
    rollouts_path: Path,
    capture_dir: "Path | None",
    dataset: str,
) -> "list[tuple[Trajectory, Health]]":
    results = []
    for _, row in read_jsonl(rollouts_path):
        trajectory = map_rollout(row, dataset=dataset)
        capture = load_wire_record(capture_dir, trajectory)
        health = verify(
            trajectory,
            token_semantics=TOKEN_SEMANTICS,
            capture=capture,
        )
        results.append((trajectory, health))
    return results


def _write_outputs(
    out_dir: Path,
    results: "list[tuple[Trajectory, Health]]",
    summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / HEALTH_FILENAME).open("w", encoding="utf-8") as handle:
        for trajectory, health in results:
            record = {
                "session_id": trajectory.session_id,
                "identity": asdict(trajectory.identity)
                if trajectory.identity is not None
                else None,
                "health": asdict(health),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    (out_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / REPORT_FILENAME).write_text(
        health_report(summary), encoding="utf-8"
    )


def _compare_summary(summary: dict, ground_truth_path: Path) -> int:
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    keys = sorted(set(summary) | set(ground_truth))
    width = max(len(key) for key in keys)
    mismatches = 0
    print()
    print(f"summary diff vs {ground_truth_path}")
    print(f"{'field'.ljust(width)}  {'ours':>12}  {'ground truth':>12}  match")
    for key in keys:
        ours = summary.get(key, "<absent>")
        theirs = ground_truth.get(key, "<absent>")
        match = ours == theirs
        mismatches += 0 if match else 1
        print(
            f"{key.ljust(width)}  {_cell(ours):>12}  {_cell(theirs):>12}  "
            f"{'OK' if match else 'DIFF'}"
        )
    print(
        f"summary fields: {len(keys) - mismatches}/{len(keys)} match"
        + ("" if mismatches == 0 else f" ({mismatches} differ)")
    )
    return 0 if mismatches == 0 else 1


def _cell(value: Any) -> str:
    return "null" if value is None else str(value)


def _compare_atif(
    results: "list[tuple[Trajectory, Health]]",
    atif_path: Path,
) -> int:
    ground_truth = {}
    for _, row in read_jsonl(atif_path):
        extra = row.get("extra") or {}
        ground_truth[str(row["session_id"])] = extra.get("health") or {}
    health_fields = [item.name for item in fields(Health)]
    mismatched_rollouts = 0
    compared = 0
    missing = []
    for trajectory, health in results:
        theirs = ground_truth.get(trajectory.session_id)
        if theirs is None:
            missing.append(trajectory.session_id)
        else:
            compared += 1
            # The evalpipeline ATIF writer omits None-valued fields;
            # an absent ground-truth key means None.
            diffs = [
                (name, getattr(health, name), theirs.get(name))
                for name in health_fields
                if getattr(health, name) != theirs.get(name)
            ]
            if diffs:
                mismatched_rollouts += 1
                print(f"health diff for rollout {trajectory.session_id}:")
                for name, ours, gt_value in diffs:
                    print(f"  {name}: ours={ours!r} ground_truth={gt_value!r}")
    print()
    print(f"per-rollout health diff vs {atif_path}")
    print(
        f"  {compared} rollouts compared, "
        f"{compared - mismatched_rollouts} identical, "
        f"{mismatched_rollouts} differ, {len(missing)} without ground truth"
    )
    ours_dirty = sorted(
        trajectory.session_id
        for trajectory, health in results
        if not health.is_clean
    )
    theirs_dirty = sorted(
        session_id
        for session_id, health in ground_truth.items()
        if health.get("is_clean") is False
    )
    print(f"  dirty rollouts (ours):         {ours_dirty}")
    print(f"  dirty rollouts (ground truth): {theirs_dirty}")
    print(
        "  dirty sets "
        + ("MATCH" if ours_dirty == theirs_dirty else "DIFFER")
    )
    clean = (
        mismatched_rollouts == 0
        and not missing
        and ours_dirty == theirs_dirty
    )
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
