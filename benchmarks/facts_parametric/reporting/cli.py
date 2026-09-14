# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build, validate, and describe a run package.

    python -m benchmarks.<benchmark>.reporting.cli build --run-dir results/full --stem <run stem> \
        --run-id <id> --model <model> --package results/<benchmark>-<model>-<id> --run-info run-info.json \
        [--calibration-dir results/full/calibration] [--paper paper.pdf]
    python -m benchmarks.<benchmark>.reporting.cli validate --package <dir>

``run-info.json`` carries the run description that no artifact records on its own (endpoint type, sampling,
generation limits, judge identities, gym revision, timestamps). The package is deterministic given the same inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import normalize as normalize_module
from .common import git_revision, load_run
from .package import build_package, validate_package
from .schema import CalibrationSummary


def _readme(normalized, package_name: str) -> str:
    primary = normalized.primary_metric()
    return (
        f"# {package_name}\n\n"
        f"Run package for {normalized.run['model']} on {normalized.benchmark['display_name']} "
        f"(run `{normalized.run['run_id']}`).\n\n"
        f"- Primary metric: {primary.name} = {primary.value} (denominator {primary.denominator}).\n"
        f"- Scored {normalized.outcomes.scored_rollouts} of {normalized.outcomes.expected_rollouts} expected rollouts.\n"
        "- `manifest/`: run manifest and the normalized run every other file derives from.\n"
        "- `raw/`: rollouts, failures sidecar, materialized inputs, aggregate metrics, judge receipts.\n"
        "- `calibration/`: upstream-vs-Gym case ledger and summary.\n"
        "- `blade/`: metrics sidecar, D1 funnel metrics, D2 anchor facts, D3 shallow baseline, BLADE report.\n"
        "- `prose/`: the grounding context given to the report writer and the generated report (md/html/pdf).\n"
        "- `checksums.sha256`: SHA-256 of every file; `validate` recomputes them.\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--stem", required=True, help="Output stem used by gym eval run (e.g. facts_parametric)")
    build.add_argument("--run-id", required=True)
    build.add_argument("--model", required=True)
    build.add_argument("--package", type=Path, required=True)
    build.add_argument(
        "--run-info", type=Path, required=True, help="JSON with endpoint/sampling/judge/timestamp details"
    )
    build.add_argument(
        "--calibration-dir", type=Path, default=None, help="Directory with upstream-vs-gym.jsonl and summary.json"
    )
    build.add_argument("--paper", type=Path, default=None)
    build.add_argument("--num-repeats", type=int, default=1)
    build.add_argument("--generated-at", default=None, help="Fixed timestamp for deterministic packages")
    build.add_argument(
        "--notes", type=Path, default=None, help="Markdown inspection notes copied to calibration/inspection-notes.md"
    )
    validate = sub.add_parser("validate")
    validate.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "validate":
        problems = validate_package(args.package)
        if problems:
            print("\n".join(problems))
            sys.exit(1)
        print(f"package {args.package} validates")
        return

    artifacts = load_run(args.run_dir, args.stem)
    run_info = json.loads(args.run_info.read_text(encoding="utf-8"))
    run_info.setdefault("run_id", args.run_id)
    run_info.setdefault("model", args.model)
    run_info.setdefault("gym_revision", git_revision(Path(__file__).resolve().parents[3]))
    if args.generated_at:
        run_info["package_generated_at"] = args.generated_at
    calibration = None
    calibration_jsonl = None
    calibration_summary = None
    if args.calibration_dir and (args.calibration_dir / "summary.json").exists():
        calibration_summary = json.loads((args.calibration_dir / "summary.json").read_text(encoding="utf-8"))
        calibration = (
            CalibrationSummary.model_validate(calibration_summary["normalized"])
            if "normalized" in calibration_summary
            else None
        )
        calibration_jsonl = args.calibration_dir / "upstream-vs-gym.jsonl"
    normalized = normalize_module.normalize(
        artifacts, run_info=run_info, calibration=calibration, num_repeats=args.num_repeats
    )
    manifest = {
        "run_id": args.run_id,
        "model": args.model,
        "benchmark": normalized.benchmark,
        "run": run_info,
        "outcomes": normalized.outcomes.model_dump(mode="json"),
        "key_metrics": {
            m.id: {"value": m.value, "numerator": m.numerator, "denominator": m.denominator}
            for m in normalized.metrics
            if m.kind in ("primary", "component")
        },
        "inputs": {
            "rollouts": str(artifacts.rollouts_path),
            "failures": str(artifacts.failures_path),
            "materialized_inputs": str(artifacts.materialized_path),
            "aggregate_metrics": str(artifacts.aggregate_path),
        },
    }
    package = build_package(
        normalized,
        package_dir=args.package,
        manifest=manifest,
        rollouts_path=artifacts.rollouts_path,
        failures_path=artifacts.failures_path,
        materialized_inputs_path=artifacts.materialized_path,
        aggregate_metrics_path=artifacts.aggregate_path,
        receipts_extractor=normalize_module.receipts_for,
        calibration_jsonl=calibration_jsonl,
        calibration_summary=calibration_summary,
        paper_pdf=args.paper,
        readme_text=_readme(normalized, args.package.name),
        notes_path=args.notes,
    )
    problems = validate_package(package)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"built and validated {package}")


if __name__ == "__main__":  # pragma: no cover
    main()
