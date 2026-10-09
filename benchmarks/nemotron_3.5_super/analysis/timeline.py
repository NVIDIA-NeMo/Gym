# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline latency analysis of a nemotron_3.5_super run.

    python benchmarks/nemotron_3.5_super/analysis/timeline.py build   --run-dir <replica dir> [--tokenize-url URL]
    python benchmarks/nemotron_3.5_super/analysis/timeline.py check   --run-dir <replica dir>
    python benchmarks/nemotron_3.5_super/analysis/timeline.py analyze --run-dir <replica or experiment dir>
    python benchmarks/nemotron_3.5_super/analysis/timeline.py report  --experiment-dir <experiment dir>

The package lives beside this file; the directory name has a dot, so it is put on the path here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from ng_latency.build import build_run_timeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-rollout latency timeline tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="project rollouts.jsonl into spans and tokenise tool text")
    build.add_argument("--run-dir", required=True, type=Path)
    build.add_argument("--tokenize-url", default=None, help="http://<engine>:<port>/tokenize (or the policy /v1 url)")
    build.add_argument("--rollouts-name", default="rollouts.jsonl")
    check = sub.add_parser("check", help="cross-check the timeline against Gym's and the harness's own counters")
    check.add_argument("--run-dir", required=True, type=Path)
    check.add_argument("--rollouts-name", default="rollouts.jsonl")
    analyze = sub.add_parser(
        "analyze", help="derived metrics, run summary and the timelines page from a run directory"
    )
    analyze.add_argument("--run-dir", required=True, type=Path, help="replica or experiment dir")
    analyze.add_argument("--no-timelines", action="store_true", help="skip the per-rollout timelines page")
    analyze.add_argument("--rollouts-name", default="rollouts.jsonl")
    report = sub.add_parser("report", help="benchmark summary JSON and HTML page for one experiment (all replicas)")
    report.add_argument("--experiment-dir", required=True, type=Path, help="experiment dir holding replica_*")
    report.add_argument("--no-occupancy", action="store_true", help="skip the run timeline section")
    report.add_argument("--rollouts-name", default="rollouts.jsonl")
    args = parser.parse_args(argv)

    if args.command == "build":
        from ng_latency.tokenize import tokenize_url_from_policy_url

        tokenize_url = tokenize_url_from_policy_url(args.tokenize_url) if args.tokenize_url else None
        summary = build_run_timeline(args.run_dir, tokenize_url=tokenize_url, rollouts_name=args.rollouts_name)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "analyze":
        from ng_latency.analyze import run_analysis

        try:
            result = run_analysis(args.run_dir, timelines=not args.no_timelines, rollouts_name=args.rollouts_name)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        summary = result["summary"]
        missing = summary["coverage"]["rows_without_record"]
        if missing:
            print(
                f"warning: {missing} row(s) in rollouts.jsonl have no timeline record (stale rollouts_timeline.jsonl?);"
                " re-run `timeline.py build` to analyse them",
                file=sys.stderr,
            )
        print(
            json.dumps(
                {
                    "trials": summary["trials"],
                    "agent_kinds": summary["agent_kinds"],
                    "by_outcome": summary["by_outcome"],
                    "flags": summary["flags"],
                    "coverage": summary["coverage"],
                    "outputs": len(result["outputs"]),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report":
        from ng_latency.report import ReportError, write_report

        try:
            result = write_report(
                args.experiment_dir, rollouts_name=args.rollouts_name, with_occupancy=not args.no_occupancy
            )
        except ReportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0

    from ng_latency.check import run_checks

    report = run_checks(args.run_dir, rollouts_name=args.rollouts_name)
    print(
        json.dumps(
            {"passed": report["passed"], "failed_checks": report["failed_checks"], "warnings": report["warnings"]},
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
