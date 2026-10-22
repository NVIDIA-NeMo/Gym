# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dashboard import build_workspace, panel_specifications
from .demo import canonical_rows, demo_records, gym_rows
from .export import log_wandb, write_report
from .inputs import read_records
from .models import Options
from .suite import ANALYSES, analyze


def _tracking_arguments(parser):
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online"),
        default=None,
        help="Explicitly export; omitted means entirely local analysis",
    )
    parser.add_argument("--entity")
    parser.add_argument("--project", default="standard-analyses-prototype")
    parser.add_argument("--name", default="standard-analyses")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local standard evaluation analyses and W&B export")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate a labeled synthetic demonstration")
    demo.add_argument("--output", type=Path, required=True)
    _tracking_arguments(demo)
    analyze_cmd = commands.add_parser("analyze", help="Analyze local canonical or Gym rollout files")
    analyze_cmd.add_argument("inputs", type=Path, nargs="+")
    analyze_cmd.add_argument("--format", choices=("canonical", "gym"), default="canonical")
    analyze_cmd.add_argument("--run-id")
    analyze_cmd.add_argument("--benchmark")
    analyze_cmd.add_argument("--config-id")
    analyze_cmd.add_argument("--max-rows", type=int, default=2000)
    analyze_cmd.add_argument("--disable", action="append", choices=[fn.__name__ for fn in ANALYSES], default=[])
    analyze_cmd.add_argument("--slow-call-seconds", type=float, default=300)
    analyze_cmd.add_argument("--output", type=Path, required=True)
    _tracking_arguments(analyze_cmd)
    workspace = commands.add_parser("workspace", help="Build or install a reusable project saved view")
    workspace.add_argument("--entity", required=True)
    workspace.add_argument("--project", required=True)
    workspace.add_argument("--namespace", default="efficiency")
    workspace.add_argument("--output", type=Path, required=True)
    workspace.add_argument("--save", action="store_true", help="Create a new saved view in W&B")
    args = parser.parse_args(argv)
    try:
        if args.command == "workspace":
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "panel-specifications.json").write_text(json.dumps(panel_specifications(), indent=2) + "\n")
            view = build_workspace(args.entity, args.project, namespace=args.namespace)
            (args.output / "workspace.json").write_text(
                view._to_model().model_dump_json(by_alias=True, indent=2) + "\n"
            )
            if args.save:
                print(view.save().url)
            else:
                print(f"Workspace built locally: {args.output / 'workspace.json'}")
            return 0
        if args.command == "demo":
            args.output.mkdir(parents=True, exist_ok=True)
            for filename, rows in (
                ("demo.jsonl", canonical_rows()),
                ("gym-demo.jsonl", gym_rows()),
            ):
                (args.output / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = analyze(demo_records())
        else:
            if args.format == "gym" and not all((args.run_id, args.benchmark, args.config_id)):
                parser.error("Gym input requires --run-id, --benchmark and --config-id")
            records = read_records(
                args.inputs,
                format=args.format,
                run_id=args.run_id,
                benchmark=args.benchmark,
                config_id=args.config_id,
            )
            report = analyze(
                records,
                Options(max_rows=args.max_rows, slow_call_s=args.slow_call_seconds),
                disabled=args.disable,
            )
        write_report(report, args.output)
        (args.output / "panel-specifications.json").write_text(json.dumps(panel_specifications(), indent=2) + "\n")
        print(f"Report: {args.output / 'report.html'}")
        for result in report.results:
            print(f"  {result.name}: {result.status}")
        if args.wandb_mode:
            import wandb

            with wandb.init(
                project=args.project,
                entity=args.entity,
                name=args.name,
                mode=args.wandb_mode,
                dir=str(args.output),
                job_type="standard-analyses",
            ) as run:
                log_wandb(report, run)
                reference = {"mode": args.wandb_mode, "id": run.id, "url": run.url}
            (args.output / "wandb-run.json").write_text(json.dumps(reference, indent=2) + "\n")
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
