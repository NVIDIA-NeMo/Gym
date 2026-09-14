# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ``build_package`` command line: benchmark modules supply ``normalize`` and defaults."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import git_revision
from .package import build_package, validate_package
from .schema import NormalizedRun


Normalizer = Callable[..., NormalizedRun]


def build_main(normalize: Normalizer, *, benchmark_id: str, description: str, argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--failures", type=Path, default=None)
    parser.add_argument("--materialized-inputs", type=Path, default=None)
    parser.add_argument("--aggregate-metrics", type=Path, default=None)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared benchmark JSONL used for the run")
    parser.add_argument("--paper-pdf", type=Path, default=None, help="Locally fetched paper PDF (hashed, not copied)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint-type", required=True)
    parser.add_argument("--gym-revision", default=None)
    parser.add_argument(
        "--run-meta",
        type=Path,
        default=None,
        help="Optional JSON with extra run fields (limits, concurrency, timestamps)",
    )
    parser.add_argument("--generated-at", default=None, help="ISO timestamp for deterministic rebuilds")
    parser.add_argument("--output", type=Path, required=True, help="Package directory to (re)create")
    args = parser.parse_args(argv)

    run_meta: dict[str, Any] = json.loads(args.run_meta.read_text(encoding="utf-8")) if args.run_meta else {}
    repo_dir = Path(__file__).resolve().parents[3]
    run = normalize(
        rollouts_path=args.rollouts,
        failures_path=args.failures,
        materialized_inputs_path=args.materialized_inputs,
        aggregate_metrics_path=args.aggregate_metrics,
        calibration_dir=args.calibration_dir,
        dataset_path=args.dataset,
        paper_pdf=args.paper_pdf,
        run_id=args.run_id,
        model=args.model,
        endpoint_type=args.endpoint_type,
        gym_revision=args.gym_revision or git_revision(repo_dir) or "unknown",
        run_meta=run_meta,
    )
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    package_dir = build_package(
        run,
        output_dir=args.output,
        package_name=f"{benchmark_id}-{args.model.split('/')[-1].lower()}-{args.run_id}",
        generated_at=generated_at,
        rollouts=args.rollouts,
        failures=args.failures,
        materialized_inputs=args.materialized_inputs,
        aggregate_metrics=args.aggregate_metrics,
        calibration_dir=args.calibration_dir,
        paper_pdf=args.paper_pdf,
    )
    problems = validate_package(package_dir)
    if problems:
        raise SystemExit("package validation failed:\n- " + "\n- ".join(problems))
    print(f"package written and validated: {package_dir}")
    return package_dir
