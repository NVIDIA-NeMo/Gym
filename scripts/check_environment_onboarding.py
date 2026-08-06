# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI entry point for the migration-safe environment onboarding gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo_gym.config_types import ConfigError
from nemo_gym.environment_ci import (
    ChangedFile,
    changed_files_from_git,
    run_enforced_verifier_checks,
    run_environment_ci_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the complete environment catalog and optionally enforce onboarding checks for "
            "new, changed, and shared-component-dependent units."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", help="Git base revision used to discover changed files.")
    parser.add_argument("--head-ref", default="HEAD", help="Git head revision (default: HEAD).")
    parser.add_argument(
        "--changed-file",
        action="append",
        type=Path,
        default=[],
        help="Explicit changed path when Git refs are unavailable; repeat as needed.",
    )
    parser.add_argument(
        "--enforce-changes",
        action="store_true",
        help="Fail checks for changed units/dependents plus schema and version locks.",
    )
    parser.add_argument(
        "--run-verifier-tests",
        action="store_true",
        help=(
            "Run each enforced manifest's canonical offline verifier scorer test. No tests run in report-only mode."
        ),
    )
    parser.add_argument(
        "--verifier-timeout",
        type=float,
        default=300,
        help="Per-manifest offline verifier scorer timeout in seconds (default: 300).",
    )
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report.")
    return parser


def _render_summary(report) -> None:
    payload = report.to_dict()
    summary = payload["summary"]
    coverage = payload["coverage"]
    print(
        f"Environment onboarding gate ({payload['mode']}): "
        f"{summary['total']} units, {summary['enforced']} enforced, "
        f"{summary['units_with_errors']} with diagnostics."
    )
    print(
        f"Manifest coverage: {coverage['with_manifest']}/{coverage['total']} "
        f"({coverage['percent']:.1f}%); {coverage['without_manifest']} grandfathered/no-manifest."
    )
    for unit in report.units:
        if not unit.errors and not unit.enforced:
            continue
        prefix = "ENFORCE" if unit.enforced else "REPORT"
        print(f"[{prefix}] {unit.kind}:{unit.name}")
        for error in unit.errors:
            print(f"  ERROR: {error}")
        for warning in unit.warnings:
            print(f"  NOTE: {warning}")
        for check in unit.verifier_checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  VERIFIER {status}: {check.test_path}::{check.node}")
            if not check.passed and check.output:
                for line in check.output.splitlines():
                    print(f"    {line}")
    for error in report.schema_errors:
        print(f"[SCHEMA] {error}")
    for error in report.lock_violations:
        print(f"[VERSION] {error}")
    print("PASS" if report.passed else "FAIL")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.base_ref:
            changes = changed_files_from_git(repo_root, args.base_ref, args.head_ref)
        else:
            changes = tuple(ChangedFile(path=path) for path in args.changed_file)
        report = run_environment_ci_gate(
            repo_root,
            changes=changes,
            enforce_changes=args.enforce_changes,
        )
        if args.run_verifier_tests:
            run_enforced_verifier_checks(report, timeout_seconds=args.verifier_timeout)
    except ConfigError as error:
        print(f"Environment onboarding gate could not run: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _render_summary(report)
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
