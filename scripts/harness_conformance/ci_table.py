# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and render retained CI measurements in a job with read-only credentials."""

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import download_artifacts
from .checks import ci_result
from .documentation import atomic_write
from .registry import HARNESSES
from .rendering import render_table, validate_row
from .scenarios import SCENARIOS


def render_ci(*, artifacts: Path, commit: str, output: Path, run_url: str) -> None:
    """Write one table file; absent or malformed measurements remain explicitly unavailable."""
    rows, statuses = {}, {}
    for harness in HARNESSES:
        result = ci_result(harness, artifacts=artifacts, sha=commit)
        statuses[harness] = result["status"] if result else "unavailable"
        if statuses[harness] == "unavailable":
            continue
        try:
            summary = json.loads((artifacts / harness / "probes" / "conformance_summary.json").read_text())
            if (
                summary["runner_status"] != "completed"
                or summary["full_suite"] is not True
                or set(summary["harnesses"]) != {harness}
            ):
                raise ValueError("incomplete probe suite")
            row = summary["harnesses"][harness]
            if [s["scenario"] for s in row["scenarios"]] != [s.name for s in SCENARIOS]:
                raise ValueError("incomplete scenario coverage")
            if row["verdict"] != ("fulfilled" if statuses[harness] == "passed" else "not_fulfilled"):
                raise ValueError("summary differs from CI result")
            validate_row(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            statuses[harness] = "unavailable"
            message = str(exc).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
            print(f"::warning::{harness} measurement unavailable: {message}", file=sys.stderr)
            continue
        rows[harness] = row
    block = render_table(
        {
            "gym_commit": commit,
            "harnesses": rows,
            "statuses": statuses,
            "generated_at": datetime.now(UTC).isoformat(),
            "run_url": run_url,
        }
    )
    atomic_write(output, block + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="source GitHub repository")
    parser.add_argument("--run-id", required=True, help="source CI run ID")
    parser.add_argument("--run-attempt", required=True, help="source CI run attempt")
    parser.add_argument("--commit", required=True, help="SHA matched against the downloaded CI results")
    parser.add_argument("--output", required=True, type=Path, help="rendered table.mdx file")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="conformance-table-") as directory:
        artifacts = Path(directory)
        download_artifacts(
            list(HARNESSES), repo=args.repo, run_id=args.run_id, attempt=args.run_attempt, output=artifacts
        )
        render_ci(
            artifacts=artifacts,
            commit=args.commit,
            output=args.output,
            run_url=f"https://github.com/{args.repo}/actions/runs/{args.run_id}/attempts/{args.run_attempt}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
