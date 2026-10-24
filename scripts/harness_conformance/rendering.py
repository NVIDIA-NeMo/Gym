# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared table layout for fresh local probes and retained CI measurements."""

import re
from datetime import UTC, datetime

from nemo_gym.harness_capabilities.results import render_matrices

from .documentation import BEGIN, END
from .registry import HARNESSES


def validate_row(row: dict) -> None:
    """Validate Gym-owned check metadata before inserting it into Markdown."""
    for scenario in row["scenarios"]:
        if scenario["verdict"] not in ("fulfilled", "not_fulfilled"):
            raise ValueError("invalid scenario verdict in documentation results")
        for identifier in [scenario["scenario"], *(c["id"] for c in scenario["checks"])]:
            if not re.fullmatch(r"[a-z][a-z0-9_.-]*", identifier):
                raise ValueError("invalid check or scenario identifier in documentation results")
        for check in scenario["checks"]:
            if check["kind"] not in ("schema", "semantic", "behavioral", "health") or check["status"] not in (
                "pass",
                "fail",
                "not_applicable",
            ):
                raise ValueError("invalid check kind or status in documentation results")


def render_table(report: dict) -> str:
    """Render measured checks with a short source/freshness footer."""
    commit = report["gym_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("table requires a full Gym commit SHA")
    statuses = report.get("statuses", {})
    rows = report["harnesses"]
    lines = [BEGIN, "", "| Harness | P0 | Passing scenarios |", "| --- | --- | --- |"]
    for harness in HARNESSES:
        row = rows.get(harness)
        status = statuses.get(harness)
        if status is None:
            status = "not_run" if row is None else "passed" if row["verdict"] == "fulfilled" else "failed"
        count = (
            "—"
            if row is None
            else f"{sum(s['verdict'] == 'fulfilled' for s in row['scenarios'])}/{len(row['scenarios'])}"
        )
        label = {"passed": "Passed", "failed": "Failed", "unavailable": "Unavailable", "not_run": "Not measured"}[
            status
        ]
        lines.append(f"| `{harness}` | {label} | {count} |")
    for row in rows.values():
        validate_row(row)
    if rows:
        matrices = render_matrices(rows).replace("# Harness checks by scenario\n", "", 1)
        for harness in rows:
            matrices = matrices.replace(f"## {harness}\n", f"### `{harness}`\n", 1)
        lines += ["", matrices]
    footer = f"[Gym `{commit[:12]}`](https://github.com/NVIDIA-NeMo/Gym/commit/{commit})"
    if report.get("generated_at"):
        stamp = datetime.fromisoformat(report["generated_at"]).astimezone(UTC)
        footer = f"Updated {stamp:%Y-%m-%d %H:%M} UTC · " + footer
    if report.get("run_url"):
        footer += f" · [Results]({report['run_url']})"
    lines += ["", "<sub>" + footer + "</sub>", "", END]
    return "\n".join(lines)
