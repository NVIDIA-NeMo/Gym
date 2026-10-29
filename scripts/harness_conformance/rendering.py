# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared table layout for fresh local probes and retained CI measurements."""

import re
from datetime import UTC, datetime

from nemo_gym.rollouts import PriorityTier, gate_passes, render_matrices

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
            if check["tier"] not in ("P0", "P1", "P2"):
                raise ValueError("invalid check tier in documentation results")


def scenario_results(scenarios: list[dict], *, tier: PriorityTier) -> tuple[list[str], list[str]]:
    """Partition applicable scenarios by their checks in this tier, independently of other tiers."""
    passing, not_passing = [], []
    for scenario in scenarios:
        checks = scenario["checks"]
        if any(check["tier"] == tier and check["status"] != "not_applicable" for check in checks):
            target = passing if gate_passes(checks, tier=tier) else not_passing
            target.append(scenario["scenario"])
    return passing, not_passing


def scenario_count(scenarios: list[dict], *, tier: PriorityTier) -> str:
    passing, not_passing = scenario_results(scenarios, tier=tier)
    total = len(passing) + len(not_passing)
    if not total:
        return "—"
    # Fern's launch intent uses the site's NVIDIA accent; minimal keeps the fill soft.
    intent = "error" if not_passing else "launch"
    return f'<Badge intent="{intent}" minimal>{len(passing)}/{total}</Badge>'


def scenario_details(scenarios: list[dict]) -> str:
    lines = []
    for tier in ("P0", "P1", "P2"):
        passing, not_passing = scenario_results(scenarios, tier=tier)
        if not passing and not not_passing:
            continue
        lines += [f"**{tier}**", ""]
        for label, names in (("Passing", passing), ("Not passing", not_passing)):
            listed = ", ".join(f"`{name}`" for name in names) or "None"
            lines.append(f"- **{label}:** {listed}")
        lines.append("")
    return "\n".join(lines)


def render_table(report: dict) -> str:
    """Render measured checks with a short source/freshness footer."""
    commit = report["gym_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("table requires a full Gym commit SHA")
    rows = report["harnesses"]
    for row in rows.values():
        validate_row(row)
    lines = [BEGIN, "", "| Harness | P0 | P1 | P2 |", "| --- | --- | --- | --- |"]
    for harness in HARNESSES:
        row = rows.get(harness)
        scenarios = row["scenarios"] if row else []
        counts = [scenario_count(scenarios, tier=tier) for tier in ("P0", "P1", "P2")]
        lines.append(f"| `{harness}` | " + " | ".join(counts) + " |")
    if rows:
        matrices = render_matrices(rows).replace("# Harness checks by scenario\n", "", 1)
        for harness, row in rows.items():
            details = scenario_details(row["scenarios"])
            matrices = matrices.replace(f"## {harness}\n", f"### `{harness}`\n\n{details}", 1)
        lines += ["", '<Accordion title="Detailed checks by harness">', "", matrices, "", "</Accordion>"]
    footer = f"[Gym `{commit[:12]}`](https://github.com/NVIDIA-NeMo/Gym/commit/{commit})"
    if report.get("generated_at"):
        stamp = datetime.fromisoformat(report["generated_at"]).astimezone(UTC)
        footer = f"Updated {stamp:%Y-%m-%d %H:%M} UTC · " + footer
    if report.get("run_url"):
        footer += f" · [Results]({report['run_url']})"
    lines += ["", "<sub>" + footer + "</sub>", "", END]
    return "\n".join(lines)
