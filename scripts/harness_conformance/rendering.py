# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared table layout for fresh local probes and retained CI measurements."""

import re
from datetime import UTC, datetime

from nemo_gym.rollouts.harness_capabilities import gate_passes, render_matrices

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


def scenario_tiers(scenario: dict) -> dict[str, bool]:
    """Evaluate applicable check tiers; the runner's gating verdict is separate."""
    checks = scenario["checks"]
    return {
        tier: gate_passes(checks, tier=tier)
        for tier in ("P0", "P1", "P2")
        if any(check["tier"] == tier and check["status"] != "not_applicable" for check in checks)
    }


def scenario_counts(scenarios: list[dict]) -> list[str]:
    results = [tiers for scenario in scenarios if (tiers := scenario_tiers(scenario))]
    if not results:
        return ["—"] * 4
    passing = sum(all(tiers.values()) for tiers in results)
    intent = "launch" if passing == len(results) else "error"
    counts = [f'<Badge intent="{intent}" minimal>{passing}/{len(results)}</Badge>']
    first_failures = [next((tier for tier, passed in tiers.items() if not passed), None) for tiers in results]
    for tier in ("P0", "P1", "P2"):
        count = first_failures.count(tier)
        counts.append(f'<Badge intent="error" minimal>{count}</Badge>' if count else "0")
    return counts


def scenario_details(scenarios: list[dict]) -> str:
    lines = []
    for scenario in scenarios:
        tiers = scenario_tiers(scenario)
        if not tiers:
            result = "— (no applicable checks)"
        else:
            symbol = "✓" if all(tiers.values()) else "✗"
            result = (
                symbol + " (" + ", ".join(f"{tier}:{'✓' if passed else '✗'}" for tier, passed in tiers.items()) + ")"
            )
        if scenario.get("unsupported_reason"):
            result += " — not run: declared unsupported"
        lines.append(f"- `{scenario['scenario']}`: {result}")
    return "\n".join(lines) + "\n\n"


def render_table(report: dict) -> str:
    """Render measured checks with a short source/freshness footer."""
    commit = report["gym_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("table requires a full Gym commit SHA")
    rows = report["harnesses"]
    for row in rows.values():
        validate_row(row)
    lines = [
        BEGIN,
        "",
        "| Harness | Scenarios passed | First failure: P0 | First failure: P1 | First failure: P2 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for harness in HARNESSES:
        row = rows.get(harness)
        scenarios = row["scenarios"] if row else []
        counts = scenario_counts(scenarios)
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
