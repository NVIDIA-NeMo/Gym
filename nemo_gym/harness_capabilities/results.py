# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent check results. TE labels describe results; they never schedule checks."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal

from .checks import Check, Kind, PriorityTier


Status = Literal["pass", "fail", "not_applicable"]


@dataclass
class CheckResult:
    id: str
    kind: Kind
    status: Status
    tier: PriorityTier
    evidence: tuple[str, ...] = ()
    locations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)


class Results:
    """Combine repeated evaluations of one rule without hiding failures or blocked checks."""

    def __init__(self) -> None:
        self.rows: dict[str, CheckResult] = {}

    def add(
        self,
        check_id: str,
        kind: Kind,
        status: Status,
        *,
        tier: PriorityTier,
        evidence: tuple[str, ...] = (),
        location: str = "",
        reason: str = "",
        blocked_by: Iterable[str] = (),
    ) -> bool:
        rank = {"not_applicable": 0, "pass": 1, "fail": 2}
        row = self.rows.get(check_id)
        if row is None:
            row = self.rows[check_id] = CheckResult(check_id, kind, status, tier, tuple(evidence))
        elif row.kind != kind or row.evidence != tuple(evidence) or row.tier != tier:
            raise ValueError(f"conflicting definition for check {check_id}")
        if rank[status] > rank[row.status]:
            row.status = status
        for target, values in ((row.locations, [location]), (row.reasons, [reason]), (row.blocked_by, blocked_by)):
            for value in values:
                if value and value not in target:
                    target.append(value)
        return status == "pass"

    def run(self, check: Check) -> bool:
        """Execute a typed rule; excluded and blocked rules never evaluate their input."""
        blocked = [key for key in check.depends_on if key not in self.rows or self.rows[key].status != "pass"]
        evaluation = None
        status: Status
        if not check.applies:
            status = "not_applicable"
        elif not check.available or blocked:
            status = "fail"
        else:
            evaluation = check.evaluate()
            status = "pass" if evaluation.passed else "fail"
        reason = ""
        if status == "fail":
            if not check.available:
                reason = "required input is unavailable; " + check.reason
            elif blocked:
                reason = "blocked by " + ", ".join(blocked) + "; " + check.reason
            else:
                reason = check.reason
        passed = self.add(
            check.id,
            check.kind,
            status,
            tier=check.tier,
            evidence=check.evidence,
            location=check.location,
            reason=reason,
            blocked_by=blocked,
        )
        if evaluation is not None and status == "fail":
            for location in evaluation.locations:
                if location not in self.rows[check.id].locations:
                    self.rows[check.id].locations.append(location)
        return passed

    def dump(self) -> list[dict]:
        return [asdict(row) for row in self.rows.values()]


def gate_passes(checks: list[dict], *, tier: PriorityTier = "P0") -> bool:
    """Require all applicable checks in the selected tier, independently of TE labels."""
    selected = [c for c in checks if c["tier"] == tier and c["status"] != "not_applicable"]
    return bool(selected) and all(c["status"] == "pass" for c in selected)


def render_matrices(harnesses: dict) -> str:
    """Render saved check results; never infer a pass from an absent failure message."""
    labels = {"pass": "✓", "fail": "✗", "not_applicable": "—"}
    lines = [
        "# Harness checks by scenario",
        "",
        "✓ Pass · ✗ Requirement not demonstrated (invalid, missing, blocked, or no result) · — Not applicable.",
        "",
    ]
    for harness, row in harnesses.items():
        scenarios = row["scenarios"]
        keys = list(dict.fromkeys(c["id"] for s in scenarios for c in s.get("checks", [])))
        lines += [
            f"## {harness}",
            "",
            "| Check | " + " | ".join(s["scenario"] for s in scenarios) + " |",
            "|---|" + "---|" * len(scenarios),
        ]
        for kind in ("schema", "semantic", "behavioral"):
            group = []
            for key in keys:
                found = [next((c for c in s.get("checks", []) if c["id"] == key), None) for s in scenarios]
                if next(c["kind"] for c in found if c) != kind:
                    continue
                cells = [labels[c["status"]] if c else "✗" for c in found]
                group.append(f"| `{key}` | " + " | ".join(cells) + " |")
            if group:
                lines.append(f"| **{kind.title()}** |" + " |" * len(scenarios))
                lines.extend(group)
        if not keys:
            lines.append("| No check results available | " + " | ".join("✗" for _ in scenarios) + " |")
        lines.append("")
    return "\n".join(lines)
