# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent check results. TE labels describe results; they never schedule checks."""

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal


Status = Literal["pass", "fail", "not_assessed", "not_applicable"]
Kind = Literal["schema", "semantic", "behavioral"]


@dataclass
class CheckResult:
    id: str
    kind: Kind
    status: Status
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
        evidence: tuple[str, ...] = (),
        location: str = "",
        reason: str = "",
        blocked_by: Iterable[str] = (),
    ) -> bool:
        rank = {"not_applicable": 0, "pass": 1, "not_assessed": 2, "fail": 3}
        row = self.rows.get(check_id)
        if row is None:
            row = self.rows[check_id] = CheckResult(check_id, kind, status, tuple(evidence))
        elif row.kind != kind or row.evidence != tuple(evidence):
            raise ValueError(f"conflicting definition for check {check_id}")
        if rank[status] > rank[row.status]:
            row.status = status
        for target, values in ((row.locations, [location]), (row.reasons, [reason]), (row.blocked_by, blocked_by)):
            for value in values:
                if value and value not in target:
                    target.append(value)
        return status == "pass"

    def check(
        self,
        check_id: str,
        kind: Kind,
        condition: bool | Callable[[], bool],
        *,
        evidence: tuple[str, ...] = (),
        location: str = "",
        reason: str = "",
        applies: bool = True,
        available: bool = True,
        depends_on: tuple[str, ...] = (),
    ) -> bool:
        blocked = [key for key in depends_on if key not in self.rows or self.rows[key].status != "pass"]
        if not applies:
            status = "not_applicable"
        elif not available or blocked:
            status = "not_assessed"
        else:
            passed = condition() if callable(condition) else condition
            status = "pass" if passed else "fail"
        return self.add(
            check_id,
            kind,
            status,
            evidence=evidence,
            location=location,
            reason=reason if status in ("fail", "not_assessed") else "",
            blocked_by=blocked,
        )

    def dump(self) -> list[dict]:
        return [asdict(row) for row in self.rows.values()]


def render_matrices(harnesses: dict) -> str:
    """Render saved check results; never infer a pass from an absent failure message."""
    labels = {"pass": "✓", "fail": "✗", "not_assessed": "?", "not_applicable": "—"}
    lines = [
        "# Harness checks by scenario",
        "",
        "✓ Pass · ✗ Fail · ? Not assessed (blocked or unavailable) · — Not applicable.",
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
                cells = [labels[c["status"]] if c else "?" for c in found]
                group.append(f"| `{key}` | " + " | ".join(cells) + " |")
            if group:
                lines.append(f"| **{kind.title()}** |" + " |" * len(scenarios))
                lines.extend(group)
        if not keys:
            lines.append("| No check results available | " + " | ".join("?" for _ in scenarios) + " |")
        lines.append("")
    return "\n".join(lines)
