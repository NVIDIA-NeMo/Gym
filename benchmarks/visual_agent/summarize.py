# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a visual_agent rollouts JSONL by category and mode.

python benchmarks/visual_agent/summarize.py results/<experiment>.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _mean(values: List[float]) -> str:
    return f"{mean(values):.3f}" if values else "-"


def summarize(rows: List[Dict[str, Any]]) -> str:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['category']}/{row['mode']}"].append(row)
        groups[f"all/{row['mode']}"].append(row)
        groups["all/all"].append(row)

    header = (
        f"{'group':28s} {'n':>4s} {'pointwise':>9s} {'reward':>7s} {'rubric':>7s} {'sim':>6s} "
        f"{'gate_fail':>9s} {'judge_fail':>10s} {'masked':>6s} {'hack':>5s} {'in_tok(M)':>9s}"
    )
    lines = [header, "-" * len(header)]
    for name in sorted(groups, key=lambda g: (g.startswith("all"), g)):
        rs = groups[name]
        scored = [r for r in rs if not r.get("mask_sample")]
        lines.append(
            f"{name:28s} {len(rs):4d} {_mean([r['pointwise_reward'] for r in scored]):>9s} "
            f"{_mean([r['reward'] for r in scored]):>7s} "
            f"{_mean([r['rubric_score'] for r in scored if r.get('rubric_score') is not None]):>7s} "
            f"{_mean([r['similarity'] for r in scored if r.get('similarity') is not None]):>6s} "
            f"{_mean([0.0 if r.get('gate_passed') else 1.0 for r in rs]):>9s} "
            f"{_mean([1.0 if r.get('judge_status') == 'failed' else 0.0 for r in rs]):>10s} "
            f"{_mean([1.0 if r.get('mask_sample') else 0.0 for r in rs]):>6s} "
            f"{_mean([1.0 if r.get('hack_detected') else 0.0 for r in rs]):>5s} "
            f"{_mean([(r.get('response', {}).get('usage') or {}).get('input_tokens', 0) / 1e6 for r in rs]):>9s}"
        )

    groupwise = [r for r in rows if r.get("groupwise_status") == "applied"]
    statuses: Dict[str, int] = defaultdict(int)
    for r in rows:
        statuses[r.get("groupwise_status", "?")] += 1
    lines.append("")
    lines.append(f"groupwise status: {dict(statuses)}")
    if groupwise:
        tiers: Dict[int, int] = defaultdict(int)
        for r in groupwise:
            tiers[r.get("groupwise_tier")] += 1
        lines.append(
            f"groupwise tiers: {dict(tiers)}; mean |reward - pointwise| = "
            f"{mean(abs(r['reward'] - r['pointwise_reward']) for r in groupwise):.3f}"
        )
    reasons: Dict[str, int] = defaultdict(int)
    for r in rows:
        for reason in r.get("gate_reasons") or []:
            reasons[reason.split(":")[0][:60]] += 1
    if reasons:
        lines.append(
            "gate failure reasons: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()]
    print(summarize(rows))


if __name__ == "__main__":
    main()
