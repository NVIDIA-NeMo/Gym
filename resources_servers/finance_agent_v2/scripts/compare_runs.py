#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two or more finance_agent_v2 rollout JSONLs by per-question reward.

Built for the cache-validation experiment: an empty-cache run and a full-cache run
should be identical, while a run against a perturbed copy of the cache flags the
questions whose answer depends on an absolute adjusted price level.

Rows match on the harness task/repeat index (question text as fallback). Judge
failures are reported separately, since a 0.0 there is not a meaningful "incorrect".

Each cell shows Partial Credit next to the unweighted pass fraction, because they
miss different things: Partial Credit is pinned at 0.0 for any question with a failed
dealbreaker and so hides a flip inside an already-gated question, while the unweighted
fraction ignores what each criterion is worth. Pre-Aug-2026 runs stored an
all-or-nothing reward and are flagged as legacy, since putting a binary reward in the
Partial Credit column invites reading a scoring change as a model regression.

Usage:
    python compare_runs.py A.jsonl B.jsonl [C.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key_of(row: dict, idx: int) -> str:
    """Match rows across runs, preferring the harness's task/repeat indices.

    The question-text fallback is for hand-assembled files, whitespace-normalized
    because upstream reflowed 3 of the 27 public questions in Aug 2026 without
    rewording them, and raw text would report those as MISSING in both runs.
    """
    task_index = row.get("_ng_task_index")
    if task_index is not None:
        repeat = row.get("_ng_rollout_index")
        return f"task_{task_index}" + (f" r{repeat}" if repeat is not None else "")
    q = row.get("question")
    if isinstance(q, str) and q.strip():
        return " ".join(q.split())
    return f"__row_{idx}"


def main(paths: list[str]) -> int:
    if len(paths) < 2:
        print(__doc__)
        return 2

    runs = {Path(p).name: load(p) for p in paths}
    names = list(runs)

    legacy = [
        name
        for name, rows in runs.items()
        if rows and not any(r.get("rubric_partial_credit") is not None for r in rows)
    ]
    if legacy:
        print("WARNING: no rubric_partial_credit in " + ", ".join(legacy))
        print("         Those runs predate weighted scoring; their reward is all-or-nothing")
        print("         and is not comparable to Partial Credit. Rescore them first with")
        print("         scripts/rescore_rubrics.py.\n")

    # question -> {run_name: (partial_credit, rubric_fraction, judge_error)}
    table: dict[str, dict[str, tuple]] = {}
    for name, rows in runs.items():
        for i, r in enumerate(rows):
            partial_credit = r.get("rubric_partial_credit")
            if partial_credit is None:
                partial_credit = r.get("reward")
            table.setdefault(key_of(r, i), {})[name] = (
                partial_credit,
                r.get("rubric_fraction"),
                r.get("judge_error"),
            )

    n_match = n_diff = n_judgefail = 0
    print(f"{'question':60}  " + "  ".join(f"{n[:16]:>16}" for n in names))
    print("-" * (62 + 18 * len(names)))
    for q, per_run in table.items():
        cells = []
        outcomes = []
        any_judge_fail = False
        for n in names:
            val = per_run.get(n)
            if val is None:
                cells.append(f"{'MISSING':>16}")
                outcomes.append(None)
                continue
            partial_credit, fraction, jerr = val
            if jerr:
                any_judge_fail = True
                cells.append(f"{'judge_fail':>16}")
            elif fraction is None:
                cells.append(f"{partial_credit!s:>16}")
            else:
                shown = f"{partial_credit:.2f}" if isinstance(partial_credit, (int, float)) else str(partial_credit)
                cells.append(f"{f'{shown} ({fraction:.2f})':>16}")
            outcomes.append((partial_credit, fraction))
        present = [x for x in outcomes if x is not None]
        differs = len(set(present)) > 1
        if any_judge_fail:
            n_judgefail += 1
        if differs:
            n_diff += 1
        else:
            n_match += 1
        flag = "  <-- DIFF" if differs else ""
        print(f"{q[:60]:60}  " + "  ".join(cells) + flag)

    print("-" * (62 + 18 * len(names)))
    print(f"matched (same partial credit + fraction):    {n_match}")
    print(f"DIFFERENT outcome across runs:               {n_diff}")
    print(f"rows with a judge failure in >=1 run:        {n_judgefail}")

    def mean_of(name: str, index: int) -> str:
        vals = [
            v[index]
            for v in (per_run.get(name) for per_run in table.values())
            if v is not None and not v[2] and isinstance(v[index], (int, float))
        ]
        return f"{sum(vals) / len(vals):.3f} (n={len(vals)})" if vals else "n/a"

    for label, index in (("mean partial credit", 0), ("mean rubric_fraction", 1)):
        print(f"\n{label} (judge-failures excluded):")
        for n in names:
            print(f"  {n}: {mean_of(n, index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
