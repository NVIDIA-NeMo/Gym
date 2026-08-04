#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two or more finance_agent_v2 rollout JSONLs by per-question reward.

Used for the cache-validation experiment:
  A = empty-cache run, B = full-cache run  -> expect identical (cache fidelity)
  C = run against a perturbed COPY of the cache (adjusted columns rescaled by
      perturb_price_cache.py) -> mismatches flag questions whose answer depends
      on an absolute *adjusted* price level.

Rows are matched across files by their `question` text (falls back to line
order). Judge failures (`judge_error` set) are reported separately, since a 0.0
there is not a meaningful "incorrect".

Runs are compared on `rubric_fraction` as well as `reward`. Reward is
all-or-nothing, so on its own it would hide a perturbation that flips one
criterion of a question that was already failing another.

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
    q = row.get("question")
    return q if isinstance(q, str) and q else f"__row_{idx}"


def main(paths: list[str]) -> int:
    if len(paths) < 2:
        print(__doc__)
        return 2

    runs = {Path(p).name: load(p) for p in paths}
    names = list(runs)

    # question -> {run_name: (reward, rubric_fraction, judge_error)}
    table: dict[str, dict[str, tuple]] = {}
    for name, rows in runs.items():
        for i, r in enumerate(rows):
            table.setdefault(key_of(r, i), {})[name] = (
                r.get("reward"),
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
            reward, fraction, jerr = val
            if jerr:
                any_judge_fail = True
                cells.append(f"{'judge_fail':>16}")
            elif fraction is None:
                cells.append(f"{reward!s:>16}")
            else:
                cells.append(f"{f'{reward} ({fraction:.2f})':>16}")
            outcomes.append((reward, fraction))
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
    print(f"matched (same reward + fraction across runs): {n_match}")
    print(f"DIFFERENT outcome across runs:               {n_diff}")
    print(f"rows with a judge failure in >=1 run:        {n_judgefail}")

    def mean_of(name: str, index: int) -> str:
        vals = [
            v[index]
            for v in (per_run.get(name) for per_run in table.values())
            if v is not None and not v[2] and isinstance(v[index], (int, float))
        ]
        return f"{sum(vals) / len(vals):.3f} (n={len(vals)})" if vals else "n/a"

    for label, index in (("mean reward", 0), ("mean rubric_fraction", 1)):
        print(f"\n{label} (judge-failures excluded):")
        for n in names:
            print(f"  {n}: {mean_of(n, index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
