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
order). Judge failures (judge_error set / judge_rating is null) are reported
separately, since a 0.0 there is not a meaningful "incorrect".

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

    # question -> {run_name: (reward, rating, judge_error)}
    table: dict[str, dict[str, tuple]] = {}
    for name, rows in runs.items():
        for i, r in enumerate(rows):
            table.setdefault(key_of(r, i), {})[name] = (
                r.get("reward"),
                r.get("judge_rating"),
                r.get("judge_error"),
            )

    base = names[0]
    n_match = n_diff = n_judgefail = 0
    print(f"{'question':60}  " + "  ".join(f"{n[:16]:>16}" for n in names))
    print("-" * (62 + 18 * len(names)))
    for q, per_run in table.items():
        cells = []
        rewards = []
        any_judge_fail = False
        for n in names:
            val = per_run.get(n)
            if val is None:
                cells.append(f"{'MISSING':>16}")
                rewards.append(None)
                continue
            reward, rating, jerr = val
            if jerr or rating is None:
                any_judge_fail = True
                cells.append(f"{'judge_fail':>16}")
            else:
                cells.append(f"{reward!s:>16}")
            rewards.append(reward)
        present = [x for x in rewards if x is not None]
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
    print(f"matched (same reward across runs): {n_match}")
    print(f"DIFFERENT reward across runs:      {n_diff}")
    print(f"rows with a judge failure in >=1 run: {n_judgefail}")

    def mean_reward(name: str) -> str:
        vals = [
            v[0]
            for v in (per_run.get(name) for per_run in table.values())
            if v is not None and not v[2] and v[1] is not None and isinstance(v[0], (int, float))
        ]
        return f"{sum(vals) / len(vals):.3f} (n={len(vals)})" if vals else "n/a"

    print("\nmean reward (judge-failures excluded):")
    for n in names:
        print(f"  {n}: {mean_reward(n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
