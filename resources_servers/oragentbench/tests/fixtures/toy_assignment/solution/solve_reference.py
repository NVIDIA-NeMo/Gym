# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference solver for the synthetic toy assignment task (brute force; optimum is 12)."""

import argparse
import csv
import itertools
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    costs = {}
    with open(Path(args.env_dir) / "data" / "costs.csv") as f:
        for row in csv.DictReader(f):
            costs[(row["worker"], row["job"])] = float(row["cost"])
    workers = sorted({w for w, _ in costs})
    jobs = sorted({j for _, j in costs})
    best = min(itertools.permutations(jobs), key=lambda perm: sum(costs[(w, j)] for w, j in zip(workers, perm)))
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["worker", "job"])
        writer.writerows(zip(workers, best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
