# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarise repeated NMRArena sweeps and compare them with a published mean ± SD.

Upstream reports each LLM as the mean and sample SD (ddof=1) over three sweeps of
the 105 molecules, for Top-1 %, Top-10 % and the rank-1 Tanimoto conditional on the
rank-1 candidate parsing. This script computes the same three numbers from each
rollout file, summarises them the same way, and runs Welch's t-test between our
summary and the published one. The task set is fixed and shared, so decoding
variance across sweeps is the only source of disagreement the test addresses;
with three runs a side the test has little power and a non-significant result is
not evidence of agreement.

Usage::

    python resources_servers/nmrarena/scripts/summarize_runs.py \
        --run results/nmrarena/gemini_run1.jsonl --run ... \
        --published top1=27.6,1.9 --published top10=32.4,1.9 --published tanimoto=0.59,0.02 \
        --published-n 3 --output results/nmrarena/gemini_summary.json
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Optional

from scipy import stats


METRICS = ("top1", "top10", "tanimoto")


def per_run_metrics(rows: list[dict]) -> dict:
    """Upstream's three numbers for one sweep, plus the counts behind them."""
    n = len(rows)
    tani = [r["tanimoto_top1"] for r in rows if isinstance(r.get("tanimoto_top1"), (int, float))]
    return {
        "n": n,
        "top1": 100.0 * sum(r["top1"] for r in rows) / n,
        "top10": 100.0 * sum(r["top10"] for r in rows) / n,
        "tanimoto": statistics.mean(tani) if tani else None,
        "tanimoto_n": len(tani),
        "answered": sum(r["answered"] for r in rows),
        "response_incomplete": sum(1 for r in rows if r.get("response_incomplete")),
        "harness_failures": sum(1 for r in rows if r.get("harness_failure")),
        "statuses": {s: sum(1 for r in rows if r.get("status") == s) for s in sorted({r.get("status") for r in rows})},
        "salvaged": sum(1 for r in rows if r.get("salvaged")),
    }


def welch(mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int) -> Optional[dict]:
    """Welch's t-test from two summaries; ``None`` when both spreads are zero."""
    var_a, var_b = sd_a**2 / n_a, sd_b**2 / n_b
    if var_a + var_b == 0:
        return None
    t = (mean_a - mean_b) / math.sqrt(var_a + var_b)
    df = (var_a + var_b) ** 2 / (
        (var_a**2 / (n_a - 1) if n_a > 1 else 0.0) + (var_b**2 / (n_b - 1) if n_b > 1 else 0.0)
    )
    p = 2 * stats.t.sf(abs(t), df)
    return {"t": t, "df": df, "p_two_sided": p, "difference": mean_a - mean_b}


def summarise(per_run: list[dict], published: dict, published_n: int) -> dict:
    out = {}
    for metric in METRICS:
        values = [r[metric] for r in per_run if r[metric] is not None]
        if not values:
            continue
        entry = {
            "runs": values,
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n_runs": len(values),
        }
        if metric in published and entry["sd"] is not None:
            pub_mean, pub_sd = published[metric]
            entry["published"] = {"mean": pub_mean, "sd": pub_sd, "n_runs": published_n}
            entry["welch"] = welch(entry["mean"], entry["sd"], len(values), pub_mean, pub_sd, published_n)
        out[metric] = entry
    return out


def _published(value: str) -> tuple[str, tuple[float, float]]:
    name, rest = value.split("=", 1)
    mean, sd = (float(x) for x in rest.split(","))
    if name not in METRICS:
        raise argparse.ArgumentTypeError(f"metric must be one of {METRICS}")
    return name, (mean, sd)


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="Rollout JSONL of one sweep; repeat per sweep")
    parser.add_argument("--published", action="append", type=_published, default=[], help="metric=mean,sd")
    parser.add_argument("--published-n", type=int, default=3, help="Sweeps behind the published SD")
    parser.add_argument("--output", default=None, help="Write the JSON summary here")
    args = parser.parse_args(argv)

    per_run = []
    for path in args.run:
        raw = Path(path).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        per_run.append(dict(per_run_metrics(rows), file=path, sha256=hashlib.sha256(raw).hexdigest()))
    report = {"per_run": per_run, "summary": summarise(per_run, dict(args.published), args.published_n)}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, file=sys.stdout)
    return report


if __name__ == "__main__":
    main()
