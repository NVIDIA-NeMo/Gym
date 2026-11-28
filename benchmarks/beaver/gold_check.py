# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gold check for BEAVER: run every gold query the way the grader does and count the outcomes.

Each gold query runs through the ``beaver_sql`` grader's own ``MySqlRunner`` (the grader's user, 10 seconds,
``KILL QUERY`` on a timeout), once, and is classed as:

- ``nonempty``: ran and returned rows. A correct model query can match it.
- ``empty``: ran and returned no rows. Any query that returns nothing, or fails, matches it.
- ``error`` (with MySQL's error number) or ``timeout``: officially an empty result too, so the same holds.

It also reports how long the gold queries take: one close to the 10-second limit can time out under load, which
changes its task's class from run to run. Writes ``results.jsonl`` (id, domain, class, error, seconds) and prints
counts. Never prints a query or a result.

Run: ``python -m benchmarks.beaver.gold_check <prepared jsonl> <out_dir> --dsn <reader dsn> [--concurrency N]``, with
the DSN's password in ``BEAVER_MYSQL_PASSWORD`` (never in the DSN).
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from resources_servers.beaver_sql.ex_acc import MySqlRunner, MySqlUnavailableError


def gold_class(status: str, empty: bool) -> str:
    if status == "ok":
        return "empty" if empty else "nonempty"
    return status


async def check(
    rows: List[Dict[str, Any]], dsn: str, concurrency: int, password: Optional[str] = None
) -> List[Dict[str, Any]]:
    runner = MySqlRunner(dsn, password=password, max_concurrency=concurrency)

    async def one(row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            outcome = await runner.run(row["db_id"], row["gt_sql"])
        except MySqlUnavailableError as e:
            return {"id": row["id"], "domain": row.get("domain"), "class": "unavailable", "error": str(e)[:200]}
        return {
            "id": row["id"],
            "domain": row.get("domain"),
            "class": gold_class(outcome.status, outcome.empty),
            "error": outcome.error,
            "seconds": round(outcome.seconds, 3),
            "rows": None if outcome.frame is None else len(outcome.frame),
        }

    return await asyncio.gather(*(one(r) for r in rows))


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"tasks": len(results), "by_class": dict(Counter(r["class"] for r in results))}
    for domain in sorted({r["domain"] for r in results if r.get("domain")}):
        summary[f"by_class/{domain}"] = dict(Counter(r["class"] for r in results if r.get("domain") == domain))
    summary["errors"] = dict(Counter(r["error"] for r in results if r["class"] == "error"))
    seconds = sorted(r["seconds"] for r in results if "seconds" in r)
    if seconds:
        summary["seconds"] = {
            "median": seconds[len(seconds) // 2],
            "p99": seconds[min(len(seconds) - 1, int(len(seconds) * 0.99))],
            "max": seconds[-1],
            "over_5s": sum(s > 5 for s in seconds),
            "over_8s": sum(s > 8 for s in seconds),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prepared", help="Prepared BEAVER rows (benchmarks/beaver/data/beaver_*.jsonl)")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--dsn", required=True, help="The grader's MySQL DSN (read-only user; password in BEAVER_MYSQL_PASSWORD)"
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Queries at a time; the grader's default is 8")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.prepared).read_text().splitlines() if line.strip()]
    start = time.monotonic()
    results = asyncio.run(check(rows, args.dsn, args.concurrency, os.environ.get("BEAVER_MYSQL_PASSWORD")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = summarize(results)
    summary["wall_seconds"] = round(time.monotonic() - start, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
