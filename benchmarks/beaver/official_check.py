# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the ``beaver_sql`` grader against BEAVER's official evaluator on the same predictions.

Downloads the official ``eval/utils/ex_acc.py`` at a pinned commit of https://github.com/beaverbench/beaver into a
local cache (the repository has no license file, so nothing from it is committed here) and scores every prediction
twice on the same MySQL server:

- officially: ``execute_sql_with_timeout`` for the gold and the predicted query, a failure as an empty DataFrame, then
  ``compare_results``, as in the official ``evaluate_ex_acc.py``;
- with Gym's grader: ``MySqlRunner`` and ``results_match`` from ``resources_servers/beaver_sql/ex_acc.py``.

Predictions come from a rollouts file (any rows with ``id`` and ``model_output``; the SQL is taken with the grader's
own ``extract_sql_from_response``) or, with ``--gold-as-prediction``, are the gold queries themselves. Writes
``results.jsonl`` (id, both verdicts, timings) and prints agreement counts. A disagreement where any query took more
than 8 seconds is reported apart: it may be timing near the 10-second limit, not a rule difference. When the rollouts
carry a ``reward`` (another grader's verdict, such as auto-ontology-eval's), that is compared with the official
verdict too.

Run: ``python -m benchmarks.beaver.official_check <prepared jsonl> <out_dir> --dsn <reader dsn>
(--rollouts <jsonl> | --gold-as-prediction) [--concurrency N]``.
"""

import argparse
import asyncio
import importlib.util
import json
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from resources_servers.beaver_sql.app import extract_sql_from_response
from resources_servers.beaver_sql.ex_acc import MySqlRunner, missing_prediction, results_match
from resources_servers.beaver_sql.setup_beaver_mysql import parse_mysql_dsn


OFFICIAL_COMMIT = "81a2a88a1a2e236281c5d09b3171edd7580f8ec9"
OFFICIAL_URL = f"https://raw.githubusercontent.com/beaverbench/beaver/{OFFICIAL_COMMIT}/eval/utils/ex_acc.py"
DEFAULT_CACHE = Path(__file__).parent / "data" / "official"
BORDERLINE_S = 8.0


def load_official(cache_dir: Path):
    """The official ``ex_acc`` module, downloaded once into ``cache_dir`` (git-ignored)."""
    path = cache_dir / f"ex_acc_{OFFICIAL_COMMIT[:9]}.py"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(OFFICIAL_URL, timeout=60) as response:
            path.write_bytes(response.read())
    spec = importlib.util.spec_from_file_location("beaver_official_ex_acc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_verdict(official, creds: Dict[str, Any], gold_sql: str, pred_sql: Optional[str]) -> Dict[str, Any]:
    """One task scored as the official ``evaluate_ex_acc.py`` does."""
    import pandas as pd

    start = time.monotonic()
    gold_df, gold_err = official.execute_sql_with_timeout(gold_sql, creds)
    gold_s = time.monotonic() - start
    if gold_df is None:
        gold_df = pd.DataFrame()
    pred_df, pred_err, pred_s = None, None, 0.0
    if pred_sql:
        start = time.monotonic()
        pred_df, pred_err = official.execute_sql_with_timeout(pred_sql, creds)
        pred_s = time.monotonic() - start
    if pred_df is None:
        pred_df = pd.DataFrame()
    match, message = official.compare_results(pred_df, gold_df)
    return {
        "match": bool(match),
        "message": message,
        "gold_empty": bool(gold_df.empty),
        "gold_error": gold_err,
        "pred_error": pred_err,
        "seconds": round(max(gold_s, pred_s), 3),
    }


async def gym_verdict(runner: MySqlRunner, db_id: str, gold_sql: str, pred_sql: Optional[str]) -> Dict[str, Any]:
    gold = await runner.run(db_id, gold_sql)
    pred = missing_prediction() if pred_sql is None else await runner.run(db_id, pred_sql)
    return {
        "match": results_match(gold, pred),
        "gold_status": gold.status,
        "gold_empty": gold.empty,
        "pred_status": pred.status,
        "seconds": round(max(gold.seconds, pred.seconds), 3),
    }


def tasks_from(prepared: List[Dict[str, Any]], rollouts: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    gold = {row["id"]: row for row in prepared}
    if rollouts is None:
        return [{"id": r["id"], "db_id": r["db_id"], "gt_sql": r["gt_sql"], "pred_sql": r["gt_sql"]} for r in prepared]
    tasks = []
    for i, r in enumerate(rollouts):
        task_id = str(r["id"])
        if task_id not in gold:
            continue
        tasks.append(
            {
                "id": task_id,
                "rollout": r.get("_ng_rollout_index", i),
                "db_id": gold[task_id]["db_id"],
                "gt_sql": gold[task_id]["gt_sql"],
                "pred_sql": extract_sql_from_response(r.get("model_output")),
                "other_reward": r.get("reward"),
            }
        )
    return tasks


async def check(tasks: List[Dict[str, Any]], dsn: str, official, concurrency: int) -> List[Dict[str, Any]]:
    runner = MySqlRunner(dsn, max_concurrency=concurrency)
    params = parse_mysql_dsn(dsn)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    loop = asyncio.get_running_loop()

    async def one(task: Dict[str, Any]) -> Dict[str, Any]:
        creds = {k: params[k] for k in ("host", "port", "user", "password")}
        creds["database"] = task["db_id"]
        off = await loop.run_in_executor(pool, official_verdict, official, creds, task["gt_sql"], task["pred_sql"])
        gym = await gym_verdict(runner, task["db_id"], task["gt_sql"], task["pred_sql"])
        result = {
            "id": task["id"],
            "rollout": task.get("rollout"),
            "official": off["match"],
            "gym": gym["match"],
            "official_message": off["message"],
            "gold_empty_official": off["gold_empty"],
            "gold_empty_gym": gym["gold_empty"],
            "pred_status_gym": gym["pred_status"],
            "has_prediction": task["pred_sql"] is not None,
            "borderline": max(off["seconds"], gym["seconds"]) > BORDERLINE_S,
        }
        if task.get("other_reward") is not None:
            result["other"] = task["other_reward"] > 0
        return result

    try:
        return await asyncio.gather(*(one(t) for t in tasks))
    finally:
        pool.shutdown(wait=True)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    disagree = [r for r in results if r["official"] != r["gym"]]
    summary: Dict[str, Any] = {
        "tasks": len(results),
        "official_matches": sum(r["official"] for r in results),
        "gym_matches": sum(r["gym"] for r in results),
        "agree": len(results) - len(disagree),
        "disagree": len(disagree),
        "disagree_borderline_timing": sum(r["borderline"] for r in disagree),
        "nonempty_gold_official": sum(not r["gold_empty_official"] for r in results),
        "nonempty_gold_gym": sum(not r["gold_empty_gym"] for r in results),
        "nonempty_gold_matches_official": sum(r["official"] and not r["gold_empty_official"] for r in results),
        "nonempty_gold_matches_gym": sum(r["gym"] and not r["gold_empty_gym"] for r in results),
    }
    if any("other" in r for r in results):
        other = [r for r in results if "other" in r]
        summary["other_matches"] = sum(r["other"] for r in other)
        summary["other_vs_official"] = dict(
            Counter(f"other={r['other']} official={r['official']}" for r in other if r["other"] != r["official"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prepared", help="Prepared BEAVER rows (benchmarks/beaver/data/beaver_*.jsonl)")
    parser.add_argument("out_dir")
    parser.add_argument("--dsn", required=True, help="The grader's MySQL DSN (read-only user)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rollouts", help="Rollouts with id and model_output")
    source.add_argument("--gold-as-prediction", action="store_true", help="Score each gold query against itself")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--official-cache", default=str(DEFAULT_CACHE))
    args = parser.parse_args()

    read = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]  # noqa: E731
    tasks = tasks_from(read(args.prepared), None if args.gold_as_prediction else read(args.rollouts))
    official = load_official(Path(args.official_cache))
    start = time.monotonic()
    results = asyncio.run(check(tasks, args.dsn, official, args.concurrency))
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
