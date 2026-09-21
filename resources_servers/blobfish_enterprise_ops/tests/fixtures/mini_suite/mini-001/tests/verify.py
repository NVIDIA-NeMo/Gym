#!/usr/bin/env python3
"""Deterministic verifier: reads MINI_EVIDENCE_PATH, writes reward.json to VERIFIER_LOG_DIR."""

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
task = json.loads((HERE / "task.json").read_text())
evidence = json.loads(Path(os.environ["MINI_EVIDENCE_PATH"]).read_text())
snapshot, trace, expected = evidence["snapshot"], evidence["trace"], task["expected"]
orders = {o["order_id"]: o for o in snapshot["orders"]}
subs = {s["task_id"]: s for s in snapshot["submissions"]}
checks = {
    "investigated": any(
        t["tool"] == "orders.get" and t["success"] and t["arguments"].get("order_id") == expected["order_id"]
        for t in trace
    ),
    "shipped": orders.get(expected["order_id"], {}).get("status") == expected["status"],
    "submitted": abs(float(subs.get(task["task_id"], {}).get("order_total", -1)) - expected["order_total"]) < 1e-6,
    "untouched": all(orders[o]["status"] == "packed" for o in task["untouched"]),
}
score = sum(checks.values()) / len(checks)
verdict = {
    "task_id": task["task_id"],
    "checks": checks,
    "reward": score,
    "strict_pass": all(checks.values()),
    "failed": [k for k, v in checks.items() if not v],
}
logdir = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
logdir.mkdir(parents=True, exist_ok=True)
(logdir / "reward.json").write_text(json.dumps({"reward": score}))
print(json.dumps(verdict, sort_keys=True))
