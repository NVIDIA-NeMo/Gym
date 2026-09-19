"""Tiny synthetic world for adapter tests: two orders in SQLite, three tools, a full trace."""

from __future__ import annotations

import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any


TABLES = ("orders", "lines", "submissions")


def seed_database(task: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    if path.exists():
        path.unlink()
    cx = sqlite3.connect(path)
    cx.executescript(
        "CREATE TABLE orders(order_id TEXT PRIMARY KEY, customer TEXT, status TEXT);"
        "CREATE TABLE lines(order_id TEXT, sku TEXT, qty INTEGER, unit_price REAL);"
        "CREATE TABLE submissions(task_id TEXT PRIMARY KEY, order_total REAL);"
    )
    for row in task["seed"]["orders"]:
        cx.execute("INSERT INTO orders VALUES (?, ?, ?)", (row["order_id"], row["customer"], row["status"]))
    for row in task["seed"]["lines"]:
        cx.execute(
            "INSERT INTO lines VALUES (?, ?, ?, ?)", (row["order_id"], row["sku"], row["qty"], row["unit_price"])
        )
    cx.commit()
    cx.close()
    return path


class MiniWorld:
    def __init__(self, task: dict[str, Any], database_path: Path):
        self.task = task
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self.trace: list[dict[str, Any]] = []

    @classmethod
    def fresh(cls, task: dict[str, Any], database_path: Path) -> "MiniWorld":
        return cls(task, seed_database(task, database_path))

    def close(self) -> None:
        self.connection.close()

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {t: [dict(r) for r in self.connection.execute(f"SELECT * FROM {t} ORDER BY 1")] for t in TABLES}

    def call_tool(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        a = dict(arguments or {})
        try:
            if tool == "orders.get":
                order = self.connection.execute("SELECT * FROM orders WHERE order_id=?", (a["order_id"],)).fetchone()
                if order is None:
                    raise ValueError("record not found")
                lines = [
                    dict(r) for r in self.connection.execute("SELECT * FROM lines WHERE order_id=?", (a["order_id"],))
                ]
                result: dict[str, Any] = {**dict(order), "lines": lines}
            elif tool == "orders.update_status":
                if (
                    self.connection.execute(
                        "UPDATE orders SET status=? WHERE order_id=?", (a["status"], a["order_id"])
                    ).rowcount
                    != 1
                ):
                    raise ValueError("record not found")
                result = {"ok": True}
            elif tool == "benchmark.submit_answer":
                if a.get("task_id") != self.task["task_id"]:
                    raise ValueError("task not found")
                self.connection.execute(
                    "INSERT OR REPLACE INTO submissions VALUES (?, ?)", (a["task_id"], float(a["order_total"]))
                )
                result = {"ok": True}
            else:
                raise ValueError(f"unknown tool: {tool}")
            self.connection.commit()
            success = True
        except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            self.connection.rollback()
            result = {"error": str(exc)}
            success = False
        self.trace.append(
            {
                "index": len(self.trace),
                "tool": tool,
                "arguments": deepcopy(a),
                "success": success,
                "result": deepcopy(result),
            }
        )
        return result
