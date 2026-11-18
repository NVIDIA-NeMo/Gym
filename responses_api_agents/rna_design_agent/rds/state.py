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
"""Campaign state: candidates in SQLite, every step in a JSONL trace.

A controller sees a bounded view, never the whole table. The trace stores each step whole, untruncated.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id          INTEGER PRIMARY KEY,
    sequence    TEXT NOT NULL UNIQUE,
    round       INTEGER NOT NULL,
    source      TEXT NOT NULL,
    scores      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS candidates_round ON candidates(round);
"""


def _producer_of(metric: str) -> str:
    """The registered tool whose name prefixes this score key, if any."""
    from . import registry

    candidates = [spec.name for spec in registry.all_tools() if metric.startswith(f"{spec.name}.")]
    return max(candidates, key=len) if candidates else "unknown"


def _missing_scores(rows: list[dict], objective: dict[str, Any]) -> dict[str, Any]:
    needed = [objective["rank_candidates_by"]] if objective.get("rank_candidates_by") else []
    needed += [k for k in (objective.get("gates") or {}) if k not in needed]
    report = {}
    for metric in needed:
        n = sum(1 for r in rows if metric not in r["scores"])
        if n:
            report[metric] = {"candidates_missing": n, "produced_by": _producer_of(metric)}
    return report


@dataclass
class StateView:
    """What a controller sees for one round. Bounded on purpose."""

    round: int
    budget: dict[str, Any]
    funnel: dict[str, int]
    top: list[dict[str, Any]]
    population: dict[str, Any]
    recent_failures: list[str] = field(default_factory=list)
    #: What the campaign is ranked on, and what a candidate must pass to count.
    objective: dict[str, Any] = field(default_factory=dict)
    #: How to name candidates already in the state, for a tool's `sequences`.
    selectors: dict[str, str] = field(default_factory=dict)
    #: Per needed metric: how many candidates lack it, and which tool produces it.
    missing_scores: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """The state a controller reads. An unset budget limit shows as "unlimited", since 0 reads as spent."""
        return json.dumps(
            {
                "objective": self.objective,
                "sequence_selectors": self.selectors,
                "missing_scores": self.missing_scores,
                "round": self.round,
                "budget": _readable_budget(self.budget),
                "funnel": self.funnel,
                "top_candidates": self.top,
                "population": self.population,
                "recent_failures": self.recent_failures,
            },
            indent=1,
            sort_keys=True,
        )


def _readable_budget(budget: dict) -> dict:
    limits = {k: (v if v else "unlimited") for k, v in (budget.get("limits") or {}).items()}
    spend = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in (budget.get("spend") or {}).items()}
    return {"limits": limits, "spend": spend}


class CampaignState:
    """Candidates, scores and the trace for one campaign run."""

    def __init__(self, directory: str | Path, direction: str = "max") -> None:
        # Which end of the ranked metric is best, so `@top:N` and the view list a minimised metric best first.
        self.best_is_high = direction == "max"
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "state.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.trace_path = self.dir / "trace.jsonl"
        self.round = 0

    # --- candidates ---------------------------------------------------
    def add_candidates(self, sequences: list[str], source: str, limit: int | None = None) -> int:
        """Insert sequences, ignoring ones already seen, at most `limit` new ones. Returns how many are new."""
        before = self.count()
        for sequence in sequences:
            if limit is not None and self.count() - before >= limit:
                break
            self.db.execute(
                "INSERT OR IGNORE INTO candidates(sequence, round, source) VALUES (?, ?, ?)",
                (sequence, self.round, source),
            )
        self.db.commit()
        return self.count() - before

    def score(self, scores_by_sequence: dict[str, dict]) -> int:
        """Merge scores into existing candidates. Unknown sequences are ignored.

        Returns how many candidates changed, so re-scoring with the same tool does not read as progress.
        """
        updated = 0
        for sequence, new_scores in scores_by_sequence.items():
            row = self.db.execute("SELECT scores FROM candidates WHERE sequence = ?", (sequence,)).fetchone()
            if row is None:
                continue
            existing = json.loads(row["scores"])
            merged = {**existing, **new_scores}
            if merged == existing:
                continue
            self.db.execute("UPDATE candidates SET scores = ? WHERE sequence = ?", (json.dumps(merged), sequence))
            updated += 1
        self.db.commit()
        return updated

    def count(self, where: str = "") -> int:
        sql = "SELECT COUNT(*) FROM candidates" + (f" WHERE {where}" if where else "")
        return int(self.db.execute(sql).fetchone()[0])

    def candidates(self, limit: int = 0) -> list[dict]:
        sql = "SELECT sequence, round, source, scores FROM candidates ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [{**dict(r), "scores": json.loads(r["scores"])} for r in self.db.execute(sql)]

    def select(self, selector: str, rank_by: str = "", by_tool: str = "") -> list[str]:
        """Resolve a `@...` selector into the sequences it names, so a controller need not quote them back."""
        name, _, argument = selector.lstrip("@").partition(":")
        rows = self.candidates()
        if name == "unscored":
            # Unscored by the tool being called, not by the ranked metric.
            if by_tool:
                rows = [r for r in rows if not any(k.startswith(f"{by_tool}.") for k in r["scores"])]
            else:
                rows = [r for r in rows if rank_by not in r["scores"]]
        elif name == "top":
            rows = [r for r in rows if rank_by in r["scores"]]
            rows.sort(key=lambda r: r["scores"][rank_by], reverse=self.best_is_high)
        elif name != "all":
            raise ValueError(f"unknown selector {selector!r}; use @all, @unscored or @top:N")
        if argument:
            rows = rows[: int(argument)]
        if not rows:
            # Fail with the way out, or a controller reads an empty selection as success and repeats it.
            if not self.count():
                raise ValueError(
                    f"{selector!r} selected no candidates: there are none yet. "
                    f"Nothing can be scored before something is designed — "
                    f"call the generator, or write sequences of your own and "
                    f"pass them to a scorer, which is how they enter the "
                    f"population."
                )
            remedy = (
                "every candidate already carries this tool's score — generate more, or call a different scorer"
                if by_tool
                else "@all returns every candidate; @top:N needs candidates scored on the ranked metric first"
            )
            raise ValueError(
                f"{selector!r} selected no candidates "
                f"({self.count()} in state"
                + (
                    f"; {by_tool} has already scored them all). " if by_tool else "). "
                )  # no metric name: it may be hidden
                + remedy
            )
        return [r["sequence"] for r in rows]

    # --- trace ---------------------------------------------------------
    def record(self, kind: str, **payload: Any) -> None:
        """Append one event, whole. Nothing here is truncated."""
        event = {"ts": time.time(), "round": self.round, "kind": kind, **payload}
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    # --- the controller's view -----------------------------------------
    def view(
        self,
        budget_snapshot: dict,
        rank_by: str,
        top_n: int = 10,
        failures: list[str] | None = None,
        objective: dict[str, Any] | None = None,
        selectors: dict[str, str] | None = None,
    ) -> StateView:
        rows = self.candidates()
        scored = [r for r in rows if rank_by in r["scores"]]
        scored.sort(key=lambda r: r["scores"][rank_by], reverse=self.best_is_high)
        values = [r["scores"][rank_by] for r in scored]
        population: dict[str, Any] = {"n": len(rows), "n_scored": len(scored)}
        if values:
            population[f"{rank_by}_median"] = values[len(values) // 2]
            population[f"{rank_by}_best"] = values[0]
        # When the campaign hides its scoring, the view must not rank by it or name it.
        shown = (objective or {}).get("rank_candidates_by")
        if not shown:
            population = {"n": len(rows), "n_measured": len(scored)}
        return StateView(
            round=self.round,
            budget=budget_snapshot,
            funnel={"candidates": len(rows), "scored": len(scored)},
            top=[
                {
                    "sequence": (r["sequence"] if len(r["sequence"]) <= 40 else r["sequence"][:40] + "..."),
                    "source": r["source"],
                    "scores": r["scores"],
                }
                for r in (scored if shown else rows)[:top_n]
            ],
            population=population,
            recent_failures=(failures or [])[-5:],
            objective={
                **(objective or {}),
                **({"unscored_candidates": len(rows) - len(scored)} if shown else {"candidates_so_far": len(rows)}),
            },
            selectors=selectors or {},
            missing_scores=_missing_scores(rows, objective or {}) if shown else {},
        )

    def close(self) -> None:
        self.db.close()
