#!/usr/bin/env python3
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
"""Check agentic_turn_simulation_judge row files against the row schema and print their composition.

Per row (the first failing row stops the run with its line number): the agent_ref name (unless --agent ''), the
TaskData schema with no undeclared top-level key, a prompt of exactly one system and one user message with text
content and no tools, a user message holding the <step_under_evaluation> block, non-empty row_id and group_id (unique
within a file), and gold_fields values the evaluation schema allows.

Then counts per file and over all files: rows, gold_ladder balance (overall and per split), rows by label.tier,
meta.source, meta.family, meta.harness, meta.split, gold_n, prompt size, and consistency counts that do not stop the
run: gold_p_yes on the other side of 0.5 than gold_ladder (expected only for hand-labelled rows), gold_fields whose
ladder differs from gold_ladder, the same prompt in several rows, a group_id in more than one split.

    PYTHONPATH=<Gym root> python validate_rows.py train.jsonl validation.jsonl --report counts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nemo_gym.data import normalize_task_fields
from resources_servers.agentic_turn_simulation_judge.task_data import TaskData
from resources_servers.agentic_turn_simulation_judge.verdict import VERDICT_FIELDS, correct_ladder, evaluation_schema


AGENT = "agentic_turn_simulation_judge_simple_agent"
STEP_TAG = "<step_under_evaluation>"
SLICE_KEYS = (("tier", "label"), ("source", "meta"), ("family", "meta"), ("harness", "meta"), ("split", "meta"))


class RowError(ValueError):
    """A row that does not match the schema."""


def _text(message: Any) -> str:
    if not isinstance(message, dict):
        raise RowError(f"input item is not an object: {type(message).__name__}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RowError(f"{message.get('role')} message content must be a non-empty string")
    return content


def check_row(row: Any, *, agent: str) -> TaskData:
    """The row's validated task data; raises RowError naming the first problem."""
    if not isinstance(row, dict):
        raise RowError("row is not a JSON object")
    if agent and (row.get("agent_ref") or {}).get("name") != agent:
        raise RowError(f"agent_ref {row.get('agent_ref')!r}, wanted name {agent!r}")
    fields, conflicts = normalize_task_fields(row)
    if conflicts:
        raise RowError(f"task fields given twice with different values: {conflicts}")
    try:
        task = TaskData.model_validate(fields)
    except ValidationError as e:
        raise RowError(str(e).replace("\n", "; ")[:500]) from e
    if task.model_extra:
        raise RowError(f"undeclared top-level keys {sorted(task.model_extra)}")

    params = row.get("responses_create_params")
    if not isinstance(params, dict):
        raise RowError("responses_create_params is missing")
    if params.get("tools"):
        raise RowError("responses_create_params.tools must be empty: the judge prompt lists the tools as text")
    messages = params.get("input")
    if not isinstance(messages, list) or [m.get("role") if isinstance(m, dict) else None for m in messages] != [
        "system",
        "user",
    ]:
        raise RowError("responses_create_params.input must be [system message, user message]")
    _text(messages[0])
    if STEP_TAG not in _text(messages[1]):
        raise RowError(f"user message has no {STEP_TAG} block")

    for key in ("row_id", "group_id"):
        if not (getattr(task, key) or "").strip():
            raise RowError(f"{key} is missing")
    if task.gold_fields is not None:
        properties = evaluation_schema()["properties"]
        for key in VERDICT_FIELDS:
            value = getattr(task.gold_fields, key)
            if value is not None and value not in properties[key]["enum"]:
                raise RowError(f"gold_fields.{key} {value!r} is not one of {properties[key]['enum']}")
    return task


def _prompt_key(row: dict[str, Any]) -> str:
    system, user = row["responses_create_params"]["input"]
    digest = hashlib.sha256()
    for text in (system["content"], user["content"]):
        digest.update(hashlib.sha256(text.encode()).digest())
    return digest.hexdigest()


def _split(task: TaskData) -> str:
    return (task.meta.split if task.meta is not None else None) or "none"


def _percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class Tally:
    """Counts over a set of rows."""

    def __init__(self) -> None:
        self.rows = 0
        self.gold = Counter()
        self.gold_by_split: dict[str, Counter] = defaultdict(Counter)
        self.slices: dict[str, Counter] = {name: Counter() for name, _ in SLICE_KEYS}
        self.gold_n = Counter()
        self.prompt_chars: list[int] = []
        self.p_yes_disagrees = 0
        self.fields_ladder_disagrees = 0

    def add(self, row: dict[str, Any], task: TaskData) -> None:
        self.rows += 1
        self.gold[task.gold_ladder] += 1
        sections = {"label": task.label, "meta": task.meta}
        for name, section in SLICE_KEYS:
            value = getattr(sections[section], name, None) if sections[section] is not None else None
            self.slices[name][str(value) if value is not None else "none"] += 1
        self.gold_by_split[_split(task)][task.gold_ladder] += 1
        self.gold_n[str(task.gold_n)] += 1
        system, user = row["responses_create_params"]["input"]
        self.prompt_chars.append(len(system["content"]) + len(user["content"]))
        if task.gold_p_yes is not None and task.gold_p_yes != 0.5:
            if (task.gold_p_yes > 0.5) != (task.gold_ladder == "yes"):
                self.p_yes_disagrees += 1
        if task.gold_fields is not None:
            ladder = correct_ladder(task.gold_fields.model_dump())
            if ladder is not None and ladder != task.gold_ladder:
                self.fields_ladder_disagrees += 1

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "gold_ladder": dict(self.gold.most_common()),
            "gold_yes_rate": round(self.gold["yes"] / self.rows, 4) if self.rows else None,
            "gold_ladder_by_split": {k: dict(v.most_common()) for k, v in sorted(self.gold_by_split.items())},
            **{f"by_{name}": dict(counter.most_common()) for name, counter in self.slices.items()},
            "gold_n": dict(sorted(self.gold_n.items())),
            "prompt_chars": {
                "p50": _percentile(self.prompt_chars, 0.5),
                "p90": _percentile(self.prompt_chars, 0.9),
                "max": max(self.prompt_chars, default=None),
            },
            "p_yes_disagrees_with_gold": self.p_yes_disagrees,
            "gold_fields_ladder_disagrees_with_gold": self.fields_ladder_disagrees,
        }


def validate_files(paths: list[Path], *, agent: str = AGENT) -> dict[str, Any]:
    """Counts per file and over all files; raises SystemExit naming the first row that fails."""
    report: dict[str, Any] = {}
    total = Tally()
    prompts: Counter = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        tally = Tally()
        seen_ids: set[str] = set()
        with open(path, "rb", buffering=1 << 22) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    task = check_row(row, agent=agent)
                    if task.row_id in seen_ids:
                        raise RowError(f"duplicate row_id {task.row_id!r}")
                except (json.JSONDecodeError, RowError) as e:
                    raise SystemExit(f"{path}:{n}: {e}") from e
                seen_ids.add(task.row_id)
                tally.add(row, task)
                total.add(row, task)
                prompts[_prompt_key(row)] += 1
                # A row without meta.split is in its file's split.
                split = _split(task)
                group_splits[task.group_id].add(path.name if split == "none" else split)
        report[path.name] = tally.summary()
    leaked = sorted(g for g, splits in group_splits.items() if len(splits) > 1)
    report["all"] = {
        **total.summary(),
        "duplicate_prompts": sum(c - 1 for c in prompts.values() if c > 1),
        "groups_in_multiple_splits": len(leaked),
        "groups_in_multiple_splits_examples": leaked[:5],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--agent", default=AGENT, help=f"expected agent_ref name (default {AGENT}; '' skips it)")
    parser.add_argument("--report", type=Path, help="write the counts as JSON (mode 0600)")
    args = parser.parse_args()
    report = validate_files(args.files, agent=args.agent)
    for name, counts in report.items():
        print(name, json.dumps(counts))
    if args.report:
        with os.fdopen(os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            f.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
