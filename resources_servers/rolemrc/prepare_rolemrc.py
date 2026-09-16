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
"""Build the RoleMRC Gym datasets from ``Junrulu/RoleMRC``.

Downloads the test split and writes two JSONL files:

* ``data/test.jsonl``       — all rows, scored in ``reference`` mode.
* ``data/test_judge.jsonl`` — rows whose ``task`` has a 5-aspect judge config,
  scored in ``judge`` mode.

Each row becomes a Gym task: the RoleMRC conversation goes into
``responses_create_params.input`` (so the model is prompted with the full
multi-turn context), and ``reference`` / ``task`` / ``dimension`` are carried
as extra fields consumed by ``app.py``'s verify().

Usage::

    python resources_servers/rolemrc/prepare_rolemrc.py
    # or read a pre-downloaded local file:
    ROLEMRC_LOCAL_JSONL=/path/roleMRC_test.jsonl python .../prepare_rolemrc.py

Field names vary across releases of the dataset, so the messages field is taken
as the first present of (question, conversations, prompt, messages) and the gold
reply as the first present of (reference, chosen, answer, target).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from app import _EVALUATION_CONFIG, _task_dimension


_HF_DATASET = "Junrulu/RoleMRC"
_HF_TEST_FILE = "roleMRC_test.jsonl"
_LOCAL_PATH_ENV = "ROLEMRC_LOCAL_JSONL"
_ALLOW_DRIFT_ENV = "ROLEMRC_ALLOW_DATASET_DRIFT"

# Task histogram of ``roleMRC_test.jsonl``: 14 slices of 100 rows = the paper's
# "1.4k testing samples", with four slices split into base / -special-content /
# -special-format / -refused variants that still sum to 100 each.
#
# Asserted on every prep run so a truncated download or a revised dataset fails
# loudly instead of quietly shifting the scores. The judge denominators are
# derived from it, and ``AvgWeighted`` is computed over them -- see
# ``app.py:_judge_rollups``.
_EXPECTED_TASK_COUNTS: Dict[str, int] = {
    "role_related_mrc_answer_no_narration": 100,
    "role_related_mrc_answer_no_narration-refused": 20,
    "role_related_mrc_answer_no_narration-special-content": 58,
    "role_related_mrc_answer_no_narration-special-format": 22,
    "role_related_mrc_answer_with_narration": 100,
    "role_related_mrc_answer_with_narration-refused": 22,
    "role_related_mrc_answer_with_narration-special-content": 58,
    "role_related_mrc_answer_with_narration-special-format": 20,
    "role_related_mrc_refused_no_narration": 100,
    "role_related_mrc_refused_no_narration-2ndrefused": 100,
    "role_related_mrc_refused_with_narration": 100,
    "role_related_mrc_refused_with_narration-2ndrefused": 100,
    "role_unrelated_mrc_answer_no_narration": 100,
    "role_unrelated_mrc_answer_with_narration": 100,
    "role_unrelated_mrc_refused_no_narration": 100,
    "role_unrelated_mrc_refused_no_narration-2ndanswer": 100,
    "role_unrelated_mrc_refused_with_narration": 100,
    "role_unrelated_mrc_refused_with_narration-2ndanswer": 100,
}

_MESSAGES_FIELDS = ("question", "conversations", "prompt", "messages")
_REFERENCE_FIELDS = ("reference", "chosen", "answer", "target")

_DATA_DIR = Path(__file__).parent / "data"
_REFERENCE_AGENT = "rolemrc_simple_agent"
_JUDGE_AGENT = "rolemrc_judge_simple_agent"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_rolemrc() -> List[Dict[str, Any]]:
    local = os.environ.get(_LOCAL_PATH_ENV)
    if local and os.path.isfile(local):
        rows = _read_jsonl(local)
        print(f"RoleMRC: loaded {len(rows)} rows from {local}")
        return rows

    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET, data_files=_HF_TEST_FILE, split="train")
    rows = [dict(row) for row in ds]
    print(f"RoleMRC: loaded {len(rows)} rows from hf://{_HF_DATASET}/{_HF_TEST_FILE}")
    return rows


def _pick_field(row: Dict[str, Any], candidates) -> Any:
    for name in candidates:
        if name in row and row[name] not in (None, "", []):
            return row[name]
    return None


def _normalize_messages(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for turn in turns:
        role = str(turn.get("role", "user")).lower()
        out.append({"role": role, "content": str(turn.get("content", ""))})
    return out


def _row_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_turns = _pick_field(row, _MESSAGES_FIELDS) or []
    if not isinstance(raw_turns, list):
        raw_turns = []
    return _normalize_messages(raw_turns)


def _to_task(row: Dict[str, Any], agent_name: str, *, carry_conversation: bool = False) -> Dict[str, Any]:
    messages = _row_messages(row)
    reference = _pick_field(row, _REFERENCE_FIELDS) or ""
    task = row.get("task", "")
    task_row: Dict[str, Any] = {
        "responses_create_params": {"input": messages},
        "reference": str(reference),
        "task": task,
        "dimension": _task_dimension(task),
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
    }
    if carry_conversation:
        # The judge prompt quotes the whole conversation, and `verifier_metadata`
        # is the one field an external runner is guaranteed to forward to
        # /verify intact (see _conversation_messages in app.py).
        task_row["verifier_metadata"] = {"conversation": messages}
    return task_row


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"RoleMRC: wrote {len(rows)} rows -> {path}")


def check_task_histogram(rows: List[Dict[str, Any]]) -> List[str]:
    """Return human-readable differences between ``rows`` and the known-good split."""
    actual = Counter(row.get("task", "") for row in rows)
    problems: List[str] = []
    for task in sorted(set(actual) | set(_EXPECTED_TASK_COUNTS)):
        want = _EXPECTED_TASK_COUNTS.get(task, 0)
        got = actual.get(task, 0)
        if want != got:
            problems.append(f"  {task}: expected {want}, got {got}")
    return problems


def judge_call_counts(rows: List[Dict[str, Any]]) -> "Counter[str]":
    """Judge calls per aspect — the denominators the per-aspect means use.

    A row fires one call per aspect its ``task`` maps to, and the two on-scene
    tasks that pair knowledge range with style fire two, so this is strictly
    larger than the row count.
    """
    counts: "Counter[str]" = Counter()
    for row in rows:
        for aspect_name, _template, _fmt in _EVALUATION_CONFIG.get(row.get("task", ""), []):
            counts[aspect_name] += 1
    return counts


def main() -> None:
    rows = _load_rolemrc()

    problems = check_task_histogram(rows)
    if problems:
        message = "RoleMRC: dataset does not match the expected split:\n" + "\n".join(problems)
        if os.environ.get(_ALLOW_DRIFT_ENV):
            print(f"{message}\n(continuing: {_ALLOW_DRIFT_ENV} is set)")
        else:
            raise SystemExit(f"{message}\n\nSet {_ALLOW_DRIFT_ENV}=1 to build anyway.")

    reference_rows = [_to_task(row, _REFERENCE_AGENT) for row in rows]
    judge_rows = [
        _to_task(row, _JUDGE_AGENT, carry_conversation=True) for row in rows if row.get("task") in _EVALUATION_CONFIG
    ]

    _write_jsonl(_DATA_DIR / "test.jsonl", reference_rows)
    _write_jsonl(_DATA_DIR / "test_judge.jsonl", judge_rows)

    calls = judge_call_counts(rows)
    total = sum(calls.values())
    print(f"RoleMRC: judge calls per aspect ({total} over {len(judge_rows)} rows) — the per-aspect denominators:")
    for aspect_name, count in sorted(calls.items()):
        print(f"  {aspect_name:<24} {count}")


if __name__ == "__main__":
    main()
