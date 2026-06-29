# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate PinchBench Gym datasets.

Each line is one PinchBench task, referenced by `task_id` — the full task
definition (prompt, grading code, assets) is reconstructed inside the container
from the skill at run time, so the JSONL stays tiny. The 5-task example.jsonl is
committed; full.jsonl (147 tasks) is read from the task manifest of a PinchBench
skill checkout (the skill is not vendored — see Dockerfile.benchmark):

    git clone -b v2.0.0 https://github.com/pinchbench/skill /tmp/pb-skill
    PINCHBENCH_SKILL_DIR=/tmp/pb-skill python responses_api_agents/pinchbench/dataset_preprocess.py
"""

import json
import os
from pathlib import Path

import yaml


_DIR = Path(__file__).resolve().parent
_DATA = _DIR / "data"
# The skill is not vendored; point at a checkout of github.com/pinchbench/skill@v2.0.0
# (nvidia-pinchbench.patch applied) to (re)generate the full task list.
_SKILL_DIR = os.environ.get("PINCHBENCH_SKILL_DIR")

# 5 representative tasks (mix of grading types) for the committed smoke set.
EXAMPLE_TASKS = [
    "task_sanity",  # automated
    "task_calendar",  # automated
    "task_todo_list_cleanup",  # automated
    "task_daily_summary",  # hybrid
    "task_email",  # llm_judge
]


def _record(task_id: str) -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "<placeholder; real prompt loaded from the task .md at run time>"}]
        },
        "verifier_metadata": {"task_id": task_id},
    }


def _all_task_ids() -> list[str]:
    if not _SKILL_DIR:
        raise SystemExit(
            "Set PINCHBENCH_SKILL_DIR to a checkout of github.com/pinchbench/skill@v2.0.0 "
            "to regenerate full.jsonl (the committed example.jsonl needs no skill)."
        )
    manifest = yaml.safe_load((Path(_SKILL_DIR) / "tasks" / "manifest.yaml").read_text())
    ids: list[str] = list(manifest.get("run_first", []))
    for cat_ids in (manifest.get("categories") or {}).values():
        for tid in cat_ids or []:
            if tid not in ids:
                ids.append(tid)
    return ids


def _write(path: Path, task_ids: list[str]) -> None:
    with path.open("w") as f:
        for tid in task_ids:
            f.write(json.dumps(_record(tid), separators=(",", ":")) + "\n")
    print(f"wrote {len(task_ids)} tasks -> {path}")


def main() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)
    _write(_DATA / "example.jsonl", EXAMPLE_TASKS)  # committed smoke set (no skill needed)
    if _SKILL_DIR:
        _write(_DATA / "full.jsonl", _all_task_ids())  # gitignored; needs the skill checkout
    else:
        print("PINCHBENCH_SKILL_DIR unset -> skipping full.jsonl (set it to regenerate)")


if __name__ == "__main__":
    main()
