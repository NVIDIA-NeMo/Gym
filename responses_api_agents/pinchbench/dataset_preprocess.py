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

Each line carries the task's human-readable prompt (extracted from the skill's task
`.md` `## Prompt` section) in `input`, plus `verifier_metadata.task_id`. `task_id` is
the authoritative selector: at run time benchmark.py loads the full task (prompt +
assets + grading) from the skill BY task_id (run_task.sh `--suite`), and you pick which
tasks to run by including only the rows you want. The 5-task example.jsonl is committed;
full.jsonl (147 tasks) follows the skill's task manifest. Regenerate from a (non-vendored)
skill checkout (see Dockerfile.benchmark):

    git clone -b v2.0.0 https://github.com/pinchbench/skill /tmp/pb-skill
    PINCHBENCH_SKILL_DIR=/tmp/pb-skill python responses_api_agents/pinchbench/dataset_preprocess.py
"""

import json
import os
import re
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


def _prompt_for(task_id: str) -> str:
    """The task's human-readable prompt = the `## Prompt` section of its skill `.md`."""
    md = (Path(_SKILL_DIR) / "tasks" / f"{task_id}.md").read_text()
    m = re.search(r"##\s*Prompt\s*\n(.*?)(?:\n##\s|\Z)", md, re.S)
    return (m.group(1).strip() if m else "").strip()


def _record(task_id: str) -> dict:
    # `input` carries the real prompt for transparency / readability; `task_id` is the
    # authoritative selector — run_task.sh runs `benchmark.py --suite <task_id>`, which
    # loads the full task (prompt + assets + grading) from the skill, so the dataset stays
    # tiny and you subset by simply choosing which rows to include.
    return {
        "responses_create_params": {"input": [{"role": "user", "content": _prompt_for(task_id)}]},
        "verifier_metadata": {"task_id": task_id},
    }


def _all_task_ids() -> list[str]:
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
    if not _SKILL_DIR:
        raise SystemExit(
            "Set PINCHBENCH_SKILL_DIR to a checkout of github.com/pinchbench/skill@v2.0.0 to "
            "(re)generate the datasets with real prompts. The committed example.jsonl is "
            "self-contained and needs no skill to *use*."
        )
    _DATA.mkdir(parents=True, exist_ok=True)
    _write(_DATA / "example.jsonl", EXAMPLE_TASKS)  # committed smoke set
    _write(_DATA / "full.jsonl", _all_task_ids())  # gitignored (147 tasks)


if __name__ == "__main__":
    main()
