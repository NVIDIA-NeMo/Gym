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
"""Prepare the ARC-AGI-2 public evaluation set (120 tasks: 167 test inputs on GitHub, 172 in the ARC Prize 2026 files).

Downloads the task files from the ARC-AGI-2 GitHub repository at a pinned commit and writes one row per test input
to two files that differ only in how the prompt is built:

- `arc_agi_2_tools.jsonl` for `arc_code_agent` (the agent writes its own prompt; `responses_create_params` holds a
  placeholder), and
- `arc_agi_2_no_tools.jsonl` for `simple_agent` with `prompts/no_tools.yaml` (`question` holds the puzzle as digits).

Every row carries `examples`, `test_input` and `expected_output`, which the `arc_code` resources server grades by exact
match, plus `task_id` (`<puzzle>_<test index>`), `puzzle_id` and `test_index` for task-level scoring (`score.py`).
"""

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
TOOLS_FPATH = DATA_DIR / "arc_agi_2_tools.jsonl"
NO_TOOLS_FPATH = DATA_DIR / "arc_agi_2_no_tools.jsonl"

REPO = "arcprize/ARC-AGI-2"
REF = "main"  # resolved to a commit at download time and recorded in data/source.json
TASK_DIR = "data/evaluation"


def grid_to_text(grid) -> str:
    """One row per line, digits separated by a space (the format the `<answer>` block uses)."""
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def question_text(examples, test_input) -> str:
    parts = []
    for i, pair in enumerate(examples, 1):
        parts.append(
            f"Example {i} input:\n{grid_to_text(pair['input'])}\n\nExample {i} output:\n{grid_to_text(pair['output'])}"
        )
    parts.append(f"Test input:\n{grid_to_text(test_input)}")
    return "\n\n".join(parts)


def download(ref: str = REF):
    """{puzzle_id: task} for every evaluation task, and the commit they were read at.

    Reads the repository archive (not the GitHub API, which rate-limits shared cluster addresses). GitHub writes the
    archive's commit into the tar's global header comment.
    """
    with urllib.request.urlopen(f"https://github.com/{REPO}/archive/{ref}.tar.gz", timeout=120) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    commit = archive.pax_headers.get("comment", ref)
    tasks = {}
    for member in archive.getmembers():
        parts = member.name.split("/")
        if member.isfile() and "/".join(parts[1:-1]) == TASK_DIR and parts[-1].endswith(".json"):
            tasks[parts[-1][: -len(".json")]] = json.load(archive.extractfile(member))
    if not tasks:
        raise RuntimeError(f"no tasks under {TASK_DIR} in {REPO}@{ref}")
    return tasks, commit


def rows(tasks):
    for puzzle_id in sorted(tasks):
        task = tasks[puzzle_id]
        examples = [{"input": p["input"], "output": p["output"]} for p in task["train"]]
        for k, test in enumerate(task["test"]):
            yield {
                "task_id": f"{puzzle_id}_{k}",
                "puzzle_id": puzzle_id,
                "test_index": k,
                "examples": examples,
                "test_input": test["input"],
                "expected_output": test["output"],
            }


def write(tasks, source: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(TOOLS_FPATH, "w") as tools, open(NO_TOOLS_FPATH, "w") as no_tools:
        for row in rows(tasks):
            placeholder = {"input": [{"role": "user", "content": "arc"}]}  # arc_code_agent writes the prompt itself
            tools.write(json.dumps({**row, "responses_create_params": placeholder}) + "\n")
            no_tools.write(json.dumps({**row, "question": question_text(row["examples"], row["test_input"])}) + "\n")
            n += 1
    (DATA_DIR / "source.json").write_text(json.dumps({**source, "tasks": len(tasks), "rows": n}, indent=1) + "\n")
    print(f"Wrote {n} rows from {len(tasks)} tasks to {TOOLS_FPATH} and {NO_TOOLS_FPATH}")
    return NO_TOOLS_FPATH


def prepare() -> Path:
    """Download the evaluation tasks and write both row files. Returns the no-tools file path."""
    tasks, commit = download()
    return write(tasks, {"repo": REPO, "commit": commit, "dir": TASK_DIR})


def from_challenges(challenges: Path, solutions: Path) -> Path:
    """Build the rows from a challenges file and its solutions file in the ARC Prize (Kaggle) layout instead."""
    chal, sol = json.loads(challenges.read_text()), json.loads(solutions.read_text())
    tasks = {
        pid: {"train": t["train"], "test": [{"input": x["input"], "output": y} for x, y in zip(t["test"], sol[pid])]}
        for pid, t in chal.items()
    }
    return write(tasks, {"challenges": str(challenges), "solutions": str(solutions)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--challenges", type=Path, help="ARC Prize challenges JSON (with --solutions) instead of GitHub"
    )
    parser.add_argument("--solutions", type=Path)
    args = parser.parse_args()
    if args.challenges:
        from_challenges(args.challenges, args.solutions)
    else:
        prepare()
