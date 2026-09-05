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
"""Prepare the full 147-task PinchBench benchmark dataset.

The PinchBench skill is not vendored (same convention as `harbor_agent` / `mini_swe_agent`:
pin an upstream ref, don't copy task files). This clones it at the SAME ref the per-task
image bakes in — `Dockerfile.benchmark`'s `PINCHBENCH_SKILL_REF` — so the prompts written
here describe the tasks `benchmark.py` actually loads inside the sandbox.

Row shape mirrors `responses_api_agents/pinchbench/dataset_preprocess.py` (which generates
the committed 5-task `data/example.jsonl`): the task's human-readable `## Prompt` section in
`input`, plus `verifier_metadata.task_id`. `task_id` is the authoritative selector — at run
time `run_task.sh` passes it to `benchmark.py --suite`, which loads the full task (prompt +
assets + grading) from the skill. `_assert_matches_example_jsonl` re-derives the 5 example
rows and compares them byte-for-byte against that committed file, so the two generators
cannot silently diverge and upstream prompt drift is caught here rather than mid-run.

Set `PINCHBENCH_SKILL_DIR` to an existing checkout at `SKILL_REF` to skip the clone.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "pinchbench_benchmark.jsonl"

AGENT_DIR = BENCHMARK_DIR.parents[1] / "responses_api_agents" / "pinchbench"
EXAMPLE_JSONL_FPATH = AGENT_DIR / "data" / "example.jsonl"

SKILL_REPO_URL = "https://github.com/pinchbench/skill"
# Keep in lockstep with `PINCHBENCH_SKILL_REF` in
# responses_api_agents/pinchbench/Dockerfile.benchmark.
SKILL_REF = "v2.0.0"
# The v2.0.0 manifest's task count; upstream drift surfaces here rather than as a
# silently shorter run.
EXPECTED_TASK_COUNT = 147

_PROMPT_SECTION_RE = re.compile(r"##\s*Prompt\s*\n(.*?)(?:\n##\s|\Z)", re.S)


def _prompt_for(skill_dir: Path, task_id: str) -> str:
    """The task's human-readable prompt = the `## Prompt` section of its skill `.md`."""
    md = (skill_dir / "tasks" / f"{task_id}.md").read_text()
    match = _PROMPT_SECTION_RE.search(md)
    return (match.group(1).strip() if match else "").strip()


def _record_line(skill_dir: Path, task_id: str) -> str:
    record = {
        "responses_create_params": {"input": [{"role": "user", "content": _prompt_for(skill_dir, task_id)}]},
        "verifier_metadata": {"task_id": task_id},
    }
    return json.dumps(record, separators=(",", ":")) + "\n"


def _all_task_ids(skill_dir: Path) -> list[str]:
    """Every task in the skill's manifest, in manifest order (`run_first` first)."""
    manifest = yaml.safe_load((skill_dir / "tasks" / "manifest.yaml").read_text())
    task_ids: list[str] = list(manifest.get("run_first", []))
    for category_task_ids in (manifest.get("categories") or {}).values():
        for task_id in category_task_ids or []:
            if task_id not in task_ids:
                task_ids.append(task_id)
    return task_ids


def _assert_matches_example_jsonl(skill_dir: Path) -> None:
    """Re-derive the committed example rows from this checkout and require an exact match.

    Guards both directions: a drifted skill checkout, and this script diverging from the
    agent-side `dataset_preprocess.py` that produced `example.jsonl`.
    """
    expected_lines = EXAMPLE_JSONL_FPATH.read_text().splitlines(keepends=True)
    for line_number, expected_line in enumerate(expected_lines, 1):
        task_id = json.loads(expected_line)["verifier_metadata"]["task_id"]
        actual_line = _record_line(skill_dir, task_id)
        if actual_line != expected_line:
            raise ValueError(
                f"Regenerated row for '{task_id}' does not match {EXAMPLE_JSONL_FPATH} line {line_number}. "
                f"The skill checkout is not at {SKILL_REF}, or the row format has diverged from "
                "responses_api_agents/pinchbench/dataset_preprocess.py."
            )


def _write_from_skill(skill_dir: Path) -> None:
    _assert_matches_example_jsonl(skill_dir)

    task_ids = _all_task_ids(skill_dir)
    if len(task_ids) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TASK_COUNT} PinchBench tasks in the {SKILL_REF} manifest, "
            f"found {len(task_ids)}; the skill checkout may not be at {SKILL_REF}."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w") as out:
        for task_id in task_ids:
            out.write(_record_line(skill_dir, task_id))
    print(f"Wrote {len(task_ids)} PinchBench tasks to {OUTPUT_FPATH}")


def _clone_skill(target_dir: Path) -> None:
    print(f"Cloning {SKILL_REPO_URL} at {SKILL_REF}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", SKILL_REF, SKILL_REPO_URL, str(target_dir)],
        check=True,
    )


def prepare() -> Path:
    """Clone (or reuse) the pinned skill, write every manifest task. Returns the JSONL path."""
    preexisting_skill_dir = os.environ.get("PINCHBENCH_SKILL_DIR")
    if preexisting_skill_dir:
        skill_dir = Path(preexisting_skill_dir)
        if not (skill_dir / "tasks" / "manifest.yaml").is_file():
            raise FileNotFoundError(
                f"PINCHBENCH_SKILL_DIR={skill_dir} is not a PinchBench skill checkout "
                "(tasks/manifest.yaml is missing)."
            )
        _write_from_skill(skill_dir)
        return OUTPUT_FPATH

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "pinchbench-skill"
        _clone_skill(skill_dir)
        _write_from_skill(skill_dir)

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
