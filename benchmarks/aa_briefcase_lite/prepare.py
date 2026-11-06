# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare AA-Briefcase-Lite rows from a pinned local public checkout.

Each row carries the rendered system and user prompts and the two task tools, so the Stirrup agent needs no
benchmark-specific code. Only task execution metadata goes into a row; the checks, rubrics and judge prompts stay
in the checkout the resources server reads.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "aa_briefcase_lite.jsonl"
TOOLS_FPATH = BENCHMARK_DIR.parent / "gdpval" / "tools.json"
# The two tools of AA's protocol, among the tools the GDPVal resources server serves; web tools are left out.
TOOL_NAMES = ("finish", "abandon_task_finish")
DATASET_ENV = "AA_BRIEFCASE_LITE_DATASET_DIR"
REVISION_ENV = "AA_BRIEFCASE_LITE_REVISION"
REVISION_MARKER = ".aa-briefcase-lite-revision"
# Public dataset commit, not a credential.
PINNED_REVISION = "4dec557b47d43867a1648c0974db1d8208c8b677"  # pragma: allowlist secret
# The turn cap AA's system prompt names; config.yaml sets the agent's agent_max_turns to the same value.
AGENT_MAX_TURNS = 500


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dataset_revision(dataset_dir: Path) -> str:
    marker = dataset_dir / REVISION_MARKER
    if marker.is_file():
        revision = marker.read_text(encoding="utf-8").strip()
    else:
        try:
            revision = subprocess.run(
                ["git", "-C", str(dataset_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"AA-Briefcase-Lite dataset must be a Git checkout or carry {REVISION_MARKER}: {dataset_dir}"
            ) from exc
    expected = os.environ.get(REVISION_ENV, PINNED_REVISION)
    if revision != expected:
        raise RuntimeError(f"AA-Briefcase-Lite revision mismatch: expected {expected}, found {revision}")
    return revision


def _dataset_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    candidate = (root / relative.rstrip("/")).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes AA-Briefcase-Lite dataset: {relative}")
    exists = candidate.is_dir() if directory else candidate.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"AA-Briefcase-Lite {kind} not found: {candidate}")
    return candidate


def _render(path: Path, values: Dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    unresolved = sorted(set(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", text)))
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders in {path}: {unresolved}")
    return text


def _validate_task(root: Path, task: dict) -> None:
    for relative in task["shared_files"]:
        _dataset_path(root, relative, directory=relative.endswith("/"))
        if not relative.startswith("source_files/shared/"):
            raise ValueError(f"Invalid shared input path: {relative}")
    for relative in task["week_files"]:
        _dataset_path(root, relative, directory=relative.endswith("/"))
        if not relative.startswith("source_files/week/"):
            raise ValueError(f"Invalid week input path: {relative}")
    for filename in task["deliverable_filenames"]:
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError(f"Invalid deliverable filename: {filename}")


def build_prompts(root: Path, task: dict) -> tuple[str, str]:
    """The system and user prompt of one task, from the checkout's released prompt files."""
    _validate_task(root, task)
    finish = {"finish_tool_name": "finish", "abandon_task_finish": "abandon_task_finish"}
    system = _render(_dataset_path(root, "prompts/eval_system.txt"), {"max_turns": str(AGENT_MAX_TURNS), **finish})
    user = _render(
        _dataset_path(root, "prompts/eval_submission.txt"),
        {
            "scenario_overview": _dataset_path(root, task["scenario_overview_path"]).read_text(encoding="utf-8"),
            "week_overview": _dataset_path(root, task["week_overview_path"]).read_text(encoding="utf-8"),
            "task": _dataset_path(root, task["task_md_path"]).read_text(encoding="utf-8"),
            "expected_output_filenames": "\n".join(f"- `{filename}`" for filename in task["deliverable_filenames"]),
            **finish,
        },
    )
    return system, user


def task_tools() -> list[dict]:
    tools = {tool["name"]: tool for tool in json.loads(TOOLS_FPATH.read_text())}
    return [tools[name] for name in TOOL_NAMES]


def _validate_task_files(dataset_dir: Path, tasks: list[dict]) -> None:
    source_lists = {(tuple(task["shared_files"]), tuple(task["week_files"])) for task in tasks}
    if len(source_lists) != 1:
        raise ValueError("Lite tasks are expected to share one identical shared/week source pool")

    for task in tasks:
        paths = [
            task["task_md_path"],
            task["scenario_overview_path"],
            task["week_overview_path"],
            *task["shared_files"],
            *task["week_files"],
        ]
        for relative in paths:
            candidate = (dataset_dir / relative.rstrip("/")).resolve()
            if not candidate.is_relative_to(dataset_dir) or not candidate.exists():
                raise FileNotFoundError(f"Invalid or missing dataset path for {task['task_id']}: {relative}")


def prepare() -> Path:
    dataset_value = os.environ.get(DATASET_ENV)
    if not dataset_value:
        raise RuntimeError(f"Set {DATASET_ENV} to the pinned ArtificialAnalysis/AA-Briefcase-Lite checkout")
    dataset_dir = Path(dataset_value).resolve()
    tasks_path = dataset_dir / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"AA-Briefcase-Lite tasks file not found: {tasks_path}")

    revision = _dataset_revision(dataset_dir)
    tasks = _read_jsonl(tasks_path)
    if {task["task_id"] for task in tasks} != {"w1_t1", "w1_t2", "w1_t3", "w1_t4"}:
        raise ValueError("Expected the four public AA-Briefcase-Lite task IDs")
    _validate_task_files(dataset_dir, tasks)
    tools = task_tools()

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w", encoding="utf-8") as output:
        for task in tasks:
            system, user = build_prompts(dataset_dir, task)
            record = {
                "responses_create_params": {
                    "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "tools": tools,
                },
                "task_id": task["task_id"],
                "week": task["week"],
                "dataset_revision": revision,
                "deliverable_filenames": task["deliverable_filenames"],
            }
            output.write(json.dumps(record) + "\n")

    print(f"Wrote {len(tasks)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
