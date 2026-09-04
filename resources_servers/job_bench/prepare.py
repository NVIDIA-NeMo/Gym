# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the JobBench dataset into a control-plane cache and a Gym JSONL.

Upstream ``setup.sh`` pulls ``JobBench/job-bench`` from Hugging Face and lays the
two splits out as ``dataset/main`` and ``dataset/easy``. This does the same into a
gitignored cache under the resources server, then writes the model-visible JSONL.

Rubrics, task cards and ``files_required_to_search/`` stay in the cache: the JSONL
carries only the task instructions the agent is allowed to see plus the task ID
the resources server uses to look the answer key back up.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from resources_servers.job_bench.task_store import (
    EXPECTED_TASK_COUNTS,
    OUTPUT_DIR,
    SEARCH_FILES_DIR,
    SPLITS,
    TASK_FOLDER_DIR,
    WORKSPACE_DIR,
    JobBenchTask,
    discover_tasks,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]

DATASET_REPO_ID = "JobBench/job-bench"
DEFAULT_CACHE_DIR = PACKAGE_DIR / "data" / "cache" / "tasks"
DEFAULT_BENCHMARK_DIR = NEMO_GYM_ROOT / "benchmarks" / "job_bench" / "data"

# Upstream ships the main split under `dataset/` and the easy split under
# `dataset_easy/`; setup.sh renames them to main/ and easy/.
UPSTREAM_SPLIT_DIRS = {"main": "dataset", "easy": "dataset_easy"}

EXAMPLE_TASK_COUNT = 5


def build_prompt(task: JobBenchTask, *, include_search_files: bool) -> str:
    """Reproduce the upstream OpenCode runner prompt against sandbox paths.

    Mirrors ``eval/run_benchmark_opencode.sh``, with the runner's ``/tmp``
    workspace replaced by the sandbox workspace the resources server seeds.
    """
    search_files_line = (
        f"\n- Reference files that must be searched for are available in: {SEARCH_FILES_DIR}"
        if include_search_files
        else ""
    )
    return f"""=== TASK FOLDER ===
{TASK_FOLDER_DIR}

=== INSTRUCTIONS ===
1. Read the TASK_INSTRUCTIONS.txt file in the task folder above
2. Based on the Reference Files section in TASK_INSTRUCTIONS.txt, read the corresponding files from the same task folder using appropriate tools.
3. Complete the task as specified in TASK_INSTRUCTIONS.txt
4. Only save the final deliverables to the output directory specified below. Do not save any intermediate or temporary files.

=== OUTPUT DIRECTORY ===
{OUTPUT_DIR}

IMPORTANT:
- All reference files are in the task folder: {TASK_FOLDER_DIR}{search_files_line}
- Only save the final deliverables to the output directory {OUTPUT_DIR}. Do not save any intermediate or temporary files.
- You MUST only access files within {WORKSPACE_DIR} or search online for new reference files if you find needed. Do NOT access any files or directories in this system outside of this path.
- If you encounter ambiguous or conflicting information, analyze the conflict, explain your reasoning, and justify the approach you choose.
- If a file cannot be read directly (e.g., .xlsx, .docx, .db, .pptx), use appropriate tools, MCP servers, or code to extract and process its contents."""


def build_row(task: JobBenchTask, *, include_search_files: bool) -> dict:
    """Build one Gym JSONL row for a prepared task."""
    return {
        "task_id": task.task_id,
        "responses_create_params": {
            "input": [{"role": "user", "content": build_prompt(task, include_search_files=include_search_files)}]
        },
        "verifier_metadata": {
            "task_id": task.task_id,
            "split": task.split,
            "profession": task.profession,
            "task_name": task.task_name,
            # Carried for analysis and to detect a cache that drifted from the
            # JSONL; the rubric text itself never leaves the control plane.
            "num_rubrics": len(task.rubrics),
            "max_score": task.max_score,
            "rubrics_sha256": task.rubrics_sha256,
        },
    }


def download_dataset(cache_dir: Path, *, splits: tuple[str, ...], force: bool) -> None:
    """Fetch the requested splits from Hugging Face into ``cache_dir``."""
    missing = [split for split in splits if not (cache_dir / split).is_dir()]
    if not missing and not force:
        print(f"JobBench cache already populated at {cache_dir}; pass --force to re-download.")
        return
    if force:
        for split in splits:
            shutil.rmtree(cache_dir / split, ignore_errors=True)
        missing = list(splits)

    from huggingface_hub import snapshot_download

    staging = cache_dir.parent / "_hf_snapshot"
    staging.mkdir(parents=True, exist_ok=True)
    patterns = [f"{UPSTREAM_SPLIT_DIRS[split]}/**" for split in missing]
    print(f"Downloading {DATASET_REPO_ID} splits {missing} into {staging} ...")
    snapshot_download(
        DATASET_REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(staging),
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in missing:
        source = staging / UPSTREAM_SPLIT_DIRS[split]
        if not source.is_dir():
            raise RuntimeError(f"Expected {source} in the Hugging Face snapshot for split {split!r}")
        destination = cache_dir / split
        shutil.rmtree(destination, ignore_errors=True)
        shutil.move(str(source), str(destination))
        print(f"  {split}: {destination}")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prepare(
    *,
    split: str = "main",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    include_search_files: bool = False,
    download: bool = True,
    force: bool = False,
    write_example: bool = True,
) -> tuple[Path, Path]:
    """Populate the cache and write the benchmark JSONL; return (cache_dir, jsonl_path)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown JobBench split {split!r}; expected one of {SPLITS}")

    if download:
        download_dataset(cache_dir, splits=(split,), force=force)

    tasks = discover_tasks(cache_dir, split)
    expected = EXPECTED_TASK_COUNTS[split]
    if len(tasks) != expected:
        raise RuntimeError(f"Prepared {len(tasks)} tasks for split {split!r}, expected {expected}")

    rows = [build_row(task, include_search_files=include_search_files) for task in tasks.values()]
    jsonl_path = benchmark_dir / f"job_bench_{split}_benchmark.jsonl"
    write_jsonl(rows, jsonl_path)
    print(f"Wrote {len(rows)} tasks to {jsonl_path}")

    if write_example:
        example_path = PACKAGE_DIR / "data" / "example.jsonl"
        write_jsonl(rows[:EXAMPLE_TASK_COUNT], example_path)
        print(f"Wrote {min(EXAMPLE_TASK_COUNT, len(rows))} tasks to {example_path}")

    return cache_dir, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="main")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--include-search-files",
        action="store_true",
        help="Tell the agent that files_required_to_search/ is mounted in the sandbox. "
        "Off by default to match the upstream runner, which withholds those files.",
    )
    parser.add_argument("--no-download", dest="download", action="store_false", help="Use the existing cache as-is.")
    parser.add_argument("--force", action="store_true", help="Wipe and re-download the split.")
    parser.add_argument("--no-example", dest="write_example", action="store_false")
    args = parser.parse_args()

    prepare(
        split=args.split,
        cache_dir=args.cache_dir,
        benchmark_dir=args.benchmark_dir,
        include_search_files=args.include_search_files,
        download=args.download,
        force=args.force,
        write_example=args.write_example,
    )


if __name__ == "__main__":
    main()
