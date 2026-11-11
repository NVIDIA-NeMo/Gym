# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare WANDR: the pinned task packages, their shared verifier runtime and the Gym benchmark rows.

Sources, both pinned to the same upstream commit:
- https://huggingface.co/datasets/perplexity-ai/wandr: split ``test`` lists the 500 scored tasks in canonical
  order, their output files and the item fields of each file. Split ``smoke`` is one unscored task, kept for
  ``resources_servers/wandr/data/example.jsonl`` and left out of the benchmark rows.
- https://github.com/perplexityai/wandr (``datasets/wandr/<task>``): each task's instruction and verifier package.
  The packages carry identical copies of the evaluator runtime ``tests/wandr_core``; one copy is kept.
"""

import hashlib
import json
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "wandr_benchmark.jsonl"
# A fixed 100-task subset for cheaper regular evaluations (config_subset100.yaml): 100 task names drawn at random
# (Python random.Random(0).sample of the 500 benchmark rows). Prepare writes its rows in benchmark order.
SUBSET100_FPATH = BENCHMARK_DIR / "subset100.txt"
SUBSET100_OUTPUT_FPATH = DATA_DIR / "wandr_benchmark_subset100.jsonl"
TASKS_DIR = DATA_DIR / "tasks"
CORE_DIR = DATA_DIR / "wandr_core"

SOURCE_COMMIT = "ccb0baeb96f1c77a48e47f92122c57479ee99700"
HF_REVISION = "066ee3d71cd332cb2fb4f49146cad7f9c7249f86"
HF_SPLIT_URL = "https://huggingface.co/datasets/perplexity-ai/wandr/resolve/{revision}/data/{split}.jsonl"
GITHUB_TARBALL_URL = f"https://codeload.github.com/perplexityai/wandr/tar.gz/{SOURCE_COMMIT}"
TASK_DIR_PATTERN = re.compile(rf"/blob/{SOURCE_COMMIT}/datasets/wandr/([^/]+)/instruction\.md$")
# Files of a task package the Gym verifier runs; the Harbor wrapper (test.sh) and Docker environment are left out.
TASK_FILES = {"instruction.md", "task.toml"}
TEST_FILES = {"verify.py", "rewards.py", "submissions.py", "artifacts.py", "utils.py", "manifest.json"}

SYSTEM_PROMPT = """You are a research agent that collects evidence-backed records from the web.

Tools:
- web_search(query, max_results): web search results, each with a title, URL, date and snippet.
- fetch_page(url, start): the readable text of a page, a slice at a time. To read on, call it again with start set to the previous end.
- write_records(file, records): saves records to one of the task's output files. Each call appends records and never replaces earlier ones. It returns how many records it accepted and why it rejected any.

The task below asks for JSONL output files. Do not print the records: save every record with write_records, with the exact file name the task gives. Save records in batches while you work, so that your work is kept if you run out of steps. Do not save the same record twice. Every output file the task names needs records: if one stays empty, the task scores zero.

Each record is a JSON object with "item", "url", "excerpts" and "answer", as the task describes. The grader fetches every url and checks the record against that page, so take excerpts from pages you have read. When the task is complete, reply with a short summary of what you saved."""

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web. Returns results with title, url, date and a text snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "Number of results, 1 to 20 (default 10)."},
            },
            "required": ["query"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "fetch_page",
        "description": (
            "Fetch a web page or PDF and return its readable text from character `start`. "
            "The reply gives `end` and `total_chars`; call again with start=end to read the next part."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The http(s) URL to fetch."},
                "start": {"type": "integer", "description": "Character offset to start from (default 0)."},
            },
            "required": ["url"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "write_records",
        "description": (
            "Append records to one of the task's JSONL output files. Each record is an object with "
            "`item` (object), `url` (string), `excerpts` (list of strings) and `answer` (object). "
            "Returns the number of accepted records and the reason for each rejected one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "The output file name the task gives, e.g. results_x.jsonl.",
                },
                "records": {"type": "array", "items": {"type": "object"}, "description": "Records to append."},
            },
            "required": ["file", "records"],
        },
        "strict": False,
    },
]


def _download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=600) as response, path.open("wb") as stream:
        while chunk := response.read(1 << 20):
            stream.write(chunk)


def _load_tasks(path: Path, expected_count: int) -> list[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["source_commit"] != SOURCE_COMMIT:
            raise ValueError(f"{row['task_id']}: source commit {row['source_commit']} is not {SOURCE_COMMIT}")
        match = TASK_DIR_PATTERN.search(row["instruction_source_url"])
        if match is None:
            raise ValueError(f"{row['task_id']}: no task package in {row['instruction_source_url']}")
        nodes = json.loads(row["task_tree_json"])
        tasks.append(
            {
                "task_name": match.group(1),
                "task_id": row["task_id"],
                "required_output_files": row["required_output_files"],
                "item_fields": {node["required_output_file"]: node["item_fields"] for node in nodes},
            }
        )
    if len(tasks) != expected_count:
        raise ValueError(f"expected {expected_count} WANDR tasks in {path.name}, found {len(tasks)}")
    return tasks


def _extract(tarball: Path, task_names: set[str], tasks_dir: Path, core_dir: Path) -> None:
    core_hashes: dict[str, str] = {}
    with tarfile.open(tarball, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if ".." in parts or PurePosixPath(member.name).is_absolute():
                raise ValueError(f"unsafe path in the WANDR tarball: {member.name}")
            if not member.isfile() or len(parts) < 5 or parts[1:3] != ("datasets", "wandr"):
                continue
            task, relative = parts[3], PurePosixPath(*parts[4:])
            if relative.parts[:2] == ("tests", "wandr_core"):
                target = core_dir.joinpath(*relative.parts[2:])
            elif task not in task_names:
                continue
            elif (
                str(relative) in TASK_FILES
                or (relative.parts[0] == "tests" and len(relative.parts) == 2 and relative.name in TEST_FILES)
                or relative.parts[:2] == ("tests", "wandr_task")
            ):
                target = tasks_dir / task / relative
            else:
                continue
            content = archive.extractfile(member).read()
            if target.is_relative_to(core_dir):
                digest = hashlib.sha256(content).hexdigest()
                if core_hashes.setdefault(str(target), digest) != digest:
                    raise ValueError(f"{member.name} differs from the other tasks' copies of wandr_core")
                if target.exists():
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    missing = sorted(name for name in task_names if not (tasks_dir / name / "tests" / "verify.py").exists())
    if missing or not (core_dir / "src" / "pipeline.py").exists():
        raise ValueError(f"task packages are incomplete; missing tasks: {missing[:5]}")


def _select_subset(rows: list[dict], names: list[str]) -> list[dict]:
    """The rows of the named tasks, in benchmark order; every name must name exactly one benchmark task."""
    if len(set(names)) != len(names):
        raise ValueError("the subset lists a duplicate task name")
    unknown = sorted(set(names) - {row["task_name"] for row in rows})
    if unknown:
        raise ValueError(f"the subset lists unknown tasks: {unknown}")
    return [row for row in rows if row["task_name"] in set(names)]


def prepare() -> Path:
    """Download the pinned sources and write the benchmark rows; returns the rows' path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wandr_prepare_", dir=DATA_DIR) as scratch:
        scratch_dir = Path(scratch)
        splits = {}
        for split, expected_count in (("test", 500), ("smoke", 1)):
            _download(HF_SPLIT_URL.format(revision=HF_REVISION, split=split), scratch_dir / f"{split}.jsonl")
            splits[split] = _load_tasks(scratch_dir / f"{split}.jsonl", expected_count)
        tasks, all_tasks = splits["test"], splits["test"] + splits["smoke"]
        _download(GITHUB_TARBALL_URL, scratch_dir / "wandr.tar.gz")
        tasks_dir, core_dir = scratch_dir / "tasks", scratch_dir / "wandr_core"
        _extract(scratch_dir / "wandr.tar.gz", {task["task_name"] for task in all_tasks}, tasks_dir, core_dir)

        index = {
            "source_commit": SOURCE_COMMIT,
            "files": {
                name: {"task": task["task_name"], "item_fields": fields}
                for task in all_tasks
                for name, fields in task["item_fields"].items()
            },
        }
        (tasks_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
        rows = []
        for task in tasks:
            instruction = (tasks_dir / task["task_name"] / "instruction.md").read_text(encoding="utf-8")
            rows.append(
                {
                    "responses_create_params": {
                        "input": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": instruction},
                        ],
                        "tools": TOOLS,
                    },
                    "task_name": task["task_name"],
                    "task_id": task["task_id"],
                    "required_output_files": task["required_output_files"],
                }
            )
        subset_rows = _select_subset(rows, SUBSET100_FPATH.read_text().split())
        rows_path, subset_path = scratch_dir / OUTPUT_FPATH.name, scratch_dir / SUBSET100_OUTPUT_FPATH.name
        for path, selected in ((rows_path, rows), (subset_path, subset_rows)):
            path.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")

        # Replace earlier output only after every download and check passed.
        for final, staged in ((TASKS_DIR, tasks_dir), (CORE_DIR, core_dir)):
            if final.exists():
                old = scratch_dir / f"old_{final.name}"
                final.rename(old)
            staged.rename(final)
        rows_path.replace(OUTPUT_FPATH)
        subset_path.replace(SUBSET100_OUTPUT_FPATH)
    print(f"Wrote {len(tasks)} WANDR tasks to {OUTPUT_FPATH} and {len(subset_rows)} to {SUBSET100_OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
