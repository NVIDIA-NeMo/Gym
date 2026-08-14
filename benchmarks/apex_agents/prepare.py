# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the gated Apex Agents evaluation split for ``gym eval prepare``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nemo_gym.global_config import get_hf_token


OUTPUT_FPATH = Path("benchmarks/apex_agents/data/apex_agents_benchmark.jsonl")
WORLD_CACHE_DIR = Path("benchmarks/apex_agents/data/world_cache")
DATASET_REPO = "mercor/apex-agents"
SERVICE_MAP = {
    "FMP": "fmp",
    "Edgar SEC": "edgar",
}

GRADING_TARGETS = {
    "message_in_console": {
        "scope": "console",
        "expected_file_type": "Final Answer Only (No Files)",
        "extensions": [],
    },
    "make_new_doc": {
        "scope": "files",
        "expected_file_type": "Word Documents (.docx, .doc)",
        "extensions": [".doc", ".docx"],
    },
    "edit_existing_doc": {
        "scope": "files",
        "expected_file_type": "Word Documents (.docx, .doc)",
        "extensions": [".doc", ".docx"],
    },
    "make_new_sheet": {
        "scope": "files",
        "expected_file_type": "Spreadsheets (.xlsx, .xls, .xlsm)",
        "extensions": [".csv", ".xls", ".xlsm", ".xlsx"],
    },
    "edit_existing_sheet": {
        "scope": "files",
        "expected_file_type": "Spreadsheets (.xlsx, .xls, .xlsm)",
        "extensions": [".csv", ".xls", ".xlsm", ".xlsx"],
    },
    "make_new_slide_deck": {
        "scope": "files",
        "expected_file_type": "Presentations (.pptx, .ppt)",
        "extensions": [".ppt", ".pptx"],
    },
    "edit_existing_slide_deck": {
        "scope": "files",
        "expected_file_type": "Presentations (.pptx, .ppt)",
        "extensions": [".ppt", ".pptx"],
    },
}


def rubric_with_grading_targets(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach held-out evaluation scope to every independently graded criterion."""
    target = GRADING_TARGETS.get(
        str(task.get("expected_output") or ""),
        {
            "scope": "both",
            "expected_file_type": "All output (modified files and final message in console)",
            "extensions": [],
        },
    )
    return [dict(criterion, grading_target=dict(target)) for criterion in task.get("rubric") or []]


def convert_task(task: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
    """Keep task-visible fields separate from held-out verifier metadata."""
    foundry_services = sorted(
        {SERVICE_MAP[app["service_name"]] for app in world.get("apps") or [] if app.get("service_name") in SERVICE_MAP}
    )
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": task["prompt"]}],
        },
        "task_id": task["task_id"],
        "world_id": task["world_id"],
        "domain": task.get("domain"),
        "foundry_services": foundry_services,
        "verifier_metadata": {
            "task_name": task.get("task_name"),
            "expected_output": task.get("expected_output"),
            "rubric": rubric_with_grading_targets(task),
            "gold_response": task.get("gold_response"),
            "gold_response_type": task.get("gold_response_type"),
        },
    }


def prepare_rows(tasks_path: Path, worlds_path: Path, output: Path, *, limit: int | None = None) -> int:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    worlds = {world["world_id"]: world for world in json.loads(worlds_path.read_text(encoding="utf-8"))}
    rows = [convert_task(task, worlds.get(task["world_id"], {})) for task in tasks]
    if limit is not None:
        rows = rows[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def prefetch_worlds(worlds_path: Path, *, cache_dir: Path, hf_token: str | None = None) -> int:
    """Download every unique world ZIP into the offline cache used at runtime."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    worlds = json.loads(worlds_path.read_text(encoding="utf-8"))
    world_ids = sorted({world["world_id"] for world in worlds})
    for world_id in world_ids:
        kwargs: dict[str, Any] = {
            "repo_id": DATASET_REPO,
            "filename": f"world_files_zipped/{world_id}.zip",
            "repo_type": "dataset",
            "cache_dir": str(cache_dir),
        }
        if hf_token:
            kwargs["token"] = hf_token
        hf_hub_download(**kwargs)
    return len(world_ids)


def prepare(
    tasks_path: Path | None = None,
    worlds_path: Path | None = None,
    output: Path = OUTPUT_FPATH,
    *,
    limit: int | None = None,
    world_cache_dir: Path = WORLD_CACHE_DIR,
    hf_token: str | None = None,
) -> Path:
    """Prepare Gym rows and the offline world cache required for rollouts."""
    if tasks_path is None or worlds_path is None:
        from huggingface_hub import hf_hub_download

        download_kwargs: dict[str, Any] = {
            "repo_id": DATASET_REPO,
            "repo_type": "dataset",
        }
        if hf_token:
            download_kwargs["token"] = hf_token
        tasks_path = Path(hf_hub_download(filename="tasks_and_rubrics.json", **download_kwargs))
        worlds_path = Path(hf_hub_download(filename="world_descriptions.json", **download_kwargs))

    count = prepare_rows(tasks_path, worlds_path, output, limit=limit)
    print(f"wrote {count} Apex Agents rows to {output}")
    world_count = prefetch_worlds(worlds_path, cache_dir=world_cache_dir, hf_token=hf_token)
    print(f"cached {world_count} Apex Agents world ZIPs")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--worlds", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--world-cache-dir", type=Path, default=WORLD_CACHE_DIR)
    args = parser.parse_args()
    if (args.tasks is None) != (args.worlds is None):
        parser.error("--tasks and --worlds must be supplied together")
    prepare(
        tasks_path=args.tasks,
        worlds_path=args.worlds,
        output=args.output,
        limit=args.limit,
        world_cache_dir=args.world_cache_dir,
        hf_token=get_hf_token(),
    )


if __name__ == "__main__":
    main()
