# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the pinned Agentic-VBench tasks as Harbor task folders for Gym's `harbor` environment server.

The 100 pinned tasks are written into the dataset entry `environments/harbor/agentic_vbench/`, where the generic
`harbor` environment server seeds each task's image as a sandbox for any sandboxed harness and runs the task's own
verifier. Each task gets the adjustments in `staging.py` (Hub judge protocol, assembly judge guard, longer
golden-asset retries, inconclusive verdicts for verifier-side infrastructure failures); the records of what changed
land under `.records/<task>/` in the entry.

Environment: AGENTIC_VBENCH_ROOT names an existing pinned checkout (else it is fetched into the cache root),
AGENTIC_VBENCH_HARBOR_FAMILIES selects families (comma-separated; default all four) and
AGENTIC_VBENCH_JUDGE_PROTOCOL selects the judge protocol (default nvinference-native, the upstream judge pair).
"""

import os
import shutil
from pathlib import Path

from benchmarks.agentic_vbench.checkout import BENCHMARK_FAMILIES, ensure_checkout, inventory
from benchmarks.agentic_vbench.staging import (
    DEFAULT_JUDGE_PROTOCOL,
    JUDGE_PROTOCOLS,
    stage_assembly_judge_guard,
    stage_judge,
    stage_materials,
)


GYM_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
# Where tasks are adjusted before they replace the entry's previous ones (benchmarks/agentic_vbench/.gitignore
# keeps data/ out of git).
STAGING_DIR = BENCHMARK_DIR / "data"
ENTRY_NAME = "agentic_vbench"
ENTRY_DIR = GYM_ROOT / "environments" / "harbor" / ENTRY_NAME
# The files an entry commits beside its tasks: what the repository's .gitignore keeps under environments/harbor/.
ENTRY_FILES = ("README.md", "dataset.toml")
ENTRY_SUFFIXES = (".yaml",)
RECORDS_DIR = ".records"
FAMILIES_ENV_VAR = "AGENTIC_VBENCH_HARBOR_FAMILIES"
PROTOCOL_ENV_VAR = "AGENTIC_VBENCH_JUDGE_PROTOCOL"


def adapt_task(source: Path, destination: Path, *, protocol: str, records: Path) -> None:
    """Write ``source`` (a pinned task folder, under its family folder) at ``destination`` with the adjustments.

    The three stages each copy the task only when they change it, so an untouched task is the upstream folder
    byte for byte. Every stage reads the previous stage's output and writes beside it, so no stage copies onto
    its own input.
    """
    if protocol not in JUDGE_PROTOCOLS:
        raise ValueError(f"Unknown judge protocol {protocol!r}; one of {sorted(JUDGE_PROTOCOLS)}")
    records.mkdir(parents=True, exist_ok=True)
    work = destination.parent / f".{destination.name}.stages"
    if work.exists():
        shutil.rmtree(work)
    try:
        staged = stage_assembly_judge_guard(source, staging=_fresh(work / "guard"), output=records)
        staged = stage_judge(staged, staging=_fresh(work / "judge"), output=records, protocol=protocol)
        staged = stage_materials(staged, staging=_fresh(work / "materials"))
        shutil.copytree(staged, destination)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _fresh(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _is_entry_file(path: Path) -> bool:
    return path.is_file() and (path.name in ENTRY_FILES or path.suffix in ENTRY_SUFFIXES)


def write_entry(task_dirs: list[Path], entry: Path = ENTRY_DIR, *, protocol: str = DEFAULT_JUDGE_PROTOCOL) -> Path:
    """Write the adjusted tasks into ``entry`` as a flat folder of task folders, keeping the entry's committed files.

    The tasks are prepared in a staging folder, the entry's committed files are copied in, and the result replaces
    the entry by renaming, so an interrupted preparation leaves the previous tasks in place.
    """
    staging = STAGING_DIR / f".{entry.name}.staging"
    previous = STAGING_DIR / f".{entry.name}.previous"
    if previous.exists():
        if entry.exists():
            shutil.rmtree(previous)
        else:
            # A preparation stopped between its two renames: the entry is still here.
            previous.rename(entry)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    names = [task.name for task in task_dirs]
    if len(set(names)) != len(names):
        raise ValueError("Task folder names must be unique across families")
    for source in task_dirs:
        adapt_task(source, staging / source.name, protocol=protocol, records=staging / RECORDS_DIR / source.name)
    if entry.exists():
        for path in entry.iterdir():
            if _is_entry_file(path):
                shutil.copy2(path, staging / path.name)
        entry.rename(previous)
    entry.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(entry)
    if previous.exists():
        shutil.rmtree(previous)
    return entry


def selected_task_dirs(root: Path, families: list[str]) -> list[Path]:
    """The pinned task folders of ``families``, validated against the complete inventory."""
    unknown = sorted(set(families) - set(BENCHMARK_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families {unknown}; one of {sorted(BENCHMARK_FAMILIES)}")
    tasks = inventory(root)
    return [
        root / "tasks" / f"agentic_vbench_{task['family']}" / task_id
        for task_id, task in sorted(tasks.items())
        if task["family"] in families
    ]


def prepare() -> Path:
    root = ensure_checkout(Path(os.environ["AGENTIC_VBENCH_ROOT"]) if os.environ.get("AGENTIC_VBENCH_ROOT") else None)
    setting = os.environ.get(FAMILIES_ENV_VAR, "")
    families = [name.strip() for name in setting.split(",") if name.strip()] or sorted(BENCHMARK_FAMILIES)
    protocol = os.environ.get(PROTOCOL_ENV_VAR, DEFAULT_JUDGE_PROTOCOL)
    task_dirs = selected_task_dirs(root, families)
    entry = write_entry(task_dirs, protocol=protocol)
    print(f"Prepared {len(task_dirs)} Agentic-VBench tasks ({', '.join(families)}; judge {protocol}) at {entry}")
    return entry


if __name__ == "__main__":
    prepare()
