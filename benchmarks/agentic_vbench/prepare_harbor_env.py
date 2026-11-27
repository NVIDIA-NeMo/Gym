# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write Agentic-VBench tasks as Harbor task folders for Gym's `harbor` environment server.

The 100 pinned tasks are written into the dataset entry `environments/harbor/agentic_vbench/`, where the generic
`harbor` environment server seeds each task's image as a sandbox for any sandboxed harness and runs the task's own
verifier. Each task gets the adjustments in `staging.py` (Hub judge protocol, assembly judge guard, bounded longer
golden-asset retries, exit 75 for verifier-side infrastructure failures, `/workspace/output` as the judged and repair
steps' artifact) and a `task_data.json` naming its family; the records of what changed land under `.records/<task>/`
in the entry.

`--tasks-root DIR` writes a root of tasks in the pinned layout (`tasks/agentic_vbench_<family>/<task>/`) that is not
the pinned checkout, such as generated training tasks of the four families, into the training entry
`environments/harbor/agentic_vbench_train/` instead (`--entry` names another), with the same adjustments.

Environment: AGENTIC_VBENCH_ROOT names an existing pinned checkout (else it is fetched into the cache root),
AGENTIC_VBENCH_HARBOR_FAMILIES selects families (comma-separated; default all four) and
AGENTIC_VBENCH_JUDGE_PROTOCOL selects the judge protocol (default nvinference-upstream, the upstream judge pair).
The command-line flags of the same names take precedence.
"""

import argparse
import json
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
    stage_opencode_plugins,
)


GYM_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
# Where tasks are adjusted before they replace the entry's previous ones (benchmarks/agentic_vbench/.gitignore
# keeps data/ out of git).
STAGING_DIR = BENCHMARK_DIR / "data"
ENTRY_NAME = "agentic_vbench"
# The training entry: the same task shape and adjustments for a root of tasks that is not the pinned benchmark.
TRAIN_ENTRY_NAME = "agentic_vbench_train"
ENTRIES_DIR = GYM_ROOT / "environments" / "harbor"
ENTRY_DIR = ENTRIES_DIR / ENTRY_NAME
# The task's own typed field (`task_data.json`, typed by the entry's `task_data.py`): its family, which the
# benchmark's score weighs by and which `family_metrics.py` groups by.
TASK_DATA_FILE = "task_data.json"
FAMILY_FOLDER_PREFIX = "agentic_vbench_"
# The files an entry commits beside its tasks: what the repository's .gitignore keeps under environments/harbor/.
ENTRY_FILES = ("README.md", "dataset.toml", "task_data.py")
ENTRY_SUFFIXES = (".yaml",)
RECORDS_DIR = ".records"
FAMILIES_ENV_VAR = "AGENTIC_VBENCH_HARBOR_FAMILIES"
PROTOCOL_ENV_VAR = "AGENTIC_VBENCH_JUDGE_PROTOCOL"


def task_family(task_dir: Path) -> str:
    """The family a task folder sits under in the pinned layout (``tasks/agentic_vbench_<family>/<task>``)."""
    folder = task_dir.parent.name
    family = folder.removeprefix(FAMILY_FOLDER_PREFIX)
    if folder == family or family not in BENCHMARK_FAMILIES:
        raise ValueError(
            f"{task_dir}: a task folder sits under tasks/{FAMILY_FOLDER_PREFIX}<family>/ with family one of "
            f"{sorted(BENCHMARK_FAMILIES)}"
        )
    return family


def adapt_task(source: Path, destination: Path, *, protocol: str, records: Path) -> None:
    """Write ``source`` (a task folder under its family folder) at ``destination`` with the adjustments.

    The three adjustment stages each copy the task only when they change it; the plugin stage copies every task.
    So a task none of the adjustments touches is the source folder plus ``task_data.json``, which names the task's
    family, and the OpenCode plugins under each step's ``workdir/.opencode-plugins/``. Every stage reads the previous
    stage's output and writes beside it, so no stage copies onto its own input.
    """
    if protocol not in JUDGE_PROTOCOLS:
        raise ValueError(f"Unknown judge protocol {protocol!r}; one of {sorted(JUDGE_PROTOCOLS)}")
    family = task_family(source)
    if (source / TASK_DATA_FILE).exists():
        raise ValueError(f"{source}: the task ships its own {TASK_DATA_FILE}; the entry writes the family there")
    records.mkdir(parents=True, exist_ok=True)
    work = destination.parent / f".{destination.name}.stages"
    if work.exists():
        shutil.rmtree(work)
    try:
        staged = stage_assembly_judge_guard(source, staging=_fresh(work / "guard"), output=records)
        staged = stage_judge(staged, staging=_fresh(work / "judge"), output=records, protocol=protocol)
        staged = stage_materials(staged, staging=_fresh(work / "materials"), output=records)
        staged = stage_opencode_plugins(staged, staging=_fresh(work / "plugins"), output=records)
        shutil.copytree(staged, destination)
        (destination / TASK_DATA_FILE).write_text(json.dumps({"family": family}) + "\n")
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


def _check_families(families: list[str]) -> None:
    unknown = sorted(set(families) - set(BENCHMARK_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families {unknown}; one of {sorted(BENCHMARK_FAMILIES)}")


def selected_task_dirs(root: Path, families: list[str]) -> list[Path]:
    """The pinned task folders of ``families``, validated against the complete inventory."""
    _check_families(families)
    tasks = inventory(root)
    return [
        root / "tasks" / f"{FAMILY_FOLDER_PREFIX}{task['family']}" / task_id
        for task_id, task in sorted(tasks.items())
        if task["family"] in families
    ]


def custom_task_dirs(root: Path, families: list[str]) -> list[Path]:
    """The task folders of ``families`` under a root in the pinned layout that is not the pinned checkout.

    A training root holds generated tasks of the four families; nothing pins their count or revision, so the
    only checks are the layout, one ``task.toml`` per task and task names unique across families (a task folder
    may be a link to the task's real folder).
    """
    _check_families(families)
    task_dirs: list[Path] = []
    for family in sorted(families):
        family_dir = root / "tasks" / f"{FAMILY_FOLDER_PREFIX}{family}"
        if not family_dir.is_dir():
            continue
        task_dirs.extend(sorted(path.parent for path in family_dir.glob("*/task.toml")))
    if not task_dirs:
        raise ValueError(f"No tasks of {sorted(families)} under {root / 'tasks'}")
    names = [task.name for task in task_dirs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Task folder names must be unique across families: {duplicates}")
    return task_dirs


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tasks-root",
        type=Path,
        help="A root of tasks in the pinned layout that is not the pinned checkout (written to the training entry).",
    )
    parser.add_argument(
        "--entry",
        help=f"The entry under environments/harbor/ to write ({ENTRY_NAME}, or {TRAIN_ENTRY_NAME} with --tasks-root).",
    )
    parser.add_argument(
        "--families",
        default=os.environ.get(FAMILIES_ENV_VAR, ""),
        help="Comma-separated families (default all four).",
    )
    parser.add_argument(
        "--judge-protocol",
        default=os.environ.get(PROTOCOL_ENV_VAR, DEFAULT_JUDGE_PROTOCOL),
        choices=sorted(JUDGE_PROTOCOLS),
        help="How the repurpose tasks are judged.",
    )
    return parser.parse_args()


def prepare(
    *, tasks_root: Path | None = None, entry_name: str | None = None, families: list[str] | None = None, protocol: str
) -> Path:
    """Write the selected tasks into the entry and return it."""
    families = families or sorted(BENCHMARK_FAMILIES)
    if tasks_root is None:
        root = ensure_checkout(
            Path(os.environ["AGENTIC_VBENCH_ROOT"]) if os.environ.get("AGENTIC_VBENCH_ROOT") else None
        )
        task_dirs = selected_task_dirs(root, families)
        entry_name = entry_name or ENTRY_NAME
    else:
        task_dirs = custom_task_dirs(tasks_root.resolve(), families)
        entry_name = entry_name or TRAIN_ENTRY_NAME
    if entry_name == ENTRY_NAME and tasks_root is not None:
        raise ValueError(f"{ENTRY_NAME} is the pinned benchmark; write a task root to {TRAIN_ENTRY_NAME} or --entry")
    entry = write_entry(task_dirs, ENTRIES_DIR / entry_name, protocol=protocol)
    source = "pinned checkout" if tasks_root is None else str(tasks_root)
    print(
        f"Prepared {len(task_dirs)} Agentic-VBench tasks ({', '.join(families)}; judge {protocol}) from {source} "
        f"at {entry}"
    )
    return entry


def main() -> None:
    args = _arguments()
    families = [name.strip() for name in args.families.split(",") if name.strip()]
    prepare(tasks_root=args.tasks_root, entry_name=args.entry, families=families, protocol=args.judge_protocol)


if __name__ == "__main__":
    main()
