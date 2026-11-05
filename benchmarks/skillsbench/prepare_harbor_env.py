# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt the official BenchFlow SkillsBench task set to Harbor task folders for Gym's `harbor` environment server.

`prepare.py` writes the task tree for the OpenHands path through `harbor_agent_general`; this script reuses its
checkout and task conversion and writes the tasks of each skill condition into that condition's dataset entry.
"""

import os
import re
import shutil
from pathlib import Path

from benchmarks.skillsbench.prepare import (
    GYM_ROOT,
    IMAGE_MANIFEST_ENV_VAR,
    SKILL_MODE_ENV_VAR,
    SOURCE_ENV_VAR,
    adapt_task,
    fetch_official_tasks,
    load_image_manifest,
    task_definition,
    task_toml,
)


BENCHMARK_DIR = Path(__file__).parent
# Where a condition's tasks are converted before they replace the entry's previous ones.
STAGING_DIR = BENCHMARK_DIR / "data"
# Each condition is a dataset entry committed under environments/harbor/ (its README.md and dataset.toml), so
# `gym eval run skillsbench ...` resolves it by name; the task folders this script writes beside them are
# gitignored. The entry name is the taskset name in rows and results.
ENTRIES_DIR = GYM_ROOT / "environments" / "harbor"
DATASETS = {"with-skill": "skillsbench", "no-skill": "skillsbench_no_skill"}
ENTRY_FILES = ("README.md", "dataset.toml")
# Where the with-skill condition's skills land at seed: the environment server uploads the task's
# `environment/skills/` there, and harnesses read the path from the task context.
SKILLS_DIR = "/root/.agents/skills"
# Agent skill locations a task Dockerfile may bake skills into; the no-skill condition clears them.
AGENT_SKILL_PATHS = (
    "/root/.agents/skills",
    "/root/.codex/skills",
    "/root/.claude/skills",
    "/root/.gemini/skills",
    "/root/.opencode/skills",
    "/root/.opencode/skill",
    "/root/.goose/skills",
    "/root/.factory/skills",
    "/root/.openhands/skills",
)
_BAKES_SKILLS = re.compile(r"^\s*(COPY|ADD)\s.*\bskills\b", re.IGNORECASE | re.MULTILINE)
PINNED_IMAGE_MANIFEST = BENCHMARK_DIR / "images.json"


def _adapt_task(source: Path, destination: Path, skill_mode: str, docker_image: str | None = None) -> None:
    # The shared conversion (prompt, build context, verifier and oracle wrappers) without its Dockerfile skill
    # installation: this server uploads the skills at seed instead.
    adapt_task(source, destination, "no-skill", docker_image)
    definition, _ = task_definition(source / "task.md")
    environment = destination / "environment"
    dockerfile = environment / "Dockerfile"
    image = docker_image
    if skill_mode == "no-skill" and _BAKES_SKILLS.search(dockerfile.read_text()):
        # Both conditions share one image, and this one has skills built in: clear them before the agent starts.
        # With a published image the Dockerfile becomes that image plus the removal, which the server applies at
        # seed without a build; without one the removal ends the task's own build.
        removal = f"RUN rm -rf {' '.join(AGENT_SKILL_PATHS)}\n"
        if docker_image:
            dockerfile.write_text(f"FROM {docker_image}\n{removal}")
            image = None
        else:
            dockerfile.write_text(
                dockerfile.read_text().rstrip("\n") + f"\n\n# SkillsBench no-skill condition.\n{removal}"
            )
    if skill_mode == "no-skill" and not _BAKES_SKILLS.search(dockerfile.read_text()):
        # Nothing builds from the skills any more, so the no-skill task does not carry them at all.
        shutil.rmtree(environment / "skills", ignore_errors=True)
    skills_dir = SKILLS_DIR if skill_mode == "with-skill" else None
    (destination / "task.toml").write_text(
        task_toml(source.name, definition, image, skills_dir=skills_dir, current_schema=True)
    )


def _image_manifest(source_dir: Path, task_names: set[str]) -> dict[str, str] | None:
    """The published image per task: from SKILLSBENCH_IMAGE_MANIFEST, else the pinned manifest for the pinned
    checkout. ``none`` (or a custom checkout without a manifest) builds each task from its Dockerfile instead."""
    setting = os.environ.get(IMAGE_MANIFEST_ENV_VAR)
    if setting == "none" or (setting is None and os.environ.get(SOURCE_ENV_VAR)):
        return None
    return load_image_manifest(setting or PINNED_IMAGE_MANIFEST, source_dir, task_names)


def _write_dataset(task_dirs: list[Path], skill_mode: str, images: dict[str, str] | None) -> Path:
    """Write one condition's tasks into its entry, replacing the previous ones only once every task converted; the
    entry's committed files stay."""
    folder = ENTRIES_DIR / DATASETS[skill_mode]
    staging = STAGING_DIR / f".{folder.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for source in task_dirs:
        _adapt_task(source, staging / source.name, skill_mode, images[source.name] if images else None)
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.name in ENTRY_FILES:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for task in staging.iterdir():
        task.rename(folder / task.name)
    staging.rmdir()
    return folder


def prepare() -> list[Path]:
    """Build the Harbor dataset entries from an official SkillsBench checkout: both conditions, or the one
    SKILLSBENCH_SKILL_MODE names."""
    source_override = os.environ.get(SOURCE_ENV_VAR)
    source_dir = Path(source_override).expanduser().resolve() if source_override else fetch_official_tasks()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Set {SOURCE_ENV_VAR} to the official SkillsBench tasks/ directory: {source_dir}")
    task_dirs = sorted(path for path in source_dir.iterdir() if (path / "task.md").is_file())
    if not task_dirs:
        raise ValueError(f"No official SkillsBench task.md files found in {source_dir}")
    skill_mode = os.environ.get(SKILL_MODE_ENV_VAR)
    if skill_mode is not None and skill_mode not in DATASETS:
        raise ValueError(f"{SKILL_MODE_ENV_VAR} must be 'with-skill' or 'no-skill', got {skill_mode!r}")

    images = _image_manifest(source_dir, {task.name for task in task_dirs})
    folders = []
    for mode in [skill_mode] if skill_mode else list(DATASETS):
        folder = _write_dataset(task_dirs, mode, images)
        print(f"Prepared {len(task_dirs)} official SkillsBench tasks ({mode}) at {folder}")
        folders.append(folder)
    return folders


if __name__ == "__main__":
    prepare()
