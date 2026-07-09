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
"""Prepare the SWE-Atlas Codebase QnA benchmark data.

Reads the per-task directories under the SWE-Atlas repo's ``data/qa/`` (each with
``task.toml``, ``tests/prompt.txt``, ``tests/rubrics.json``) and writes one Gym
benchmark row per task. Each row carries the codebase ``question`` at the top
level (the ``prompt_config`` renders it into the model input) plus a
``verifier_metadata`` block consumed by the ``swe_atlas_qna`` rubric-judge
resources server (``rubrics``, ``problem_statement``, and task identity).

The SWE-Atlas data is not publicly hosted as a flat download, so the source repo
must be available locally. Point the script at it via, in priority order:

1. the ``swe_atlas_dir`` kwarg (``+prepare_script_args.swe_atlas_dir=...``),
2. the ``SWE_ATLAS_DIR`` environment variable,
3. a shallow ``git clone`` of ``REPO_URL`` into a local cache (fallback).
"""

import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "swe_atlas_qna_benchmark.jsonl"

# Public SWE-Atlas repository, used only as a fallback when no local checkout is
# supplied. Cloned shallowly into ``data/_swe_atlas_src`` (gitignored).
REPO_URL = "https://github.com/scaleapi/SWE-Atlas.git"
_CLONE_DIR = DATA_DIR / "_swe_atlas_src"


def _sif_basename(docker_image: str) -> Optional[str]:
    """Suggested .sif basename derived from the container image tag."""
    if not docker_image:
        return None
    tag = docker_image.split(":")[-1] if ":" in docker_image else docker_image
    return f"{tag}.sif"


def _resolve_source_dir(swe_atlas_dir: Optional[str]) -> Path:
    """Locate a SWE-Atlas checkout, cloning it as a last resort."""
    candidate = swe_atlas_dir or os.environ.get("SWE_ATLAS_DIR")
    if candidate:
        source = Path(candidate).expanduser()
        if not (source / "data" / "qa").is_dir():
            raise FileNotFoundError(f"{source} does not contain a data/qa directory — is this the SWE-Atlas repo?")
        return source

    if not (_CLONE_DIR / "data" / "qa").is_dir():
        print(f"No SWE_ATLAS_DIR set; cloning {REPO_URL} into {_CLONE_DIR} ...")
        _CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(_CLONE_DIR)],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Could not obtain the SWE-Atlas data. Set the SWE_ATLAS_DIR environment variable (or pass "
                "+prepare_script_args.swe_atlas_dir=/path/to/SWE-Atlas) to a local checkout of the SWE-Atlas repo."
            ) from exc
    return _CLONE_DIR


def _convert_task(task_dir: Path) -> dict:
    """Build one Gym benchmark row from a SWE-Atlas QnA task directory."""
    task_meta = tomllib.loads((task_dir / "task.toml").read_text())
    metadata = task_meta.get("metadata", {})
    environment = task_meta.get("environment", {})

    problem_statement = (task_dir / "tests" / "prompt.txt").read_text().strip()
    rubrics = json.loads((task_dir / "tests" / "rubrics.json").read_text())
    instance_id = task_dir.name
    docker_image = environment.get("docker_image", "")

    return {
        # Top-level ``question`` is rendered into the model input by prompt_config.
        "question": problem_statement,
        "instance_id": instance_id,
        "verifier_metadata": {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "rubrics": rubrics,
            "category": metadata.get("category"),
            "language": metadata.get("language"),
            "repository": metadata.get("repository"),
            "base_commit": metadata.get("base_commit"),
            "docker_image": docker_image,
            "sif_basename": _sif_basename(docker_image),
        },
    }


def prepare(swe_atlas_dir: Optional[str] = None) -> Path:
    """Convert the SWE-Atlas QnA tasks and write the benchmark JSONL. Returns the path."""
    source_dir = _resolve_source_dir(swe_atlas_dir)
    qa_dir = source_dir / "data" / "qa"

    task_dirs = sorted(d for d in qa_dir.iterdir() if d.is_dir() and d.name.startswith("task-"))
    if not task_dirs:
        raise SystemExit(f"No task-* directories found under {qa_dir}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for task_dir in task_dirs:
            fout.write(json.dumps(_convert_task(task_dir)) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
