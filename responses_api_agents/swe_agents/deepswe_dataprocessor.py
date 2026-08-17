#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""DeepSWE (Harbor task format) -> NeMo-Gym JSONL dataprocessor.

DeepSWE tasks (https://deepswe.datacurve.ai/) ship as Harbor task directories:

    <task_id>/
      task.toml          # metadata: repo, base commit, language, timeouts
      instruction.md     # the prompt the agent sees (problem statement)
      tests/test.sh      # Harbor verifier entry point (graded at eval time)
      tests/test.patch   # hidden test additions, applied at grading time
      solution/solution.patch   # reference (golden) patch, held out from agent

This module converts those task directories into the JSONL schema the
``swe_agents`` wrapper consumes (one problem per line). The verifier artifacts
(``test.sh``, ``test.patch``) and the golden ``solution.patch`` are packed into
``responses_create_params.metadata.instance_dict`` so that the eval container
(see ``DeepSWEDatasetProcessor`` in ``app.py``) is fully self-contained and the
golden-patch validation path (``verify_golden_patch``) has a ``patch`` to apply.

Run as a script to materialize a dataset::

    python -m responses_api_agents.swe_agents.deepswe_dataprocessor \
        --tasks-dir temp/deep-swe/tasks \
        --images-dir temp/deep-swe/singularity_images \
        --output-jsonl data/deepswe.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


try:  # Python 3.11+ stdlib.
    import tomllib

    def _load_toml(path: Path) -> Dict[str, Any]:
        with path.open("rb") as f:
            return tomllib.load(f)

except ModuleNotFoundError:  # Fall back to tomlkit (a project dependency).
    import tomlkit

    def _load_toml(path: Path) -> Dict[str, Any]:
        return tomlkit.parse(path.read_text(encoding="utf-8"))


# Identifies DeepSWE rows to the swe_agents wrapper (dataset_name dispatch).
DEEPSWE_DATASET_NAME = "deepswe"
# DeepSWE images clone the repo into /app (see the task Dockerfiles).
DEEPSWE_WORKSPACE_PATH = "/app"
# Image filename convention from temp/deep-swe/pull.sh: deepswe.<task_id_lower>.sif
DEEPSWE_IMAGE_TEMPLATE = "deepswe.{task_id_lower}.sif"

# Tasks whose PRE-EXISTING (base/regression) test suite fails in the published
# image with NO patch applied — i.e. "broken baselines" caused by library
# version drift in datacurve's images (verified via the base-at-base-commit
# screen, scripts/golden_patch_val/deepswe_base_screen.sh). Harbor scores
# base+new, so these can never reach reward 1.0 regardless of the solution, and
# datacurve's README notes solution.patch "is never used at grading time".
# Excluded by default so they don't add unscorable noise to eval/training;
# override with include_broken_baseline=True / --include-broken-baseline.
KNOWN_BROKEN_BASELINE = {
    "langchain-request-coalescing",  # base: repr snapshot drift (langchain-core); needs snapshot regen
}

# Per-task environment repair run in the eval BEFORE the verifier, to restore a
# repo-pinned dependency the mirror image drifted from (it built with the latest
# dep instead of the repo's pin). Verified to green the base suite. Needs host
# network at eval time (eval runs without network isolation). Tasks listed here
# are NOT in KNOWN_BROKEN_BASELINE — they're fixable, not unscorable.
KNOWN_BASELINE_FIX = {
    # image shipped polars 1.40.1; skrub's pyproject pins polars==1.5.0.
    # With the pin restored, the base suite passes (2469 passed).
    "skrub-duration-encoding": "pip install 'polars==1.5.0'",
    # image shipped pandas 3.0.3 + polars 1.40.1. pandas<3 fixes most; polars<1.40
    # avoids the 1.40 dataframe-interchange DeprecationWarning (filterwarnings=error)
    # and the changed get_column error message. Resolves to polars 1.39.3 ->
    # full base green (10089 passed).
    "narwhals-rolling-window-suite": "pip install 'pandas<3' 'polars<1.40'",
    # apptainer injects LD_LIBRARY_PATH=/.singularity.d/libs; deno 2.x refuses to
    # spawn subprocesses (the help/completion snapshot tests do) with it set under
    # --allow-run. Unsetting it (persists via the brace-group wrapper) -> base
    # green (459 passed). Not a dependency pin.
    "cliffy-config-file-parsing": "unset LD_LIBRARY_PATH",
}


def _repo_slug(repository_url: str) -> str:
    """https://github.com/abs-lang/abs(.git) -> abs-lang/abs."""
    slug = repository_url.strip().rstrip("/")
    if slug.endswith(".git"):
        slug = slug[: -len(".git")]
    parts = [p for p in slug.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return slug


class DeepSWEDataProcessor:
    """Convert a directory of DeepSWE Harbor tasks into Gym JSONL rows."""

    def __init__(
        self,
        tasks_dir: Path | str,
        images_dir: Optional[Path | str] = None,
        require_image: bool = True,
        exclude_broken_baseline: bool = False,
        agent_ref_name: str = "swe_agents_val",
        split: str = "test",
        model: str = "model",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_output_tokens: int = 32768,
    ) -> None:
        self.tasks_dir = Path(tasks_dir)
        self.images_dir = Path(images_dir) if images_dir is not None else None
        self.require_image = require_image
        self.exclude_broken_baseline = exclude_broken_baseline
        self.agent_ref_name = agent_ref_name
        self.split = split
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_task_dirs(self) -> List[Path]:
        """Every immediate subdir of tasks_dir that contains a task.toml."""
        if not self.tasks_dir.is_dir():
            raise FileNotFoundError(f"tasks_dir does not exist: {self.tasks_dir}")
        return sorted(p.parent for p in self.tasks_dir.glob("*/task.toml"))

    def image_path(self, task_id: str) -> Optional[Path]:
        if self.images_dir is None:
            return None
        return self.images_dir / DEEPSWE_IMAGE_TEMPLATE.format(task_id_lower=task_id.lower())

    def has_image(self, task_id: str) -> bool:
        img = self.image_path(task_id)
        return bool(img and img.is_file() and img.stat().st_size > 0)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def build_instance_dict(self, task_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
        task_id = meta["task_id"]
        return {
            "instance_id": task_id,
            "dataset_name": DEEPSWE_DATASET_NAME,
            "repo": _repo_slug(meta.get("repository_url", "")),
            "repository_url": meta.get("repository_url", ""),
            "base_commit": meta.get("base_commit_hash", ""),
            "language": meta.get("language", ""),
            "problem_statement": meta["problem_statement"],
            # Verifier artifacts (consumed only by the eval container).
            "patch": meta["solution_patch"],  # golden patch -> verify_golden_patch
            "test_patch": meta["test_patch"],
            "test_sh": meta["test_sh"],
            "workspace_path": DEEPSWE_WORKSPACE_PATH,
            # Optional env-repair command run before the verifier (empty for most).
            "baseline_fix": KNOWN_BASELINE_FIX.get(task_id, ""),
        }

    def build_row(self, task_dir: Path) -> Optional[Dict[str, Any]]:
        toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        test_sh_path = task_dir / "tests" / "test.sh"
        test_patch_path = task_dir / "tests" / "test.patch"
        solution_patch_path = task_dir / "solution" / "solution.patch"

        for required in (toml_path, instruction_path, test_sh_path, test_patch_path, solution_patch_path):
            if not required.is_file():
                print(f"[deepswe] SKIP {task_dir.name}: missing {required.relative_to(task_dir)}", file=sys.stderr)
                return None

        toml_data = _load_toml(toml_path)
        metadata_section = toml_data.get("metadata", {})
        task_id = metadata_section.get("task_id") or task_dir.name

        if self.require_image and not self.has_image(task_id):
            print(f"[deepswe] SKIP {task_id}: no image at {self.image_path(task_id)}", file=sys.stderr)
            return None

        if self.exclude_broken_baseline and task_id in KNOWN_BROKEN_BASELINE:
            print(f"[deepswe] SKIP {task_id}: known broken baseline (unscorable in image)", file=sys.stderr)
            return None

        meta = {
            "task_id": task_id,
            "repository_url": metadata_section.get("repository_url", ""),
            "base_commit_hash": metadata_section.get("base_commit_hash", ""),
            "language": metadata_section.get("language", ""),
            "problem_statement": self._read(instruction_path),
            "test_sh": self._read(test_sh_path),
            "test_patch": self._read(test_patch_path),
            "solution_patch": self._read(solution_patch_path),
        }

        instance_dict = self.build_instance_dict(task_dir, meta)

        return {
            "responses_create_params": {
                "input": [],
                "metadata": {
                    "instance_id": task_id,
                    "base_commit": meta["base_commit_hash"],
                    "dataset_name": DEEPSWE_DATASET_NAME,
                    "split": self.split,
                    "problem_statement": meta["problem_statement"],
                    "golden_patch": meta["solution_patch"],
                    "instance_dict": json.dumps(instance_dict),
                },
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
            },
            "agent_ref": {"type": "responses_api_agents", "name": self.agent_ref_name},
            # Top-level convenience mirrors (match the SWE-bench JSONL layout).
            "instance_id": task_id,
            "repo": instance_dict["repo"],
            "base_commit": meta["base_commit_hash"],
            "patch": meta["solution_patch"],
            "test_patch": meta["test_patch"],
            "problem_statement": meta["problem_statement"],
            "language": meta["language"],
        }

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        for task_dir in self.discover_task_dirs():
            row = self.build_row(task_dir)
            if row is not None:
                yield row

    def write_jsonl(self, output_jsonl: Path | str, limit: Optional[int] = None) -> int:
        output_jsonl = Path(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with output_jsonl.open("w", encoding="utf-8") as f:
            for row in self.iter_rows():
                if limit is not None and n >= limit:
                    break
                f.write(json.dumps(row) + "\n")
                n += 1
        return n


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert DeepSWE Harbor tasks to NeMo-Gym JSONL.")
    p.add_argument("--tasks-dir", required=True, help="Directory of DeepSWE task dirs (each with task.toml).")
    p.add_argument(
        "--images-dir",
        default=None,
        help="Directory of deepswe.<task_id>.sif images. Used to filter tasks (unless --no-require-image).",
    )
    p.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    p.add_argument(
        "--no-require-image",
        action="store_true",
        help="Include tasks even if no matching .sif image is present.",
    )
    p.add_argument(
        "--exclude-broken-baseline",
        action="store_true",
        help="Drop tasks in KNOWN_BROKEN_BASELINE (e.g. langchain). Off by default — the full set is emitted.",
    )
    p.add_argument("--agent-ref-name", default="swe_agents_val", help="agent_ref.name baked into each row.")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="model")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-output-tokens", type=int, default=32768)
    p.add_argument("--limit", type=int, default=None, help="Cap the number of rows (smoke tests).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    processor = DeepSWEDataProcessor(
        tasks_dir=args.tasks_dir,
        images_dir=args.images_dir,
        require_image=not args.no_require_image,
        exclude_broken_baseline=args.exclude_broken_baseline,
        agent_ref_name=args.agent_ref_name,
        split=args.split,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )
    n = processor.write_jsonl(args.output_jsonl, limit=args.limit)
    print(f"[deepswe] wrote {n} rows to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
