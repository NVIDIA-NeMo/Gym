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
"""
Write a minimal setup.sh containing only the server dependency section.

This can write one file directly (--output) or populate every Harbor task under
--task-root with environment/files/setup.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SETUP_SH_CONTENT = """#!/bin/bash
# Ensure server deps (Python/uvicorn) for Harbor.
set -e
if ! python3 -c "import uvicorn, fastapi" 2>/dev/null; then
  echo "[harbor] Installing server dependencies (Python/uvicorn)..." >&2
  if python3 -m pip install uvicorn fastapi 2>/dev/null; then
    :
  elif python3 -m pip install --user uvicorn fastapi 2>/dev/null; then
    :
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq 2>/dev/null && apt-get install -y -qq python3-uvicorn python3-fastapi python3-pydantic 2>/dev/null || true
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache py3-uvicorn 2>/dev/null || true
  fi
  if ! python3 -c "import uvicorn, fastapi" 2>/dev/null && command -v pip3 >/dev/null 2>&1; then
    pip3 install --break-system-packages uvicorn fastapi 2>/dev/null || pip3 install uvicorn fastapi 2>/dev/null || true
  fi
fi
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a minimal setup.sh with uvicorn/fastapi bootstrap only.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--output",
        type=Path,
        help="Path to write setup.sh (e.g. /path/to/environment/files/setup.sh).",
    )
    mode.add_argument(
        "--task-root",
        type=Path,
        help="Root directory containing Harbor task folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned writes only.",
    )
    return parser.parse_args()


def is_task_dir(task_dir: Path) -> bool:
    return task_dir.is_dir() and (task_dir / "task.toml").is_file() and (task_dir / "environment").is_dir()


def find_tasks(root: Path) -> list[Path]:
    tasks: list[Path] = []
    seen: set[Path] = set()
    for sub in sorted(root.iterdir()):
        if is_task_dir(sub):
            resolved = sub.resolve()
            if resolved not in seen:
                seen.add(resolved)
                tasks.append(sub)
            continue
        if not sub.is_dir():
            continue
        for nested in sorted(sub.iterdir()):
            if is_task_dir(nested):
                resolved = nested.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    tasks.append(nested)
    return tasks


def write_setup_file(output_path: Path, force: bool, dry_run: bool) -> bool:
    if output_path.exists() and not force:
        print(f"SKIP {output_path} (exists; use --force to overwrite)")
        return False
    if dry_run:
        print(f"PLAN {output_path}")
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(SETUP_SH_CONTENT, encoding="utf-8")
    print(f"OK   {output_path}")
    return True


def main() -> None:
    args = parse_args()
    if args.output is not None:
        wrote = write_setup_file(args.output, force=args.force, dry_run=args.dry_run)
        if not wrote:
            sys.exit(1)
        return

    task_root = args.task_root
    if task_root is None or not task_root.is_dir():
        print(f"Error: task root not found: {task_root}", file=sys.stderr)
        sys.exit(2)

    tasks = find_tasks(task_root)
    if not tasks:
        print(f"No task directories found under: {task_root}")
        return

    print(f"Found {len(tasks)} task(s) under: {task_root}")
    wrote = 0
    skipped = 0
    for task_dir in tasks:
        output_path = task_dir / "environment" / "files" / "setup.sh"
        if write_setup_file(output_path, force=args.force, dry_run=args.dry_run):
            wrote += 1
        else:
            skipped += 1

    print(f"Done. wrote={wrote} skipped={skipped}")
    if skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
