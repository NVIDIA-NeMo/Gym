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
Rewrite Harbor task.toml files from a build-and-push manifest.

Intended to run on a machine that does NOT need Docker.
Consumes the manifest produced by build_and_push_images.py and writes
[environment].docker_image values into task.toml files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


def update_task_toml_docker_image(task_toml_path: Path, image_ref: str) -> None:
    lines = task_toml_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_env = False
    updated = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_env = stripped == "[environment]"
        if in_env and stripped.startswith("docker_image"):
            prefix = line.split("docker_image", 1)[0]
            out.append(f'{prefix}docker_image = "{image_ref}"')
            updated = True
        else:
            out.append(line)

    if not updated:
        appended: list[str] = []
        in_env = False
        inserted = False
        for line in out:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if in_env and not inserted:
                    appended.append(f'docker_image = "{image_ref}"')
                    inserted = True
                in_env = stripped == "[environment]"
            appended.append(line)
        if in_env and not inserted:
            appended.append(f'docker_image = "{image_ref}"')
            inserted = True
        if not inserted:
            appended.extend(["", "[environment]", f'docker_image = "{image_ref}"'])
        out = appended

    task_toml_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def remap_path(path_str: str, remaps: list[tuple[str, str]]) -> str:
    """Apply path prefix remappings in order. First match wins."""
    for src, dst in remaps:
        if path_str.startswith(src):
            return dst + path_str[len(src) :]
    return path_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite task.toml docker_image values from manifest.")
    parser.add_argument("--manifest-in", type=Path, required=True, help="Manifest JSON from build_and_push_images.py.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned rewrites only.")
    parser.add_argument(
        "--path-remap",
        metavar="SRC:DST",
        action="append",
        default=[],
        help="Remap path prefixes in manifest (e.g. /home/user:/lustre/.../user). Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest_in.is_file():
        print(f"Error: manifest not found: {args.manifest_in}", file=sys.stderr)
        sys.exit(2)

    remaps: list[tuple[str, str]] = []
    for remap_str in args.path_remap:
        if ":" not in remap_str:
            print(f"Error: --path-remap must be SRC:DST, got: {remap_str}", file=sys.stderr)
            sys.exit(2)
        src, dst = remap_str.split(":", 1)
        remaps.append((src, dst))
        print(f"Path remap: {src} -> {dst}")

    manifest: dict[str, Any] = json.loads(args.manifest_in.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if not tasks:
        print("No tasks found in manifest.")
        return

    rewrites: list[tuple[str, Path, str, bool, Optional[str]]] = []
    for item in tasks:
        task_name = item.get("task_name")
        task_toml_path_raw = item.get("task_toml_path")
        if task_toml_path_raw and remaps:
            task_toml_path_raw = remap_path(task_toml_path_raw, remaps)
        task_toml_path = Path(task_toml_path_raw) if task_toml_path_raw else None
        docker_image = item.get("docker_image")
        success = bool(item.get("success"))
        error = item.get("error")
        if not task_name or not docker_image or task_toml_path is None:
            continue
        rewrites.append((task_name, task_toml_path, docker_image, success, error))

    if not rewrites:
        print("No valid task rewrite entries in manifest.")
        return

    failures = 0
    for task_name, task_toml_path, docker_image, build_success, build_error in rewrites:
        if not build_success:
            print(f"SKIP {task_name}: build/push failed in manifest ({build_error})")
            failures += 1
            continue
        if not task_toml_path.is_file():
            print(f"SKIP {task_name}: task.toml not found at {task_toml_path}")
            failures += 1
            continue

        if args.dry_run:
            print(f"PLAN {task_name}: {task_toml_path} -> {docker_image}")
            continue

        update_task_toml_docker_image(task_toml_path, docker_image)
        print(f"OK   {task_name}: {task_toml_path} -> {docker_image}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
