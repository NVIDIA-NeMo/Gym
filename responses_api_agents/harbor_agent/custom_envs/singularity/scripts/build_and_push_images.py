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
Build and push Docker images from Harbor task Dockerfiles.

Intended to run on a machine that has Docker + registry access.
This script writes a JSON manifest that can be consumed by
rewrite_task_tomls.py on a different machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskInfo:
    task_name: str
    task_dir: Path
    environment_dir: Path
    dockerfile_path: Path
    task_toml_path: Path

    @property
    def image_name(self) -> str:
        safe = self.task_name.replace("_", "-").lower()
        safe = "".join(ch for ch in safe if ch.isalnum() or ch in "-._")
        return f"hb__{safe}"[:128]


@dataclass
class BuildResult:
    task_name: str
    success: bool
    local_tag: str
    remote_ref: str
    task_toml_path: str
    error: str | None = None


def run_cmd(cmd: list[str], timeout_sec: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout_sec}s: {' '.join(cmd)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Command error: {exc}"
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return False, err[:2000]
    return True, ""


def find_tasks(input_dir: Path) -> list[TaskInfo]:
    tasks: list[TaskInfo] = []
    seen: set[str] = set()

    def try_add(task_dir: Path) -> bool:
        if not task_dir.is_dir():
            return False
        env_dir = task_dir / "environment"
        task_toml = task_dir / "task.toml"
        if not env_dir.is_dir() or not task_toml.is_file():
            return False
        dockerfile = env_dir / "Dockerfile"
        if not dockerfile.is_file():
            return False

        task_name = task_dir.name
        if task_name in seen:
            return False
        seen.add(task_name)

        tasks.append(
            TaskInfo(
                task_name=task_name,
                task_dir=task_dir,
                environment_dir=env_dir,
                dockerfile_path=dockerfile,
                task_toml_path=task_toml,
            )
        )
        return True

    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        if try_add(sub):
            continue
        for nested in sorted(sub.iterdir()):
            try_add(nested)

    return tasks


def build_and_push_task(task: TaskInfo, registry: str, image_tag: str, timeout_sec: int) -> BuildResult:
    local_tag = f"{task.image_name}:{image_tag}"
    remote_ref = f"{registry}/{task.image_name}:{image_tag}"

    ok, err = run_cmd(
        [
            "docker",
            "build",
            "-t",
            local_tag,
            "-f",
            str(task.dockerfile_path),
            str(task.environment_dir),
        ],
        timeout_sec=timeout_sec,
    )
    if not ok:
        return BuildResult(
            task.task_name, False, local_tag, remote_ref, str(task.task_toml_path), f"Build failed: {err}"
        )

    ok, err = run_cmd(["docker", "tag", local_tag, remote_ref], timeout_sec=timeout_sec)
    if not ok:
        return BuildResult(
            task.task_name, False, local_tag, remote_ref, str(task.task_toml_path), f"Tag failed: {err}"
        )

    ok, err = run_cmd(["docker", "push", remote_ref], timeout_sec=timeout_sec)
    if not ok:
        return BuildResult(
            task.task_name, False, local_tag, remote_ref, str(task.task_toml_path), f"Push failed: {err}"
        )

    return BuildResult(task.task_name, True, local_tag, remote_ref, str(task.task_toml_path))


def build_and_push_shared(
    source_task: TaskInfo,
    all_tasks: list[TaskInfo],
    image_stem: str,
    registry: str,
    image_tag: str,
    timeout_sec: int,
) -> list[BuildResult]:
    safe = image_stem.replace("_", "-").lower()
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in "-._")
    safe = safe[:128] or "shared-image"
    local_tag = f"hb__{safe}:{image_tag}"
    remote_ref = f"{registry}/hb__{safe}:{image_tag}"

    ok, err = run_cmd(
        [
            "docker",
            "build",
            "-t",
            local_tag,
            "-f",
            str(source_task.dockerfile_path),
            str(source_task.environment_dir),
        ],
        timeout_sec=timeout_sec,
    )
    if ok:
        ok, err = run_cmd(["docker", "tag", local_tag, remote_ref], timeout_sec=timeout_sec)
    if ok:
        ok, err = run_cmd(["docker", "push", remote_ref], timeout_sec=timeout_sec)

    results: list[BuildResult] = []
    for task in all_tasks:
        if ok:
            results.append(BuildResult(task.task_name, True, local_tag, remote_ref, str(task.task_toml_path)))
        else:
            results.append(
                BuildResult(
                    task.task_name,
                    False,
                    local_tag,
                    remote_ref,
                    str(task.task_toml_path),
                    f"Shared image build/push failed: {err}",
                )
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push Harbor task images, then emit a manifest.")
    parser.add_argument("--input", type=Path, required=True, help="Task root directory.")
    parser.add_argument("--registry", type=str, required=True, help="Registry prefix, e.g. ghcr.io/org/harbor-images.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output JSON manifest path.")
    parser.add_argument("--image-tag", type=str, default="latest", help="Image tag (default: latest).")
    parser.add_argument("--task-name", type=str, default=None, help="Only process one task.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Process first N tasks.")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers (per-task mode only).")
    parser.add_argument("--timeout-sec", type=int, default=1800, help="Timeout per docker command.")
    parser.add_argument(
        "--shared-image-subfolder",
        type=str,
        default=None,
        help="Build one shared image named after this subfolder and reuse for all tasks in it.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        print(f"Error: input directory not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    target_root = args.input
    if args.shared_image_subfolder:
        target_root = args.input / args.shared_image_subfolder
        if not target_root.is_dir():
            print(f"Error: shared-image subfolder not found: {target_root}", file=sys.stderr)
            sys.exit(2)

    tasks = find_tasks(target_root)
    if args.task_name:
        tasks = [t for t in tasks if t.task_name == args.task_name]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    if not tasks:
        print("No tasks with environment/Dockerfile + task.toml found.")
        return

    print(f"Found {len(tasks)} task(s).")
    if args.dry_run:
        if args.shared_image_subfolder:
            shared_ref = (
                f"{args.registry}/hb__{args.shared_image_subfolder.replace('_', '-').lower()}:{args.image_tag}"
            )
            print(f"Shared mode image ref: {shared_ref}")
        for t in tasks:
            ref = (
                f"{args.registry}/hb__{args.shared_image_subfolder.replace('_', '-').lower()}:{args.image_tag}"
                if args.shared_image_subfolder
                else f"{args.registry}/{t.image_name}:{args.image_tag}"
            )
            print(f"- {t.task_name}")
            print(f"    Dockerfile: {t.dockerfile_path}")
            print(f"    Context:    {t.environment_dir}")
            print(f"    task.toml:  {t.task_toml_path}")
            print(f"    image_ref:  {ref}")
        return

    results: list[BuildResult] = []
    if args.shared_image_subfolder:
        source_task = sorted(tasks, key=lambda x: x.task_name)[0]
        results = build_and_push_shared(
            source_task=source_task,
            all_tasks=tasks,
            image_stem=args.shared_image_subfolder,
            registry=args.registry,
            image_tag=args.image_tag,
            timeout_sec=args.timeout_sec,
        )
    else:
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(build_and_push_task, task, args.registry, args.image_tag, args.timeout_sec): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:  # noqa: BLE001
                        results.append(
                            BuildResult(
                                task_name=task.task_name,
                                success=False,
                                local_tag=f"{task.image_name}:{args.image_tag}",
                                remote_ref=f"{args.registry}/{task.image_name}:{args.image_tag}",
                                task_toml_path=str(task.task_toml_path),
                                error=str(exc),
                            )
                        )
        else:
            for task in tasks:
                results.append(build_and_push_task(task, args.registry, args.image_tag, args.timeout_sec))

    for r in sorted(results, key=lambda x: x.task_name):
        if r.success:
            print(f"OK   {r.task_name} -> {r.remote_ref}")
        else:
            print(f"FAIL {r.task_name}: {r.error}")

    manifest: dict[str, Any] = {
        "input": str(args.input),
        "target_root": str(target_root),
        "registry": args.registry,
        "image_tag": args.image_tag,
        "shared_image_subfolder": args.shared_image_subfolder,
        "total": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "tasks": [
            {
                "task_name": r.task_name,
                "task_toml_path": r.task_toml_path,
                "docker_image": r.remote_ref,
                "success": r.success,
                "error": r.error,
            }
            for r in sorted(results, key=lambda x: x.task_name)
        ],
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {args.manifest_out}")

    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
