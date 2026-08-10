#!/usr/bin/env python3
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

"""Fetch the pinned DeepSWE v1.1 tasks and materialize Gym input JSONL."""

import json
import subprocess
from pathlib import Path


REPOSITORY = "https://github.com/datacurve-ai/deep-swe.git"
COMMIT = "e016041a6ccf8da29906afc9a3f5a8df940a1f78"  # pragma: allowlist secret
EXPECTED_TASKS = 113
HERE = Path(__file__).resolve().parent
CHECKOUT = HERE / "data" / "deep-swe"
OUTPUT = HERE / "data" / "deep_swe_benchmark.jsonl"


def _run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def ensure_checkout() -> None:
    if not (CHECKOUT / ".git").exists():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", REPOSITORY, str(CHECKOUT))
    _run("git", "fetch", "origin", COMMIT, cwd=CHECKOUT)
    _run("git", "checkout", "--detach", COMMIT, cwd=CHECKOUT)
    actual = _run("git", "rev-parse", "HEAD", cwd=CHECKOUT)
    if actual != COMMIT:
        raise RuntimeError(f"DeepSWE checkout mismatch: expected {COMMIT}, got {actual}")


def task_names() -> list[str]:
    tasks_root = CHECKOUT / "tasks"
    required = (
        "task.toml",
        "instruction.md",
        "pre_artifacts.sh",
        "environment/Dockerfile",
        "tests/Dockerfile",
        "tests/test.sh",
        "tests/grader.py",
    )
    names = sorted(path.name for path in tasks_root.iterdir() if path.is_dir())
    invalid = [name for name in names if any(not (tasks_root / name / rel).exists() for rel in required)]
    if invalid:
        raise RuntimeError(f"DeepSWE tasks are incomplete: {invalid[:5]}")
    if len(names) != EXPECTED_TASKS:
        raise RuntimeError(f"Expected {EXPECTED_TASKS} DeepSWE tasks, found {len(names)}")
    return names


def materialize(names: list[str]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as output:
        for name in names:
            row = {
                "instance_id": f"deep_swe::{name}",
                "responses_create_params": {"input": []},
                "agent_ref": {"name": "deep_swe"},
            }
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    ensure_checkout()
    names = task_names()
    materialize(names)
    print(f"Prepared {len(names)} DeepSWE tasks at {OUTPUT}")


if __name__ == "__main__":
    main()
