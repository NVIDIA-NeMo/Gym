#!/usr/bin/env python3
"""Fetch the pinned DeepSWE v1.1 tasks and materialize Gym input JSONL."""

import json
import subprocess
from pathlib import Path


REPOSITORY = "https://github.com/datacurve-ai/deep-swe.git"
COMMIT = "e016041a6ccf8da29906afc9a3f5a8df940a1f78"
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
