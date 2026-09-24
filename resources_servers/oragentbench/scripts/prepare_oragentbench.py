# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the ORAgentBench benchmark rows from a pinned upstream checkout.

Clones https://github.com/ORAgentBench/ORAgentBench at ``PINNED_COMMIT`` (or verifies an
existing checkout is at that commit), loads every ``harbor_tasks/<task>/`` through the same
parser the server uses, checks the corpus against the published manifest (107 tasks; 32 easy /
41 medium / 34 hard) and only then writes one Gym row per task. Optionally builds the task
container images: the shared base image from ``docker/Dockerfile`` next to this server and
one image per task from upstream's own ``environment/Dockerfile``.

Nothing is written when any task fails to load or the counts do not match; ``--limit`` is the
one supported way to prepare a deliberate subset (rows are written in sorted task order).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import DIFFICULTIES, load_task  # noqa: E402


REPO_URL = "https://github.com/ORAgentBench/ORAgentBench.git"
PINNED_COMMIT = "c9eb952435a4352f33daa2a35efe0f8c76d31b28"  # pragma: allowlist secret  (git commit, not a credential)
EXPECTED_COUNTS = {"easy": 32, "medium": 41, "hard": 34}
EXPECTED_TOTAL = 107

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_DIR.parents[1]
DEFAULT_CHECKOUT_DIR = SERVER_DIR / "data" / "ORAgentBench"
DEFAULT_OUTPUT = SERVER_DIR / "data" / "benchmark.jsonl"
BASE_DOCKERFILE = SERVER_DIR / "docker" / "Dockerfile"
BASE_IMAGE_TAG = "oragentbench-base:py311-scip"  # the tag every upstream task Dockerfile builds FROM
AGENT_NAME = "oragentbench_agent"


def _run(argv: List[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout


def ensure_checkout(checkout_dir: Path, commit: str = PINNED_COMMIT) -> Path:
    """Return a checkout of upstream at ``commit``, cloning or fetching as needed."""
    if not (checkout_dir / ".git").exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--quiet", REPO_URL, str(checkout_dir)])
    head = _run(["git", "rev-parse", "HEAD"], cwd=checkout_dir).strip()
    if head != commit:
        _run(["git", "fetch", "--quiet", "origin"], cwd=checkout_dir)
        _run(["git", "checkout", "--quiet", commit], cwd=checkout_dir)
        head = _run(["git", "rev-parse", "HEAD"], cwd=checkout_dir).strip()
    if head != commit:
        raise RuntimeError(f"checkout at {checkout_dir} is at {head}, expected {commit}")
    return checkout_dir


def load_difficulty(checkout_dir: Path) -> Dict[str, str]:
    payload = json.loads((checkout_dir / "difficulty.json").read_text())
    bands: Dict[str, str] = {}
    for task, info in payload["tasks"].items():
        band = str(info.get("band", "")).lower()
        if band not in DIFFICULTIES:
            raise ValueError(f"difficulty.json: task {task!r} has band {band!r}")
        bands[task] = band
    return bands


def image_tag(task_dir_name: str, commit: str = PINNED_COMMIT) -> str:
    return f"oragentbench/{task_dir_name.lower()}:{commit[:12]}"


def load_rows(checkout_dir: Path, task_folder_root: Optional[Path] = None) -> List[dict]:
    """Load every task; raise unless the corpus matches the published manifest exactly."""
    bands = load_difficulty(checkout_dir)
    tasks_dir = checkout_dir / "harbor_tasks"
    task_dirs = sorted(p for p in tasks_dir.iterdir() if p.is_dir())
    rows: List[dict] = []
    failures: List[str] = []
    for task_dir in task_dirs:
        try:
            task = load_task(task_dir)
            band = bands[task_dir.name]
        except Exception as exc:  # noqa: BLE001 - every failure is reported, then the run fails closed
            failures.append(f"{task_dir.name}: {type(exc).__name__}: {exc}")
            continue
        folder = task_dir
        if task_folder_root is not None:
            folder = Path(task_folder_root) / task_dir.name
        rows.append(
            {
                "responses_create_params": {"input": [{"role": "user", "content": task.steps[0].instruction}]},
                "task_name": task.name,
                "docker_image": image_tag(task_dir.name),
                "task_folder": str(folder),
                "difficulty": band,
                "num_steps": len(task.steps),
                "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
            }
        )
    if failures:
        raise RuntimeError(f"{len(failures)} task(s) failed to load:\n  " + "\n  ".join(failures))
    counts = Counter(row["difficulty"] for row in rows)
    if len(rows) != EXPECTED_TOTAL or dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"loaded {len(rows)} tasks with strata {dict(counts)}; expected {EXPECTED_TOTAL} with {EXPECTED_COUNTS}"
        )
    return rows


def build_images(checkout_dir: Path, task_dir_names: List[str], commit: str = PINNED_COMMIT) -> None:
    print(f"Building {BASE_IMAGE_TAG} from {BASE_DOCKERFILE}", file=sys.stderr)
    _run(["docker", "build", "--quiet", "-t", BASE_IMAGE_TAG, "-f", str(BASE_DOCKERFILE), str(BASE_DOCKERFILE.parent)])
    for index, name in enumerate(task_dir_names, start=1):
        tag = image_tag(name, commit)
        print(f"[{index}/{len(task_dir_names)}] docker build {tag}", file=sys.stderr)
        _run(["docker", "build", "--quiet", "-t", tag, str(checkout_dir / "harbor_tasks" / name / "environment")])


def corpus_digest(rows: List[dict]) -> str:
    """SHA-256 over the written rows, for the run record."""
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return number


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkout-dir", type=Path, default=DEFAULT_CHECKOUT_DIR)
    parser.add_argument("--limit", type=_positive_int, default=None, help="Write only the first N tasks (sorted).")
    parser.add_argument("--build-images", action="store_true", help="Build the base and per-task Docker images.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    checkout_dir = ensure_checkout(args.checkout_dir.resolve())
    try:
        task_folder_root = (checkout_dir / "harbor_tasks").relative_to(REPO_ROOT)
    except ValueError:
        task_folder_root = checkout_dir / "harbor_tasks"
    rows = load_rows(checkout_dir, task_folder_root)
    print(f"Loaded {len(rows)} tasks at {PINNED_COMMIT}; corpus digest sha256:{corpus_digest(rows)}", file=sys.stderr)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.build_images:
        build_images(checkout_dir, [Path(row["task_folder"]).name for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
