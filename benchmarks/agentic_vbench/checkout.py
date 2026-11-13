# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned Agentic-VBench checkout: fetch it once and list its 100 benchmark tasks."""

import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPO_URL = "https://github.com/PhiloLabs/agentic-vbench.git"
REVISION = "410a75fe3dae37de4344fc9e5317da089505330a"
# The four leaderboard families and their task counts; the checkout's separate "understanding" area is not part
# of the benchmark.
BENCHMARK_FAMILIES = {"repair": 18, "assembly": 18, "sequencing": 28, "repurpose": 36}


def cache_root() -> Path:
    override = os.environ.get("AGENTIC_VBENCH_CACHE_ROOT")
    if override:
        return Path(override).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home.parent / "agentic_vbench"


def default_checkout_root() -> Path:
    return cache_root() / f"agentic-vbench-{REVISION[:12]}"


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialize concurrent fetches where the filesystem supports flock.

    Some network mounts refuse flock; then the staging-directory rename below is the only guard, and a
    concurrent loser simply redoes the work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        locked = True
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            locked = False
        try:
            yield
        finally:
            if locked:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ensure_checkout(root: Path | None = None, repo_url: str | None = None) -> Path:
    """Materialize the pinned revision at ``root`` (the cache root by default); reuse a matching checkout."""
    root = (root or default_checkout_root()).expanduser()
    repo_url = repo_url or os.environ.get("AGENTIC_VBENCH_REPO_URL") or REPO_URL
    with _locked(root.parent / f"{root.name}.lock"):
        if (root / ".git").exists():
            head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            if head != REVISION:
                raise ValueError(f"Existing checkout {root} is at {head}, expected {REVISION}")
            return root
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Checkout target is not empty: {root}")
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
        try:
            _git("init", "--quiet", cwd=staging)
            _git("remote", "add", "origin", repo_url, cwd=staging)
            try:
                _git("fetch", "--quiet", "--depth", "1", "origin", REVISION, cwd=staging)
            except subprocess.CalledProcessError:
                # Some servers refuse fetching an unadvertised commit; take the full history.
                _git("fetch", "--quiet", "origin", cwd=staging)
            _git("checkout", "--quiet", "--detach", REVISION, cwd=staging)
            if (root / ".git").exists():
                shutil.rmtree(staging)  # an unlocked concurrent caller won
            else:
                staging.replace(root)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return root


def inventory(root: Path) -> dict[str, dict]:
    """The 100 pinned tasks by task id: family, prompt digest and revision; raises on any drift from the pin."""
    root = root.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Agentic-VBench {REVISION}, found {revision}")
    changed = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "tasks"], text=True)
    if changed.strip():
        raise ValueError("Benchmark tasks differ from the pinned checkout")
    tasks: dict[str, dict] = {}
    family_dirs = (root / "tasks" / f"agentic_vbench_{family}" for family in BENCHMARK_FAMILIES)
    for path in sorted(task for family_dir in family_dirs for task in family_dir.glob("*/task.toml")):
        family = path.parent.parent.name.removeprefix("agentic_vbench_")
        task_id = path.parent.name
        if task_id in tasks:
            raise ValueError(f"Duplicate task id across families: {path}")
        prompt = (path.parent / "steps/solve/instruction.md").read_text()
        tasks[task_id] = {
            "task_id": task_id,
            "family": family,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "benchmark_revision": revision,
        }
    counts = Counter(task["family"] for task in tasks.values())
    if counts != BENCHMARK_FAMILIES:
        raise ValueError(f"Expected the complete 100-task inventory, got {dict(counts)}")
    return tasks
