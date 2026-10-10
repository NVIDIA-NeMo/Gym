# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize Terminal Bench 2.1 as a Harbor-shaped taskset in Gym's datasets folder.

The upstream tasks run as-is, except for the handful of `tests/test.sh` and
`solution/solve.sh` fixes the `terminal_bench_2_1` resources server applies at run time
(package pins that rotted upstream). Here those fixes become files, so the `harbor`
resources server needs no task-specific code. The pinned commit and the patched
tasks are recorded in `manifest.toml`.

    python benchmarks/terminal_bench_2_1/prepare_harbor_taskset.py [--datasets-dir DIR]
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from nemo_gym.tasks.harbor.hub import MANIFEST_FILE, datasets_dir
from resources_servers.terminal_bench_2_1.app import GOLDEN_PATCH_SOLVE_SH_PATCHES, TEST_SH_PATCHES


REPO_URL = "https://github.com/harbor-framework/terminal-bench-2-1"
# Upstream commit the `terminal_bench_2_1` server was baselined against.
PINNED_COMMIT = "7131e4375048a0e408a8fb404b5f499d726b695b"
TASKSET = "terminal-bench-2-1"


def _apply(path: Path, patches: list[tuple[str, str]]) -> bool:
    """Apply the server's string patches; ones upstream has since fixed no longer match and are skipped."""
    text = path.read_text()
    patched = text
    for old, new in patches:
        patched = patched.replace(old, new)
    if patched != text:
        path.write_text(patched)
        return True
    return False


def prepare(root: Path | None = None, *, commit: str = PINNED_COMMIT) -> Path:
    root = Path(root) if root is not None else datasets_dir()
    folder = root / TASKSET
    if folder.exists():
        return folder
    root.mkdir(parents=True, exist_ok=True)
    patched: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="tb21-", dir=root) as tmp:
        clone = Path(tmp) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO_URL, str(clone)], check=True
        )
        subprocess.run(["git", "checkout", "--quiet", commit], cwd=clone, check=True)
        staged = Path(tmp) / TASKSET
        staged.mkdir()
        for task_dir in sorted((clone / "tasks").iterdir()):
            if not (task_dir / "task.toml").is_file():
                continue
            shutil.copytree(task_dir, staged / task_dir.name, symlinks=False)
            name = f"terminal-bench/{task_dir.name}"
            files = []
            if name in TEST_SH_PATCHES and _apply(staged / task_dir.name / "tests" / "test.sh", TEST_SH_PATCHES[name]):
                files.append("tests/test.sh")
            if name in GOLDEN_PATCH_SOLVE_SH_PATCHES and _apply(
                staged / task_dir.name / "solution" / "solve.sh", GOLDEN_PATCH_SOLVE_SH_PATCHES[name]
            ):
                files.append("solution/solve.sh")
            if files:
                patched[task_dir.name] = files
        lines = [
            "[dataset]",
            f"name = {json.dumps(TASKSET)}",
            'version = "2.1"',
            'source = "git"',
            f"git_url = {json.dumps(REPO_URL)}",
            f"git_commit_id = {json.dumps(commit)}",
            "",
            "# Files changed from upstream so the tasks build and verify today.",
            "[patched]",
        ]
        lines += [f"{json.dumps(task)} = {json.dumps(files)}" for task, files in sorted(patched.items())]
        (staged / MANIFEST_FILE).write_text("\n".join(lines) + "\n")
        shutil.move(str(staged), str(folder))
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=Path, default=None)
    args = parser.parse_args()
    print(prepare(args.datasets_dir))
