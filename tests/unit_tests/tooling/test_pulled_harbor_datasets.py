# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Harbor dataset pulled into environments/harbor/<entry>/ stays out of the wheel and out of git."""

import os
import shutil
import subprocess
import tomllib
from fnmatch import fnmatchcase
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
ENTRY = "environments/harbor/terminal_bench_4"
# The committed entry's files: git tracks them and the wheel ships them as package data of environments.harbor.
ENTRY_FILES = {f"{ENTRY}/README.md", f"{ENTRY}/harbor_sandbox.yaml", f"{ENTRY}/dataset.toml"}
# What `gym dataset pull terminal_bench_4` writes beside them: the dataset's manifest, a dataset-level task data
# module and the task folders, whose own Python and YAML must stay out too; and the Harbor registry cache.
PULLED_FILES = {
    f"{ENTRY}/manifest.toml",
    f"{ENTRY}/task_data.py",
    f"{ENTRY}/hello_task/task.toml",
    f"{ENTRY}/hello_task/instruction.md",
    f"{ENTRY}/hello_task/environment/Dockerfile",
    f"{ENTRY}/hello_task/environment/docker-compose.yaml",
    f"{ENTRY}/hello_task/tests/test_outputs.py",
    f"{ENTRY}/hello_task/solution/solve.sh",
    "environments/harbor/.harbor/registry.json",
}


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A checkout whose committed entry folder holds a pulled dataset."""
    for relative in (*ENTRY_FILES, *PULLED_FILES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return tmp_path


def test_package_discovery_leaves_out_a_pulled_harbor_dataset(checkout: Path) -> None:
    find = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]["find"]
    # setuptools reads this table as namespace discovery: every folder whose name has no "." is a package when its
    # dotted path matches an include pattern and no exclude pattern (fnmatch). setuptools is not installed in the
    # core test environment, so the test applies those rules instead of calling setuptools.find_namespace_packages.
    assert find.get("namespaces", True)
    folders = [path.relative_to(checkout).parts for path in checkout.rglob("*") if path.is_dir()]
    candidates = {".".join(parts) for parts in folders if not any("." in part for part in parts)}
    packages = {
        name
        for name in candidates
        if any(fnmatchcase(name, pattern) for pattern in find["include"])
        and not any(fnmatchcase(name, pattern) for pattern in find["exclude"])
    }
    # environments.harbor stays a package, so the committed entry files still ship as its package data.
    assert packages == {"environments", "environments.harbor"}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_gitignore_ignores_a_pulled_harbor_dataset_but_not_its_entry_files(checkout: Path) -> None:
    shutil.copyfile(ROOT / ".gitignore", checkout / ".gitignore")
    # Only the repository's .gitignore decides: no repository named by GIT_DIR (as in a git hook), no user-wide
    # ignore file.
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    git = ["git", "-C", str(checkout), "-c", f"core.excludesFile={os.devnull}"]
    subprocess.run([*git, "init", "--quiet"], check=True, env=env)
    status = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=all", "--ignored"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.splitlines()
    untracked = {line[3:] for line in status if line.startswith("?? ")}
    ignored = {line[3:] for line in status if line.startswith("!! ")}
    assert untracked == {".gitignore", *ENTRY_FILES}
    assert ignored == PULLED_FILES
