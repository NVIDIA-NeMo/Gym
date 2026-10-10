# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tomllib
from pathlib import Path

from benchmarks.terminal_bench_2_1 import prepare_harbor_taskset as module
from nemo_gym.tasks.harbor import discover_tasks


def _fake_upstream(root: Path) -> tuple[str, str]:
    """A tiny repo shaped like terminal-bench-2-1: tasks/<name>/{task.toml,instruction.md,tests,solution,environment}."""
    repo = root / "upstream"
    for name, test_body in (("mteb-retrieve", "uv run -w mteb==1.36.8 pytest\n"), ("plain", "echo 1\n")):
        task = repo / "tasks" / name
        (task / "tests").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "environment").mkdir()
        (task / "task.toml").write_text('schema_version = "1.4"\n[environment]\ndocker_image = "img:1"\n')
        (task / "instruction.md").write_text("do it\n")
        (task / "tests" / "test.sh").write_text(test_body)
        (task / "solution" / "solve.sh").write_text("echo solved\n")
        (task / "environment" / "Dockerfile").write_text("FROM img:1\nRUN true\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "uploadpack.allowFilter", "true"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "init"], check=True
    )
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return repo.as_uri(), head.stdout.strip()


def test_prepare_applies_server_patches_and_records_them(tmp_path, monkeypatch):
    url, head = _fake_upstream(tmp_path)
    monkeypatch.setattr(module, "REPO_URL", url)

    folder = module.prepare(tmp_path / "datasets", commit=head)

    assert folder == tmp_path / "datasets" / "terminal-bench-2-1"
    tasks = {task.task_id: task for task in discover_tasks(folder)}
    assert set(tasks) == {"mteb-retrieve", "plain"}
    patched = (folder / "mteb-retrieve" / "tests" / "test.sh").read_text()
    assert "--index https://download.pytorch.org/whl/cpu" in patched
    assert (folder / "plain" / "tests" / "test.sh").read_text() == "echo 1\n"
    manifest = tomllib.loads((folder / "manifest.toml").read_text())
    assert manifest["dataset"]["git_commit_id"] == head
    assert manifest["patched"] == {"mteb-retrieve": ["tests/test.sh"]}
    # A second call keeps the existing folder.
    assert module.prepare(tmp_path / "datasets", commit=head) == folder


def test_patches_that_no_longer_match_are_skipped(tmp_path):
    path = tmp_path / "test.sh"
    path.write_text("echo unchanged\n")
    assert module._apply(path, [("missing text", "replacement")]) is False
    assert path.read_text() == "echo unchanged\n"
