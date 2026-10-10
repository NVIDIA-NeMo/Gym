# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.deepswe_external1.inline_task import InlineTask
from resources_servers.deepswe_external1.task_data import TaskData, TaskFiles
from resources_servers.deepswe_external1.tests.test_app import make_server


def test_row_is_self_contained_and_preserves_file_text(task: InlineTask) -> None:
    row = task.data.model_dump()
    restored = InlineTask(TaskData.model_validate(row))
    assert restored.config == task.config
    assert restored.data.files == task.data.files
    assert not hasattr(restored, "task_dir")
    assert not hasattr(restored, "paths")
    verifier = restored.data.files.verification_files(b"candidate patch\n")
    assert set(verifier) == {
        "/tests/test.sh",
        "/tests/test.patch",
        "/tests/grader.py",
        "/tests/config.json",
        "/logs/artifacts/model.patch.b64",
    }
    assert base64.b64decode(verifier["/logs/artifacts/model.patch.b64"], validate=True) == b"candidate patch\n"
    assert restored.data.files.solution_files() == {
        "/solution/solve.sh": task.data.files.solve_script,
        "/solution/solution.patch": task.data.files.solution_patch,
    }


@pytest.mark.parametrize("field,value", [("task_id", "../escape"), ("base_commit", "not-a-commit"), ("workdir", "/")])
def test_required_identifiers_are_validated(task: InlineTask, field: str, value: str) -> None:
    with pytest.raises(ValueError):
        TaskData.model_validate(task.data.model_dump() | {field: value})


def test_file_text_is_not_silently_normalized(task: InlineTask) -> None:
    files = TaskFiles.model_validate(task.data.files.model_dump() | {"test_patch": "é\r\n"})
    assert files.verification_files(b"")["/tests/test.patch"].encode() == b"\xc3\xa9\r\n"
    assert base64.b64decode(files.verification_files(b"\xff")["/logs/artifacts/model.patch.b64"]) == b"\xff"


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for the patch round-trip")
@pytest.mark.asyncio
@pytest.mark.parametrize("shadow_source", [None, "cwd", "pythonpath"])
async def test_real_git_patch_preserves_binary_deletion_symlink_and_executable_mode(
    task: InlineTask, tmp_path: Path, shadow_source: str | None
) -> None:
    repo = tmp_path / "A"
    repo.mkdir()

    def git(*args: str, cwd: Path = repo) -> bytes:
        return subprocess.run(
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
        ).stdout

    git("init", "-q")
    (repo / ".gitignore").write_text("ignored/\n")
    (repo / "tracked.txt").write_text("base\n")
    (repo / "deleted.txt").write_text("remove me\n")
    (repo / "executable.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "binary.dat").write_bytes(b"\x00base\xff")
    (repo / "latin1.txt").write_bytes(b"caf\xe9\n")
    (repo / "link").symlink_to("tracked.txt")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").decode().strip()
    (repo / "tracked.txt").write_text("committed\n")
    (repo / "deleted.txt").unlink()
    (repo / "executable.sh").chmod(0o755)
    (repo / "binary.dat").write_bytes(b"\x00changed\xff")
    (repo / "latin1.txt").write_bytes(b"caf\xe9 au lait\n")
    (repo / "new.txt").write_text("new\n")
    (repo / "link").unlink()
    (repo / "link").symlink_to("new.txt")
    git("add", "-A")
    git("commit", "-qm", "solution")
    (repo / "tracked.txt").write_text("uncommitted is not transferred\n")
    (repo / "untracked.txt").write_text("not submitted\n")
    (repo / "ignored").mkdir()
    (repo / "ignored/cache").write_text("cache\n")
    # Exercise the exact diff argv selected by the resource-server collector, with
    # a temporary repository instead of a privileged /app mount on the test host.
    command = task.config.verifier.collect[0].command
    assert "--no-ext-diff --no-textconv --no-color" in command and command.endswith(
        "HEAD -- . > /logs/artifacts/model.patch"
    )
    diff = command[command.index("git diff") :].split(" > ", 1)[0].replace(task.data.base_commit, base)
    patch = subprocess.run(shlex.split(diff), cwd=repo, check=True, capture_output=True).stdout
    assert b"GIT binary patch" in patch and b"new mode 100755" in patch
    with pytest.raises(UnicodeDecodeError):
        patch.decode("utf-8")
    # Exercise inline provisioning and the real staging command, not just an
    # independent base64 round trip that could miss a broken consumer.
    staged = tmp_path / "staged"
    for path, content in task.data.files.verification_files(patch).items():
        target = staged / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    environment = os.environ.copy()
    if shadow_source is not None:
        shadow_dir = repo if shadow_source == "cwd" else tmp_path / "pythonpath"
        shadow_dir.mkdir(exist_ok=True)
        for name in ("base64", "pathlib"):
            (shadow_dir / f"{name}.py").write_text('raise RuntimeError("task-local module imported")\n')
        if shadow_source == "pythonpath":
            environment["PYTHONPATH"] = str(shadow_dir)

    async def execute(command, *, timeout_s):
        command = command.replace("/logs/artifacts", shlex.quote(str(staged / "logs/artifacts"))).replace(
            "/tests", shlex.quote(str(staged / "tests"))
        )
        completed = subprocess.run(
            ["sh", "-c", command], cwd=repo, env=environment, capture_output=True, text=True, timeout=timeout_s
        )
        return SimpleNamespace(return_code=completed.returncode, stderr=completed.stderr)

    await make_server(task)._stage_verifier(AsyncMock(exec=AsyncMock(side_effect=execute)), task, patch)
    transported = (staged / "logs/artifacts/model.patch").read_bytes()
    assert transported == patch
    fresh = tmp_path / "B"
    git("clone", "--quiet", "--no-hardlinks", str(repo), str(fresh), cwd=tmp_path)
    git("checkout", "--quiet", "--detach", base, cwd=fresh)
    subprocess.run(["git", "apply", "--binary", "-"], cwd=fresh, input=transported, check=True, capture_output=True)
    assert (fresh / "tracked.txt").read_text() == "committed\n"
    assert (fresh / "binary.dat").read_bytes() == b"\x00changed\xff"
    assert (fresh / "latin1.txt").read_bytes() == b"caf\xe9 au lait\n"
    assert not (fresh / "deleted.txt").exists()
    assert (fresh / "executable.sh").stat().st_mode & 0o111
    assert (fresh / "link").is_symlink() and (fresh / "link").readlink() == Path("new.txt")
    assert not (fresh / "untracked.txt").exists() and not (fresh / "ignored").exists()
