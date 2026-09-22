# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from resources_servers.deepswe_external1.prepare_examples import describe_assets, materialize_task, task_row
from resources_servers.deepswe_external1.task_store import PreparedTask, PreparedTaskStore, TaskDefinition


def test_store_and_rows_are_content_bound(task: PreparedTask) -> None:
    store = PreparedTaskStore(task.task_dir.parent, expected_task_count=1)
    assert len(store) == 1 and store.task_ids == (task.definition.task_id,)
    loaded = store.get(task.definition.task_id)
    assert next(iter(store)).definition == task.definition
    row = task_row(loaded)
    assert row["responses_create_params"]["input"][0]["content"] == task.instruction
    assert row["task_fingerprint"] == task.definition.fingerprint()
    assert "solution" not in row and "tests" not in row
    changed = task.definition.model_copy(update={"verifier_image": "different-image"})
    assert changed.fingerprint() != task.definition.fingerprint()


@pytest.mark.parametrize("kind", ["bytes", "mode", "symlink", "missing"])
def test_assets_cannot_change_after_preparation(task: PreparedTask, kind: str) -> None:
    asset = task.task_dir / "tests/test.patch"
    if kind == "bytes":
        asset.write_bytes(b"changed")
    elif kind == "mode":
        asset.chmod(0o700)
    elif kind == "symlink":
        asset.unlink()
        asset.symlink_to(task.task_dir / "solution/solution.patch")
    else:
        asset.unlink()
    with pytest.raises(ValueError):
        task.validate_assets()


def test_preparation_refuses_overwrite_and_undeclared_paths(task: PreparedTask) -> None:
    again = materialize_task(task.task_dir, task.task_dir.parent, task.definition)
    assert again.definition == task.definition
    with pytest.raises(ValueError, match="overwrite"):
        materialize_task(task.task_dir, task.task_dir.parent, task.definition.model_copy(update={"image": "changed"}))
    with pytest.raises(ValueError, match="Undeclared"):
        task.asset_path("../outside")
    with pytest.raises(ValueError, match="exactly"):
        TaskDefinition.model_validate(task.definition.model_dump() | {"assets": {}})


def test_missing_root_wrong_count_and_metadata(task: PreparedTask, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PreparedTaskStore(tmp_path / "absent", expected_task_count=1)
    with pytest.raises(ValueError, match="Expected"):
        PreparedTaskStore(task.task_dir.parent, expected_task_count=2)
    metadata = task.task_dir / "task.json"
    definition = json.loads(metadata.read_text())
    definition["task_id"] = "different"
    metadata.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match="disagree"):
        PreparedTask(task.task_dir)


def test_native_grader_base_must_match(task: PreparedTask) -> None:
    config = task.task_dir / "tests/config.json"
    config.write_text(json.dumps({"base_commit": "f" * 40}))
    definition = task.definition.model_copy(update={"assets": describe_assets(task.task_dir)})
    (task.task_dir / "task.json").write_text(definition.model_dump_json())
    with pytest.raises(ValueError, match="native grader"):
        PreparedTask(task.task_dir)


def test_symlinked_source_directory_or_asset_is_rejected(task: PreparedTask, tmp_path: Path) -> None:
    source = tmp_path / "alias"
    source.symlink_to(task.task_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        describe_assets(source)
    with pytest.raises(ValueError, match="symlink"):
        PreparedTask(source)
    asset = task.task_dir / "tests/test.patch"
    asset.unlink()
    asset.symlink_to(task.task_dir / "solution/solution.patch")
    with pytest.raises(ValueError, match="non-symlink"):
        describe_assets(task.task_dir)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for the patch round-trip")
def test_real_git_patch_preserves_binary_deletion_symlink_and_executable_mode(
    task: PreparedTask, tmp_path: Path
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
    (repo / "link").symlink_to("tracked.txt")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").decode().strip()
    (repo / "tracked.txt").write_text("committed\n")
    (repo / "deleted.txt").unlink()
    (repo / "executable.sh").chmod(0o755)
    (repo / "binary.dat").write_bytes(b"\x00changed\xff")
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
    diff = command[command.index("git diff") :].split(" > ", 1)[0].replace(task.definition.base_commit, base)
    patch = subprocess.run(shlex.split(diff), cwd=repo, check=True, capture_output=True).stdout
    assert b"GIT binary patch" in patch and b"new mode 100755" in patch
    fresh = tmp_path / "B"
    git("clone", "--quiet", "--no-hardlinks", str(repo), str(fresh), cwd=tmp_path)
    git("checkout", "--quiet", "--detach", base, cwd=fresh)
    subprocess.run(["git", "apply", "--binary", "-"], cwd=fresh, input=patch, check=True, capture_output=True)
    assert (fresh / "tracked.txt").read_text() == "committed\n"
    assert (fresh / "binary.dat").read_bytes() == b"\x00changed\xff"
    assert not (fresh / "deleted.txt").exists()
    assert (fresh / "executable.sh").stat().st_mode & 0o111
    assert (fresh / "link").is_symlink() and (fresh / "link").readlink() == Path("new.txt")
    assert not (fresh / "untracked.txt").exists() and not (fresh / "ignored").exists()
