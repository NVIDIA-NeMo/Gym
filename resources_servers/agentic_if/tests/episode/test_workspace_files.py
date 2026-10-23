# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial actor files are validated, bound to receipts, and staged as the task user."""

import os
import subprocess

import pytest
from pydantic import ValidationError

from resources_servers.agentic_if.episode.opencode_replay import restore_workspace
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import stage_workspace_files
from resources_servers.agentic_if.task_variants.runtime import (
    ContinuationSpec,
    RuntimeSpec,
    seal_runtime_row,
    validate_runtime,
)
from resources_servers.agentic_if.tests.support import LocalTransport, exported


def spec(files):
    return RuntimeSpec(harness="opencode", task_backend="swe", workspace_files=files)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/",
        "//tmp/x",
        "/tmp/../x",
        "/tmp/.git/config",
        "../x",
        "a/../x",
        "./x",
        "a//x",
        "a/",
        ".git/config",
        "a/.git",
        "a\\b",
        "a\0b",
    ],
)
def test_rejects_unsafe_workspace_paths(path):
    with pytest.raises(ValidationError):
        spec({path: "text"})


def test_rejects_file_directory_collision():
    with pytest.raises(ValidationError):
        spec({"a": "file", "a/b": "child"})


@pytest.mark.asyncio
@pytest.mark.parametrize("child", [False, True])
async def test_absolute_relative_collisions_are_rejected_before_any_write(tmp_path, child):
    target = tmp_path / "state"
    files = {"state": "first", str(target / "nested" if child else target): "second"}
    sandbox = BoundSandbox(LocalTransport(), user=os.getuid(), workdir=str(tmp_path), timeout_s=60)
    with pytest.raises(RuntimeError, match="workspace"):
        await stage_workspace_files(sandbox, files)
    assert not target.exists()


@pytest.mark.parametrize(
    "files",
    [
        {".env": "秘密\n", "empty": ""},
        {"/etc/example/.env": "literal state\n", "/app/.agents/skills/demo/SKILL.md": "skill"},
    ],
    ids=["relative", "absolute"],
)
def test_workspace_contents_stay_literal_and_bound_to_configuration_receipt(files):
    path, value = next(iter(files.items()))
    row = seal_runtime_row(
        {"instance_id": "task", "responses_create_params": {"input": [{"role": "user", "content": "Read files"}]}},
        spec(files),
    )
    assert validate_runtime(row).workspace_files[path] == value
    row["agentic_if"]["workspace_files"][path] += "changed"
    with pytest.raises(ValueError, match="receipt"):
        validate_runtime(row)


@pytest.mark.asyncio
@pytest.mark.parametrize("default_cwd", [False, True])
async def test_staging_preserves_exact_bytes_and_identity(tmp_path, monkeypatch, default_cwd):
    monkeypatch.chdir(tmp_path)
    transport = LocalTransport()
    sandbox = BoundSandbox(transport, user=os.getuid(), workdir=None if default_cwd else str(tmp_path), timeout_s=60)
    files = {
        ".env": "秘密 λ\n",
        "AGENTS.md": "literal ${tool}\n",
        ".agents/skills/demo/SKILL.md": "skill",
        "memory/a.md": "",
        "docs/a.rst": "rst\n",
        # An absolute file is staged at its declared path, even outside the working directory.
        str(tmp_path.parent / (tmp_path.name + "-outside") / "config"): "exact λ\n",
        # Larger than Linux's 128 KiB limit on one command-line argument.
        "large.txt": "x" * (256 * 1024),
    }
    await stage_workspace_files(sandbox, files)
    assert all((tmp_path / path).read_bytes() == text.encode("utf-8") for path, text in files.items())
    assert all((tmp_path / path).stat().st_uid == os.getuid() for path in files)
    # One write, as the task user, that reads the uploaded files rather than taking them on its command line.
    ((command, user),) = [call for call in transport.calls if call[0].startswith("python3 ")]
    assert user == os.getuid() and "literal ${tool}" not in command
    # Only the private staging directory's chmod and removal run as root, and the directory is gone afterwards.
    assert {user for command, user in transport.calls if command.startswith(("chmod ", "rm "))} == {"root"}
    (staging,) = {command.split()[-1] for command, _ in transport.calls if command.startswith("umask 077; mkdir -p ")}
    assert staging.startswith("/tmp/nemo-gym-agentic-if-workspace-") and not os.path.exists(staging)


@pytest.mark.asyncio
@pytest.mark.parametrize("leaf", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
async def test_staging_rejects_existing_symlink(tmp_path, leaf, absolute):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (root / "link").symlink_to(target / "file" if leaf else target)
    sandbox = BoundSandbox(LocalTransport(), user=os.getuid(), workdir=str(root), timeout_s=60)
    path = "link" if leaf else "link/file"
    if absolute:
        path = str(root / path)
    with pytest.raises(RuntimeError, match="workspace"):
        await stage_workspace_files(sandbox, {path: "must not escape"})
    assert not (target / "file").exists()


def repository(path, files):
    """A new Git repository at `path` whose base commit holds `files`; returns its git runner."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    for name, content in files.items():
        (path / name).write_text(content)
    git("add", ".")
    git("commit", "-qm", "base")
    return git


async def restore(tmp_path, sandbox, files, *, workspace_diff="", staging=None):
    """Restore a recorded prefix and its workspace diff over the staged `files`; returns the receipt."""

    prefix = exported("root")
    prefix["info"]["directory"] = str(tmp_path)
    continuation = ContinuationSpec(
        mode="restore",
        prefix_export=prefix,
        workspace_diff=workspace_diff,
        source_task_sha256="a" * 64,
        source_tools_sha256="b" * 64,
    )
    return await restore_workspace(sandbox, spec(files), continuation, staging=str(staging or tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
async def test_restore_retains_initial_files(tmp_path, absolute):
    repository(tmp_path, {"tracked": "original"})
    sandbox = BoundSandbox(LocalTransport(), user=os.getuid(), workdir=str(tmp_path), timeout_s=60)
    files = {"tracked": "initial override", "memory/state.md": "initial memory\n"}
    if absolute:
        files = {str(tmp_path / path): content for path, content in files.items()}
    await stage_workspace_files(sandbox, files)
    await restore(tmp_path, sandbox, files)
    assert {path: (tmp_path / path).read_text() for path in files} == files


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
async def test_restore_applies_recorded_changes_to_seed_files(tmp_path, absolute):
    git = repository(tmp_path, {".gitignore": ".env\n", "tracked": "base", "deleted": "base"})
    files = {
        "tracked": "seed",
        "deleted": "seed",
        "added": "seed",
        ".env": "secret\n",
        ".agents/skills/demo/SKILL.md": "skill\n",
    }
    if absolute:
        files = {str(tmp_path / path): content for path, content in files.items()}
    sandbox = BoundSandbox(LocalTransport(), user=os.getuid(), workdir=str(tmp_path), timeout_s=60)
    await stage_workspace_files(sandbox, files)
    (tmp_path / "tracked").write_text("prefix tracked")
    (tmp_path / "added").write_text("prefix added")
    (tmp_path / "deleted").unlink()
    git("add", "-N", "--ignore-removal", ".")
    diff = git("diff", "--binary", "--full-index")
    git("reset", "-q")
    git("checkout", "--", ".")
    (tmp_path / "added").unlink()
    await stage_workspace_files(sandbox, files)
    staging = tmp_path.parent / (tmp_path.name + "-staging")
    staging.mkdir()
    receipt = await restore(tmp_path, sandbox, files, workspace_diff=diff, staging=staging)
    assert (tmp_path / "tracked").read_text() == "prefix tracked"
    assert (tmp_path / "added").read_text() == "prefix added"
    assert not (tmp_path / "deleted").exists()
    assert (tmp_path / ".env").read_text() == "secret\n"
    assert (tmp_path / ".agents/skills/demo/SKILL.md").read_text() == "skill\n"
    assert receipt["workspace_matches_diff"] is True
