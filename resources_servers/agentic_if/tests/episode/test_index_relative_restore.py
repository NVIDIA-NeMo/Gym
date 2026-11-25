# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A prefix whose diffs were taken against the Git index restores when the row says so (``workspace_diff_base``)."""

import os
import shutil
import subprocess

import pytest
from pydantic import ValidationError

from resources_servers.agentic_if.episode.opencode_replay import restore_workspace
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import workspace_base
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec
from resources_servers.agentic_if.tests.support import LocalTransport, exported


# The prefix's shell calls: one changes a tracked file, the next stages a rename and a removal with git.
COMMANDS = ["printf draft > kept.txt", "git mv moved.txt renamed.txt && git rm -q removed.txt"]


def git(repo, *args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def index_relative_diff(repo, scratch) -> str:
    """The workspace diff as opencode_sandboxed_agent recorded it: plain `git diff` against a copy of the index, with
    untracked files added to the copy as intent-to-add."""
    copy = scratch / "index"
    shutil.copy(repo / ".git" / "index", copy)
    env = os.environ | {"GIT_INDEX_FILE": str(copy)}
    git(repo, "add", "-N", "--ignore-removal", ".", env=env)
    return git(repo, "diff", env=env)


def bash_part(index: int, command: str) -> dict:
    """A completed shell call that printed nothing when it was recorded."""
    state = {"status": "completed", "input": {"command": command}, "output": "", "metadata": {}, "time": {"end": 3}}
    return {"type": "tool", "tool": "bash", "callID": f"call-{index}", "state": state}


async def restore(tmp_path, **options):
    """Record the prefix in a repository, return that repository to its commit, and restore the prefix there."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "T"]):
        git(repo, *args)
    (repo / "moved.txt").write_text("line one\nline two\n")
    (repo / "removed.txt").write_text("gone\n")
    (repo / "kept.txt").write_text("kept\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    sandbox = BoundSandbox(LocalTransport(), user=None, workdir=str(repo), timeout_s=60)
    base = await workspace_base(sandbox)
    for command in COMMANDS:
        subprocess.run(["/bin/sh", "-c", command], cwd=repo, check=True)
    # Then the model's edits, which replay does not rerun: the renamed file, the changed one, and a new one.
    (repo / "renamed.txt").write_text("line one\nline two, edited\n")
    (repo / "kept.txt").write_text("kept\nfinal\n")
    (repo / "new.txt").write_text("new\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recorded = index_relative_diff(repo, scratch)
    git(repo, "reset", "-q", "--hard")
    git(repo, "clean", "-fdq")
    prefix = exported("root")
    prefix["info"]["directory"] = str(repo)
    prefix["messages"][1]["parts"] = [bash_part(index, command) for index, command in enumerate(COMMANDS)]
    continuation = ContinuationSpec(
        mode="restore",
        prefix_export=prefix,
        workspace_diff=recorded,
        source_task_sha256="a" * 64,
        source_tools_sha256="b" * 64,
        **options,
    )
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    receipt = await restore_workspace(sandbox, spec, continuation, base=base, staging=str(scratch))
    return repo, receipt


@pytest.mark.asyncio
async def test_an_index_relative_diff_does_not_apply_against_the_base_tree(tmp_path):
    # The diff records the renamed file as modified; against the base tree, that file does not exist.
    with pytest.raises(RuntimeError, match="recorded prefix workspace diff does not apply"):
        await restore(tmp_path)


@pytest.mark.asyncio
async def test_a_row_can_restore_against_the_index_the_replayed_commands_leave(tmp_path):
    repo, receipt = await restore(tmp_path, workspace_diff_base="index")
    assert receipt == {
        "mode": "restore",
        "replayed_commands": 2,
        "command_output_mismatch_indices": [],
        "workspace_matches_diff": True,
    }
    assert (repo / "renamed.txt").read_text() == "line one\nline two, edited\n"
    assert (repo / "kept.txt").read_text() == "kept\nfinal\n"
    assert (repo / "new.txt").read_text() == "new\n"
    assert not (repo / "moved.txt").exists() and not (repo / "removed.txt").exists()
    # What the prefix staged stays staged, as it was when the prefix was recorded.
    assert git(repo, "diff", "--cached", "--name-status", "-M").splitlines() == [
        "D\tremoved.txt",
        "R100\tmoved.txt\trenamed.txt",
    ]


def test_the_option_is_sealed_only_when_set():
    fields = {
        "mode": "restore",
        "prefix_export": exported("root"),
        "workspace_diff": "",
        "source_task_sha256": "a" * 64,
        "source_tools_sha256": "b" * 64,
    }
    assert "workspace_diff_base" not in ContinuationSpec(**fields).model_dump(mode="json")
    dumped = ContinuationSpec(**fields, workspace_diff_base="index").model_dump(mode="json")
    assert dumped["workspace_diff_base"] == "index"
    assert ContinuationSpec.model_validate(dumped).workspace_diff_base == "index"
    with pytest.raises(ValidationError):
        ContinuationSpec(**fields, workspace_diff_base="tree")
