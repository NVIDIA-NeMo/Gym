# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A prefix whose diffs were taken against the Git index (``workspace_diff_base``): its restore, and the continued
session's changes, told against the workspace seed restored rather than against the prefix's last diff (which leaves out
what the prefix staged)."""

import os
import shutil
import subprocess

import pytest

from resources_servers.agentic_if.episode.opencode_replay import restore_prefix_workspace, restore_workspace
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.session_evidence import with_seed_facts
from resources_servers.agentic_if.episode.workspace import snapshot_workspace, workspace_base
from resources_servers.agentic_if.if_grading import engine
from resources_servers.agentic_if.if_grading.checkers.run import diff_files, evidence_steps
from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence
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


async def repository(tmp_path):
    """The task repository at its commit, its sandbox and its base tree."""
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
    return repo, sandbox, await workspace_base(sandbox)


async def restore(tmp_path, **options):
    """Record the prefix in a repository, return that repository to its commit, and restore the prefix there."""
    repo, sandbox, base = await repository(tmp_path)
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


# A prefix that renames and removes tracked files with git: a shell call changes a tracked file, the next stages a
# rename and a removal, then the model edits the renamed file and writes a new one. Each entry: call ID, tool, input,
# file it writes.
PREFIX = [
    ("p0", "bash", {"command": "printf draft > kept.txt"}, None),
    ("p1", "bash", {"command": "git mv moved.txt renamed.txt && git rm -q removed.txt"}, None),
    ("p2", "edit", {"filePath": "renamed.txt"}, ("renamed.txt", "line one\nline two, edited\n")),
    ("p3", "write", {"filePath": "new.txt"}, ("new.txt", "new\n")),
]


def tool_part(ident, tool, args, diff):
    """A completed OpenCode tool part stamped with the workspace diff after it."""
    state = {
        "status": "completed",
        "input": args,
        "output": "",
        "metadata": {"nemo_gym_profile": {"workspace_diff": diff}},
    }
    return {"type": "tool", "callID": ident, "tool": tool, "state": state | {"time": {"end": 3}}}


def tool_call(ident, tool, args):
    return {
        "call_id": ident,
        "invocation_id": "root",
        "tool": tool,
        "args": args,
        "model_tool": tool,
        "model_args": args,
    }


def steps_of(calls):
    """Each (call, diff) as an assistant step and its result."""
    out = []
    for call, diff in calls:
        out.append({"role": "assistant", "calls": [call]})
        out.append(
            {
                "role": "tool",
                "tool": call["tool"],
                "call_id": call["call_id"],
                "invocation_id": "root",
                "workspace_diff": diff,
            }
        )
    return out


async def continued_session(tmp_path):
    """Record the prefix in a repository, return the repository to its commit and seed the episode as `_prepare` does
    (initial snapshot, restore, snapshot of the restored workspace). The continued session then runs a command that
    changes nothing and edits kept.txt. Returns the evidence, the prefix's last diff and the seed."""
    repo, sandbox, base = await repository(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    prefix, recorded = exported("root"), []
    for ident, tool, args, written in PREFIX:
        if written is None:
            subprocess.run(["/bin/sh", "-c", args["command"]], cwd=repo, check=True)
        else:
            (repo / written[0]).write_text(written[1])
        recorded.append((tool_call(ident, tool, args), index_relative_diff(repo, scratch)))
    prefix["info"]["directory"] = str(repo)
    prefix["messages"][1]["parts"] = [
        tool_part(call["call_id"], call["tool"], call["args"], diff) for call, diff in recorded
    ]
    git(repo, "reset", "-q", "--hard")
    git(repo, "clean", "-fdq")

    seed = {"workdir": str(repo), "workspace_initial_diff": await snapshot_workspace(sandbox, base)}
    continuation = {
        "mode": "restore",
        "prefix_export": prefix,
        "workspace_diff": recorded[-1][1],
        "source_task_sha256": "a" * 64,
        "source_tools_sha256": "b" * 64,
        "workspace_diff_base": "index",
    }
    spec = RuntimeSpec(harness="opencode", task_backend="swe", continuation=continuation)
    seed["replay"] = await restore_prefix_workspace(sandbox, spec, base=base)
    seed["workspace_restored_diff"] = await snapshot_workspace(sandbox, base)

    live = [(tool_call("l1", "bash", {"command": "git status --short"}), await snapshot_workspace(sandbox, base))]
    (repo / "kept.txt").write_text("edited\n")
    live.append((tool_call("l2", "edit", {"filePath": "kept.txt"}), await snapshot_workspace(sandbox, base)))
    steps = [{"role": "user", "text": "Fix it"}, *steps_of(recorded)]
    start = len(steps)
    steps += [{"role": "user", "text": "Which file would you change next? Change nothing yet."}, *steps_of(live)]
    steps.append({"role": "assistant", "text": "renamed.txt."})
    evidence = SessionEvidence(
        status="complete",
        steps=[{"id": f"s{index}", "session_id": "root", **step} for index, step in enumerate(steps)],
        continuation_start=start,
    )
    return with_seed_facts(evidence, seed), recorded[-1][1], seed


def marks(evidence):
    return {call["call_id"]: call for step in evidence_steps(evidence) for call in step.get("calls", [])}


class Judge:
    """No check here asks a judge."""

    config = load_config()
    version, hash = "test-judge", "test-hash"

    def evaluate(self, **kwargs):
        raise AssertionError("a deterministic check asked the judge")


# exploratory_question_no_change: after the mid-run question, change no file until the user agrees
NO_CHANGE_AFTER_QUESTION = ResolvedCheck(
    id="no_change",
    kind="calls",
    parameters={"value": {"mode": "never", "tool": "any", "changes_repo": True}},
    scope={"activation": "continuation_start", "places": ["after_event"]},
)


@pytest.mark.asyncio
async def test_what_a_prefix_staged_is_no_change_of_the_continued_session_and_its_own_change_is(tmp_path):
    evidence, last_recorded, seed = await continued_session(tmp_path)
    assert seed["replay"]["workspace_matches_diff"] is True
    # The restored workspace shows the staged rename and removal, which the prefix's own last diff leaves out.
    restored = evidence.facts["workspace_restored_diff"]
    assert {"moved.txt", "removed.txt"} <= set(diff_files(restored)) - set(diff_files(last_recorded))
    marked = marks(evidence)
    assert (marked["l1"]["changed_repo"], marked["l1"]["changed_paths"]) == (False, [])
    assert (marked["l2"]["changed_repo"], marked["l2"]["changed_paths"]) == (True, ["kept.txt"])
    result = engine.grade([NO_CHANGE_AFTER_QUESTION], evidence, judge=Judge()).checks[0]
    assert result.status == "fail" and result.explanation == "called: edit kept.txt"
