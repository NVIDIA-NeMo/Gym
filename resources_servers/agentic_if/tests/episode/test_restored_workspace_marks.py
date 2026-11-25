# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The continued session's changes are told against the workspace seed restored, not against the prefix's last diff.

A prefix recorded with its diffs taken against the Git index (``ContinuationSpec.workspace_diff_base``) leaves out what
it staged; this harness's diffs, taken against the base tree, show it. Only seed's snapshot of the restored workspace
says what the continued session started from.
"""

import os
import shutil
import subprocess

import pytest

from resources_servers.agentic_if.episode.opencode_replay import restore_prefix_workspace
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.session_evidence import with_seed_facts
from resources_servers.agentic_if.episode.workspace import snapshot_workspace, workspace_base
from resources_servers.agentic_if.if_grading import engine
from resources_servers.agentic_if.if_grading.checkers.run import diff_files, evidence_steps
from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.git_diff import workspace_patches_equal
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec
from resources_servers.agentic_if.tests.support import LocalTransport, exported


def git(repo, *args, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def index_relative_diff(repo, scratch):
    """opencode_sandboxed_agent's per-call diff: a plain `git diff` against a copy of the index, with untracked files
    added to the copy as intent-to-add."""
    copy = scratch / "index"
    shutil.copy2(repo / ".git" / "index", copy)
    env = os.environ | {"GIT_INDEX_FILE": str(copy)}
    git(repo, "add", "-N", "--ignore-removal", ".", env=env)
    return git(repo, "diff", env=env)


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


async def continued_session(tmp_path, *, live_edit):
    """Record the prefix in a repository, return the repository to its commit and seed the episode as `_prepare` does
    (initial snapshot, restore, snapshot of the restored workspace). The continued session then runs a command that
    changes nothing and, with `live_edit`, edits kept.txt. Returns the evidence, the prefix's last diff and the seed."""
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "T"]):
        git(repo, *args)
    (repo / "moved.txt").write_text("line one\nline two\n")
    (repo / "removed.txt").write_text("gone\n")
    (repo / "kept.txt").write_text("kept\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    sandbox = BoundSandbox(LocalTransport(), user=None, workdir=str(repo), timeout_s=60)
    base = await workspace_base(sandbox)

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
    if live_edit:
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
async def test_what_a_prefix_staged_is_no_change_of_the_continued_session(tmp_path):
    evidence, last_recorded, seed = await continued_session(tmp_path, live_edit=False)
    assert seed["replay"]["workspace_matches_diff"] is True
    # The restored workspace shows the staged rename and removal, which the prefix's own last diff leaves out.
    restored = evidence.facts["workspace_restored_diff"]
    assert {"moved.txt", "removed.txt"} <= set(diff_files(restored)) - set(diff_files(last_recorded))
    assert not workspace_patches_equal(restored, last_recorded)
    marked = marks(evidence)
    assert (marked["l1"]["changed_repo"], marked["l1"]["changed_paths"]) == (False, [])
    result = engine.grade([NO_CHANGE_AFTER_QUESTION], evidence, judge=Judge()).checks[0]
    assert result.status == "pass", result.explanation


@pytest.mark.asyncio
async def test_a_continued_session_s_own_change_after_such_a_prefix_is_marked(tmp_path):
    evidence, _, _ = await continued_session(tmp_path, live_edit=True)
    marked = marks(evidence)
    assert (marked["l1"]["changed_repo"], marked["l1"]["changed_paths"]) == (False, [])
    assert (marked["l2"]["changed_repo"], marked["l2"]["changed_paths"]) == (True, ["kept.txt"])
    result = engine.grade([NO_CHANGE_AFTER_QUESTION], evidence, judge=Judge()).checks[0]
    assert result.status == "fail" and result.explanation == "called: edit kept.txt"
