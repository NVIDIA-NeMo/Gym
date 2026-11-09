# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay must restore real workspace state and preserve OpenCode's continuation boundaries."""

import os
import subprocess
from copy import deepcopy

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.episode.opencode_evidence import opencode_evidence
from resources_servers.agentic_if.episode.opencode_replay import (
    _workspace_diff,
    continued_session,
    restore_prefix_workspace,
    restore_workspace,
)
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import workspace_base
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec
from resources_servers.agentic_if.tests.support import REPLAY_RECEIPT, LocalTransport, exported


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [None, ""])
async def test_empty_workspace_diff_accepts_none_or_empty_output(output):
    class EmptyOutput:
        async def exec(self, **kwargs):
            return SandboxExecResult(stdout=output, stderr=None, return_code=0)

    sandbox = BoundSandbox(EmptyOutput(), user="agent", workdir="/app", timeout_s=60)
    assert await _workspace_diff(sandbox, base="0" * 40) == ""


def repository(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (path / "file").write_text("before")
    git("add", ".")
    git("commit", "-qm", "base")
    return git


def continuation(prefix, diff):
    return ContinuationSpec(
        mode="restore",
        prefix_export=prefix,
        workspace_diff=diff,
        source_task_sha256="a" * 64,
        source_tools_sha256="b" * 64,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_ms", [None, 100])
async def test_restore_replays_shell_as_agent_and_restores_binary_diff(tmp_path, timeout_ms):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = repository(repo)
    binary = bytes([0, 42, 255, 0])
    (repo / "new.bin").write_bytes(binary)
    git("add", "-N", ".")
    diff = git("diff", "--binary", "--full-index")
    git("reset", "-q")
    (repo / "new.bin").unlink()
    prefix = exported("root")
    prefix["info"]["directory"] = str(repo)
    arguments = {"command": "printf changed > file; printf drift"}
    if timeout_ms is not None:
        arguments.update(command=arguments["command"] + "; sleep 5; printf late > ../late", timeout=timeout_ms)
    prefix["messages"][1]["parts"] = [
        {
            "type": "tool",
            "tool": "bash",
            "callID": "first",
            "state": {
                "status": "completed",
                "input": arguments,
                "output": "original",
                "metadata": {},
                "time": {"end": 3},
            },
        }
    ]
    transport = LocalTransport()
    sandbox = BoundSandbox(transport, user="agent", workdir=str(repo), timeout_s=60)
    staging = tmp_path / "staging"
    staging.mkdir()
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    base = git("rev-parse", "HEAD^{tree}").strip()
    receipt = await restore_workspace(sandbox, spec, continuation(prefix, diff), base=base, staging=str(staging))
    assert (repo / "new.bin").read_bytes() == binary
    assert (repo / "file").read_text() == "before"
    assert receipt["workspace_matches_diff"] is True
    assert receipt["command_output_mismatch_indices"] == [0]
    assert not (tmp_path / "late").exists()
    assert all(
        user == "agent" for command, user in transport.calls if "git " in command or "printf changed" in command
    )


@pytest.mark.asyncio
async def test_seed_restore_checks_the_recorded_diff_and_removes_its_staging(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = repository(repo)("rev-parse", "HEAD^{tree}").strip()
    prefix = exported("root")
    prefix["messages"][1]["parts"] = [
        {
            "type": "tool",
            "tool": "edit",
            "callID": "edit",
            "state": {
                "status": "completed",
                "input": {"filePath": "file"},
                "output": "ok",
                "metadata": {"nemo_gym_profile": {"workspace_diff": "diff --git a/file b/file\n"}},
            },
        }
    ]
    transport = LocalTransport()
    sandbox = BoundSandbox(transport, user=None, workdir=str(repo), timeout_s=60)
    spec = RuntimeSpec(harness="opencode", task_backend="swe", continuation=continuation(prefix, "").model_dump())
    with pytest.raises(ValueError, match="disagrees with the recorded prefix"):
        await restore_prefix_workspace(sandbox, spec, base=base)
    prefix["messages"][1]["parts"][0]["state"]["metadata"]["nemo_gym_profile"]["workspace_diff"] = ""
    spec = RuntimeSpec(harness="opencode", task_backend="swe", continuation=continuation(prefix, "").model_dump())
    with pytest.raises(RuntimeError, match="Git repository"):
        await restore_prefix_workspace(sandbox, spec, base=None)
    receipt = await restore_prefix_workspace(sandbox, spec, base=base)
    assert receipt["workspace_matches_diff"] is True
    (staging,) = {command.split()[-1] for command, _ in transport.calls if command.startswith("umask 077; mkdir -p ")}
    assert staging.startswith("/tmp/nemo-gym-agentic-if-replay-") and not os.path.exists(staging)


def bash_prefix(repo, command):
    """A recorded prefix whose one completed bash call ran `command` in `repo` and printed nothing."""
    prefix = exported("root")
    prefix["info"]["directory"] = str(repo)
    state = {"status": "completed", "input": {"command": command}, "output": "", "metadata": {}, "time": {"end": 3}}
    prefix["messages"][1]["parts"] = [{"type": "tool", "tool": "bash", "callID": "bash", "state": state}]
    return prefix


@pytest.mark.asyncio
async def test_restore_reproduces_the_recorded_files_when_the_replayed_commands_stage_and_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = repository(repo)
    sandbox = BoundSandbox(LocalTransport(), user=None, workdir=str(repo), timeout_s=60)
    base = await workspace_base(sandbox)
    # The prefix committed an edit of `file` and then wrote a binary file, which no command replays.
    command = "printf committed > file && git add file && git commit -qm prefix"
    subprocess.run(["/bin/sh", "-c", command], cwd=repo, check=True)
    (repo / "new.bin").write_bytes(bytes([0, 42, 255, 0]))
    recorded = await _workspace_diff(sandbox, base=base)
    git("reset", "-q", "--hard", "HEAD~1")
    (repo / "new.bin").unlink()
    staging = tmp_path / "staging"
    staging.mkdir()
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    prefix = bash_prefix(repo, command)
    receipt = await restore_workspace(sandbox, spec, continuation(prefix, recorded), base=base, staging=str(staging))
    assert (repo / "file").read_text() == "committed"
    assert (repo / "new.bin").read_bytes() == bytes([0, 42, 255, 0])
    assert receipt["workspace_matches_diff"] is True and receipt["command_output_mismatch_indices"] == []
    # What the replayed commands committed stays committed.
    assert git("log", "-1", "--format=%s").strip() == "prefix"


@pytest.mark.asyncio
async def test_restore_supports_a_repository_without_commits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    sandbox = BoundSandbox(LocalTransport(), user=None, workdir=str(repo), timeout_s=60)
    base = await workspace_base(sandbox)
    (repo / "recorded.txt").write_text("recorded")
    recorded = await _workspace_diff(sandbox, base=base)
    (repo / "recorded.txt").unlink()
    staging = tmp_path / "staging"
    staging.mkdir()
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    prefix = bash_prefix(repo, "printf replayed > replayed.txt")
    receipt = await restore_workspace(sandbox, spec, continuation(prefix, recorded), base=base, staging=str(staging))
    assert (repo / "recorded.txt").read_text() == "recorded" and not (repo / "replayed.txt").exists()
    assert receipt["workspace_matches_diff"] is True


def continued_root(prefix):
    """The root session after the harness imported the prefix and continued it with the row's user message."""
    document = deepcopy(prefix)
    for message in exported("root", text="Fresh decision")["messages"]:
        message["info"]["id"] += "-fresh"
        message["info"]["time"]["created"] += 10
        document["messages"].append(message)
    return document


def test_prefix_then_real_continuation_uses_same_opencode_session():
    prefix = exported("root", text="Earlier decision")
    session = continued_session(
        continuation(prefix, ""),
        session_id="root",
        export=continued_root(prefix),
        receipt=REPLAY_RECEIPT,
        artifacts={"prefix_export": "/artifacts/prefix-export.json"},
    )
    assert session.prefix_ids == frozenset({"u", "a"})
    assert session.receipt["prefix_messages"] == 2 and session.receipt["tool_outputs_match"] is True
    evidence = session.attach(opencode_evidence(continued_root(prefix)))
    assert evidence.status == "complete"
    assert evidence.continuation_start == 2
    assert evidence.steps[1].text == "Earlier decision" and evidence.steps[-1].text == "Fresh decision"
    assert evidence.facts["replay"]["prefix_messages"] == 2
    assert evidence.artifacts["prefix_export"] == "/artifacts/prefix-export.json"


@pytest.mark.parametrize("defect", ["tool_output", "workspace", "no_fresh_user"])
def test_restored_history_must_match_the_recorded_prefix(defect):
    prefix = exported("root", text="Earlier decision")
    prefix["messages"][1]["parts"].append(
        {
            "type": "tool",
            "tool": "read",
            "callID": "read-one",
            "state": {"status": "completed", "input": {"filePath": "/app/file"}, "output": "recorded"},
        }
    )
    root = continued_root(prefix)
    receipt = dict(REPLAY_RECEIPT)
    if defect == "tool_output":
        root["messages"][1]["parts"][-1]["state"]["output"] = "changed"
    elif defect == "workspace":
        receipt["workspace_matches_diff"] = False
    else:
        root = deepcopy(prefix)
    session = continued_session(
        continuation(prefix, ""), session_id="root", export=root, receipt=receipt, artifacts={}
    )
    evidence = session.attach(opencode_evidence(root))
    assert evidence.status == "incomplete"
    assert {
        "tool_output": "restored_prefix_history_diverged",
        "workspace": "replay_workspace_diverged",
        "no_fresh_user": "missing_continuation_user_message",
    }[defect] in evidence.issues
