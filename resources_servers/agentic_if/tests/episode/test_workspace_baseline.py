# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial image/seed state is not an effect of the actor's first tool call."""

import os
import shutil
import subprocess
from shlex import quote

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import snapshot_workspace, workspace_base
from resources_servers.agentic_if.if_grading.checkers.run import diff_files, evidence_steps, mark_repo_changes
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence
from resources_servers.agentic_if.tests.support import LocalTransport


INITIAL = "diff --git a/.env b/.env\nnew file mode 100644\n--- /dev/null\n+++ b/.env\n@@ -0,0 +1 @@\n+seed\n"


def test_baseline_excludes_seed_but_preserves_changes_and_deletion():
    changed = INITIAL.replace("+seed", "+changed")
    calls = [{"tool": "bash", "args": {"command": "opaque"}} for _ in range(3)]
    marked = mark_repo_changes(calls, [INITIAL, changed, ""], initial_diff=INITIAL)
    assert marked[0]["changed_repo"] is False
    assert marked[0]["changed_paths"] == []
    for call in marked[1:]:
        assert call["changed_repo"] is True
        assert call["changed_paths"] == [".env"]
        assert call["created_paths"] == []


def test_evidence_uses_initial_snapshot_before_recorded_prefix_and_live_calls():
    evidence = SessionEvidence(
        status="complete",
        facts={"workspace_initial_diff": INITIAL},
        steps=[
            {
                "id": "m",
                "session_id": "root",
                "role": "assistant",
                "calls": [
                    {
                        "call_id": "c",
                        "invocation_id": "r",
                        "tool": "bash",
                        "args": {"command": "ls"},
                        "model_tool": "bash",
                        "model_args": {"command": "ls"},
                    },
                ],
            },
            {
                "id": "t",
                "session_id": "root",
                "role": "tool",
                "tool": "bash",
                "call_id": "c",
                "invocation_id": "r",
                "workspace_diff": INITIAL,
            },
        ],
    )
    call = evidence_steps(evidence)[0]["calls"][0]
    assert call["changed_repo"] is False
    assert call["changed_paths"] == []
    # Evidence recorded without a baseline is not silently reinterpreted.
    evidence.facts.clear()
    assert evidence_steps(evidence)[0]["calls"][0]["changed_paths"] == [".env"]


@pytest.mark.asyncio
async def test_snapshot_observes_real_dirty_tree_without_mutating_index(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args])

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "tracked").write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "base")
    (tmp_path / ".env").write_text("seed\n")
    (tmp_path / "tracked").write_text("initial dirty\n")
    before_status = git("status", "--porcelain")
    before_index = (tmp_path / ".git/index").read_bytes()
    transport = LocalTransport()
    sandbox = BoundSandbox(transport, workdir=str(tmp_path), user="1001:1001", timeout_s=30)
    base = await workspace_base(sandbox)
    initial = await snapshot_workspace(sandbox, base)
    assert "a/.env" in initial and "+initial dirty" in initial
    await sandbox.exec(command="ls -la", timeout_s=30)
    unchanged = await snapshot_workspace(sandbox, base)
    assert unchanged == initial
    assert (tmp_path / ".git/index").read_bytes() == before_index
    assert git("status", "--porcelain") == before_status
    assert all(user == "1001:1001" for _, user in transport.calls)
    (tmp_path / "new").write_bytes(b"\x00new binary\xff")
    final = await snapshot_workspace(sandbox, base)
    marked = mark_repo_changes([{"tool": "bash", "args": {}}], [final], initial_diff=initial)[0]
    assert marked["created_paths"] == ["new"]
    assert marked["changed_paths"] == ["new"]


@pytest.mark.asyncio
async def test_staging_and_commits_do_not_change_what_a_snapshot_says_about_the_files(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "tracked").write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "base")
    sandbox = BoundSandbox(LocalTransport(), workdir=str(tmp_path), user=None, timeout_s=30)
    base = await workspace_base(sandbox)
    assert base == git("rev-parse", "HEAD^{tree}").strip()
    (tmp_path / "tracked").write_text("changed\n")
    (tmp_path / "new.bin").write_bytes(b"\x00binary\xff")
    unstaged = await snapshot_workspace(sandbox, base)
    git("add", "tracked", "new.bin")
    staged = await snapshot_workspace(sandbox, base)
    git("commit", "-qm", "work")
    committed = await snapshot_workspace(sandbox, base)
    assert unstaged == staged == committed
    assert "+changed" in unstaged and "GIT binary patch" in unstaged
    # Staging and committing are no file changes.
    calls = [{"tool": "bash", "args": {}} for _ in range(2)]
    marked = mark_repo_changes(calls, [staged, committed], initial_diff=unstaged)
    assert [(c["changed_repo"], c["changed_paths"]) for c in marked] == [(False, []), (False, [])]


@pytest.mark.asyncio
async def test_a_repository_without_commits_is_observed_against_the_empty_tree(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True)

    git("init", "-q")
    (tmp_path / "staged").write_text("staged\n")
    (tmp_path / "untracked").write_text("untracked\n")
    git("add", "staged")
    index, status = (tmp_path / ".git/index").read_bytes(), git("status", "--porcelain")
    sandbox = BoundSandbox(LocalTransport(), workdir=str(tmp_path), user=None, timeout_s=30)
    base = await workspace_base(sandbox)
    assert base == git("hash-object", "-t", "tree", "/dev/null").strip()
    files = diff_files(await snapshot_workspace(sandbox, base))
    assert sorted(files) == ["staged", "untracked"]
    assert all("new file mode" in section for section, _ in files.values())
    assert (tmp_path / ".git/index").read_bytes() == index and git("status", "--porcelain") == status


@pytest.mark.asyncio
async def test_a_head_naming_a_missing_commit_leaves_the_base_unknown(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    branch = subprocess.check_output(["git", "-C", str(tmp_path), "symbolic-ref", "HEAD"], text=True).strip()
    width = len(
        subprocess.check_output(["git", "-C", str(tmp_path), "hash-object", "-t", "tree", "/dev/null"]).strip()
    )
    (tmp_path / ".git" / branch).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git" / branch).write_text("1" * width + "\n")
    sandbox = BoundSandbox(LocalTransport(), workdir=str(tmp_path), user=None, timeout_s=30)
    assert await workspace_base(sandbox) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["no_repository", "bare", "no_git", "unsupported_version", "invalid_git_dir", "corrupt_index"]
)
async def test_unavailable_or_unreadable_repository_is_an_unknown_snapshot_not_a_clean_tree(
    tmp_path, monkeypatch, state
):
    if state == "no_git":
        monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    elif state != "no_repository":
        subprocess.run(["git", "init", "-q", *(["--bare"] if state == "bare" else []), str(tmp_path)], check=True)
    if state == "unsupported_version":
        subprocess.run(["git", "-C", str(tmp_path), "config", "core.repositoryformatversion", "99"], check=True)
    elif state == "invalid_git_dir":
        monkeypatch.setenv("GIT_DIR", str(tmp_path / "missing-git-dir"))
    elif state == "corrupt_index":
        (tmp_path / ".git/index").write_bytes(b"broken")
    sandbox = BoundSandbox(LocalTransport(), workdir=str(tmp_path), user=None, timeout_s=30)
    assert await snapshot_workspace(sandbox, await workspace_base(sandbox)) is None
    if state == "corrupt_index":
        # The task can still start, and its broken index is not repaired.
        assert (tmp_path / ".git/index").read_bytes() == b"broken"


@pytest.mark.asyncio
async def test_temporary_index_lock_failure_is_not_an_unknown_repository(tmp_path, monkeypatch):
    real_git = shutil.which("git")
    subprocess.run([real_git, "init", "-q", str(tmp_path)], check=True)
    tools = tmp_path / "tools"
    tools.mkdir()
    wrapper = tools / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = add ]; then\n'
        '  touch "$GIT_INDEX_FILE.lock" || exit 1\n'
        f'  {quote(real_git)} "$@"\n'
        "  result=$?\n"
        '  rm -f "$GIT_INDEX_FILE.lock"\n'
        '  exit "$result"\n'
        "fi\n"
        f'exec {quote(real_git)} "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    sandbox = BoundSandbox(LocalTransport(), workdir=str(tmp_path), user=None, timeout_s=30)
    base = await workspace_base(sandbox)
    with pytest.raises(RuntimeError, match="snapshot"):
        await snapshot_workspace(sandbox, base)


def test_unknown_initial_snapshot_does_not_attribute_seed_state_to_first_call():
    changed = INITIAL.replace("+seed", "+changed")
    calls = [{"tool": "bash", "args": {"command": "opaque"}} for _ in range(4)]
    marked = mark_repo_changes(calls, [None, INITIAL, INITIAL, changed], initial_diff=None)
    assert [c["changed_repo"] for c in marked] == [None, None, False, True]
    assert [c["changed_paths"] for c in marked] == [[], [], [], [".env"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        SandboxExecResult(stdout="", stderr="index write failed", return_code=1),
        SandboxExecResult(stdout="partial", stderr="timed out", return_code=124),
        SandboxExecResult(stdout="", stderr="signal: killed", return_code=137),
        SandboxExecResult(stdout="", stderr="transport error", return_code=78, error_type="transport"),
    ],
)
async def test_snapshot_failure_is_not_a_clean_tree(result):
    class Failed:
        async def exec(self, **kwargs):
            return result

    sandbox = BoundSandbox(Failed(), workdir="/app", user=None, timeout_s=30)
    with pytest.raises(RuntimeError, match="base tree"):
        await workspace_base(sandbox)
    with pytest.raises(RuntimeError, match="snapshot"):
        await snapshot_workspace(sandbox, "0" * 40)


@pytest.mark.asyncio
async def test_a_base_probe_that_prints_no_tree_fails_and_no_base_takes_no_snapshot():
    class Printed:
        async def exec(self, **kwargs):
            return SandboxExecResult(stdout="HEAD^{tree}\n", stderr="", return_code=0)

    sandbox = BoundSandbox(Printed(), workdir="/app", user=None, timeout_s=30)
    with pytest.raises(RuntimeError, match="base tree"):
        await workspace_base(sandbox)
    assert await snapshot_workspace(sandbox, None) is None


@pytest.mark.asyncio
async def test_snapshot_uses_replay_budget_for_a_large_initial_tree():
    class Slow:
        async def exec(self, **kwargs):
            assert kwargs["timeout_s"] == 600
            assert kwargs["user"] == "agent"
            return SandboxExecResult(stdout=INITIAL, stderr="", return_code=0)

    sandbox = BoundSandbox(Slow(), workdir="/app", user="agent", timeout_s=1800)
    assert await snapshot_workspace(sandbox, "0" * 40) == INITIAL
