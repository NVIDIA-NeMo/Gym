# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.swe_rebench.verification import (
    VerificationInputs,
    build_eval_script,
    run_verification,
    verification_files,
)
from resources_servers.swebench import start_state
from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup


class LocalSandbox:
    """Runs sandbox commands on a local repository with the real git and scripts."""

    async def exec(self, command, cwd=None, timeout_s=None):
        proc = subprocess.run(["bash", "-c", command], cwd=cwd, capture_output=True, text=True)
        return SimpleNamespace(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)

    async def upload(self, local_path, remote_path):
        shutil.copy(local_path, remote_path)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def commit(repo: Path, message: str, date: str, author: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qam", message],
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(repo.parent),
            "GIT_AUTHOR_NAME": author,
            "GIT_AUTHOR_EMAIL": "a@example.com",
            "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_NAME": author,
            "GIT_COMMITTER_EMAIL": "a@example.com",
            "GIT_COMMITTER_DATE": date,
        },
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """History first -> base, a tag, and a setup patch that edits a file the base added."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / ".gitconfig"))
    monkeypatch.setattr(start_state, "SETUP_PATCH_PATH", str(tmp_path / "setup.diff"))
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    (path / "a.py").write_text("x = 1\n")
    git(path, "add", ".")
    commit(path, "first", "2020-01-01T00:00:00+0000", "First Author")
    (path / "a.py").write_text("x = 2\n")
    (path / "b.py").write_text("def f():\n    return 1\n")
    git(path, "add", ".")
    commit(path, "base: add f\n\nLonger body.", "2021-06-01T12:00:00+0000", "Base Author")
    git(path, "tag", "v1")
    (path / "b.py").write_text("def f():\n    return 2\n")
    setup = git(path, "diff")
    git(path, "checkout", "--", "b.py")
    (tmp_path / "setup.diff").write_text(setup)
    (path / "__pycache__").mkdir()
    (path / "__pycache__" / "b.cpython-313.pyc").write_bytes(b"stale")
    return path


async def seed(repo: Path, *, squash: bool) -> str:
    sandbox = LocalSandbox()
    original = await start_state.seeded_head(sandbox, str(repo))
    await start_state.commit_start_state(sandbox, str(repo), squash_history=squash)
    await apply_anti_cheat_setup(sandbox, str(repo), "toy", "test")
    await start_state.assert_unreachable(sandbox, str(repo), original)
    await start_state.normalize_tree(sandbox, str(repo))
    return original


@pytest.mark.asyncio
async def test_squashed_start_state_hides_the_applied_patch(repo):
    base_meta = git(repo, "log", "-1", "--format=%an|%ae|%aD|%cn|%cD|%B")
    await seed(repo, squash=True)
    assert (repo / "b.py").read_text() == "def f():\n    return 2\n"
    assert git(repo, "rev-list", "--all", "--count").strip() == "1"
    assert git(repo, "log", "-1", "--format=%an|%ae|%aD|%cn|%cD|%B") == base_meta
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "tag") == ""
    assert not (repo / "__pycache__").exists()
    assert len({(repo / name).stat().st_mtime for name in ("a.py", "b.py")}) == 1


@pytest.mark.asyncio
async def test_unsquashed_start_state_keeps_history_but_replaces_the_base(repo):
    first = git(repo, "rev-parse", "HEAD~1").strip()
    original = await seed(repo, squash=False)
    assert git(repo, "rev-parse", "HEAD~1").strip() == first
    assert git(repo, "rev-parse", "HEAD").strip() != original
    assert git(repo, "log", "--format=%s") == "base: add f\nfirst\n"


@pytest.mark.asyncio
async def test_a_surviving_original_commit_fails_the_seed(repo):
    sandbox = LocalSandbox()
    original = await start_state.seeded_head(sandbox, str(repo))
    await start_state.commit_start_state(sandbox, str(repo), squash_history=True)
    with pytest.raises(RuntimeError, match="survived"):
        await start_state.assert_unreachable(sandbox, str(repo), original)


def test_grading_applies_the_start_state_exactly_before_the_agent_patch():
    inputs = VerificationInputs(
        instance_id="toy", repo="o/r", base_commit="abc", patch="agent\n", setup_patch="setup\n"
    )
    script = build_eval_script(inputs)
    setup_line = start_state.apply_setup_patch_command("setup\n")
    assert setup_line in script
    assert script.index(setup_line) < script.index("/tmp/nemo_gym_patch.diff")
    assert verification_files(inputs)[start_state.SETUP_PATCH_PATH] == "setup\n"
    assert start_state.SETUP_PATCH_PATH not in verification_files(
        VerificationInputs(instance_id="toy", repo="o/r", base_commit="abc", patch="agent\n")
    )


@pytest.mark.asyncio
async def test_a_start_state_that_does_not_apply_is_incomplete_not_a_zero():
    class Failing:
        async def exec(self, command, timeout_s=None):
            return SimpleNamespace(
                stdout=start_state.SETUP_PATCH_FAILED_MARK + "\n",
                stderr="patch failed",
                return_code=start_state.SETUP_PATCH_FAILED_EXIT,
            )

    inputs = VerificationInputs(instance_id="toy", repo="o/r", base_commit="abc", patch="", setup_patch="s\n")
    result = await run_verification(Failing(), inputs, parser=lambda log: {})
    assert result.completed is False and result.error == start_state.SETUP_PATCH_FAILED_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("setup_patch", ["", "s\n"])
async def test_a_test_command_exiting_96_keeps_the_normal_verdict(setup_patch):
    class TestsExit:
        def __init__(self, code):
            self.code = code

        async def exec(self, command, timeout_s=None):
            return SimpleNamespace(stdout="FAILED some_test\n", stderr="", return_code=self.code)

    inputs = VerificationInputs(
        instance_id="toy",
        repo="o/r",
        base_commit="abc",
        patch="",
        setup_patch=setup_patch,
        fail_to_pass=["some_test"],
    )
    parser = lambda log: {"some_test": "FAILED"}  # noqa: E731
    at_96 = await run_verification(TestsExit(start_state.SETUP_PATCH_FAILED_EXIT), inputs, parser=parser)
    at_1 = await run_verification(TestsExit(1), inputs, parser=parser)
    assert at_96.error != start_state.SETUP_PATCH_FAILED_ERROR
    assert vars(at_96) == vars(at_1)
    assert at_96.completed and not at_96.resolved


@pytest.mark.asyncio
async def test_the_real_grading_script_stops_at_a_setup_patch_that_does_not_apply(repo, monkeypatch):
    from resources_servers.swe_rebench import verification

    monkeypatch.setattr(verification, "repo_directory", lambda _: str(repo))
    exits = []

    class RunsTheScript:
        """Runs the eval script exactly as verification_files ships it, on the local repository."""

        def __init__(self, inputs):
            self.files = verification.verification_files(inputs)

        async def exec(self, command, timeout_s=None):
            for path, content in self.files.items():
                if path == start_state.SETUP_PATCH_PATH:
                    Path(path).write_text(content)
            proc = subprocess.run(
                ["bash", "-c", self.files["/tmp/nemo_gym_eval.sh"]], cwd=repo, capture_output=True, text=True
            )
            exits.append(proc.returncode)
            return SimpleNamespace(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)

    def inputs(setup_patch):
        return VerificationInputs(
            instance_id="toy",
            repo="o/r",
            base_commit=git(repo, "rev-parse", "HEAD").strip(),
            patch="",
            setup_patch=setup_patch,
            test_cmd=["echo TESTS_RAN"],
        )

    applies = Path(start_state.SETUP_PATCH_PATH).read_text()
    stale = "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-nope\n+x\n"
    failed = await run_verification(RunsTheScript(inputs(stale)), inputs(stale), parser=lambda log: {})
    assert exits[-1] == start_state.SETUP_PATCH_FAILED_EXIT
    assert start_state.SETUP_PATCH_FAILED_MARK in failed.test_output
    assert "TESTS_RAN" not in failed.test_output
    assert failed.completed is False and failed.error == start_state.SETUP_PATCH_FAILED_ERROR

    # Control: the same script with a setup patch that applies reaches the tests and grades.
    graded = await run_verification(RunsTheScript(inputs(applies)), inputs(applies), parser=lambda log: {})
    assert exits[-1] == 0 and "TESTS_RAN" in graded.test_output
    assert graded.completed is True and graded.error is None
    assert (repo / "b.py").read_text() == "def f():\n    return 2\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["timeout", "sandbox"])
async def test_a_failed_probe_does_not_count_as_a_pruned_original(error_type):
    class ProbeFails:
        async def exec(self, command, cwd=None, timeout_s=None):
            if "rev-parse" in command:
                return SimpleNamespace(stdout="seeded\n", stderr="", return_code=0)
            return SimpleNamespace(stdout=None, stderr="provider error", return_code=-1, error_type=error_type)

    with pytest.raises(RuntimeError, match=f"Could not verify.*{error_type}"):
        await start_state.assert_unreachable(ProbeFails(), "/w", "original")


@pytest.mark.asyncio
async def test_seeding_gives_every_directory_the_tracked_files_time(tmp_path, monkeypatch):
    """Applying the setup patch touches the directories it edits or creates; none may stand out."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / ".gitconfig"))
    monkeypatch.setattr(start_state, "SETUP_PATCH_PATH", str(tmp_path / "setup.diff"))
    repo = tmp_path / "repo"
    (repo / "pkg" / "old").mkdir(parents=True)
    (repo / "pkg" / "edit").mkdir()
    (repo / "pkg" / "old" / "x.py").write_text("a\n")
    (repo / "pkg" / "edit" / "y.py").write_text("b\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    commit(repo, "base", "2021-06-01T12:00:00+0000", "Base Author")
    (repo / "pkg" / "edit" / "y.py").write_text("b\nc\n")
    (repo / "pkg" / "new" / "sub").mkdir(parents=True)
    (repo / "pkg" / "new" / "sub" / "a.py").write_text("n\n")
    git(repo, "add", ".")
    (tmp_path / "setup.diff").write_text(git(repo, "diff", "--cached"))
    git(repo, "reset", "-q", "--hard")
    shutil.rmtree(repo / "pkg" / "new", ignore_errors=True)
    for path in [repo, *repo.rglob("*")]:
        if ".git" not in path.relative_to(repo).parts:
            os.utime(path, (1_000_000_000, 1_000_000_000))

    await seed(repo, squash=True)

    assert (repo / "pkg" / "new" / "sub" / "a.py").read_text() == "n\n"
    assert git(repo, "status", "--porcelain") == ""
    stamp = (repo / "pkg" / "old" / "x.py").stat().st_mtime
    directories = [repo, *(p for p in repo.rglob("*") if p.is_dir() and ".git" not in p.relative_to(repo).parts)]
    assert {d.stat().st_mtime for d in directories} == {stamp}


@pytest.mark.asyncio
async def test_rebench_seeds_the_start_state_and_extracts_only_the_agent_work(repo, tmp_path, monkeypatch):
    from resources_servers.swe_rebench import app

    pristine = tmp_path / "pristine"
    shutil.copytree(repo, pristine)
    setup = (tmp_path / "setup.diff").read_text()
    monkeypatch.setattr(app, "repo_directory", lambda _: str(repo))

    async def create_sandbox(body, files=None):
        for path, content in (files or {}).items():
            Path(path).write_text(content)
        sandbox = LocalSandbox()
        sandbox._handle = SimpleNamespace(sandbox_id="local")
        return sandbox

    async def stop_sandbox(sandbox):
        return None

    server = SimpleNamespace(
        config=SimpleNamespace(apply_anti_cheating=False, patch_capture_mode="worktree"),
        _session_id_to_sandbox={},
        _session_id_to_pristine_untracked={},
        _session_id_to_seeded_head={},
        _create_sandbox=create_sandbox,
        _stop_sandbox=stop_sandbox,
    )
    server._pristine_untracked_files = (
        lambda sandbox, workdir: app.SWERebenchResourcesServer._pristine_untracked_files(server, sandbox, workdir)
    )
    body = app.SWERebenchSeedSessionRequest(
        instance_id="toy",
        repo="o/r",
        base_commit=git(repo, "rev-parse", "HEAD").strip(),
        image_name="img",
        setup_patch=setup,
        squash_history=True,
    )
    request = SimpleNamespace(session={app.SESSION_ID_KEY: "s"})
    await app.SWERebenchResourcesServer.seed_session(server, request, body)
    assert git(repo, "rev-list", "--all", "--count").strip() == "1"

    (repo / "a.py").write_text("x = 3\n")
    capture = await app.SWERebenchResourcesServer._extract_model_patch(server, "s", str(repo), body.base_commit)
    patch = capture.patch
    assert "a.py" in patch and "b.py" not in patch

    # Grading: a fresh copy at base, then the start state, then the agent's patch.
    for diff in (setup, patch):
        subprocess.run(["git", "-C", str(pristine), "apply", "-"], input=diff, text=True, check=True)
    assert (pristine / "a.py").read_text() == "x = 3\n"
    assert (pristine / "b.py").read_text() == "def f():\n    return 2\n"
