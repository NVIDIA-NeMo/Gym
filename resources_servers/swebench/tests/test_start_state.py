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
    setup_line = f"git apply {start_state.SETUP_PATCH_PATH} || exit {start_state.SETUP_PATCH_FAILED_EXIT}"
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
            return SimpleNamespace(stdout="", stderr="patch failed", return_code=start_state.SETUP_PATCH_FAILED_EXIT)

    inputs = VerificationInputs(instance_id="toy", repo="o/r", base_commit="abc", patch="", setup_patch="s\n")
    result = await run_verification(Failing(), inputs, parser=lambda log: {})
    assert result.completed is False and "setup patch" in result.error


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
