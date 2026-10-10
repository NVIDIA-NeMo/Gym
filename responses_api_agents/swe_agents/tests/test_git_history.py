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
"""``sanitize_git_history``: the agent only sees git history reachable from HEAD (or base_commit)."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import responses_api_agents.swe_agents.app as swe_app
from nemo_gym.config_types import OmegaConf
from responses_api_agents.swe_agents.app import (
    OpenCodeHarnessProcessor,
    OpenHandsHarnessProcessor,
    SWEBenchWrapperConfig,
    _git_history_sanitize_cmd,
)
from responses_api_agents.swe_agents.tests.test_app import _make_instance_config


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, content: str, message: str) -> str:
    (repo / "lib.py").write_text(content)
    _git(repo, "add", "lib.py")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _object_exists(repo: Path, sha: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", sha], cwd=repo, capture_output=True).returncode == 0


def _sanitize(repo: Path, base_commit: str, log_path: Path) -> str:
    cmd = _git_history_sanitize_cmd(str(repo), base_commit, str(log_path))
    subprocess.run(["bash", "-c", cmd + "true"], check=True)
    return log_path.read_text()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestGitHistorySanitizer:
    def test_future_commits_become_unreachable(self, tmp_path: Path) -> None:
        """An image cloned at the upstream tip and reset to the base commit, as task images often are."""
        repo = tmp_path / "testbed"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        initial = _commit(repo, "v0\n", "initial")
        _git(repo, "tag", "-a", "v0.9", "-m", "v0.9")
        base = _commit(repo, "bug\n", "base")
        # Upstream fix after the base commit: on origin/main and a release tag.
        fix = _commit(repo, "fixed\n", "Fix the bug (#123)")
        _git(repo, "tag", "v1.0")
        _git(repo, "remote", "add", "origin", "https://example.com/upstream.git")
        _git(repo, "update-ref", "refs/remotes/origin/main", fix)
        # The same fix backported to a release branch that forked before the base commit, so it is not a
        # descendant of the base commit.
        _git(repo, "checkout", "-q", "-b", "release-0.9", initial)
        backport = _commit(repo, "fixed\n", "Backport: fix the bug (#123)")
        _git(repo, "checkout", "-q", "main")
        _git(repo, "reset", "-q", "--hard", base)  # the fix commit stays in the reflog
        (repo / "lib.py").write_text("stashed\n")
        _git(repo, "stash", "-q")
        stash = _git(repo, "rev-parse", "refs/stash")
        (repo / "notes.txt").write_text("pre-existing untracked file\n")
        # A tag pointing straight at a blob of the fixed file.
        fixed_blob = _git(repo, "rev-parse", f"{fix}:lib.py")
        _git(repo, "tag", "fixed-blob", fixed_blob)

        log = _sanitize(repo, base, tmp_path / "git_sanitize.log")

        assert "6 -> 2 commits reachable" in log
        assert _git(repo, "rev-parse", "HEAD") == base
        assert _git(repo, "symbolic-ref", "HEAD") == "refs/heads/main"
        assert _git(repo, "log", "--all", "--format=%H").split() == [base, initial]
        assert _git(repo, "for-each-ref", "--format=%(refname)").split() == ["refs/heads/main", "refs/tags/v0.9"]
        assert _git(repo, "remote") == ""
        assert _git(repo, "reflog", "--all") == ""
        for sha in (fix, backport, stash, fixed_blob):
            assert not _object_exists(repo, sha)
        # The working tree is untouched.
        assert (repo / "lib.py").read_text() == "bug\n"
        assert (repo / "notes.txt").read_text() == "pre-existing untracked file\n"

    def test_base_commit_outside_head_history_is_kept(self, tmp_path: Path) -> None:
        """A harness that resets to base_commit after the sanitizer must still find it."""
        repo = tmp_path / "testbed"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        initial = _commit(repo, "v0\n", "initial")
        base = _commit(repo, "bug\n", "base")
        future = _commit(repo, "fixed\n", "fix")
        _git(repo, "checkout", "-q", "-b", "side", initial)
        side = _commit(repo, "side\n", "side")
        _git(repo, "checkout", "-q", "--detach", side)
        _git(repo, "update-ref", "refs/heads/main", base)  # main at the base commit
        _git(repo, "branch", "future", future)

        _sanitize(repo, base, tmp_path / "git_sanitize.log")

        assert _git(repo, "rev-parse", "HEAD") == side
        assert _object_exists(repo, base)
        assert not _object_exists(repo, future)
        assert sorted(_git(repo, "for-each-ref", "--format=%(refname)").split()) == [
            "refs/heads/main",
            "refs/heads/side",
        ]

    def test_unknown_base_commit_falls_back_to_head(self, tmp_path: Path) -> None:
        repo = tmp_path / "testbed"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        base = _commit(repo, "bug\n", "base")
        future = _commit(repo, "fixed\n", "fix")
        _git(repo, "reset", "-q", "--hard", base)
        _git(repo, "tag", "v1.0", future)

        log = _sanitize(repo, "0" * 40, tmp_path / "git_sanitize.log")

        assert "2 -> 1 commits reachable" in log
        assert not _object_exists(repo, future)

    def test_missing_repository_is_a_no_op(self, tmp_path: Path) -> None:
        log = _sanitize(tmp_path / "absent", "abc", tmp_path / "absent.log")
        assert "no git repository at" in log
        (tmp_path / "plain").mkdir()
        log = _sanitize(tmp_path / "plain", "abc", tmp_path / "plain.log")
        assert "no git repository at" in log

    def test_repo_path_and_base_commit_are_quoted(self, tmp_path: Path) -> None:
        repo = tmp_path / "work space" / "repo"
        repo.mkdir(parents=True)
        _git(repo, "init", "-q", "-b", "main")
        base = _commit(repo, "bug\n", "base")
        future = _commit(repo, "fixed\n", "fix")
        _git(repo, "reset", "-q", "--hard", base)

        log = _sanitize(repo, f"$(touch {tmp_path}/injected); {base}", tmp_path / "git_sanitize.log")

        assert "git history sanitized" in log
        assert not _object_exists(repo, future)
        assert not (tmp_path / "injected").exists()


class TestSanitizeGitHistoryWiring:
    def test_default_off(self) -> None:
        assert SWEBenchWrapperConfig.model_fields["sanitize_git_history"].default is False

    @pytest.mark.parametrize("enabled", [False, True])
    def test_openhands_agent_script(self, tmp_path: Path, enabled: bool) -> None:
        config = _make_instance_config(str(tmp_path), sanitize_git_history=enabled)
        OpenHandsHarnessProcessor(config=config).get_run_command()
        script = (config.persistent_dir / f"agent_script_{config.agent_run_id}.sh").read_text()
        assert ("git_sanitize.log" in script) is enabled
        if enabled:
            # Runs in the image's repository before the harness starts.
            assert "cd /testbed" in script
            assert script.index("git_sanitize.log") < script.index("run_infer.sh")

    @pytest.mark.parametrize("enabled", [False, True])
    def test_opencode_agent_script(self, tmp_path: Path, monkeypatch, enabled: bool) -> None:
        monkeypatch.setattr(
            swe_app,
            "get_first_server_config_dict",
            lambda _global, name: type("Cfg", (), {"host": "test-host", "port": 12345, "model": "test-model"})(),
        )
        monkeypatch.setattr(swe_app, "get_global_config_dict", MagicMock(return_value=OmegaConf.create({})))
        opencode_setup_dir = tmp_path / "opencode_setup"
        opencode_setup_dir.mkdir()
        config = _make_instance_config(
            str(tmp_path),
            agent_framework="opencode",
            opencode_setup_dir=opencode_setup_dir,
            agent_framework_repo="https://example.invalid/opencode.git",
            agent_framework_commit="deadbeef",
            sanitize_git_history=enabled,
        )
        OpenCodeHarnessProcessor(config=config).get_run_command()
        script = (config.persistent_dir / f"agent_script_{config.agent_run_id}.sh").read_text()
        assert ("git_sanitize.log" in script) is enabled
        if enabled:
            assert script.index("git_sanitize.log") < script.index("run_infer.sh")
