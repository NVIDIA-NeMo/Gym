# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``sanitize_git_history``: the agent sandbox keeps only the history reachable from HEAD."""

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from responses_api_agents.anyswe_agent.app import AnySweAgent, AnySweAgentConfig, AnySweInstanceConfig
from responses_api_agents.anyswe_agent.tests.test_app import _config


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestGitHistorySanitizer:
    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _repo_with_future_fix(self, tmp_path: Path) -> tuple[Path, str, str]:
        """A task repo checked out at the base commit, with the fix reachable through a branch, a tag and a remote."""
        repo = tmp_path / "testbed"
        repo.mkdir()
        self._git(repo, "init", "-q", "-b", "main")
        (repo / "lib.py").write_text("bug\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-qm", "base")
        base = self._git(repo, "rev-parse", "HEAD")
        (repo / "lib.py").write_text("fixed\n")
        self._git(repo, "commit", "-qam", "Fix the bug")
        fix = self._git(repo, "rev-parse", "HEAD")
        self._git(repo, "tag", "v1.0")
        self._git(repo, "update-ref", "refs/remotes/origin/main", fix)
        self._git(repo, "remote", "add", "origin", "https://example.com/upstream.git")
        self._git(repo, "checkout", "-q", "--detach", base)
        (repo / "dirty.txt").write_text("image state\n")
        return repo, base, fix

    def _sanitize(self, repo: Path, timeout_s: int = 60) -> str:
        script = AnySweAgent._git_history_sanitizer_script(str(repo), timeout_s)
        return subprocess.run(["sh", "-c", script], check=True, capture_output=True, text=True).stdout

    def test_future_commits_become_unreachable(self, tmp_path: Path) -> None:
        repo, base, fix = self._repo_with_future_fix(tmp_path)

        output = self._sanitize(repo)

        assert "git history sanitized: 2 -> 1 commits reachable" in output
        assert self._git(repo, "rev-parse", "HEAD") == base
        assert self._git(repo, "log", "--all", "--format=%H").split() == [base]
        assert self._git(repo, "tag") == ""
        assert self._git(repo, "remote") == ""
        missing = subprocess.run(["git", "cat-file", "-e", fix], cwd=repo, capture_output=True)
        assert missing.returncode != 0
        # The working tree, including pre-existing image changes, is untouched.
        assert (repo / "lib.py").read_text() == "bug\n"
        assert (repo / "dirty.txt").read_text() == "image state\n"

    def test_missing_repo_is_a_no_op(self, tmp_path: Path) -> None:
        assert self._sanitize(tmp_path / "absent") == ""
        (tmp_path / "plain").mkdir()
        assert self._sanitize(tmp_path / "plain") == ""

    def test_default_off_and_repo_path_quoted(self) -> None:
        assert AnySweAgentConfig.model_fields["sanitize_git_history"].default is False
        assert "cd '/my repo'" in AnySweAgent._git_history_sanitizer_script("/my repo", 60)

    @pytest.mark.parametrize("enabled", [False, True])
    def test_agent_sandbox_runs_the_sanitizer_only_when_enabled(self, tmp_path: Path, enabled: bool) -> None:
        commands: list[str] = []

        class _FakeSandbox:
            def __init__(self, provider, spec) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def exec(self, command: str, **kwargs):
                commands.append(command)
                return SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)

            async def download(self, remote: str, local: Path) -> None:
                pass

        params = AnySweInstanceConfig(
            **_config(sanitize_git_history=enabled, sandbox_spec={"workdir": "/workspace/repo"}).model_dump(),
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://policy:8000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
            problem_info={"instance_id": "astropy__astropy-12907"},
            body={"input": "fix it", "model": "policy"},
            persistent_dir=tmp_path,
            metrics_fpath=tmp_path / "metrics.json",
            container="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        )
        params.metrics_fpath.write_text("{}")
        (tmp_path / "instruction.txt").write_text("fix it")
        (tmp_path / "agent_runner.py").write_text("")
        agent = AnySweAgent.__new__(AnySweAgent)
        object.__setattr__(
            agent, "__dict__", {"config": params, "server_client": SimpleNamespace(global_config_dict={})}
        )

        with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", _FakeSandbox):
            asyncio.run(agent._run_agent_in_sandbox(params))

        sanitizer = [command for command in commands if "git gc --prune=now" in command]
        if enabled:
            assert len(sanitizer) == 1 and "cd /workspace/repo" in sanitizer[0]
            assert (tmp_path / "git_sanitize.txt").exists()
        else:
            assert sanitizer == []
