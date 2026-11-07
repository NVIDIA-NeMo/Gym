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

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from responses_api_agents.anyswe_agent.agent_runner import _extract_patch, _snapshot_repo
from responses_api_agents.anyswe_agent.app import AnySweAgent


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def _repo(root: Path) -> tuple[Path, str, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "lib.py").write_text("bug\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "lib.py").write_text("fixed\n")
    _git(repo, "commit", "-qam", "future")
    future = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, base, future


def _command(repo: Path, base: str, root: Path) -> str:
    return "sh -c " + shlex.quote(AnySweAgent._git_history_sanitizer_script(str(repo), 60)) + " || exit 1"


@pytest.mark.parametrize(
    "layout", ["linked", "primary_with_sibling", "alternates", "separate_gitdir", "symlink_gitdir"]
)
def test_shared_storage_rejected_before_mutation(tmp_path: Path, layout: str) -> None:
    repo, base, future = _repo(tmp_path)
    task = repo
    if layout in ("linked", "primary_with_sibling"):
        sibling = tmp_path / "sibling"
        _git(repo, "worktree", "add", "--detach", str(sibling), base)
        if layout == "linked":
            task = sibling
    elif layout == "alternates":
        task = tmp_path / "task"
        _git(tmp_path, "clone", "--shared", str(repo), str(task))
        _git(task, "checkout", "--detach", base)
    elif layout == "separate_gitdir":
        task = tmp_path / "task"
        _git(tmp_path, "clone", "--separate-git-dir", str(tmp_path / "gitdir"), str(repo), str(task))
    elif layout == "symlink_gitdir":
        (repo / ".git").rename(tmp_path / "gitdir")
        (repo / ".git").symlink_to(tmp_path / "gitdir", target_is_directory=True)
    before = _git(repo, "for-each-ref").stdout
    command = _command(task, base, tmp_path)
    # A later best-effort step must not swallow the sanitizer's error and execute the agent.
    command += f" && false || true; touch {shlex.quote(str(tmp_path / 'agent-started'))}"
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "agent-started").exists()
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == future
    assert _git(repo, "for-each-ref").stdout == before
    assert _git(repo, "show", f"{future}:lib.py").stdout == "fixed\n"


def test_unreachable_kept_pack_stops_agent(tmp_path: Path) -> None:
    repo, base, future = _repo(tmp_path)
    _git(repo, "repack", "-ad")
    for pack in (repo / ".git/objects/pack").glob("*.pack"):
        pack.with_suffix(".keep").write_text("preserve pack\n")
    _git(repo, "checkout", "--detach", base)
    command = _command(repo, base, tmp_path) + f"; touch {shlex.quote(str(tmp_path / 'agent-started'))}"
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "agent-started").exists()
    # It must report the unsupported retention, not pretend the still-readable object is gone.
    assert _git(repo, "cat-file", "-e", future, check=False).returncode == 0


def test_retained_ref_is_literal(tmp_path: Path) -> None:
    repo, base, future = _repo(tmp_path)
    _git(repo, "branch", "releaseX1", future)
    _git(repo, "checkout", "-b", "release.1", base)
    subprocess.run(["sh", "-c", AnySweAgent._git_history_sanitizer_script(str(repo), 60)], check=True)
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == base
    assert _git(repo, "cat-file", "-e", future, check=False).returncode != 0


def test_ownership_error_preserves_repository(tmp_path: Path) -> None:
    repo, base, future = _repo(tmp_path)
    env = {
        **os.environ,
        "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    result = subprocess.run(
        ["sh", "-c", AnySweAgent._git_history_sanitizer_script(str(repo), 60)], env=env, capture_output=True
    )
    assert result.returncode != 0
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == future


def test_failed_fallback_cannot_report_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, base, _ = _repo(tmp_path)
    real_git = shutil.which("git")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "git"
    wrapper.write_text('#!/bin/sh\ncase "$1" in gc|init) exit 1;; esac\n' + f'exec {shlex.quote(real_git)} "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        ["sh", "-c", AnySweAgent._git_history_sanitizer_script(str(repo), 60)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "git history sanitized:" not in result.stdout


def test_fallback_preserves_ignored_tracked_files_for_patch_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, _ = _repo(tmp_path)
    unusual_name = "tracked [name]\nfile.py"
    (repo / ".gitignore").write_text("*.py\n")
    (repo / unusual_name).write_text("before\n")
    (repo / "deleted.py").write_text("deleted before cleanup\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "add", "-f", "--", unusual_name, "deleted.py")
    _git(repo, "commit", "-qm", "track ignored files")
    (repo / "deleted.py").unlink()
    (repo / "cache.py").write_text("ignored and never tracked\n")
    real_git = shutil.which("git")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "git"
    wrapper.write_text('#!/bin/sh\nif [ "$1" = gc ]; then exit 1; fi\n' + f'exec {shlex.quote(real_git)} "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])

    subprocess.run(["sh", "-c", AnySweAgent._git_history_sanitizer_script(str(repo), 60)], check=True)
    index_path = tmp_path / "capture-index"
    baseline = _snapshot_repo(repo, index_path)
    (repo / "lib.py").write_text("agent fix\n")
    (repo / unusual_name).write_text("after\n")
    patch = _extract_patch(repo, index_path, baseline)

    assert b"+agent fix\n" in patch
    assert b"+after\n" in patch
    assert set(_git(repo, "ls-files", "-z").stdout.rstrip("\0").split("\0")) == {
        ".gitignore",
        "lib.py",
        unusual_name,
    }
