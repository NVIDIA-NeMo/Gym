# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from pathlib import Path

import pytest


WRAPPER = Path(__file__).resolve().parents[1] / "openclaw" / "path_shadow_wrapper.py"


def _run_as(
    basename: str, *args, env: dict | None = None, extra_path_dirs: list[str] | None = None
) -> subprocess.CompletedProcess:
    """Invoke wrapper.py via symlink basename in a tmpdir wrapper dir."""
    import tempfile

    with tempfile.TemporaryDirectory() as wrapper_dir:
        link = Path(wrapper_dir) / basename
        os.symlink(str(WRAPPER), str(link))
        path = ":".join((extra_path_dirs or []) + [os.environ.get("PATH", "")])
        full_env = {**os.environ, "PATH": path, **(env or {})}
        return subprocess.run(
            [str(link), *args],
            capture_output=True,
            text=True,
            env=full_env,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["curl", "http://example.com"],
        ["wget", "http://example.com"],
        ["killall", "python"],
        ["shutdown", "-h", "now"],
        ["pkill", "-f", "python"],
        ["reboot"],
        ["poweroff"],
        ["halt"],
    ],
)
def test_always_deny_names_blocked(argv):
    # One case per ALWAYS_DENY exact command name in path_shadow_wrapper.py.
    r = _run_as(*argv)
    assert r.returncode == 1
    assert "blocked" in r.stderr.lower()


@pytest.mark.parametrize(
    "argv",
    [
        # git network verbs (fetch/pull/clone) — PATTERN_DENY["git"] #1
        ["git", "fetch", "origin", "main"],
        ["git", "pull"],
        ["git", "clone", "https://github.com/foo/bar.git"],
        # git remote add/set-url/etc. — PATTERN_DENY["git"] #2
        ["git", "remote", "add", "origin", "https://x"],
        # git submodule add/update/sync/init — PATTERN_DENY["git"] #3
        ["git", "submodule", "update", "--init"],
        # git command with remote http(s)/ssh URL — PATTERN_DENY["git"] #5
        ["git", "show", "https://github.com/x/y"],
        # git command with git@host: URL — PATTERN_DENY["git"] #6
        ["git", "fetch", "git@github.com:psf/requests.git"],
        # git archive --remote — PATTERN_DENY["git"] #4
        ["git", "archive", "--remote=origin", "master"],
        # git command referencing a remote-tracking ref (origin/...) — PATTERN_DENY["git"] #7
        ["git", "diff", "origin/main"],
        # rm -rf / — PATTERN_DENY["rm"] #1
        ["rm", "-rf", "/"],
        # rm of critical system directory — PATTERN_DENY["rm"] #2
        ["rm", "-rf", "/etc"],
        # dd to a block device — PATTERN_DENY["dd"]
        ["dd", "if=/dev/zero", "of=/dev/sda"],
        # tmux kill-server — PATTERN_DENY["tmux"]
        ["tmux", "kill-server"],
        # init 0 — PATTERN_DENY["init"]
        ["init", "0"],
        # kill with command substitution $(...) — PATTERN_DENY["kill"] #1
        ["kill", "$(pidof python)"],
        # kill with backtick command substitution — PATTERN_DENY["kill"] #2
        ["kill", "`pidof python`"],
        # kill with a shell variable — PATTERN_DENY["kill"] #3
        ["kill", "$PID"],
        # kill -1 (kill all user processes) — PATTERN_DENY["kill"] #4
        ["kill", "-1", "1234"],
        # kill 0 (kill process group) — PATTERN_DENY["kill"] #5
        ["kill", "0"],
        # kill with negative PID (process group) — PATTERN_DENY["kill"] #6
        ["kill", "-200"],
    ],
)
def test_pattern_deny_commands_blocked(argv):
    # One case per distinct PATTERN_DENY regex (every rule variant across git/rm/dd/tmux/init/kill is covered).
    r = _run_as(*argv)
    assert r.returncode == 1
    assert "blocked" in r.stderr.lower()


def test_install_creates_symlinks(tmp_path):
    # Run with --install to populate symlinks in a tmpdir
    target = tmp_path / "bin"
    target.mkdir()
    # The wrapper requires wrapper.py to be present in target before --install
    wrapper_copy = target / "wrapper.py"
    wrapper_copy.write_text(WRAPPER.read_text())
    result = subprocess.run(
        [sys.executable, str(wrapper_copy), "--install", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # Spot-check a sampling of expected basenames
    for name in ("git", "curl", "wget", "rm", "kill", "killall", "shutdown", "dd", "tmux", "init"):
        link = target / name
        assert link.is_symlink(), f"{name} symlink missing"
        # Relative symlink so it survives bind-mounts
        assert os.readlink(str(link)) == "wrapper.py"


def test_install_rejects_missing_wrapper_in_target(tmp_path):
    target = tmp_path / "bin"
    target.mkdir()
    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--install", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "wrapper.py must already exist" in result.stderr


def test_list_names_outputs_all_wrapped():
    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--list-names"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    names = set(result.stdout.split())
    for required in ("git", "curl", "wget", "rm", "kill", "killall", "shutdown", "dd", "tmux", "init"):
        assert required in names


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_allowed_git_command_passes_through_to_real_git():
    # A git invocation matching no deny pattern must exec the real git and
    # propagate its output + exit code. This exercises the allow path (exec_real),
    # and guards against a deny-pattern false-positive silently blocking normal git.
    r = _run_as("git", "--version")
    assert r.returncode == 0, r.stderr
    assert "git version" in r.stdout.lower()
    assert "blocked" not in r.stderr.lower()


def test_unwrapped_basename_errors(tmp_path):
    # Invoke via a symlink for a name that has no rules registered
    target = tmp_path / "bin"
    target.mkdir()
    link = target / "echo"
    os.symlink(str(WRAPPER), str(link))
    r = subprocess.run([str(link), "hello"], capture_output=True, text=True)
    assert r.returncode == 2
    assert "no rules registered" in r.stderr
