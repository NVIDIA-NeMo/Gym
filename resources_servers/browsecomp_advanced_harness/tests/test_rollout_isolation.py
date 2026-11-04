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
"""Per-rollout workspace isolation (GRPO num_repeats>1).

Pins the guarantee that two rollouts of the SAME sample cannot read each
other's workspaces. Under GRPO each /run gets a fresh session uuid, so
workspaces are sibling dirs under one workspace_root:

    workspace_root/<sid_rollout_A>/pages/...
    workspace_root/<sid_rollout_B>/pages/...

If rollout B could read A's pages, B's reward would reflect A's work and
the within-group advantage estimates would be poisoned. The confinement
is pocketshell's Workspace root (= this rollout's dir), so escapes via
relative paths, absolute paths, globs and symlinks must all fail.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as harness  # noqa: E402


class _Stub:
    def __init__(self):
        self._bash_semaphore = asyncio.Semaphore(8)


def run_bash(keystrokes, workspace, duration=10.0):
    stub = _Stub()
    bound = types.MethodType(harness.TavilySearchResourcesServer._run_bash_readonly, stub)
    return asyncio.run(bound(keystrokes, duration, str(workspace)))


SECRET = "SIBLING-ROLLOUT-SECRET-9f3a"


@pytest.fixture
def two_rollouts(tmp_path):
    """workspace_root with two sibling per-rollout workspaces of one sample."""
    ws_a = tmp_path / "sid-rollout-a"
    ws_b = tmp_path / "sid-rollout-b"
    for ws in (ws_a, ws_b):
        (ws / "pages").mkdir(parents=True)
    (ws_a / "pages" / "0001_browse_answer.txt").write_text(f"the answer is {SECRET}\n")
    (ws_b / "pages" / "0001_browse_own.txt").write_text("rollout b's own page\n")
    return ws_a, ws_b


def test_own_workspace_still_readable(two_rollouts):
    ws_a, _ = two_rollouts
    out = run_bash("cat pages/0001_browse_answer.txt", ws_a)
    assert SECRET in out and "[exit_code=0]" in out


def test_sibling_relative_path_is_blocked(two_rollouts):
    ws_a, ws_b = two_rollouts
    out = run_bash("cat ../sid-rollout-a/pages/0001_browse_answer.txt", ws_b)
    assert SECRET not in out
    assert "[exit_code=0]" not in out


def test_sibling_absolute_path_is_blocked(two_rollouts):
    ws_a, ws_b = two_rollouts
    out = run_bash(f"cat {ws_a}/pages/0001_browse_answer.txt", ws_b)
    assert SECRET not in out
    assert "[exit_code=0]" not in out


def test_sibling_glob_is_blocked(two_rollouts):
    ws_a, ws_b = two_rollouts
    for cmd in ("cat ../*/pages/*.txt", "grep -r SIBLING ..", f"grep -r SIBLING {ws_a.parent}"):
        out = run_bash(cmd, ws_b)
        assert SECRET not in out, cmd


def test_symlink_into_sibling_is_blocked(two_rollouts):
    """A symlink planted inside B pointing at A must not pierce confinement
    (Workspace.resolve follows symlinks BEFORE the containment check)."""
    ws_a, ws_b = two_rollouts
    (ws_b / "pages" / "sneaky.txt").symlink_to(ws_a / "pages" / "0001_browse_answer.txt")
    out = run_bash("cat pages/sneaky.txt", ws_b)
    assert SECRET not in out


def test_page_writer_dirs_are_distinct_per_session(tmp_path, monkeypatch):
    """_get_page_writer(sid) must key strictly on the session id: distinct
    sids -> distinct dirs, same sid -> same writer (stable within a rollout)."""
    stub = types.SimpleNamespace(
        config=types.SimpleNamespace(workspace="per_session"),
        _session_workspaces={},
        _workspace_root=str(tmp_path),
    )
    get_pw = types.MethodType(harness.TavilySearchResourcesServer._get_page_writer, stub)
    pw_a, pw_b = get_pw("sid-rollout-a"), get_pw("sid-rollout-b")
    assert pw_a.workspace != pw_b.workspace
    assert pw_a.workspace.parent == pw_b.workspace.parent == tmp_path
    assert get_pw("sid-rollout-a") is pw_a
