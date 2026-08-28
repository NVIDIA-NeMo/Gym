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
"""Integration test for the in-process bash path.

Exercises the ACTUAL patched `_run_bash_readonly` (async wrapper, semaphore,
asyncio.wait_for, truncation, legacy-guard passthrough) rather than pocketshell
in isolation -- a 2-sample eval smoke can easily make zero bash_command calls,
as run_6312596 did, so it cannot gate this change.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as harness  # noqa: E402


class _Stub:
    """Minimal stand-in: _run_bash_readonly only touches _bash_semaphore."""

    def __init__(self):
        self._bash_semaphore = asyncio.Semaphore(8)


def run_bash(keystrokes, workspace, duration=10.0):
    stub = _Stub()
    bound = types.MethodType(harness.TavilySearchResourcesServer._run_bash_readonly, stub)
    return asyncio.run(bound(keystrokes, duration, str(workspace)))


@pytest.fixture
def ws(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "0001_browse_alpha.txt").write_text("Alpha Beta\nDavid Pemantle 1983\nPhD thesis 1994\nlast line\n")
    (pages / "0002_browse_beta.txt").write_text("beta\nno match\n")
    (tmp_path / "manifest.tsv").write_text("0001\tbrowse\thttps://a.example\tAlpha\t120\n")
    return tmp_path


def test_grep_returns_output_and_exit_code(ws):
    out = run_bash('grep -i "david" pages/0001_browse_alpha.txt', ws)
    assert "David Pemantle 1983" in out
    assert "[exit_code=0]" in out


def test_bre_alternation_through_the_harness(ws):
    out = run_bash(r'grep -i "david\|phd" pages/*.txt', ws)
    assert "David Pemantle 1983" in out and "PhD thesis 1994" in out


def test_no_match_reports_exit_1(ws):
    assert "[exit_code=1]" in run_bash('grep "zzzz" pages/0001_browse_alpha.txt', ws)


def test_pipeline_and_sequencing(ws):
    out = run_bash('echo "=== A ==="; grep -c "" pages/0001_browse_alpha.txt', ws)
    assert "=== A ===" in out and "4" in out


def test_sed_range_and_arithmetic(ws):
    assert "David Pemantle 1983" in run_bash("sed -n '2,2p' pages/0001_browse_alpha.txt", ws)
    assert "2031" in run_bash("echo $((1989+42))", ws)


def test_legacy_guard_still_blocks_with_identical_string(ws):
    """The [blocked: ...] wording is what the SDG corpus and the SFT models saw."""
    for cmd, frag in [
        ("curl https://example.com", "command 'curl' is blocked"),
        ("python3 -c 'print(1)'", "command 'python3' is blocked"),
        ("cat $(ls pages)", "$(...) command substitution"),
        ("echo hi > f.txt", "output redirection to a file"),
    ]:
        out = run_bash(cmd, ws)
        assert out.startswith("[blocked: "), out[:80]
        assert frag in out, (cmd, out[:120])
        assert "[exit_code=-3]" in out


def test_stderr_section_is_rendered(ws):
    out = run_bash("cat pages/missing.txt", ws)
    assert "--- stderr ---" in out
    assert "[exit_code=" in out


def test_cannot_escape_the_workspace(ws):
    out = run_bash("cat /etc/passwd", ws)
    assert "root:" not in out
    assert "Permission denied" in out


def test_output_is_truncated_not_unbounded(ws):
    big = ws / "pages" / "big.txt"
    big.write_text("x" * 200 + "\n" * 1 + ("y" * 200 + "\n") * 5000)
    out = run_bash("cat pages/big.txt", ws)
    assert len(out) < 200_000
    assert "truncated" in out


def test_catastrophic_pattern_cannot_wedge_the_worker(ws):
    import time

    (ws / "pages" / "bomb.txt").write_text(("a" * 60 + "!\n") * 200)
    t = time.time()
    out = run_bash(r'grep -E "(a|a)*$" pages/bomb.txt', ws, duration=3.0)
    assert time.time() - t < 90
    assert "[exit_code=" in out
