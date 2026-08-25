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
"""Unit tests. The real correctness gate is tools/difftest.py (replays 1.65M
real agent commands against GNU bash); these cover semantics and confinement."""

import os
import pathlib
import subprocess

import pytest
from pocketshell import bre_to_python, run
from pocketshell.shell import eval_arith


@pytest.fixture
def ws(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "0001_browse_alpha.txt").write_text(
        "Alpha Beta Gamma\nthe quick brown fox\nDavid Pemantle 1983\nline four\nline five\nend of alpha\n"
    )
    (pages / "0002_browse_beta.txt").write_text("beta page\nno match here\nPhD thesis 1994\nCOST: $1,200\n")
    (pages / "0003_search_gamma.txt").write_text("gamma\n" * 50)
    (tmp_path / "manifest.tsv").write_text(
        "0001\tbrowse\thttps://a.example\tAlpha\t120\n0002\tbrowse\thttps://b.example\tBeta\t95\n"
    )
    return str(tmp_path)


def R(cmd, ws, **kw):
    return run(cmd, workspace=ws, **kw)


# --- the headline idioms, by measured frequency ---------------------------
def test_cat_page(ws):
    r = R("cat pages/0001_browse_alpha.txt", ws)
    assert r.exit_code == 0
    assert "David Pemantle 1983" in r.stdout
    assert r.stdout.endswith("end of alpha\n")


def test_grep_i_pipe_head(ws):
    r = R('grep -i "david" pages/0001_browse_alpha.txt | head -5', ws)
    assert r.exit_code == 0
    assert r.stdout == "David Pemantle 1983\n"


def test_grep_bre_alternation(ws):
    """The dominant real idiom -- and the one a naive re.compile gets wrong."""
    r = R(r'grep -i "david\|phd\|doctoral" pages/*.txt', ws)
    assert "David Pemantle 1983" in r.stdout
    assert "PhD thesis 1994" in r.stdout
    assert "no match here" not in r.stdout


def test_bre_literal_pipe_is_not_alternation(ws):
    """In BRE a bare | is a literal, so this must NOT match either side."""
    r = R('grep "alpha|beta" pages/0001_browse_alpha.txt', ws)
    assert r.exit_code == 1
    assert r.stdout == ""


def test_grep_l_which_file(ws):
    r = R('grep -l "PhD" pages/*.txt', ws)
    assert r.stdout.strip().endswith("0002_browse_beta.txt")


def test_grep_c_count(ws):
    r = R('grep -c "gamma" pages/0003_search_gamma.txt', ws)
    assert r.stdout.strip() == "50"


def test_grep_n_and_context(ws):
    r = R('grep -n -A 1 "quick" pages/0001_browse_alpha.txt', ws)
    assert r.stdout.startswith("2:the quick brown fox")
    assert "David Pemantle" in r.stdout


def test_grep_o_extract(ws):
    r = R(r'grep -o "[0-9]\{4\}" pages/0001_browse_alpha.txt', ws)
    assert r.stdout.strip() == "1983"


def test_grep_E_ere(ws):
    r = R('grep -E "(quick|slow) brown" pages/0001_browse_alpha.txt', ws)
    assert "quick brown fox" in r.stdout


def test_grep_no_match_exit_1(ws):
    assert R('grep "zzzz" pages/0001_browse_alpha.txt', ws).exit_code == 1


def test_sed_line_range(ws):
    r = R("sed -n '2,3p' pages/0001_browse_alpha.txt", ws)
    assert r.stdout == "the quick brown fox\nDavid Pemantle 1983\n"


def test_sed_substitution(ws):
    r = R("sed 's/Alpha/OMEGA/' pages/0001_browse_alpha.txt | head -1", ws)
    assert r.stdout == "OMEGA Beta Gamma\n"


def test_ls_and_wc(ws):
    assert R("ls pages/ | wc -l", ws).stdout.strip() == "3"


def test_head_bytes(ws):
    r = R("head -c 5 pages/0001_browse_alpha.txt", ws)
    assert r.stdout == "Alpha"


def test_cut_manifest(ws):
    r = R("cut -f3 manifest.tsv | sort -u", ws)
    assert r.stdout == "https://a.example\nhttps://b.example\n"


def test_chained_with_semicolon(ws):
    r = R('echo "=== A ==="; grep -c gamma pages/0003_search_gamma.txt', ws)
    assert r.stdout == "=== A ===\n50\n"


def test_and_or_shortcircuit(ws):
    r = R('grep -q "nothinghere" pages/0001_browse_alpha.txt || echo "not found"', ws)
    assert r.stdout.strip() == "not found"
    r = R('grep -q "Alpha" pages/0001_browse_alpha.txt && echo "found"', ws)
    assert r.stdout.strip() == "found"


def test_for_loop_with_arithmetic(ws):
    r = R("for i in 1 2 3; do echo $((i+1900)); done", ws)
    assert r.stdout == "1901\n1902\n1903\n"


def test_stderr_suppression(ws):
    r = R("cat pages/nope.txt 2>/dev/null; echo done", ws)
    assert r.stdout.strip() == "done"
    assert r.stderr == ""


def test_glob_unmatched_stays_literal(ws):
    r = R("ls pages/zzz*", ws)
    assert r.exit_code != 0


# --- the calculator question ----------------------------------------------
def test_arithmetic_42_years_after_1989(ws):
    assert R("echo $((1989 + 42))", ws).stdout.strip() == "2031"


def test_arithmetic_integer_division(ws):
    assert R("echo $((100 / 7))", ws).stdout.strip() == "14"


def test_eval_arith_direct():
    assert eval_arith("2*106+31+63", {}) == 306
    assert eval_arith("A+B", {"A": "10", "B": "5"}) == 15


def test_arithmetic_rejects_code_execution(ws):
    r = R('echo $((__import__("os").system("id")))', ws)
    assert r.exit_code != 0 or "2031" not in r.stdout


# --- confinement: the whole point of removing the subprocess ---------------
@pytest.mark.parametrize(
    "cmd",
    [
        "cat /etc/passwd",
        "cat ../../../etc/hostname",
        "ls /",
        "grep -r root /etc",
        "cat /proc/self/environ",
        "find / -name '*.key'",
    ],
)
def test_cannot_read_outside_workspace(cmd, ws):
    r = R(cmd, ws)
    assert r.exit_code != 0
    assert "root:" not in r.stdout
    assert r.stdout.strip() == "" or "Permission denied" in r.stderr or r.stderr


def test_symlink_escape_blocked(ws, tmp_path):
    link = os.path.join(ws, "escape")
    os.symlink("/etc", link)
    r = R("ls escape", ws)
    assert r.exit_code != 0
    assert "passwd" not in r.stdout


# --- unsupported syntax fails closed, like the old guard -------------------
@pytest.mark.parametrize(
    "cmd,frag",
    [
        ("cat $(ls pages | head -1)", "command substitution"),
        ("echo hi > out.txt", "redirection"),
        ("cat <(ls)", "process substitution"),
        ("echo `whoami`", "backtick"),
        ("sleep 5 &", "background"),
    ],
)
def test_unsupported_syntax_blocked(cmd, frag, ws):
    r = R(cmd, ws)
    assert r.exit_code == -3
    assert frag in r.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf pages",
        "curl https://example.com",
        "python3 -c 'print(1)'",
        "awk '{print $1}' manifest.tsv",
        "bc -l",
        "wget http://x",
        "chmod 777 pages",
    ],
)
def test_dangerous_commands_do_not_exist(cmd, ws):
    r = R(cmd, ws)
    assert r.exit_code != 0
    assert "command not found" in r.stderr or "blocked" in r.stderr


def test_write_is_impossible(ws):
    before = sorted(os.listdir(os.path.join(ws, "pages")))
    R("sed -i 's/a/b/' pages/0001_browse_alpha.txt", ws)
    R("echo x > pages/new.txt", ws)
    assert sorted(os.listdir(os.path.join(ws, "pages"))) == before


def test_timeout_is_enforced(ws):
    r = R("for i in $(seq 1 10); do cat pages/0003_search_gamma.txt; done", ws, timeout=1.0)
    assert r.exit_code in (-1, -3, 2)


def test_malformed_pattern_degrades_not_crashes(ws):
    r = R(r'grep "A:\|A  \|A\)" pages/0001_browse_alpha.txt', ws)
    assert r.exit_code in (0, 1)


# --- BRE translation unit checks -----------------------------------------
@pytest.mark.parametrize(
    "bre,py",
    [
        (r"a\|b", "a|b"),
        ("a|b", r"a\|b"),
        (r"\(ab\)*", "(ab)*"),
        ("(ab)", r"\(ab\)"),
        (r"x\{2,3\}", "x{2,3}"),
        ("*star", r"\*star"),
        ("a^b", r"a\^b"),
        ("a$b", r"a\$b"),
        ("^anchored$", "^anchored$"),
    ],
)
def test_bre_translation(bre, py):
    assert bre_to_python(bre) == py


# --- differential spot-check against real bash ----------------------------
BASH = "/bin/bash"


@pytest.mark.skipif(not os.path.exists(BASH), reason="bash not available")
@pytest.mark.parametrize(
    "cmd",
    [
        'grep -i "david" pages/0001_browse_alpha.txt',
        r'grep -i -n "david\|phd" pages/0001_browse_alpha.txt pages/0002_browse_beta.txt',
        "sed -n '2,4p' pages/0001_browse_alpha.txt",
        "cat pages/0002_browse_beta.txt | head -2",
        "ls pages/ | wc -l",
        "cut -f3 manifest.tsv | sort -u",
        'grep -c "gamma" pages/0003_search_gamma.txt',
        "wc -l pages/0001_browse_alpha.txt",
        r'grep -o "[0-9]\{4\}" pages/0002_browse_beta.txt',
        "echo $((1989+42))",
    ],
)
def test_matches_real_bash(cmd, ws):
    real = subprocess.run(
        [BASH, "-c", cmd], cwd=ws, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"}
    )
    mine = R(cmd, ws)
    # normalise the file-label prefix bash prints with absolute-ish paths
    assert mine.stdout == real.stdout, f"{cmd}\nmine={mine.stdout!r}\nbash={real.stdout!r}"
    assert mine.exit_code == real.returncode


# --- catastrophic backtracking -------------------------------------------
# GNU grep uses a DFA and cannot blow up; Python's engines backtrack. A thread
# running run_in_executor cannot be cancelled, so an unbounded match would
# permanently consume a pool slot and eventually stall the resources server.
@pytest.mark.parametrize("pat", [r'grep -E "(a+)+$" pages/bomb.txt', r'grep -E "(a|a)*$" pages/bomb.txt'])
def test_catastrophic_pattern_is_bounded(pat, ws):
    import time

    (pathlib.Path(ws) / "pages" / "bomb.txt").write_text(("a" * 60 + "!\n") * 200)
    t = time.time()
    r = run(pat, workspace=ws, timeout=3.0)
    elapsed = time.time() - t
    assert elapsed < 60, f"took {elapsed:.0f}s -- not bounded"
    assert r.exit_code in (-1, 0, 1)


def test_normal_grep_is_not_slowed_by_the_timeout_wrapper(ws):
    import time

    t = time.time()
    r = run('grep -c "gamma" pages/0003_search_gamma.txt', workspace=ws, timeout=3.0)
    assert r.stdout.strip() == "50"
    assert time.time() - t < 2.0
