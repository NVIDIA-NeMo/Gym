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
"""Terminal-answer ("message expectation") rows for the image_tools pivot env.

A row whose expected_action is the __answer__ sentinel asks the model to STOP
calling tools and answer. Unlike the generic pivot server, which pays 1.0 for
any chat message, reward here is conditional on the answer being correct --
otherwise the objective rewards giving up early.
"""
import ast
import pathlib
import re
import unicodedata

import pytest

# Importing app.py pulls in fastapi/ray/omegaconf via nemo_gym, which are not
# present outside the server venv. The reward semantics under test are pure, so
# load just those definitions from the source instead -- this keeps the test
# runnable anywhere and still fails if app.py's logic changes.
_APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
_ns = {"re": re, "unicodedata": unicodedata, "Optional": __import__("typing").Optional}
_tree = ast.parse(_APP.read_text(encoding="utf-8"))
_want_fn = {"extract_final_answer", "answers_match"}
for _node in _tree.body:
    if isinstance(_node, ast.FunctionDef) and _node.name in _want_fn:
        exec(compile(ast.Module([_node], []), "app", "exec"), _ns)
    elif isinstance(_node, ast.Assign) and any(
        getattr(t, "id", None) in ("_ANSWER_ACTION", "_BOXED_RE") for t in _node.targets
    ):
        exec(compile(ast.Module([_node], []), "app", "exec"), _ns)
    elif isinstance(_node, ast.ClassDef) and _node.name == "FailureCode":
        _src = ast.get_source_segment(_APP.read_text(encoding="utf-8"), _node)
        exec("from enum import Enum\n" + _src, _ns)

extract_final_answer = _ns["extract_final_answer"]
answers_match = _ns["answers_match"]
_ANSWER_ACTION = _ns["_ANSWER_ACTION"]
FailureCode = _ns["FailureCode"]


# --- extraction ------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>maybe 3</think>\nThe answer is \\boxed{4}", "4"),
        ("<think>x</think>\n\\boxed{(5,16)}", "(5,16)"),
        ("<think>y</think>\nFinal answer: cricket ball", "cricket ball"),
        ("<think>z</think>\nAnswer: left.", "left"),
        # last boxed wins when several appear after </think>
        ("<think>t</think>\n\\boxed{1} then \\boxed{2}", "2"),
        # no answer at all
        ("<think>still thinking</think>\n", None),
        ("", None),
    ],
)
def test_extract_final_answer(text, expected):
    assert extract_final_answer(text) == expected


def test_boxed_inside_think_is_not_an_answer():
    """Mirrors the eval harness: only the segment after the last </think> counts,
    so a boxed answer stranded in the reasoning must not be extracted."""
    assert extract_final_answer("<think>\\boxed{7} hmm</think>\n") is None


def test_no_think_tag_still_extracts():
    """rsplit returns the whole string when </think> is absent."""
    assert extract_final_answer("\\boxed{9}") == "9"


# --- matching --------------------------------------------------------------

@pytest.mark.parametrize(
    "gold,got,ok",
    [
        ("4", "4", True),
        ("D", "d", True),                 # case-insensitive
        ("cricket ball", " Cricket  Ball ", True),   # whitespace + case
        ("left", "left.", True),          # trailing period
        ("6", '"6"', True),               # stray quotes
        ("6", "9", False),
        ("(5,16)", "(5,17)", False),
        ("4", None, False),
    ],
)
def test_answers_match(gold, got, ok):
    assert answers_match(gold, got) is ok


# --- the verify branch (source-level guards) --------------------------------
#
# The branch itself lives inside the server's async verify(), which cannot be
# imported here (fastapi/ray). These guards assert the properties that would
# silently break terminal rows if app.py were edited.

_SRC = _APP.read_text(encoding="utf-8")


def test_sentinel_is_distinct_from_any_real_tool():
    """The sentinel must never collide with a real tool name; every image tool
    ends in _tool, so a dunder-style sentinel is unambiguous."""
    assert _ANSWER_ACTION == "__answer__"
    assert not _ANSWER_ACTION.endswith("_tool")


def test_answer_branch_precedes_the_no_tool_call_guard():
    """Ordering is load-bearing: if the __answer__ check came after
    `if not rollout_calls`, every terminal row would be scored
    NO_TOOL_CALL_IN_ROLLOUT (reward 0) and the objective would be unchanged."""
    answer_branch = _SRC.index(f'if expected.get("name") == _ANSWER_ACTION')
    no_call_guard = _SRC.index("if not rollout_calls:")
    assert answer_branch < no_call_guard


def test_answer_branch_covers_all_three_outcomes():
    """A terminal row must distinguish: answered correctly, answered wrongly,
    never answered, and kept calling tools."""
    branch = _SRC[_SRC.index('if expected.get("name") == _ANSWER_ACTION'):]
    branch = branch[: branch.index("if not rollout_calls:")]
    for code in (
        "TOOL_CALL_WHEN_ANSWER_EXPECTED",
        "ANSWER_MISSING",
        "ANSWER_INCORRECT",
    ):
        assert code in branch, f"{code} not handled in the terminal branch"
    assert 'state["reward"] = 1.0' in branch


def test_wrong_answer_is_not_rewarded_like_the_generic_pivot_server():
    """The generic server pays 1.0 for *any* chat message. Here reward must be
    gated on answers_match, or the objective pays the model to guess early."""
    branch = _SRC[_SRC.index('if expected.get("name") == _ANSWER_ACTION'):]
    branch = branch[: branch.index("if not rollout_calls:")]
    assert "answers_match(gold, got)" in branch


def test_module_imports_every_name_it_uses_at_module_scope():
    """Regression guard for a real failure: the helpers used `re` and
    `unicodedata` while app.py imported neither, so the resources server died at
    startup with NameError -- after py_compile passed and after these tests
    passed, because the loader above injects those modules into the namespace.
    Assert the real file imports them.
    """
    tree = ast.parse(_SRC)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)
    for name in ("re", "unicodedata", "Optional"):
        assert name in imported, f"app.py uses {name} but never imports it"
