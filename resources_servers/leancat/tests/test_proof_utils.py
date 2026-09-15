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

import pytest

from resources_servers.leancat.proof_utils import (
    check_statement_preserved,
    extract_lean_code,
    find_banned_tokens,
)


# LeanCat problem 0001, verbatim.
REFERENCE = """import Mathlib

open CategoryTheory

variable {C : Type*} [Category.{v} C]

theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) : α ≫ β = β ≫ α := by
  sorry"""

SOLVED = """import Mathlib

open CategoryTheory

variable {C : Type*} [Category.{v} C]

theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) : α ≫ β = β ≫ α := by
  ext X
  exact (α.naturality (β.app X)).symm"""


TRIVIAL = "theorem t : True := trivial"
FILE = "import Mathlib\n\ntheorem t : True := trivial"


@pytest.mark.parametrize(
    "text,expected",
    [
        # Upstream's rule: the last fenced block wins.
        ("First try:\n```lean4\nbad\n```\nActually:\n```lean4\ngood\n```", "good"),
        (f"```Lean4\n{TRIVIAL}\n```", TRIVIAL),
        (f"```\n{TRIVIAL}\n```", TRIVIAL),
        # Reasoning is dropped before a block is chosen, so a `sorry` sketched while
        # thinking cannot beat the real answer.
        (f"<think>```lean4\nsorry\n```</think>\n```lean4\n{TRIVIAL}\n```", TRIVIAL),
        # StepFun-style: the final file arrives bare after `</think>`, with no fence.
        (f"<think>```lean4\nsorry\n```</think>\n{FILE}\n", FILE),
        # DeepSeek-R1-style templates open `<think>` in the prompt; only the close is echoed.
        (f"reasoning <sketch>sorry</sketch>\n</think>\n{FILE}\n", FILE),
        # Prose after the trace is not a Lean file, so the fence inside it is all there is.
        (f"<think>```lean4\n{TRIVIAL}\n```</think>\nThat should do it.", TRIVIAL),
        # An unclosed trace is still mined for a fence, but yields nothing without one.
        ("<think>still reasoning ```lean4\nimport Mathlib\n```", "import Mathlib"),
        ("<think>still reasoning, no code", ""),
        (f"  {TRIVIAL}  ", TRIVIAL),
        ("", ""),
    ],
)
def test_extract_lean_code(text, expected):
    assert extract_lean_code(text) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ("theorem t : True := by sorry", ["sorry"]),
        ("axiom bad : False\nunsafe def f := 1", ["axiom", "unsafe"]),
        ("theorem t : True := by admit", ["admit"]),
        (SOLVED, []),
        # Comments and string literals are blanked before the scan.
        ("-- no sorry here\ntheorem t : True := trivial", []),
        ('def msg := "sorry"', []),
        # `\b` must not match inside a snake_case identifier.
        ("theorem no_sorry_needed : True := trivial", []),
    ],
)
def test_find_banned_tokens(code, expected):
    assert find_banned_tokens(code) == expected


@pytest.mark.parametrize(
    "submission",
    [
        SOLVED,
        # Re-wrapping the signature across lines is whitespace, not tampering.
        SOLVED.replace(
            "theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) : α ≫ β = β ≫ α := by",
            "theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) :\n    α ≫ β = β ≫ α := by",
        ),
        SOLVED.replace("theorem id_comm", "-- the main result\ntheorem id_comm"),
        # The LeanCat prompt explicitly allows auxiliary declarations before the target.
        SOLVED.replace("theorem id_comm", "lemma helper : True := trivial\n\ntheorem id_comm"),
    ],
    ids=["faithful", "rewrapped-signature", "added-comment", "auxiliary-lemma"],
)
def test_check_statement_preserved_accepts(submission):
    assert check_statement_preserved(REFERENCE, submission)[0]


@pytest.mark.parametrize(
    "submission,reason_contains",
    [
        # Slipping in `(h : False)` makes the goal trivially provable.
        (SOLVED.replace("(α β : (𝟭 C) ⟶ (𝟭 C))", "(α β : (𝟭 C) ⟶ (𝟭 C)) (h : False)"), "Statement fragment"),
        (SOLVED.replace("α ≫ β = β ≫ α", "α ≫ β = α ≫ β"), "Statement fragment"),
        (SOLVED.replace("theorem id_comm", "theorem id_comm'"), "Statement fragment"),
        # Dropping the binder and proving it for one concrete category would compile.
        (SOLVED.replace("variable {C : Type*} [Category.{v} C]\n\n", ""), "Preamble line missing"),
    ],
    ids=["added-hypothesis", "altered-conclusion", "renamed-theorem", "dropped-binder"],
)
def test_check_statement_preserved_rejects(submission, reason_contains):
    ok, reason = check_statement_preserved(REFERENCE, submission)
    assert not ok
    assert reason_contains in reason


def test_check_statement_preserved_across_multiple_holes():
    """Nine problems carry more than one `sorry`; each fragment must match, in order."""
    reference = "import Mathlib\n\ntheorem a : True := by\n  sorry\n\ntheorem b : False ∨ True := by\n  sorry"
    good = "import Mathlib\n\ntheorem a : True := by\n  trivial\n\ntheorem b : False ∨ True := by\n  trivial"
    assert check_statement_preserved(reference, good)[0]

    # Second theorem dropped entirely: its fragment has nowhere to match.
    assert not check_statement_preserved(reference, "import Mathlib\n\ntheorem a : True := by\n  trivial")[0]

    # Swapping them fails even though both texts are present: order is part of the file
    # the model was told to copy back.
    swapped = "import Mathlib\n\ntheorem b : False ∨ True := by\n  trivial\n\ntheorem a : True := by\n  trivial"
    assert not check_statement_preserved(reference, swapped)[0]
