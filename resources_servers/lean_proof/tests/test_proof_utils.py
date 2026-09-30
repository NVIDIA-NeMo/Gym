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

from resources_servers.lean_proof.proof_utils import (
    DECLARED_SHORTCUT_TOKENS,
    check_statement_preserved,
    check_target_statement_preserved,
    extract_lean_code,
    find_banned_declarations,
    has_unterminated_block_comment,
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
def test_find_banned_declarations(code, expected):
    assert find_banned_declarations(code) == expected


@pytest.mark.parametrize(
    "text,expected,why",
    [
        (
            "Here is the proof:\n```lean4\n" + TRIVIAL + "\n```\nThen run:\n```bash\nlake build\n```",
            TRIVIAL,
            "a trailing shell-command fence must not become the answer",
        ),
        (
            "```\n" + TRIVIAL + "\n```",
            TRIVIAL,
            "an untagged fence is still upstream's last-fence candidate",
        ),
        (
            "```lean4 " + TRIVIAL + "```",
            "```lean4 " + TRIVIAL + "```",
            "upstream requires a newline after the tag, so a single-line fence is not a block",
        ),
    ],
)
def test_extraction_follows_upstreams_fence_rule(text, expected, why):
    """Parity with `scripts/eval_common.py`: loosening the rule changes which block wins."""
    assert extract_lean_code(text) == expected, why


@pytest.mark.parametrize(
    "answer,wins",
    [
        ("open CategoryTheory\n" + TRIVIAL, True),
        # Two namespaces on one `open` is ordinary Lean (problem 0003 does it).
        ("open CategoryTheory Limits\n" + TRIVIAL, True),
        (FILE, True),
        ("variable {C : Type*} [Category C]\n" + TRIVIAL, True),
        ("open the file and read the statement carefully.", False),
        ("def the target theorem is about limits, roughly.", False),
    ],
)
def test_unfenced_answer_only_beats_a_thinking_fence_when_it_looks_like_lean(answer, wins):
    """Prose after `</think>` must not displace the fenced code inside it.

    StepFun emits its final file bare, so a bare answer has to be able to win -- but only when
    it reads as a Lean file, or a sentence starting "open the file..." would be scored as one.
    """
    text = "<think>```lean4\nsorry\n```</think>\n" + answer
    extracted = extract_lean_code(text)
    assert (extracted == answer.strip()) == wins
    if not wins:
        assert extracted == "sorry", "the fence inside the reasoning is all there is"


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


# ──────────────────────────────────────────────────────────
# The "file keeps holes on purpose" variants (formal_conjectures)
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code,expected,why",
    [
        # `sorry` and `admit` are off the list: the file is allowed to keep the open
        # conjecture it sanity-checks.
        ("theorem open_conj : True := by sorry\ntheorem t : True := trivial", [], "other holes are legal"),
        ("axiom cheat : False", ["axiom"], "an added axiom is still a shortcut"),
        ("unsafe def f := 1", ["unsafe"], "so is unsafe"),
        # Declaration-anchored, so the word inside an identifier or mid-line is fine.
        ("theorem t : True := by exact Classical.axiom_of_choice", [], "axiom inside a name"),
        ("-- axiom bad : False\ntheorem t : True := trivial", [], "commented out"),
    ],
)
def test_find_banned_declarations_declarations_only(code, expected, why):
    assert find_banned_declarations(code, DECLARED_SHORTCUT_TOKENS, declarations_only=True) == expected, why


def test_default_banned_declarations_are_unchanged_by_the_new_parameters():
    """The leancat call path must behave exactly as before."""
    assert find_banned_declarations("theorem t : True := by sorry") == ["sorry"]
    assert find_banned_declarations("axiom cheat : False") == ["axiom"]
    # The default is a word-boundary scan, so it already ignores `axiom` inside an identifier;
    # what `declarations_only` adds is ignoring it mid-line as a standalone word.
    assert find_banned_declarations("theorem t : True := by exact Classical.axiom_of_choice") == []


@pytest.mark.parametrize(
    "code,expected",
    [
        ("theorem t : True := trivial", False),
        ("/- a closed comment -/\ntheorem t : True := trivial", False),
        ("/- nested /- inner -/ still closed -/", False),
        # The failure this exists for: a mangled `-/` swallows the rest of the file.
        ("/- opened and never closed\ntheorem t : True := trivial", True),
        ("/- outer /- inner -/ never closed", True),
        # `--` line comments must not be mistaken for a block delimiter.
        ("-- /- not a block\ntheorem t : True := trivial", False),
    ],
)
def test_has_unterminated_block_comment(code, expected):
    assert has_unterminated_block_comment(code) is expected


TARGET = "theorem foo (n : ℕ) (h : 0 < n) : n ≠ 0"


@pytest.mark.parametrize(
    "submission,why",
    [
        (f"import Mathlib\n\n{TARGET} := by omega\n", "verbatim"),
        # Reformatted and re-indented.
        ("import Mathlib\n\ntheorem foo (n : ℕ)\n    (h : 0 < n) :\n    n ≠ 0 := by omega\n", "rewrapped"),
        # `lemma` is notation for `theorem`; swapping them weakens nothing.
        (f"import Mathlib\n\n{TARGET.replace('theorem', 'lemma')} := by omega\n", "lemma for theorem"),
        # Other holes in the same file are none of this check's business.
        (f"import Mathlib\n\ntheorem open_conj : True := by sorry\n\n{TARGET} := by omega\n", "other sorry"),
        # A comment mentioning something else does not count as the statement.
        (f"import Mathlib\n-- proving foo\n{TARGET} := by omega\n", "comments ignored"),
    ],
)
def test_check_target_statement_preserved_accepts(submission, why):
    preserved, reason = check_target_statement_preserved(TARGET, submission)
    assert preserved, f"{why}: {reason}"


@pytest.mark.parametrize(
    "submission,why",
    [
        ("import Mathlib\n\ntheorem foo (n : ℕ) : n ≠ 0 := by omega\n", "hypothesis dropped"),
        ("import Mathlib\n\ntheorem foo (n : ℕ) (h : 0 < n) : n ≥ 0 := by omega\n", "conclusion changed"),
        ("import Mathlib\n\ntheorem bar (n : ℕ) (h : 0 < n) : n ≠ 0 := by omega\n", "renamed"),
        ("import Mathlib\n\ntheorem unrelated : True := trivial\n", "target absent"),
        # A statement that only appears inside a comment is not a statement.
        (f"import Mathlib\n/- {TARGET} -/\ntheorem other : True := trivial\n", "only in a comment"),
    ],
)
def test_check_target_statement_preserved_rejects(submission, why):
    preserved, reason = check_target_statement_preserved(TARGET, submission)
    assert not preserved, why
    assert reason


def test_check_target_statement_preserved_requires_a_recorded_statement():
    preserved, reason = check_target_statement_preserved("", "theorem foo : True := trivial")
    assert not preserved
    assert "No target statement" in reason
