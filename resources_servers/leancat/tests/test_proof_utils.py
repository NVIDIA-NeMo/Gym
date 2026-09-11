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

from resources_servers.leancat.proof_utils import (
    check_statement_preserved,
    extract_lean_code,
    find_banned_tokens,
    split_preamble_and_body,
    strip_lean_comments_and_strings,
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


class TestExtractLeanCode:
    def test_takes_last_fenced_block(self):
        text = "First try:\n```lean4\nbad\n```\nActually:\n```lean4\ngood\n```"
        assert extract_lean_code(text) == "good"

    def test_accepts_bare_and_capitalised_fences(self):
        assert extract_lean_code("```Lean4\ntheorem t : True := trivial\n```") == "theorem t : True := trivial"
        assert extract_lean_code("```\ntheorem t : True := trivial\n```") == "theorem t : True := trivial"

    def test_unfenced_response_falls_back_to_raw_text(self):
        assert extract_lean_code("  theorem t : True := trivial  ") == "theorem t : True := trivial"

    def test_empty_response(self):
        assert extract_lean_code("") == ""


class TestStripLeanCommentsAndStrings:
    def test_line_comment_content_removed_newline_and_offsets_kept(self):
        original = "a -- sorry\nb"
        stripped = strip_lean_comments_and_strings(original)
        assert stripped == "a" + " " * (len("a -- sorry") - 1) + "\nb"
        assert len(stripped) == len(original)

    def test_nested_block_comments(self):
        # The inner `-/` must not be read as closing the outer comment, or `sorry`
        # would leak back into the checked text.
        assert "sorry" not in strip_lean_comments_and_strings("/- /- sorry -/ -/ ok")
        assert "ok" in strip_lean_comments_and_strings("/- /- sorry -/ -/ ok")

    def test_doc_comment_is_stripped(self):
        assert "sorry" not in strip_lean_comments_and_strings(
            "/-- proves it with sorry -/\ntheorem t : True := trivial"
        )

    def test_string_literal_is_stripped(self):
        assert "sorry" not in strip_lean_comments_and_strings('def s := "sorry"')

    def test_escaped_quote_does_not_end_string(self):
        assert "sorry" not in strip_lean_comments_and_strings('def s := "a\\" sorry"')

    def test_code_outside_comments_survives_verbatim(self):
        code = "theorem t : True := trivial"
        assert strip_lean_comments_and_strings(code) == code


class TestFindBannedTokens:
    def test_finds_each_shortcut(self):
        assert find_banned_tokens("theorem t : True := by sorry") == ["sorry"]
        assert find_banned_tokens("axiom bad : False\nunsafe def f := 1") == ["axiom", "unsafe"]
        assert find_banned_tokens("theorem t : True := by admit") == ["admit"]

    def test_clean_proof_has_none(self):
        assert find_banned_tokens(SOLVED) == []

    def test_ignores_comments_and_strings(self):
        assert find_banned_tokens("-- no sorry here\ntheorem t : True := trivial") == []
        assert find_banned_tokens('def msg := "sorry"') == []

    def test_identifier_containing_sorry_is_not_a_shortcut(self):
        # `\b` must not match inside a snake_case identifier.
        assert find_banned_tokens("theorem no_sorry_needed : True := trivial") == []


class TestSplitPreambleAndBody:
    def test_preamble_stops_at_first_declaration(self):
        preamble, body = split_preamble_and_body(REFERENCE)
        assert preamble == ["import Mathlib", "open CategoryTheory", "variable {C : Type*} [Category.{v} C]"]
        assert body.startswith("theorem id_comm")

    def test_file_without_declarations_is_all_preamble(self):
        preamble, body = split_preamble_and_body("import Mathlib\nopen CategoryTheory")
        assert preamble == ["import Mathlib", "open CategoryTheory"]
        assert body == ""

    def test_attribute_line_starts_the_body(self):
        _, body = split_preamble_and_body("import Mathlib\n@[simp]\ntheorem t : True := by\n  sorry")
        assert body.startswith("@[simp]")


class TestCheckStatementPreserved:
    def test_faithful_proof_passes(self):
        assert check_statement_preserved(REFERENCE, SOLVED) == (True, None)

    def test_rewrapped_signature_passes(self):
        rewrapped = SOLVED.replace(
            "theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) : α ≫ β = β ≫ α := by",
            "theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) :\n    α ≫ β = β ≫ α := by",
        )
        assert check_statement_preserved(REFERENCE, rewrapped)[0]

    def test_comments_added_by_the_model_are_ignored(self):
        commented = SOLVED.replace("theorem id_comm", "-- the main result\ntheorem id_comm")
        assert check_statement_preserved(REFERENCE, commented)[0]

    def test_auxiliary_lemma_inserted_before_target_passes(self):
        # The LeanCat prompt explicitly allows auxiliary declarations before the target.
        with_aux = SOLVED.replace(
            "theorem id_comm",
            "lemma helper : True := trivial\n\ntheorem id_comm",
        )
        assert check_statement_preserved(REFERENCE, with_aux)[0]

    def test_added_hypothesis_is_rejected(self):
        # Slipping in `(h : False)` makes the goal trivially provable; it has to fail.
        weakened = SOLVED.replace("(α β : (𝟭 C) ⟶ (𝟭 C))", "(α β : (𝟭 C) ⟶ (𝟭 C)) (h : False)")
        ok, reason = check_statement_preserved(REFERENCE, weakened)
        assert not ok
        assert "Statement fragment" in reason

    def test_altered_conclusion_is_rejected(self):
        weakened = SOLVED.replace("α ≫ β = β ≫ α", "α ≫ β = α ≫ β")
        ok, reason = check_statement_preserved(REFERENCE, weakened)
        assert not ok
        assert "Statement fragment" in reason

    def test_renamed_theorem_is_rejected(self):
        renamed = SOLVED.replace("theorem id_comm", "theorem id_comm'")
        assert not check_statement_preserved(REFERENCE, renamed)[0]

    def test_dropped_variable_binder_is_rejected(self):
        # Deleting `variable {C : Type*} [Category.{v} C]` and proving the statement for
        # some concrete category would otherwise compile and score 1.0.
        without_binder = SOLVED.replace("variable {C : Type*} [Category.{v} C]\n\n", "")
        ok, reason = check_statement_preserved(REFERENCE, without_binder)
        assert not ok
        assert "Preamble line missing" in reason

    def test_multiple_sorries_must_all_be_filled_in_place(self):
        reference = "import Mathlib\n\ntheorem a : True := by\n  sorry\n\ntheorem b : True := by\n  sorry"
        good = "import Mathlib\n\ntheorem a : True := by\n  trivial\n\ntheorem b : True := by\n  trivial"
        assert check_statement_preserved(reference, good)[0]

        # Second theorem dropped entirely: the second fragment has nowhere to match.
        partial = "import Mathlib\n\ntheorem a : True := by\n  trivial"
        assert not check_statement_preserved(reference, partial)[0]

    def test_reordered_declarations_are_rejected(self):
        # The scan is in-order, so swapping the two targets fails even though both texts
        # are present -- order is part of the file the model was told to copy.
        reference = "import Mathlib\n\ntheorem a : True := by\n  sorry\n\ntheorem b : False ∨ True := by\n  sorry"
        swapped = "import Mathlib\n\ntheorem b : False ∨ True := by\n  trivial\n\ntheorem a : True := by\n  trivial"
        assert not check_statement_preserved(reference, swapped)[0]
