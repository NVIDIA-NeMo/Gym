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

"""Unit tests for the Fine-Eval port. Each case pins a rule the verdict depends on."""

from resources_servers.combibench.fine_eval import (
    DEFAULT_HEADER,
    LeanResult,
    abbrev_types,
    answer_check,
    answer_tags,
    build_submission,
    classify_lean_result,
    extract_lean_code,
    has_forbidden_substring,
    missing_chunks,
    remove_comments,
    statement_chunks,
)
from resources_servers.combibench.lean_client import parse_verify_response


STATEMENT = """import Mathlib

abbrev hackmath_1_solution : ℕ := sorry

theorem hackmath_1 (sols : Finset (Fin 13 → Fin 2))
    (h_sols : ∀ f, f ∈ sols ↔ ((List.ofFn f).count 0 = 6)) :
    sols.card = hackmath_1_solution := by sorry"""

SOLUTION = STATEMENT.replace(":= sorry\n\ntheorem", ":= 1716\n\ntheorem").replace("by sorry", "by decide")


def _fenced(code: str, lang: str = "lean4") -> str:
    return f"```{lang}\n{code}\n```"


class TestExtraction:
    def test_last_lean4_block_wins(self) -> None:
        text = "Draft:\n" + _fenced("import Mathlib\nexample : 1 = 1 := sorry") + "\nFixed:\n" + _fenced(SOLUTION)
        assert extract_lean_code(text) == SOLUTION

    def test_lean_fence_is_a_fallback_only(self) -> None:
        text = _fenced("import Mathlib\nexample : True := trivial", lang="lean") + _fenced(SOLUTION)
        assert extract_lean_code(text) == SOLUTION
        assert extract_lean_code(_fenced(SOLUTION, lang="lean")) == SOLUTION

    def test_prose_yields_nothing(self) -> None:
        assert extract_lean_code("I believe the answer is 1716.") is None

    def test_header_is_prepended_when_imports_are_missing(self) -> None:
        code = extract_lean_code(_fenced("theorem t : True := trivial"))
        assert code == DEFAULT_HEADER + "theorem t : True := trivial"

    def test_comments_are_removed_before_any_check(self) -> None:
        """The paper's cheat case: a theorem hidden in a comment must not count."""
        hidden = (
            "import Mathlib\n\n/-\n"
            + STATEMENT.split("\n\n", 1)[1]
            + "\n-/\n-- axiom cheat : False\nexample : True := trivial"
        )
        code = extract_lean_code(_fenced(hidden))
        assert "hackmath_1" not in code
        assert not has_forbidden_substring(code)

    def test_trailing_line_comment_after_code_is_kept(self) -> None:
        assert remove_comments("theorem t : True := trivial -- note\n") == "theorem t : True := trivial -- note\n"


class TestForbidden:
    def test_axiom_anywhere_in_code_is_rejected(self) -> None:
        assert has_forbidden_substring("import Mathlib\naxiom cheat : False\n" + SOLUTION)

    def test_local_instance_is_rejected(self) -> None:
        assert has_forbidden_substring("local_instance foo : Decidable p := sorry")

    def test_clean_solution_passes(self) -> None:
        assert not has_forbidden_substring(SOLUTION)


class TestStatementCheck:
    def test_header_paragraphs_are_not_checked(self) -> None:
        chunks = statement_chunks(
            "import Mathlib\n\nopen Finset\n\nset_option autoImplicit false\n\ntheorem t : True := by sorry"
        )
        assert chunks == ["theorem t : True := by"]

    def test_solution_reproduces_every_paragraph(self) -> None:
        assert missing_chunks(SOLUTION, statement_chunks(STATEMENT)) == []

    def test_dropped_hypothesis_is_detected(self) -> None:
        tampered = SOLUTION.replace("\n    (h_sols : ∀ f, f ∈ sols ↔ ((List.ofFn f).count 0 = 6)) :", " :")
        missing = missing_chunks(tampered, statement_chunks(STATEMENT))
        assert len(missing) == 1 and missing[0].startswith("theorem hackmath_1")

    def test_changed_answer_type_is_detected(self) -> None:
        tampered = SOLUTION.replace("abbrev hackmath_1_solution : ℕ := 1716", "abbrev hackmath_1_solution : ℤ := 1716")
        assert missing_chunks(tampered, statement_chunks(STATEMENT)) == ["abbrev hackmath_1_solution : ℕ :="]

    def test_trailing_whitespace_lines_are_ignored_by_default(self) -> None:
        """Thirteen pinned statements carry lines of only spaces; copying them without is not tampering."""
        statement = "import Mathlib\n\ndef d : Prop :=\n  ∃ a b, a ≠ b ∧\n  \n  a = b\n\ntheorem t : d := by sorry"
        code = "import Mathlib\n\ndef d : Prop :=\n  ∃ a b, a ≠ b ∧\n\n  a = b\n\ntheorem t : d := by trivial"
        chunks = statement_chunks(statement)
        assert missing_chunks(code, chunks) == []
        assert missing_chunks(code, chunks, normalize_trailing_whitespace=False) != []

    def test_indentation_changes_are_still_rejected(self) -> None:
        reindented = SOLUTION.replace("\n    sols.card", "\n  sols.card")
        assert missing_chunks(reindented, statement_chunks(STATEMENT)) != []


class TestAnswers:
    def test_tags_follow_statement_order(self) -> None:
        statement = (
            "import Mathlib\n\nnoncomputable abbrev p_1_solution : ENNReal := sorry\n\n"
            "noncomputable abbrev p_2_solution : ENNReal := sorry\n\ntheorem p : True := by sorry"
        )
        assert answer_tags(statement_chunks(statement)) == ["p_1_solution", "p_2_solution"]

    def test_implicit_binder_between_name_and_type(self) -> None:
        chunks = statement_chunks("import Mathlib\n\nabbrev b_solution {k} : (Fin k → ℕ) → ℕ := sorry")
        assert answer_tags(chunks) == ["b_solution"]

    def test_proof_only_statement_has_no_tags(self) -> None:
        assert answer_tags(statement_chunks("import Mathlib\n\ntheorem t : True := by sorry")) == []

    def test_checks_are_appended_positionally(self) -> None:
        submission = build_submission("code", ["a_solution", "b_solution"], ["1", "2"])
        assert submission == (
            "code\n\nexample: a_solution = 1 := by\n  try rfl\n  try norm_num"
            "\n\nexample: b_solution = 2 := by\n  try rfl\n  try norm_num"
        )

    def test_no_ground_truth_appends_nothing(self) -> None:
        assert build_submission("code", ["a_solution"], None) == "code"

    def test_declared_types_are_read_from_the_abbrev(self) -> None:
        statement = (
            "import Mathlib\n\nnoncomputable abbrev a_solution : ENNReal := sorry\n\n"
            "abbrev b_solution {k} : (Fin k → ℕ) → ℕ := sorry\n\n"
            "abbrev c_solution : ℕ+ → ℕ+ → ℝ := sorry\n\ntheorem t : True := by sorry"
        )
        assert abbrev_types(statement_chunks(statement)) == ["ENNReal", "(Fin k → ℕ) → ℕ", "ℕ+ → ℕ+ → ℝ"]

    def test_unparseable_abbrev_falls_back_to_upstream_form(self) -> None:
        assert abbrev_types(["abbrev weird_solution"]) == [None]
        assert (
            answer_check("weird_solution", "1", None)
            == "\n\nexample: weird_solution = 1 := by\n  try rfl\n  try norm_num"
        )

    def test_ascribed_check_wraps_the_gold_answer(self) -> None:
        check = answer_check("s_solution", "fun n => ⌈√n⌉₊ - 1", "ℕ → ℕ")
        assert check == "\n\nexample: s_solution = (fun n => ⌈√n⌉₊ - 1 : ℕ → ℕ) := by\n  try rfl\n  try norm_num"
        submission = build_submission("code", ["s_solution"], ["1"], ["ℕ"])
        assert submission.endswith("example: s_solution = (1 : ℕ) := by\n  try rfl\n  try norm_num")


class TestClassification:
    def test_clean_compile_is_success(self) -> None:
        assert classify_lean_result(LeanResult(messages=[{"severity": "info", "data": "ok"}])) == "success"

    def test_error_message_fails_the_proof(self) -> None:
        result = LeanResult(messages=[{"severity": "error", "data": "unsolved goals"}])
        assert classify_lean_result(result) == "proof_failed"

    def test_sorry_warning_is_not_a_pass(self) -> None:
        result = LeanResult(messages=[{"severity": "warning", "data": "declaration uses 'sorry'"}])
        assert classify_lean_result(result) == "has_sorry"

    def test_structured_sorries_are_not_a_pass(self) -> None:
        assert classify_lean_result(LeanResult(sorries=[{"goal": "⊢ True"}])) == "has_sorry"

    def test_other_warnings_do_not_fail(self) -> None:
        result = LeanResult(messages=[{"severity": "warning", "data": "unused variable `h`"}])
        assert classify_lean_result(result) == "success"

    def test_server_timeout_is_charged_to_the_model(self) -> None:
        assert classify_lean_result(LeanResult(error="Lean REPL command timed out in 60 seconds")) == "timeout"

    def test_transport_failure_is_a_harness_fault(self) -> None:
        assert classify_lean_result(LeanResult(error="boom", transport_failure=True)) == "lean_server_error"


class TestVerifyResponseParsing:
    def test_backward_verify_shape(self) -> None:
        body = {
            "results": [
                {
                    "custom_id": "x",
                    "response": {"messages": [{"severity": "error", "data": "e"}], "sorries": [], "time": 1.5},
                }
            ]
        }
        result = parse_verify_response(body)
        assert result.error is None and result.time == 1.5
        assert result.messages == [{"severity": "error", "data": "e"}]
        assert result.transport_failure is False

    def test_missing_results_is_transport_failure(self) -> None:
        assert parse_verify_response({"detail": "Unauthorized"}).transport_failure is True
        assert parse_verify_response({"results": []}).transport_failure is True
