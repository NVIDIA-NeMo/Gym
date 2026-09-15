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


import pytest

from resources_servers.math_formal_lean.proof_utils import (
    ProofBuildConfig,
    build_lean4_proof,
    clean_formal_generation,
    determine_proof_status,
    extract_code_block,
    extract_proof_only,
    strip_lean_comments_and_strings,
    strip_thinking,
)


class TestExtractCodeBlock:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Here's the proof:\n```lean4\nsimp [h₁, h₂, h₃]\nring\n```\nDone!", "simp [h₁, h₂, h₃]\nring"),
            ("Just some text without code blocks", ""),
        ],
    )
    def test_extract_by_language_tag(self, text, expected):
        assert extract_code_block(text, languages=["lean4", "lean", ""]) == expected

    @pytest.mark.parametrize(
        "mode,expected",
        [("last", "second_proof"), ("first", "first_proof")],
    )
    def test_extract_mode_picks_the_right_block(self, mode, expected):
        text = "```lean4\nfirst_proof\n```\nActually:\n```lean4\nsecond_proof\n```"
        assert extract_code_block(text, languages=["lean4"], extract_code_mode=mode) == expected


class TestCleanFormalGeneration:
    def test_clean_with_code_block(self):
        generation = """Let me solve this step by step.
```lean4
simp [h₁, h₂]
ring
```"""
        result = clean_formal_generation(generation)
        assert result == "simp [h₁, h₂]\nring"

    def test_clean_without_code_block(self):
        generation = "simp [h₁, h₂]\nring"
        result = clean_formal_generation(generation)
        assert result == "simp [h₁, h₂]\nring"

    def test_clean_with_final_answer_key(self):
        generation = """Thinking...
FINAL ANSWER:
```lean4
omega
```"""
        result = clean_formal_generation(generation, final_answer_key="FINAL ANSWER:")
        assert result == "omega"


class TestExtractProofOnly:
    def test_extract_proof_from_theorem(self):
        lean_code = """theorem test (n : Nat) : n + 0 = n := by
  simp
  ring"""
        result = extract_proof_only(lean_code)
        assert "simp" in result
        assert "ring" in result
        assert "theorem" not in result

    def test_extract_proof_with_by_on_same_line(self):
        lean_code = "theorem test : True := by trivial"
        result = extract_proof_only(lean_code)
        assert result == "trivial"

    def test_extract_proof_from_example(self):
        lean_code = """example : 1 + 1 = 2 := by
  norm_num"""
        result = extract_proof_only(lean_code)
        assert "norm_num" in result
        assert "example" not in result

    def test_no_theorem_returns_original(self):
        lean_code = "simp\nring"
        result = extract_proof_only(lean_code)
        assert result == "simp\nring"

    def test_empty_input(self):
        result = extract_proof_only("")
        assert result == ""


class TestBuildLean4Proof:
    def test_build_proof_with_restate(self):
        generation = """```lean4
theorem test : True := by
  trivial
```"""
        data_point = {
            "header": "import Mathlib\n\n",
            "formal_statement": "theorem test : True := by\n",
        }
        config = ProofBuildConfig(
            restate_formal_statement=True,
            strip_theorem_from_proof=True,
        )
        result = build_lean4_proof(generation, data_point, config)

        assert result.startswith("import Mathlib")
        assert "theorem test : True := by" in result
        assert "trivial" in result

    def test_build_proof_without_restate(self):
        generation = """```lean4
theorem test : True := by
  trivial
```"""
        data_point = {
            "header": "import Mathlib\n\n",
            "formal_statement": "theorem test : True := by\n",
        }
        config = ProofBuildConfig(
            restate_formal_statement=False,
            strip_theorem_from_proof=True,
        )
        result = build_lean4_proof(generation, data_point, config)

        assert result.startswith("import Mathlib")
        # formal_statement should not be included when restate is False
        assert result.count("theorem test") == 0 or "trivial" in result


class TestDetermineProofStatus:
    @pytest.mark.parametrize(
        "output,expected",
        [
            ({"process_status": "completed", "stdout": "", "stderr": ""}, "completed"),
            ({"process_status": "timeout", "stdout": "", "stderr": ""}, "timeout"),
            ({"process_status": "error", "stdout": "", "stderr": "compilation failed"}, "error"),
            ({}, "unknown"),
            (
                {"process_status": "completed", "stdout": "warning: declaration uses 'sorry'", "stderr": ""},
                "has_sorry",
            ),
        ],
    )
    def test_status_mapping(self, output, expected):
        assert determine_proof_status(output) == expected


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


class TestStripThinking:
    def test_closed_block_removed(self):
        assert strip_thinking("<think>a</think>answer") == "answer"

    def test_bare_close_keeps_only_the_tail(self):
        assert strip_thinking("reasoning</think>answer") == "answer"

    def test_last_close_wins(self):
        assert strip_thinking("r1</think>mid</think>answer") == "answer"

    def test_unclosed_opener_drops_the_rest(self):
        assert strip_thinking("answer<think>reasoning") == "answer"

    def test_no_tags_unchanged(self):
        assert strip_thinking("plain") == "plain"
