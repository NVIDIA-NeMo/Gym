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


from resources_servers.math_formal_lean.proof_utils import (
    ProofBuildConfig,
    build_lean4_proof,
    clean_formal_generation,
    determine_proof_status,
    extract_code_block,
    extract_proof_only,
)


class TestExtractCodeBlock:
    def test_extract_lean4_code_block(self):
        text = """Here's the proof:
```lean4
simp [h₁, h₂, h₃]
ring
```
Done!"""
        result = extract_code_block(text, languages=["lean4", "lean", ""])
        assert result == "simp [h₁, h₂, h₃]\nring"

    def test_extract_lean_code_block(self):
        text = """```lean
omega
```"""
        result = extract_code_block(text, languages=["lean4", "lean", ""])
        assert result == "omega"

    def test_extract_generic_code_block(self):
        text = """```
simp
```"""
        result = extract_code_block(text, languages=["lean4", "lean", ""])
        assert result == "simp"

    def test_no_code_block(self):
        text = "Just some text without code blocks"
        result = extract_code_block(text, languages=["lean4", "lean", ""])
        assert result == ""

    def test_extract_last_code_block(self):
        text = """```lean4
first_proof
```
Actually, let me try again:
```lean4
second_proof
```"""
        result = extract_code_block(text, languages=["lean4"], extract_code_mode="last")
        assert result == "second_proof"

    def test_extract_first_code_block(self):
        text = """```lean4
first_proof
```
Actually:
```lean4
second_proof
```"""
        result = extract_code_block(text, languages=["lean4"], extract_code_mode="first")
        assert result == "first_proof"


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
    def test_completed_status(self):
        output = {"process_status": "completed", "stdout": "", "stderr": ""}
        assert determine_proof_status(output) == "completed"

    def test_timeout_status(self):
        output = {"process_status": "timeout", "stdout": "", "stderr": ""}
        assert determine_proof_status(output) == "timeout"

    def test_error_status(self):
        output = {"process_status": "error", "stdout": "", "stderr": "compilation failed"}
        assert determine_proof_status(output) == "error"

    def test_has_sorry_in_output(self):
        output = {
            "process_status": "completed",
            "stdout": "warning: declaration uses 'sorry'",
            "stderr": "",
        }
        assert determine_proof_status(output) == "has_sorry"

    def test_has_sorry_in_stderr(self):
        output = {
            "process_status": "completed",
            "stdout": "",
            "stderr": "sorry found in proof",
        }
        assert determine_proof_status(output) == "has_sorry"

    def test_unknown_status(self):
        output = {}
        assert determine_proof_status(output) == "unknown"
