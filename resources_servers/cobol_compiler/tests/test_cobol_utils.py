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

import shutil

import pytest
from cobol_utils import _soft_match, compile_and_test, extract_cobol_code


VALID_COBOL = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.
       PROCEDURE DIVISION.
           DISPLAY "HELLO"
           STOP RUN.
"""

VALID_COBOL_SQUARE = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SQUARE.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-LINE       PIC X(20).
       01  WS-N          PIC S9(9).
       01  WS-RESULT     PIC S9(18).
       01  WS-DISPLAY    PIC -(18)9.
       PROCEDURE DIVISION.
           ACCEPT WS-LINE
           MOVE FUNCTION NUMVAL(FUNCTION TRIM(WS-LINE))
               TO WS-N
           COMPUTE WS-RESULT = WS-N * WS-N
           MOVE WS-RESULT TO WS-DISPLAY
           DISPLAY FUNCTION TRIM(WS-DISPLAY)
           STOP RUN.
"""


class TestExtractCobolCode:
    def test_extract_from_cobol_fence(self) -> None:
        text = f"Here is the code:\n```cobol\n{VALID_COBOL}\n```"
        result = extract_cobol_code(text)
        assert result is not None
        assert "IDENTIFICATION DIVISION" in result
        assert "PROGRAM-ID" in result

    def test_extract_from_generic_fence(self) -> None:
        text = f"```\n{VALID_COBOL}\n```"
        result = extract_cobol_code(text)
        assert result is not None
        assert "IDENTIFICATION DIVISION" in result

    def test_extract_from_raw_text(self) -> None:
        """Code without fences should be extracted via IDENTIFICATION DIVISION marker."""
        result = extract_cobol_code(VALID_COBOL)
        assert result is not None
        assert "IDENTIFICATION DIVISION" in result

    def test_extract_with_think_blocks(self) -> None:
        text = f"<think>Let me think about this...</think>\n```cobol\n{VALID_COBOL}\n```"
        result = extract_cobol_code(text)
        assert result is not None
        assert "<think>" not in result

    def test_extract_with_orphaned_think(self) -> None:
        text = f"Some reasoning here...</think>\n```cobol\n{VALID_COBOL}\n```"
        result = extract_cobol_code(text)
        assert result is not None
        assert "IDENTIFICATION DIVISION" in result

    def test_extract_with_unclosed_think(self) -> None:
        text = f"<think>Some reasoning that never closes\n```cobol\n{VALID_COBOL}\n```"
        # Unclosed think block consumes everything after it, so no code extracted.
        # Just verify it doesn't crash.
        assert extract_cobol_code(text) is None

    def test_extract_no_code(self) -> None:
        result = extract_cobol_code("I don't know how to write COBOL.")
        assert result is None

    def test_extract_empty(self) -> None:
        result = extract_cobol_code("")
        assert result is None

    def test_extract_none(self) -> None:
        result = extract_cobol_code(None)
        assert result is None

    def test_extract_incomplete_cobol(self) -> None:
        """Missing PROCEDURE DIVISION should fail validation."""
        incomplete = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. INCOMPLETE.
       DATA DIVISION.
"""
        result = extract_cobol_code(f"```cobol\n{incomplete}\n```")
        assert result is None

    def test_extract_with_preamble(self) -> None:
        """Text before IDENTIFICATION DIVISION should be stripped."""
        text = "Here is my solution:\n\n" + VALID_COBOL
        result = extract_cobol_code(text)
        assert result is not None
        assert result.startswith("IDENTIFICATION DIVISION") or result.strip().startswith("IDENTIFICATION DIVISION")

    def test_extract_id_division_shorthand(self) -> None:
        """ID DIVISION (shorthand) should also be recognized."""
        code = VALID_COBOL.replace("IDENTIFICATION DIVISION", "ID DIVISION")
        result = extract_cobol_code(code)
        assert result is not None
        assert "ID DIVISION" in result


class TestCompileAndTest:
    """Tests that require GnuCOBOL to be installed."""

    pytestmark = pytest.mark.skipif(
        shutil.which("cobc") is None,
        reason="GnuCOBOL (cobc) not installed",
    )

    def test_compile_and_test_pass(self) -> None:
        test_cases = [
            {"input": "3", "expected_output": "9", "test_id": 0},
            {"input": "5", "expected_output": "25", "test_id": 1},
        ]
        result = compile_and_test(VALID_COBOL_SQUARE, test_cases)
        assert result["all_passed"] is True
        assert result["compilation_success"] is True
        assert result["tests_passed"] == 2
        assert result["tests_total"] == 2

    def test_compile_and_test_wrong_output(self) -> None:
        test_cases = [
            {"input": "3", "expected_output": "100", "test_id": 0},
        ]
        result = compile_and_test(VALID_COBOL_SQUARE, test_cases)
        assert result["all_passed"] is False
        assert result["compilation_success"] is True
        assert result["tests_passed"] == 0

    def test_compile_error(self) -> None:
        bad_code = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. BAD.
       PROCEDURE DIVISION.
           DISPLAY HELLO
           STOP RUN.
"""
        result = compile_and_test(bad_code, [{"input": "", "expected_output": "", "test_id": 0}])
        assert result["all_passed"] is False
        assert result["compilation_success"] is False
        assert len(result["compilation_errors"]) > 0

    def test_compiler_not_found(self) -> None:
        result = compile_and_test(
            VALID_COBOL_SQUARE,
            [{"input": "3", "expected_output": "9", "test_id": 0}],
            compiler_cmd="nonexistent_compiler",
        )
        assert result["all_passed"] is False
        assert result["compilation_success"] is False
        assert "not found" in result["compilation_errors"][0].lower()

    def test_hello_world(self) -> None:
        test_cases = [{"input": "", "expected_output": "HELLO", "test_id": 0}]
        result = compile_and_test(VALID_COBOL, test_cases)
        assert result["all_passed"] is True

    def test_multi_line_stdin(self) -> None:
        """Multiple ACCEPT calls with multi-line input should work correctly."""
        code = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SUM-TWO.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-LINE-A     PIC X(20).
       01  WS-LINE-B     PIC X(20).
       01  WS-A          PIC S9(9).
       01  WS-B          PIC S9(9).
       01  WS-SUM        PIC S9(9).
       01  WS-DISPLAY    PIC -(9)9.
       PROCEDURE DIVISION.
           ACCEPT WS-LINE-A
           ACCEPT WS-LINE-B
           MOVE FUNCTION NUMVAL(FUNCTION TRIM(WS-LINE-A))
               TO WS-A
           MOVE FUNCTION NUMVAL(FUNCTION TRIM(WS-LINE-B))
               TO WS-B
           ADD WS-A TO WS-B GIVING WS-SUM
           MOVE WS-SUM TO WS-DISPLAY
           DISPLAY FUNCTION TRIM(WS-DISPLAY)
           STOP RUN.
"""
        test_cases = [
            {"input": "3\n4", "expected_output": "7", "test_id": 0},
            {"input": "10\n20", "expected_output": "30", "test_id": 1},
        ]
        result = compile_and_test(code, test_cases)
        assert result["compilation_success"] is True
        assert result["all_passed"] is True
        assert result["tests_passed"] == 2

    def test_compilation_warnings_still_pass(self) -> None:
        """Code that triggers warnings but compiles should succeed."""
        # Unused variable triggers a warning but still compiles
        code = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. WARN-TEST.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-UNUSED     PIC X(10).
       01  WS-USED       PIC X(10) VALUE "OK".
       PROCEDURE DIVISION.
           DISPLAY FUNCTION TRIM(WS-USED)
           STOP RUN.
"""
        test_cases = [{"input": "", "expected_output": "OK", "test_id": 0}]
        result = compile_and_test(code, test_cases)
        assert result["compilation_success"] is True
        assert result["all_passed"] is True

    def test_compilation_warnings_field_present(self) -> None:
        """The compilation_warnings field should always be present."""
        test_cases = [{"input": "3", "expected_output": "9", "test_id": 0}]
        result = compile_and_test(VALID_COBOL_SQUARE, test_cases)
        assert "compilation_warnings" in result
        assert isinstance(result["compilation_warnings"], list)

    def test_debug_mode(self) -> None:
        """Debug mode should not change pass/fail behaviour."""
        test_cases = [
            {"input": "3", "expected_output": "9", "test_id": 0},
        ]
        result = compile_and_test(VALID_COBOL_SQUARE, test_cases, debug=True)
        assert result["all_passed"] is True

    def test_soft_match_leading_zeros(self) -> None:
        """soft_match=True should accept COBOL leading-zero output."""
        # VALID_COBOL_SQUARE uses edited picture so output is already clean,
        # but we can test by expecting a value that would only match with
        # soft comparison when the program outputs the clean value.
        test_cases = [
            {"input": "3", "expected_output": "9", "test_id": 0},
        ]
        result = compile_and_test(VALID_COBOL_SQUARE, test_cases, soft_match=True)
        assert result["all_passed"] is True
        assert result["tests_passed"] == 1

    def test_soft_match_flag_false_by_default(self) -> None:
        """Default behaviour should be strict matching."""
        test_cases = [
            {"input": "3", "expected_output": "9", "test_id": 0},
        ]
        result_default = compile_and_test(VALID_COBOL_SQUARE, test_cases)
        result_strict = compile_and_test(VALID_COBOL_SQUARE, test_cases, soft_match=False)
        assert result_default["all_passed"] == result_strict["all_passed"]


class TestSoftMatch:
    """Unit tests for _soft_match comparison logic."""

    def test_exact_match(self) -> None:
        assert _soft_match("5", "5")

    def test_leading_zeros(self) -> None:
        assert _soft_match("5", "00005")
        assert _soft_match("0", "00000")

    def test_leading_plus(self) -> None:
        assert _soft_match("5", "+00005")
        assert _soft_match("5", "+5")

    def test_negative_leading_zeros(self) -> None:
        assert _soft_match("-5", "-00005")

    def test_decimal_trailing_zeros(self) -> None:
        assert _soft_match("0.5", ".500000")
        assert _soft_match("0.5", "0.500000000")

    def test_decimal_integer_equivalence(self) -> None:
        assert _soft_match("35.0", "35")
        assert _soft_match("3", "3.0")

    def test_case_insensitive(self) -> None:
        assert _soft_match("TRUE", "true")
        assert _soft_match("FALSE", "False")

    def test_whitespace_stripping(self) -> None:
        assert _soft_match(" 5 ", "5")
        assert _soft_match("5", "  5  ")

    def test_number_list(self) -> None:
        assert _soft_match("5 10 15", "00005 00010 00015")
        assert _soft_match("5 10 15", "+5 +10 +15")
        assert _soft_match("0.0 1.0", "0.000000 1.000000")

    def test_no_false_positives_different_values(self) -> None:
        assert not _soft_match("5", "6")
        assert not _soft_match("5", "")
        assert not _soft_match("hello", "world")

    def test_no_false_positives_different_lengths(self) -> None:
        assert not _soft_match("5 10", "5 10 15")
        assert not _soft_match("0.0 1.0", "0.0")

    def test_no_false_positives_structured(self) -> None:
        assert not _soft_match("[1, 2]", "[1, 3]")

    def test_empty_strings(self) -> None:
        assert _soft_match("", "")
        assert _soft_match("  ", "  ")
        assert not _soft_match("", "5")
