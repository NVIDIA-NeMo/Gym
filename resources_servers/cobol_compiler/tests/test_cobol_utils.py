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
from cobol_utils import compile_and_test, extract_cobol_code


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
