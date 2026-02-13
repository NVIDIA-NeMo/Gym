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

"""COBOL code extraction, compilation, and test execution utilities.

Inlined from DomainForge (prompt_builder.py, verifier.py, languages/cobol.py)
to avoid a pip dependency on DomainForge.
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_cobol_code(text: Optional[str]) -> Optional[str]:
    """Extract COBOL source code from an LLM response.

    Strategy:
    1. Strip <think>/<thinking> blocks (reasoning traces).
    2. Extract from ```cobol``` or ``` markdown fences (longest block).
    3. Fall back to finding IDENTIFICATION DIVISION marker.
    4. Validate that the result contains required COBOL divisions.
    """
    if not text:
        return None

    cleaned = _strip_think_blocks(text)

    # Strategy 1: markdown code fences
    code = _extract_from_fences(cleaned)
    if code and _is_valid_cobol(code):
        return code

    # Strategy 2: find IDENTIFICATION DIVISION marker in raw text
    code = _extract_from_division_marker(cleaned)
    if code and _is_valid_cobol(code):
        return code

    return None


def _strip_think_blocks(text: str) -> str:
    """Remove <think>/<thinking> blocks from text."""
    result = text
    # Orphaned closing tags
    if "</think>" in result and "<think>" not in result:
        result = re.sub(r"^.*?</think>", "", result, flags=re.DOTALL)
    if "</thinking>" in result and "<thinking>" not in result:
        result = re.sub(r"^.*?</thinking>", "", result, flags=re.DOTALL)
    # Well-formed blocks
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
    result = re.sub(r"<thinking>.*?</thinking>", "", result, flags=re.DOTALL)
    # Unclosed blocks
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
    result = re.sub(r"<thinking>.*", "", result, flags=re.DOTALL)
    # Stray closing tags
    result = result.replace("</think>", "").replace("</thinking>", "")
    return result.strip()


def _extract_from_fences(text: str) -> Optional[str]:
    """Extract code from markdown fences, returning the longest block."""
    blocks: List[str] = []
    current_block: List[str] = []
    in_code = False

    for line in text.split("\n"):
        if "```" in line:
            if in_code:
                if current_block:
                    blocks.append("\n".join(current_block))
                current_block = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            current_block.append(line)

    # Handle unclosed fence
    if in_code and current_block:
        blocks.append("\n".join(current_block))

    if blocks:
        return max(blocks, key=len)
    return None


def _extract_from_division_marker(text: str) -> Optional[str]:
    """Extract COBOL code starting from IDENTIFICATION DIVISION."""
    upper = text.upper()
    for marker in ["IDENTIFICATION DIVISION", "ID DIVISION"]:
        idx = upper.find(marker)
        if idx >= 0:
            return text[idx:].strip()
    return None


def _is_valid_cobol(code: str) -> bool:
    """Check that code contains essential COBOL structure."""
    upper = code.upper()
    has_id = "IDENTIFICATION DIVISION" in upper or "ID DIVISION" in upper
    has_program_id = "PROGRAM-ID" in upper
    has_procedure = "PROCEDURE DIVISION" in upper
    return has_id and has_program_id and has_procedure


# ---------------------------------------------------------------------------
# Compilation and test execution
# ---------------------------------------------------------------------------


def compile_and_test(
    code: str,
    test_cases: List[Dict[str, Any]],
    compiler_cmd: str = "cobc",
    compiler_flags: Optional[List[str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Compile COBOL code and run test cases.

    Returns a dict with:
        all_passed: bool
        compilation_success: bool
        compilation_errors: list[str]
        tests_passed: int
        tests_total: int
        test_results: list[dict]
    """
    if compiler_flags is None:
        compiler_flags = ["-x", "-free", "-fdiagnostics-plain-output"]

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "program.cob"
        exe = Path(tmpdir) / "program"

        src.write_text(code)

        # ---- Compile ----
        try:
            compile_result = subprocess.run(
                [compiler_cmd] + compiler_flags + ["-o", str(exe), str(src)],
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "all_passed": False,
                "compilation_success": False,
                "compilation_errors": ["Compilation timed out"],
                "tests_passed": 0,
                "tests_total": len(test_cases),
                "test_results": [],
            }
        except FileNotFoundError:
            return {
                "all_passed": False,
                "compilation_success": False,
                "compilation_errors": [f"Compiler not found: {compiler_cmd}"],
                "tests_passed": 0,
                "tests_total": len(test_cases),
                "test_results": [],
            }

        if compile_result.returncode != 0:
            stderr_text = compile_result.stderr.decode("utf-8", errors="replace") if compile_result.stderr else ""
            errors = [line for line in stderr_text.splitlines() if line.strip()]
            return {
                "all_passed": False,
                "compilation_success": False,
                "compilation_errors": errors,
                "tests_passed": 0,
                "tests_total": len(test_cases),
                "test_results": [],
            }

        # ---- Run tests ----
        test_results: List[Dict[str, Any]] = []
        passed = 0
        consecutive_timeouts = 0

        for tc in test_cases:
            if consecutive_timeouts >= 2:
                test_results.append(
                    {
                        "test_id": tc.get("test_id", len(test_results)),
                        "passed": False,
                        "expected": tc["expected_output"],
                        "actual": "",
                        "error": "Skipped after 2 consecutive timeouts",
                    }
                )
                continue

            try:
                run_result = subprocess.run(
                    [str(exe)],
                    input=tc["input"].encode("utf-8"),
                    capture_output=True,
                    timeout=timeout,
                )
                actual = run_result.stdout.decode("utf-8", errors="replace").strip()
                expected = tc["expected_output"].strip()
                test_passed = actual == expected

                if test_passed:
                    passed += 1
                consecutive_timeouts = 0

                stderr_text = (
                    run_result.stderr.decode("utf-8", errors="replace").strip() if run_result.stderr else None
                )
                test_results.append(
                    {
                        "test_id": tc.get("test_id", len(test_results)),
                        "passed": test_passed,
                        "expected": expected,
                        "actual": actual,
                        "error": stderr_text,
                    }
                )
            except subprocess.TimeoutExpired:
                consecutive_timeouts += 1
                test_results.append(
                    {
                        "test_id": tc.get("test_id", len(test_results)),
                        "passed": False,
                        "expected": tc["expected_output"].strip(),
                        "actual": "",
                        "error": "Execution timed out",
                    }
                )

        return {
            "all_passed": passed == len(test_cases),
            "compilation_success": True,
            "compilation_errors": [],
            "tests_passed": passed,
            "tests_total": len(test_cases),
            "test_results": test_results,
        }


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------


@ray.remote
def compile_and_test_remote(
    code: str,
    test_cases: List[Dict[str, Any]],
    compiler_cmd: str = "cobc",
    compiler_flags: Optional[List[str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Ray remote wrapper for compile_and_test."""
    return compile_and_test(code, test_cases, compiler_cmd, compiler_flags, timeout)
