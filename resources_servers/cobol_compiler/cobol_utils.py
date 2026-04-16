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
# Output comparison
# ---------------------------------------------------------------------------


def _soft_match(expected: str, actual: str, rtol: float = 1e-6) -> bool:
    """Compare expected/actual with tolerance for COBOL numeric formatting.

    Handles leading zeros ("00005" == "5"), leading plus signs ("+5" == "5"),
    trailing decimal zeros ("0.500000" == "0.5"), and case differences
    ("TRUE" == "true") that are artifacts of COBOL's default DISPLAY
    formatting rather than logic errors.
    """
    e, a = expected.strip(), actual.strip()
    if e == a:
        return True

    # Case-insensitive (TRUE/true/True)
    if e.upper() == a.upper():
        return True

    # Single numeric value
    try:
        fe, fa = float(e), float(a)
        if fe == fa or abs(fe - fa) <= rtol * max(abs(fe), abs(fa), 1.0):
            return True
    except (ValueError, OverflowError):
        pass

    # Space-separated list of numeric values
    ep, ap = e.split(), a.split()
    if len(ep) == len(ap) and len(ep) > 1:
        try:
            pairs = [(float(ei), float(ai)) for ei, ai in zip(ep, ap)]
            if all(ei == ai or abs(ei - ai) <= rtol * max(abs(ei), abs(ai), 1.0) for ei, ai in pairs):
                return True
        except (ValueError, OverflowError):
            pass

    return False


# ---------------------------------------------------------------------------
# Compilation and test execution
# ---------------------------------------------------------------------------


def compile_and_test(
    code: str,
    test_cases: List[Dict[str, Any]],
    compiler_cmd: str = "cobc",
    compiler_flags: Optional[List[str]] = None,
    timeout: int = 30,
    debug: bool = False,
    soft_match: bool = False,
) -> Dict[str, Any]:
    """Compile COBOL code and run test cases.

    Args:
        soft_match: Use numeric/case-insensitive comparison instead of exact
            string matching. Tolerates COBOL formatting artifacts like leading
            zeros, plus signs, and trailing decimal zeros.

    Returns a dict with:
        all_passed: bool
        compilation_success: bool
        compilation_errors: list[str]
        compilation_warnings: list[str]
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
                "compilation_warnings": [],
                "tests_passed": 0,
                "tests_total": len(test_cases),
                "test_results": [],
            }
        except FileNotFoundError:
            return {
                "all_passed": False,
                "compilation_success": False,
                "compilation_errors": [f"Compiler not found: {compiler_cmd}"],
                "compilation_warnings": [],
                "tests_passed": 0,
                "tests_total": len(test_cases),
                "test_results": [],
            }

        stderr_text = compile_result.stderr.decode("utf-8", errors="replace") if compile_result.stderr else ""
        stderr_lines = [line for line in stderr_text.splitlines() if line.strip()]

        # GnuCOBOL may return non-zero exit code for warnings but still
        # produce an executable.  Check whether the executable exists rather
        # than relying solely on the return code.  Separate warnings from
        # real errors by parsing stderr severity tags.
        compilation_warnings: List[str] = []
        compilation_errors: List[str] = []
        for line in stderr_lines:
            if re.search(r":\s*warning\s*:", line, re.IGNORECASE):
                compilation_warnings.append(line)
            elif re.search(r":\s*note\s*:", line, re.IGNORECASE):
                compilation_warnings.append(line)
            else:
                compilation_errors.append(line)

        if not exe.exists():
            if debug:
                LOG.info("Compilation failed – no executable produced. stderr:\n%s", stderr_text)
            return {
                "all_passed": False,
                "compilation_success": False,
                "compilation_errors": compilation_errors or stderr_lines,
                "compilation_warnings": compilation_warnings,
                "tests_passed": 0,
                "tests_total": len(test_cases),
                "test_results": [],
            }

        if debug and compilation_warnings:
            LOG.info("Compilation succeeded with %d warning(s)", len(compilation_warnings))

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
                # Append trailing newline so the last COBOL ACCEPT reads
                # correctly (matches DomainForge behaviour).
                stdin_data = (tc["input"] + "\n").encode("utf-8")

                run_result = subprocess.run(
                    [str(exe)],
                    input=stdin_data,
                    capture_output=True,
                    timeout=timeout,
                )
                actual = run_result.stdout.decode("utf-8", errors="replace").strip()
                expected = tc["expected_output"].strip()
                test_passed = _soft_match(expected, actual) if soft_match else actual == expected

                if test_passed:
                    passed += 1
                consecutive_timeouts = 0

                run_stderr = run_result.stderr.decode("utf-8", errors="replace").strip() if run_result.stderr else None

                if debug and not test_passed:
                    LOG.info(
                        "Test %s FAILED: expected=%r, actual=%r",
                        tc.get("test_id", len(test_results)),
                        expected[:100],
                        actual[:100],
                    )

                test_results.append(
                    {
                        "test_id": tc.get("test_id", len(test_results)),
                        "passed": test_passed,
                        "expected": expected,
                        "actual": actual,
                        "error": run_stderr,
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
            "compilation_errors": compilation_errors,
            "compilation_warnings": compilation_warnings,
            "tests_passed": passed,
            "tests_total": len(test_cases),
            "test_results": test_results,
        }


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
def compile_and_test_remote(
    code: str,
    test_cases: List[Dict[str, Any]],
    compiler_cmd: str = "cobc",
    compiler_flags: Optional[List[str]] = None,
    timeout: int = 30,
    debug: bool = False,
    soft_match: bool = False,
) -> Dict[str, Any]:
    """Ray remote wrapper for compile_and_test."""
    return compile_and_test(code, test_cases, compiler_cmd, compiler_flags, timeout, debug, soft_match)
