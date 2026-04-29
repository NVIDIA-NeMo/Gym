# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SWE-Bench-Ext test output parsing utilities for swe_agents.

Usage from SweBenchExtDatasetProcessor.postprocess_after_run():

    from responses_api_agents.swe_agents.swe_bench_ext.utils import parse_and_check_tests

    result = parse_and_check_tests(
        test_output=log_text,
        test_framework="pytest",
        fail_to_pass=["test_a", "test_b"],
        pass_to_pass=["test_c"],
        instance_id="my-task-123",
    )
    # result["resolved"] -> bool
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from responses_api_agents.swe_agents.swe_bench_ext.parsing import (
    normalize_test_id,
    parse_test_output,
)


# Marker strings used by generate_test_run_script to delimit structured output
_TEST_OUTPUT_START = "<<<SWE_BENCH_EXT_TEST_OUTPUT_START>>>"
_TEST_OUTPUT_END = "<<<SWE_BENCH_EXT_TEST_OUTPUT_END>>>"
_RESULT_FILE_START = "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>"
_RESULT_FILE_END = "<<<SWE_BENCH_EXT_RESULT_FILE_END>>>"


def _extract_between_markers(text: str, start: str, end: str) -> Optional[str]:
    """Extract text between two markers, or None if not found."""
    s = text.find(start)
    e = text.find(end)
    if s != -1 and e != -1 and s < e:
        return text[s + len(start) : e].strip()
    return None


def _match_test_with_fuzzy(
    test_id: str,
    parsed_results: Dict[str, str],
    build_failed_packages: set,
) -> str:
    """Fuzzy-match a test ID against parsed results.

    Mirrors SweBenchExtTask._match_test_with_fuzzy from
    benchmark-swe-bench-ext/swe_bench_ext/task.py.
    """
    # Direct match
    if test_id in parsed_results:
        return parsed_results[test_id]

    # Check if this test belongs to a package that failed to build
    for pkg in build_failed_packages:
        if test_id.startswith(pkg):
            return "FAILED"

    # Substring match (normalized IDs may differ in prefix)
    for parsed_id, status in parsed_results.items():
        if test_id in parsed_id or parsed_id in test_id:
            return status

    # Try matching by last component (after last ::)
    if "::" in test_id:
        suffix = test_id.rsplit("::", 1)[-1]
        for parsed_id, status in parsed_results.items():
            if "::" in parsed_id and parsed_id.rsplit("::", 1)[-1] == suffix:
                return status

    return "NOT_FOUND"


def parse_and_check_tests(
    test_output: str,
    test_framework: str,
    fail_to_pass: List[str],
    pass_to_pass: List[str],
    instance_id: str = "",
) -> Dict[str, Any]:
    """Parse test output and check FAIL_TO_PASS / PASS_TO_PASS resolution."""
    # Try to extract result file content from markers (same as task.py)
    result_file_content = _extract_between_markers(test_output, _RESULT_FILE_START, _RESULT_FILE_END)

    if result_file_content:
        parsed = parse_test_output(result_file_content, test_framework)
        if not parsed:
            parsed = parse_test_output(test_output, test_framework)
    else:
        parsed = parse_test_output(test_output, test_framework)

    if parsed is None:
        parsed = {}

    # Normalize parsed test IDs
    parsed = {normalize_test_id(tid, test_framework): status for tid, status in parsed.items()}

    # Normalize expected test IDs
    norm_f2p = [normalize_test_id(tid, test_framework) for tid in fail_to_pass]
    norm_p2p = [normalize_test_id(tid, test_framework) for tid in pass_to_pass]

    # Handle synthetic build/compile tests
    for tid in norm_f2p + norm_p2p:
        if (tid.endswith("::build") or tid.endswith("::compile")) and tid not in parsed:
            parsed[tid] = "PASSED"

    # Identify packages that failed to build
    build_failed_packages = {pkg for pkg, status in parsed.items() if status == "FAILED" and "::" not in pkg}

    # Match FAIL_TO_PASS
    f2p_results = {}
    for tid in norm_f2p:
        f2p_results[tid] = _match_test_with_fuzzy(tid, parsed, build_failed_packages)

    # Match PASS_TO_PASS
    p2p_results = {}
    for tid in norm_p2p:
        p2p_results[tid] = _match_test_with_fuzzy(tid, parsed, build_failed_packages)

    all_f2p_passed = all(v == "PASSED" for v in f2p_results.values()) if f2p_results else False
    all_p2p_passed = all(v == "PASSED" for v in p2p_results.values())
    resolved = all_f2p_passed and all_p2p_passed

    return {
        "resolved": resolved,
        "patch_exists": True,
        "patch_successfully_applied": True,
        "fail_to_pass_results": f2p_results,
        "pass_to_pass_results": p2p_results,
        "f2p_passed": sum(1 for v in f2p_results.values() if v == "PASSED"),
        "f2p_total": len(f2p_results),
        "p2p_passed": sum(1 for v in p2p_results.values() if v == "PASSED"),
        "p2p_total": len(p2p_results),
        "parsed_count": len(parsed),
        "framework": test_framework,
    }
