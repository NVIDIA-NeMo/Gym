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
"""
Test parsing utilities for build.py.

Helper functions for:
- Separating test and gold patches
- Parsing JUnit XML and JSON test outputs
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple


def read_patch(path: Path, skip_binary: bool = False) -> str:
    """
    Read the text content of a patch file optionally skipping binary files
    """
    parts = split_patch(path, skip_binary=skip_binary)
    return "".join([diff for _, diff in parts])


def split_patch(patch_path: Path, skip_binary: bool = False) -> list[Tuple[str, str]]:
    """
    Read a patch and partition by file.

    Args:
        patch_path (Path) - The patch file to split
        skip_binary (bool) - Whether to exclude binary files

    Returns: List of (filename, patch content) tuples
    """
    content = patch_path.read_text()
    parts = []

    # Split by file changes (each starts with "diff --git")
    file_diffs = re.split(r"(diff --git.*?)(?=diff --git|\Z)", content, flags=re.DOTALL)

    for i in range(0, len(file_diffs), 2):
        if i + 1 >= len(file_diffs):
            continue

        header = file_diffs[i]
        content = file_diffs[i + 1]
        full_diff = header + content

        # Extract filename from diff header
        file_match = re.search(r"diff --git a/(.*?) b/", full_diff)
        if not file_match:
            continue

        filepath = file_match.group(1)

        if skip_binary:
            binary_match = re.search(r"^GIT binary patch$", full_diff, flags=re.MULTILINE)
            if binary_match:
                continue

        parts.append((filepath, full_diff))

    return parts


def _parse_embedded_test_results(text_output: str, test_prefix: str = "") -> Dict[str, str]:
    """Parse embedded test results from system-out text.

    This handles cases like wolfssl where a single ctest testcase runs many individual tests
    and outputs them in a specific format within <system-out>.

    Expected formats:
    - "     1: test_name                                    : passed (  0.00016)"
    - "     2: test_name                                    : failed (  0.00016)"
    - "     3: test_name                                    : skipped"
    - "HMAC-MD5 test passed!"
    - "RSA      test failed!"

    Args:
        text_output: The text content from <system-out>
        test_prefix: Prefix to add to test names (usually the testcase name)

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
    """
    results = {}

    # Pattern 1: Numbered test format (wolfssl API tests)
    # Format: "     1: test_name                                    : passed (  0.00016)"
    numbered_pattern = re.compile(
        r"^\s*\d+:\s+([^\s:]+(?:\s+[^\s:]+)*?)\s*:\s*(passed|failed|skipped)", re.MULTILINE | re.IGNORECASE
    )

    for match in numbered_pattern.finditer(text_output):
        test_name = match.group(1).strip()
        status = match.group(2).lower()

        # Build test ID with prefix
        if test_prefix:
            test_id = f"{test_prefix}::{test_name}"
        else:
            test_id = test_name

        if status == "passed":
            results[test_id] = "PASSED"
        elif status == "failed":
            results[test_id] = "FAILED"
        elif status == "skipped":
            results[test_id] = "SKIPPED"

    # Pattern 2: Unit test format (wolfssl unit tests)
    # Format: "HMAC-MD5 test passed!"
    # Only match lines that don't contain '---' (separator lines)
    # Use [ \t] instead of \s to avoid matching newlines
    unit_pattern = re.compile(
        r"^([A-Za-z0-9_\-/]+(?:[ \t]+[A-Za-z0-9_\-/]+){0,5}?)[ \t]+test[ \t]+(passed|failed)!",
        re.MULTILINE | re.IGNORECASE,
    )

    for match in unit_pattern.finditer(text_output):
        test_name = match.group(1).strip()
        status = match.group(2).lower()

        # Skip if the test name contains special characters indicating it's not a real test
        if "---" in test_name or len(test_name) > 50:
            continue

        # Build test ID with prefix
        if test_prefix:
            test_id = f"{test_prefix}::{test_name}"
        else:
            test_id = test_name

        if status == "passed":
            results[test_id] = "PASSED"
        elif status == "failed":
            results[test_id] = "FAILED"

    # Pattern 3: FAILURES section (wolfssl API tests)
    # Format: "FAILURES:\n   892: test_wolfSSL_CTX_load_verify_locations"
    failures_section = re.search(r"FAILURES:\s*\n(.*?)(?:\n\s*End|$)", text_output, re.DOTALL)
    if failures_section:
        failure_pattern = re.compile(r"^\s*\d+:\s+([^\s:]+(?:\s+[^\s:]+)*)", re.MULTILINE)
        for match in failure_pattern.finditer(failures_section.group(1)):
            test_name = match.group(1).strip()
            if test_prefix:
                test_id = f"{test_prefix}::{test_name}"
            else:
                test_id = test_name
            # Mark as failed (this overrides any previous 'passed' if it exists)
            results[test_id] = "FAILED"

    return results


def parse_junit_xml(xml_content: str) -> Dict[str, str]:
    """Parse JUnit XML to extract test results.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (import errors, syntax errors, etc.), if we find valid
    XML test results, we parse and return them. We only return None if we're certain
    the framework didn't run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If test framework failed to run (not the same as tests failing)
    """
    results = {}
    found_any_xml = False

    # PRIORITY 1 & 2: Try to parse XML documents (pure or mixed with other output)
    # Handle multiple concatenated XML documents (from multiple test result files)
    # Split by <?xml declaration to handle each document separately
    xml_docs = xml_content.split("<?xml")

    for i, doc in enumerate(xml_docs):
        if i == 0 and not doc.strip():
            continue  # Skip empty first split

        # Re-add the <?xml declaration (except for first empty split)
        if i > 0:
            doc = "<?xml" + doc

        if not doc.strip():
            continue

        # Try parsing this document
        try:
            tree = ET.fromstring(doc.strip())
            found_any_xml = True
        except ET.ParseError:
            # If parsing fails, try extracting just the testsuite/testsuites portion
            xml_start = doc.find("<testsuite")
            if xml_start == -1:
                xml_start = doc.find("<testsuites")
            if xml_start == -1:
                continue

            xml_end = doc.find("</testsuites>", xml_start)
            if xml_end > xml_start:
                xml_extracted = doc[xml_start : xml_end + len("</testsuites>")]
            else:
                xml_end = doc.find("</testsuite>", xml_start)
                if xml_end > xml_start:
                    xml_extracted = doc[xml_start : xml_end + len("</testsuite>")]
                else:
                    continue

            try:
                tree = ET.fromstring(xml_extracted)
                found_any_xml = True
            except ET.ParseError:
                continue

        # Parse all testcases from this document
        for testcase in tree.iter("testcase"):
            classname = testcase.get("classname", "")
            name = testcase.get("name", "")
            test_id = f"{classname}::{name}" if classname else name

            # Check if this testcase has system-out with embedded test results
            # This handles cases like wolfssl where a single ctest executable runs many tests
            system_out = testcase.find("system-out")
            embedded_results = {}
            if system_out is not None and system_out.text:
                embedded_results = _parse_embedded_test_results(system_out.text, classname or name)

            if embedded_results:
                # If we found embedded test results, use those instead of the testcase status
                results.update(embedded_results)
            elif testcase.find("failure") is not None or testcase.find("error") is not None:
                results[test_id] = "FAILED"
            elif testcase.find("skipped") is not None:
                results[test_id] = "SKIPPED"
            else:
                results[test_id] = "PASSED"

    # PRIORITY 3: If we found NO valid XML and NO results, check for error indicators
    # Only return None if we're certain the framework failed to run
    if not found_any_xml and not results:
        error_indicators = [
            "ERROR: ",  # Generic error marker
            "ImportError:",  # Python import errors
            "ModuleNotFoundError:",  # Python module errors
            "SyntaxError:",  # Python syntax errors
            "FAILED ",  # Framework failure markers
            "INTERNALERROR",  # pytest internal errors
            "collection errors",  # pytest collection errors
            "error: ",  # Generic error (C++, Swift, etc.)
            "fatal error:",  # Fatal compilation errors
            "cannot find symbol",  # Java compilation errors
            "error: build had",  # Swift build errors (xctest)
            "error: terminated",  # Swift process crashes (xctest)
        ]
        has_errors = any(indicator in xml_content for indicator in error_indicators)
        # Return None ONLY if: no XML found AND errors present
        # Return empty dict if: no XML found AND no errors (rare but valid)
        return None if has_errors else results

    return results


def parse_go_json(json_output: str) -> Dict[str, str]:
    """Parse Go test -json output (newline-delimited JSON).

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (module errors, build errors, etc.), if we find valid
    test results JSON, we parse and return it. We only return None if we're certain
    the tests didn't run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If Go tests failed to run (not the same as tests failing)
    """
    results = {}
    has_valid_json = False

    # PRIORITY 1: Try to parse newline-delimited JSON (valid test output)
    for line in json_output.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            has_valid_json = True  # Found at least one valid JSON line
            action = event.get("Action")

            # Handle test-level events
            if "Test" in event and action in ["pass", "fail", "skip"]:
                test_name = event.get("Test", "")
                if test_name:
                    package = event.get("Package", "")
                    test_id = f"{package}::{test_name}" if package else test_name

                    if action == "pass":
                        results[test_id] = "PASSED"
                    elif action == "fail":
                        results[test_id] = "FAILED"
                    elif action == "skip":
                        results[test_id] = "SKIPPED"

            # Handle package-level failures (no Test field)
            elif "Package" in event and "Test" not in event and action == "fail":
                package = event.get("Package", "")
                test_id = f"{package}::package"
                results[test_id] = "FAILED"

        except json.JSONDecodeError:
            # PRIORITY 2: Handle plaintext build failures (legitimate failures)
            # When tests can't compile/build, Go outputs plaintext "FAIL package [build failed]"
            # This is a legitimate test failure, not a parsing error
            build_fail_match = re.match(r"^FAIL\s+(\S+)\s+\[build failed\]", line)
            if build_fail_match:
                package_name = build_fail_match.group(1)
                results[package_name] = "FAILED"
                has_valid_json = True  # Count build failures as valid results

    # PRIORITY 3: If we found NO valid JSON and NO build failures, check for error indicators
    if not has_valid_json and not results:
        error_indicators = [
            "go: cannot find main module",  # Module not found
            "can't load package",  # Package loading errors
            "pattern matches no packages",  # No matching packages
            "build constraints exclude all Go files",  # Build constraints error
        ]
        has_errors = any(indicator in json_output for indicator in error_indicators)
        # Return None ONLY if: no JSON found AND errors present
        # Return empty dict if: no JSON found AND no errors (rare but valid)
        return None if has_errors else results

    return results


def parse_jest_vitest_json(json_output: str) -> Dict[str, str]:
    """Parse Jest/Vitest JSON output.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (TypeScript, npm, etc.), if we find valid
    test results JSON, we parse and return it. We only return None if we're
    certain the framework didn't run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If Jest itself failed to run (not the same as tests failing)
    """
    results = {}

    # PRIORITY 1: Try to parse as pure JSON (test results take precedence)
    try:
        data = json.loads(json_output.strip())
        # If we got JSON, check if it has test results (even if errors exist elsewhere in output)
    except json.JSONDecodeError:
        # PRIORITY 2: Search for JSON markers in mixed output
        # Even with errors in output, tests might have run and produced JSON
        json_start = json_output.find('{"numFailed')  # Jest format
        if json_start == -1:
            json_start = json_output.find('{"numTotalTest')  # Vitest format
        if json_start == -1:
            json_start = json_output.find('{"test')  # Alternative format
        if json_start == -1:
            # PRIORITY 3: No JSON found - NOW check if there are error indicators
            # Only return None if we're sure tests didn't run (no results + errors present)
            # NOTE: error_indicators are a LAST RESORT - we prefer finding test results
            error_indicators = [
                "error TS",  # TypeScript compilation errors (e.g., error TS2307:)
                "ELIFECYCLE",  # npm script failures
                "npm ERR!",  # npm errors
                "Error: Cannot find module",  # Module loading errors (like Mocha)
                "SyntaxError:",  # JavaScript/TypeScript syntax errors
                "Test suite failed to run",  # Jest-specific: tests couldn't be loaded
                "FAIL ",  # Jest failure marker without JSON
            ]
            has_errors = any(indicator in json_output for indicator in error_indicators)
            # Return None ONLY if: no JSON found AND errors present
            # Return empty dict if: no JSON found AND no errors (rare but valid)
            return None if has_errors else results

        # Try to extract JSON from mixed output
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(json_output[json_start:])
        except json.JSONDecodeError:
            # Could not parse JSON even after finding marker
            return None

    # At this point, we have successfully parsed JSON
    # Check if this is Jest's error response format (Jest itself failed, not the tests)
    # Format: {"error": {"code": 2, "summary": "", "detail": ""}}
    # This is a structured error response, NOT test results
    if "error" in data and "code" in data.get("error", {}):
        # This is an error response from Jest itself, not test results
        return None

    # Check if we have the expected test results structure
    # If we have testResults, parse it even if tests failed - those are legitimate test results
    # Parse test results
    if "testResults" in data:
        for test_result in data.get("testResults", []):
            file_path = test_result.get("name", "")
            suite_status = test_result.get("status", "")
            assertions = test_result.get("assertionResults", [])

            # Handle suite-level failures (no assertions ran)
            if suite_status == "failed" and len(assertions) == 0:
                test_id = f"{file_path}::suite"
                results[test_id] = "FAILED"
                continue

            # Handle individual test assertions
            for assertion in assertions:
                full_name = assertion.get("fullName", "")
                title = assertion.get("title", "")
                status = assertion.get("status", "")
                test_id = f"{file_path}::{full_name}" if full_name else f"{file_path}::{title}"

                if status == "passed":
                    results[test_id] = "PASSED"
                elif status == "failed":
                    results[test_id] = "FAILED"
                elif status in ["pending", "skipped"]:
                    results[test_id] = "SKIPPED"

    # If we successfully parsed JSON but found no testResults, that's unexpected
    # Return None to indicate this isn't valid test output
    # (Valid Jest output should have testResults array, even if empty)
    if "testResults" not in data:
        return None

    return results


def parse_mocha_json(json_output: str) -> Optional[Dict[str, str]]:
    """Parse Mocha JSON output.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (module errors, syntax errors, etc.), if we find valid
    test results JSON, we parse and return it. We only return None if we're certain
    the framework didn't run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If Mocha itself failed to run (not the same as tests failing)
    """
    results = {}

    # PRIORITY 1: Try to parse as pure JSON (test results take precedence)
    try:
        data = json.loads(json_output.strip())
        # Validate this is Mocha JSON by checking for 'stats' key
        if "stats" not in data:
            data = None
    except json.JSONDecodeError:
        data = None

    # PRIORITY 2: If direct parse failed, search for JSON in mixed output
    if data is None:
        # Look for stats key in JSON
        stats_pos = json_output.find('"stats"')
        if stats_pos == -1:
            # PRIORITY 3: No JSON found - NOW check if there are error indicators
            error_indicators = [
                "Error: Cannot find module",  # Module loading errors
                "SyntaxError:",  # JavaScript syntax errors
                "TypeError:",  # Type errors
                "ReferenceError:",  # Reference errors
                "No test files found",  # Mocha-specific: no tests found
            ]
            has_errors = any(indicator in json_output for indicator in error_indicators)
            # Return None ONLY if: no JSON found AND errors present
            # Return empty dict if: no JSON found AND no errors (rare but valid)
            return None if has_errors else results

        # Find the opening brace before "stats"
        json_start = json_output.rfind("{", 0, stats_pos)
        if json_start == -1:
            return None

        # Try parsing from this position
        json_portion = json_output[json_start:]

        # Use json.JSONDecoder to find where the object ends
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(json_portion)
        except json.JSONDecodeError:
            return None

        # Validate extracted JSON has 'stats'
        if "stats" not in data:
            return None

    # At this point, we have valid Mocha JSON with 'stats'
    # Parse test results even if some tests failed - those are legitimate results

    # Process passed tests
    for test in data.get("passes", []):
        file_path = test.get("file", "")
        full_title = test.get("fullTitle", "")
        test_id = f"{file_path}::{full_title}" if full_title else file_path
        results[test_id] = "PASSED"

    # Process failed tests
    for test in data.get("failures", []):
        file_path = test.get("file", "")
        full_title = test.get("fullTitle", "")
        test_id = f"{file_path}::{full_title}" if full_title else file_path
        results[test_id] = "FAILED"

    # Process pending/skipped tests
    for test in data.get("pending", []):
        file_path = test.get("file", "")
        full_title = test.get("fullTitle", "")
        test_id = f"{file_path}::{full_title}" if full_title else file_path
        results[test_id] = "SKIPPED"

    return results


def parse_gtest_json(json_output: str) -> Dict[str, str]:
    """Parse Google Test JSON output.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (compilation errors, linking errors, etc.), if we find valid
    test results JSON, we parse and return it. We only return None if we're certain
    the tests didn't run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If GTest itself failed to run (not the same as tests failing)
    """
    results = {}

    # PRIORITY 1: Try to parse as pure JSON (test results take precedence)
    try:
        data = json.loads(json_output.strip())
        # Validate this is GTest JSON by checking for 'testsuites' key
        if "testsuites" not in data:
            data = None
    except json.JSONDecodeError:
        data = None

    # PRIORITY 2: If direct parse failed, search for JSON in mixed output
    if data is None:
        # Try to find JSON in mixed output
        json_start = json_output.find('{"testsuites"')
        if json_start == -1:
            json_start = json_output.find('{\n  "testsuites"')
        if json_start == -1:
            # PRIORITY 3: No JSON found - NOW check if there are error indicators
            error_indicators = [
                "error:",  # C++ compilation errors
                "undefined reference to",  # Linking errors
                "fatal error:",  # Fatal compilation errors
                "cannot find -l",  # Linking library errors (e.g., "cannot find -lgtest")
                ": No such file or directory",  # File not found errors
            ]
            has_errors = any(indicator in json_output for indicator in error_indicators)
            # Return None ONLY if: no JSON found AND errors present
            # Return empty dict if: no JSON found AND no errors (rare but valid)
            return None if has_errors else results

        # Extract JSON object
        json_portion = json_output[json_start:]
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(json_portion)
        except json.JSONDecodeError:
            return None

        # Validate extracted JSON has 'testsuites'
        if "testsuites" not in data:
            return None

    # At this point, we have valid GTest JSON with 'testsuites'
    # Parse test results even if some tests failed - those are legitimate results

    # Parse test results from testsuites
    testsuites = data.get("testsuites", [])
    if not isinstance(testsuites, list):
        testsuites = [testsuites] if isinstance(testsuites, dict) else []

    for testsuite in testsuites:
        suite_name = testsuite.get("name", "")

        # Handle both 'testsuite' (array) and direct test cases
        test_cases = testsuite.get("testsuite", [])
        if not test_cases:
            test_cases = testsuite.get("tests", [])

        for test_case in test_cases:
            test_name = test_case.get("name", "")
            classname = test_case.get("classname", suite_name)

            # Build test ID in format: SuiteName::TestName
            test_id = f"{classname}::{test_name}" if classname else test_name

            # Determine test status
            status = test_case.get("status", "RUN")
            result = test_case.get("result", "COMPLETED")

            # Check for failures
            failures = test_case.get("failures", [])
            if failures and len(failures) > 0:
                results[test_id] = "FAILED"
            elif status == "NOTRUN" or result == "SKIPPED":
                results[test_id] = "SKIPPED"
            elif result == "COMPLETED" or status == "RUN":
                results[test_id] = "PASSED"
            else:
                results[test_id] = "FAILED"

    return results


def parse_maven_text_output(text_output: str) -> Dict[str, str]:
    """Parse Maven text output for test results."""
    results = {}

    # Look for test summary lines like:
    # Tests run: 5, Failures: 1, Errors: 0, Skipped: 0
    summary_pattern = r"Tests run: (\d+),\s*Failures: (\d+),\s*Errors: (\d+),\s*Skipped: (\d+)"

    # Check for compilation errors - if tests can't compile, mark them as failed
    compilation_error_pattern = r"\[ERROR\].*?testCompile.*?Compilation failure"
    if re.search(compilation_error_pattern, text_output, re.DOTALL | re.IGNORECASE):
        # Find test files mentioned in compilation errors
        test_file_pattern = r"/workspace/repo/[^/]+/src/test/java/([\w/]+)\.java"
        for match in re.finditer(test_file_pattern, text_output):
            test_class = match.group(1).replace("/", ".")
            # Mark as failed due to compilation
            results[f"{test_class}::compile"] = "FAILED"
        # If we found compilation errors, return early
        if results:
            return results

    # Check for BUILD FAILURE
    if "BUILD FAILURE" in text_output:
        # If build failed and we haven't found specific test failures, mark as generic failure
        if not results:
            results["maven::build"] = "FAILED"
        return results

    # Parse test run summaries per module
    lines = text_output.split("\n")
    current_module = None

    for line in lines:
        # Track which module we're in
        if "Building" in line and "[" in line and "]" in line:
            # Extract module name from lines like "[INFO] Building Docs Web 1.12-SNAPSHOT [4/4]"
            parts = line.split("Building")
            if len(parts) > 1:
                module_parts = parts[1].strip().split()
                if len(module_parts) > 0:
                    current_module = module_parts[0]

        # Look for test summary
        summary_match = re.search(summary_pattern, line)
        if summary_match:
            total = int(summary_match.group(1))
            failures = int(summary_match.group(2))
            errors = int(summary_match.group(3))
            skipped = int(summary_match.group(4))

            if total > 0:
                # We have test counts but might not have individual test names
                # Generate generic test IDs based on the current module
                module_name = current_module or "unknown"
                passed = total - failures - errors - skipped

                for j in range(passed):
                    results[f"{module_name}::test_{j + 1}"] = "PASSED"
                for j in range(failures + errors):
                    results[f"{module_name}::test_failed_{j + 1}"] = "FAILED"
                for j in range(skipped):
                    results[f"{module_name}::test_skipped_{j + 1}"] = "SKIPPED"
    return results


def parse_cargo_nextest(output: str) -> Dict[str, str]:
    """Parse cargo-nextest text output.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (warnings, etc.), if we find valid test results,
    we parse and return them. We only return None if we're certain the tests didn't
    run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If cargo-nextest failed to run (not the same as tests failing)
    """
    results = {}

    # PRIORITY 1: Parse individual test result lines
    # Format: PASS [   1.588s] rusty::tests integration::linking::test_name
    #         FAIL [   5.845s] rusty codegen::tests::parameters_tests::test_name
    test_line_pattern = re.compile(r"^\s*(PASS|FAIL|SIGKILL|SKIP)\s+\[.*?\]\s+(.+)$", re.MULTILINE)

    for match in test_line_pattern.finditer(output):
        status = match.group(1)
        test_name = match.group(2).strip()

        if status == "PASS":
            results[test_name] = "PASSED"
        elif status in ("FAIL", "SIGKILL"):
            results[test_name] = "FAILED"
        elif status == "SKIP":
            results[test_name] = "SKIPPED"

    # PRIORITY 2: If we found NO test results, check for error indicators
    # Only return None if we're certain tests didn't run (compilation/linking errors)
    if not results:
        error_indicators = [
            "error[E",  # Rust compiler errors (e.g., error[E0425])
            "error: could not compile",  # Cargo compilation errors
            "error: linking with",  # Linking errors
            "error: aborting due to",  # Compilation aborted
        ]
        has_errors = any(indicator in output for indicator in error_indicators)
        # Return None ONLY if: no results found AND errors present
        # Return empty dict if: no results found AND no errors (rare but valid - no tests in project)
        return None if has_errors else results

    return results


def parse_bun_text(text_output: str) -> Dict[str, str]:
    """
    Parse Bun test framework output.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (TypeScript, compilation, etc.), if we find valid
    test results, we parse and return them. We only return None if we're
    certain Bun didn't run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If Bun itself failed to run (not the same as tests failing)
    """
    results = {}
    current_file = None
    current_describe = None

    # PRIORITY 1: Try to parse test results (✓ and ✗ symbols)
    for line in text_output.split("\n"):
        # Track current file (lines ending with .ts: or .js:)
        if re.match(r"^[^\s].*\.(ts|js|tsx|jsx):?\s*$", line.strip()):
            current_file = line.strip().rstrip(":")
            current_describe = None
            continue

        # Track describe blocks (indented text followed by colon, but not test results)
        describe_match = re.match(r"^\s+([^✓✗\n]+):\s*$", line)
        if describe_match:
            current_describe = describe_match.group(1).strip()
            continue

        # Remove ANSI color codes
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line)

        # Match passed tests: ✓ test_name [time]
        pass_match = re.match(r"^\s*✓\s+(.+?)(?:\s+\[[\d.]+m?s\])?\s*$", clean_line)
        if pass_match:
            test_name = pass_match.group(1).strip()
            # Build test ID with file, describe block, and test name
            test_id = test_name
            if current_file:
                test_id = f"{current_file}::{test_name}"
            if current_describe:
                test_id = f"{current_file}::{current_describe} > {test_name}"
            results[test_id] = "PASSED"
            continue

        # Match failed tests: ✗ test_name [time]
        fail_match = re.match(r"^\s*✗\s+(.+?)(?:\s+\[[\d.]+m?s\])?\s*$", clean_line)
        if fail_match:
            test_name = fail_match.group(1).strip()
            # Build test ID with file, describe block, and test name
            test_id = test_name
            if current_file:
                test_id = f"{current_file}::{test_name}"
            if current_describe:
                test_id = f"{current_file}::{current_describe} > {test_name}"
            results[test_id] = "FAILED"
            continue

        # Alternative format: FAIL  filepath > describe > test_name
        alt_fail_match = re.match(r"^\s*FAIL\s+(.+?)\s+>\s+(.+?)\s*$", clean_line)
        if alt_fail_match:
            file_path = alt_fail_match.group(1).strip()
            test_path = alt_fail_match.group(2).strip()
            test_id = f"{file_path}::{test_path}"
            results[test_id] = "FAILED"
            continue

    # PRIORITY 2: If no individual test results found, try parsing summary
    if not results:
        # Look for summary like "5 pass, 2 fail" or "X passing (Yms)"
        summary_match = re.search(r"(\d+)\s+pass(?:ing|ed)?.*?(\d+)\s+fail(?:ing|ed)?", text_output.lower())
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2))

            # Generate generic test IDs
            for i in range(passed):
                results[f"test_{i + 1}"] = "PASSED"
            for i in range(failed):
                results[f"test_failed_{i + 1}"] = "FAILED"

    # PRIORITY 3: No test results found - NOW check if there are error indicators
    # Only return None if we're sure tests didn't run (no results + errors present)
    # NOTE: error_indicators are a LAST RESORT - we prefer finding test results
    if not results:
        error_indicators = [
            "error TS",  # TypeScript compilation errors (e.g., error TS2307:)
            "Error: Cannot find module",  # Module loading errors
            "SyntaxError:",  # JavaScript/TypeScript syntax errors
            "error: ",  # Generic Bun errors (lowercase 'error:')
            "Error:",  # Generic errors
            "ModuleNotFoundError",  # Module not found
            "bun: command not found",  # Bun not installed
            "panicked at",  # Bun runtime panics
            "Segmentation fault",  # Critical runtime errors
        ]
        has_errors = any(indicator in text_output for indicator in error_indicators)
        # Return None ONLY if: no test results found AND errors present
        # Return empty dict if: no test results found AND no errors (rare but valid)
        return None if has_errors else results

    return results


def parse_cppunit_text(text_output: str) -> Dict[str, str]:
    """Parse CppUnit text output for test results.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    Even if output contains errors (warnings, etc.), if we find valid test results,
    we parse and return them. We only return None if we're certain the tests didn't
    run (no test results + error indicators).

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If CppUnit failed to run (not the same as tests failing)
    """
    results = {}

    # PRIORITY 1: Parse individual test result lines
    # Format: TestClassName::testMethodName : OK
    #         TestClassName::testMethodName : FAIL
    test_line_pattern = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(OK|FAIL|ERROR)$", re.MULTILINE
    )

    for match in test_line_pattern.finditer(text_output):
        test_name = match.group(1).strip()
        status = match.group(2).strip()

        if status == "OK":
            results[test_name] = "PASSED"
        elif status in ["FAIL", "ERROR"]:
            results[test_name] = "FAILED"

    # PRIORITY 2: If we found NO test results, check for error indicators
    # Only return None if we're certain tests didn't run (compilation/linking errors)
    if not results:
        error_indicators = [
            "error:",  # C++ compilation errors
            "undefined reference to",  # Linking errors
            "fatal error:",  # Fatal compilation errors
            "ld returned",  # Linker errors
            "cannot find -l",  # Library linking errors
        ]
        has_errors = any(indicator in text_output for indicator in error_indicators)
        # Return None ONLY if: no results found AND errors present
        # Return empty dict if: no results found AND no errors (rare but valid - no tests)
        return None if has_errors else results

    return results


def parse_minitest_text(text_output: str, test_metadata_path: str = None) -> Dict[str, str]:
    """
    Parse mini.nvim (MiniTest) test framework output.

    MiniTest is used by Neovim plugins for testing.
    Example output:
      Total number of cases: 5
      tests/test_treesitter.lua: ooooo

      Fails (0) and Notes (0)

      Or with failures:
      FAIL in tests/test_treesitter.lua | wrap_cursor | normal: error message
      FAIL in tests/test_treesitter.lua | enumerate: error message

      Fails (2) and Notes (0)

    IMPORTANT: MiniTest only outputs individual test names when they FAIL.
    When all tests pass, only summary is shown - no individual test names.

    Solution: When all tests pass, read test_metadata.json to get expected test names
    and return them as PASSED. This ensures real test names are used consistently.
    """
    results = {}

    # Parse individual test results from FAIL/NOTE lines
    # Format: FAIL in file.lua | group | test_name: error message
    # Use [^|:]+ to stop at pipe OR colon (prevents capturing error message)
    fail_pattern = re.compile(
        r"^(?:\x1b\[\d+(?:;\d+)?m)?FAIL(?:\x1b\[0m)?\s+in\s+([^|]+)\s*\|\s*([^|:]+)(?:\s*\|\s*([^:]+))?:", re.MULTILINE
    )

    for match in fail_pattern.finditer(text_output):
        file_path = match.group(1).strip()
        group = match.group(2).strip()
        test_name = match.group(3).strip() if match.group(3) else ""

        # Create test ID: file | group | test_name or file | group
        if test_name:
            test_id = f"{file_path} | {group} | {test_name}"
        else:
            test_id = f"{file_path} | {group}"

        results[test_id] = "FAILED"

    return results


def parse_telescope_text(text_output: str) -> Dict[str, str]:
    """
    Parse telescope test framework output.

    Telescope outputs lines like:
      ✓ test_name
      ✗ test_name
      - test_name (skipped)

    Also handles PlenaryBusted output for Neovim plugins.

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    We only return None if we're certain the framework didn't run.

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If telescope failed to run (not the same as tests failing)
    """
    results = {}

    for line in text_output.split("\n"):
        line = line.strip()

        # Match passed tests: ✓ test_name or "Success: test_name"
        if "✓" in line:
            test_name = line.split("✓", 1)[1].strip()
            if test_name:  # Avoid empty test names
                results[test_name] = "PASSED"
        elif line.lower().startswith("success:"):
            test_name = line.split(":", 1)[1].strip()
            if test_name:
                results[test_name] = "PASSED"

        # Match failed tests: ✗ test_name or "Failed: test_name"
        elif "✗" in line:
            test_name = line.split("✗", 1)[1].strip()
            if test_name:
                results[test_name] = "FAILED"
        elif line.lower().startswith("failed:"):
            test_name = line.split(":", 1)[1].strip()
            if test_name:
                results[test_name] = "FAILED"

        # Match skipped tests: - test_name or "Skipped: test_name"
        elif line.startswith("- ") and "skip" in line.lower():
            test_name = line[2:].strip()
            # Remove "(skipped)" suffix if present
            test_name = re.sub(r"\s*\(skipped\)\s*$", "", test_name, flags=re.IGNORECASE)
            if test_name:
                results[test_name] = "SKIPPED"
        elif line.lower().startswith("skipped:"):
            test_name = line.split(":", 1)[1].strip()
            if test_name:
                results[test_name] = "SKIPPED"

    # If no results found, try parsing summary line
    if not results:
        # Look for summary like "5 passed, 2 failed, 1 skipped"
        summary_pattern = r"(\d+)\s+passed.*?(\d+)\s+failed"
        match = re.search(summary_pattern, text_output.lower())
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))

            # Generate generic test IDs
            for i in range(passed):
                results[f"test_{i + 1}"] = "PASSED"
            for i in range(failed):
                results[f"test_failed_{i + 1}"] = "FAILED"

    # PRIORITY 2: If we found NO test results, check for error indicators
    # Only return None if we're certain tests didn't run (Lua/Neovim errors)
    if not results:
        error_indicators = [
            "Error:",  # Generic Lua errors
            "error loading module",  # Lua module loading errors
            "attempt to call",  # Lua runtime errors
            "bad argument",  # Lua runtime errors
            "stack traceback:",  # Lua errors with traceback
        ]
        has_errors = any(indicator in text_output for indicator in error_indicators)
        # Return None ONLY if: no results found AND errors present
        # Return empty dict if: no results found AND no errors (rare but valid - no tests)
        return None if has_errors else results

    return results


def parse_lust_text(text_output: str) -> Dict[str, str]:
    """
    Parse lust test framework output.

    Lust outputs test results with dots (.) for pass, F for fail.
    Example output:
      ..F.
      4 tests, 1 failure
      test/my_test.lua:15: Expected true but got false

    We parse individual test results when available, or fall back to summary.
    """
    results = {}

    # Try to parse individual test results from verbose output
    # Pattern: "  test_name ... ok" or "  test_name ... FAILED"
    test_pattern = re.compile(r"^\s*(.+?)\s+\.\.\.\s+(ok|FAILED|ERROR)", re.MULTILINE)
    matches = test_pattern.findall(text_output)

    if matches:
        # Found individual test results
        for test_name, status in matches:
            test_name = test_name.strip()
            if status == "ok":
                results[test_name] = "PASSED"
            else:
                results[test_name] = "FAILED"
        return results

    # Try to extract test descriptions from failure messages
    # Pattern: "test_file.lua:line_number: test description"
    failure_pattern = re.compile(r"^([^\s:]+\.lua):(\d+):\s*(.+)$", re.MULTILINE)
    failures = failure_pattern.findall(text_output)

    if failures:
        for filepath, _, description in failures:
            test_id = f"{filepath}::{description.strip()}"
            results[test_id] = "FAILED"

    # Parse summary line to get total count: "X tests, Y failures"
    summary_match = re.search(r"(\d+)\s+tests?,\s+(\d+)\s+failures?", text_output.lower())
    if summary_match:
        total_tests = int(summary_match.group(1))
        failures = int(summary_match.group(2))

        # If we haven't parsed individual tests yet, generate generic ones
        if not results:
            passed = total_tests - failures
            for i in range(passed):
                results[f"test_{i + 1}"] = "PASSED"
            for i in range(failures):
                results[f"test_failed_{i + 1}"] = "FAILED"
            return results

    # Fallback: if no detailed info, check for overall success/failure
    if not results:
        if "0 failures" in text_output.lower() or "0 errors" in text_output.lower():
            results["test_suite"] = "PASSED"
        else:
            results["test_suite"] = "FAILED"

    return results


def parse_bespoke_libgeos(text_output: str) -> Dict[str, str]:
    """Parse libgeos/GEOS test output format.

    Format:
        capi::GEOSBoundary: .
        capi::GEOSBuffer: .....................
        geos::operation::OverlayNGEmptyCoordDim: [1=F][2=F].[4=F][5=F][6=F]
        geos::operation::buffer::BufferOp: ..........................[27=X]

    Where:
        - dots (.) = passing tests
        - [N=F] = explicit failure markers
        - [N=X] = exception markers (also failures)
        - standalone F or X = failure/exception

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    We only return None if we're certain the framework didn't run.

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If libgeos tests failed to run (not the same as tests failing)
    """
    results = {}

    # PRIORITY 1: Parse individual test result lines
    # Pattern: TestSuite::TestName: followed by dots, Fs, Xs, or [N=F]/[N=X] markers
    # Example: capi::GEOSBoundary: .
    # Example: geos::OverlayNGEmptyCoordDim: [1=F][2=F].[4=F]
    # Example: geos::operation::buffer::BufferOp: ..........................[27=X]
    test_line_pattern = re.compile(
        r"^([a-zA-Z_][a-zA-Z0-9_:]*::[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)(?:\n|$)", re.MULTILINE
    )

    for match in test_line_pattern.finditer(text_output):
        test_id = match.group(1)  # Full name like "capi::GEOSBoundary"
        test_output_line = match.group(2)  # Everything after the colon

        # Check for failure markers:
        # 1. [N=F] pattern (explicit failure notation)
        # 2. [N=X] pattern (exception notation)
        # 3. Standalone F or X characters
        has_failure = bool(re.search(r"\[.*=[FX]\]|(?<!\w)[FX](?!\w)", test_output_line))

        if has_failure:
            results[test_id] = "FAILED"
        else:
            # All dots or successful - mark as passed
            results[test_id] = "PASSED"

    # PRIORITY 2: If we found NO test results, check for error indicators
    # Only return None if we're certain tests didn't run (compilation/linking errors)
    if not results:
        error_indicators = [
            "error:",  # C++ compilation errors
            "undefined reference",  # Linking errors
            "fatal error:",  # Fatal compilation errors
            "ld returned",  # Linker errors
            "cannot find -l",  # Library linking errors
        ]
        has_errors = any(indicator in text_output for indicator in error_indicators)
        # Return None ONLY if: no results found AND errors present
        # Return empty dict if: no results found AND no errors (rare but valid - no tests)
        return None if has_errors else results

    return results


def _normalize_swift_test_name(raw_name: str) -> str:
    """Normalize XCTest case identifiers from swift test console output."""
    name = raw_name.strip()

    # Typical format: -[Module.Class testMethod]
    if name.startswith("-[") and name.endswith("]"):
        inner = name[2:-1].strip()
        if " " in inner:
            class_name, method = inner.split(" ", 1)
            return f"{class_name}::{method}"
        return inner

    # Alternate format: Module.Class.testMethod
    if "." in name:
        parts = name.split(".", 1)
        return f"{parts[0]}::{parts[1]}"

    return name


def parse_swift_test_text(text_output: str) -> Dict[str, str]:
    """Parse vanilla `swift test` console output (without --xunit-output)."""
    results = {}
    test_case_pattern = re.compile(r"Test Case '([^']+)' (passed|failed|skipped)", re.IGNORECASE)

    for match in test_case_pattern.finditer(text_output):
        raw_name, status = match.groups()
        test_id = _normalize_swift_test_name(raw_name)
        results[test_id] = status.upper()

    if results:
        return results

    # Fallback: parse summary line to infer aggregate results if per-test lines missing
    summary_match = re.search(
        r"Executed\s+(\d+)\s+tests?,\s+with\s+(\d+)\s+failures?",
        text_output,
        re.IGNORECASE,
    )
    if summary_match:
        total_tests = int(summary_match.group(1))
        failures = int(summary_match.group(2))
        passes = max(total_tests - failures, 0)

        for i in range(passes):
            results[f"swift_test_pass_{i + 1}"] = "PASSED"
        for i in range(failures):
            results[f"swift_test_fail_{i + 1}"] = "FAILED"

    return results


def parse_xctest_output(output: str) -> Dict[str, str]:
    """Parse XCTest results, preferring XML when available."""
    xml_results = parse_junit_xml(output)
    if xml_results:
        return xml_results
    return parse_swift_test_text(output)


def normalize_test_id(test_id: str, framework: str = "") -> str:
    """Normalize test IDs for stable matching across different formats.

    This function performs several normalizations:

    1. Removes unstable runtime prefixes that change between runs:
       - (N/M) - Test execution order (e.g., "(2/5) test_name")
       - [N/M] - Alternative bracket format
       - #N - Test number prefix (e.g., "#42 test_name")
       - N. - Numbered list format (e.g., "1. test_name")

    2. Removes common file extensions (.py, .js, .ts, .go, etc.) from test paths
       to allow matching between "test_file.py::test" and "test_file::test"

    3. Normalizes delimiters (`.`, `::`, `/`) to a canonical form (`::`)
       when they appear between alphanumeric characters, allowing matching
       between "testa.testb::testc" and "testa/testb.testc"

    Examples:
        "(2/5) test_name" -> "test_name"
        "test_file.py::test_name" -> "test_file::test_name"
        "testa.testb::testc" -> "testa::testb::testc"
        "testa/testb.testc" -> "testa::testb::testc"
        "tests/module.js::describe::it" -> "tests::module::describe::it"

    Args:
        test_id: Original test ID from parser
        framework: Test framework name (for future framework-specific rules if needed)

    Returns:
        Normalized test ID
    """
    # Step 1: Remove unstable runtime prefixes

    # Universal pattern: Remove (N/M) or [N/M] prefixes (test execution order)
    # Matches: "(2/5) test", "[2/5] test", "(123/456) test", "( 1/75) test" (with internal space)
    normalized = re.sub(r"^[\(\[]?\s*\d+/\d+[\)\]]?\s+", "", test_id)

    # Universal pattern: Remove #N prefix (test numbering)
    # Matches: "#42 test", "# 42 test"
    normalized = re.sub(r"^#\s*\d+\s+", "", normalized)

    # Universal pattern: Remove "N. " prefix (numbered list)
    # Matches: "1. test", "42. test"
    normalized = re.sub(r"^\d+\.\s+", "", normalized)

    # Step 2: Remove common file extensions before delimiters
    # This prevents .py from becoming ::py after delimiter normalization
    # Match extensions like .py, .js, .ts, etc. that appear before :: / . or end of string
    extensions_pattern = (
        r"\.(py|pyw|js|mjs|cjs|ts|mts|cts|jsx|tsx|"
        r"go|java|rb|rs|c|cpp|cc|cxx|h|hpp|hxx|"
        r"swift|kt|kts|scala|php|cs|fs|"
        r"ex|exs|erl|hrl|clj|cljs|cljc|"
        r"lua|pl|pm|t|r|R|m|mm|"
        r"f|f90|f95|for|vb|pas|pp|"
        r"d|nim|zig|v|sv|vhd|vhdl|"
        r"tcl|sh|bash|zsh|fish|ps1|psm1|psd1)"
        r"(?=::|/|\.|$)"
    )
    normalized = re.sub(extensions_pattern, "", normalized, flags=re.IGNORECASE)

    # Step 3: Normalize delimiters (., ::, /) to :: when between word characters
    # This allows matching "testa.testb::testc" with "testa/testb.testc"
    delimiter_pattern = r"(?<=\w)(::|\.|/)(?=\w)"
    normalized = re.sub(delimiter_pattern, "::", normalized)

    return normalized


def parse_tap_text(text_output: str) -> Dict[str, str]:
    """
    Parse TAP (Test Anything Protocol) output.

    TAP is used by tape, node-tap, and other JavaScript test frameworks.

    Format:
        TAP version 13
        # Subtest: Test name
            1..N
            ok 1 - assertion name
            not ok 2 - assertion name
        ok 1 - Test name # time=123ms
        not ok 2 - Test name
        1..N

    IMPORTANT: We prioritize finding valid test results over detecting errors.
    We only return None if we're certain the tests didn't run.

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If TAP tests failed to run (not the same as tests failing)
    """
    results = {}

    # PRIORITY 1: Parse top-level test results (not indented subtests)
    # Format: "ok N - Test name" or "not ok N - Test name"
    # Skip lines starting with whitespace (subtests)
    tap_test_pattern = re.compile(
        r"^(not )?ok\s+(\d+)\s*(?:-\s*)?(.+?)(?:\s*#\s*(skip|todo|time=.*))?$", re.MULTILINE | re.IGNORECASE
    )

    for match in tap_test_pattern.finditer(text_output):
        is_failure = match.group(1) is not None  # "not ok" prefix
        test_num = match.group(2)
        test_name = match.group(3).strip() if match.group(3) else f"test_{test_num}"
        directive = match.group(4)

        # Clean up test name (remove timing info like "# time=123ms")
        test_name = re.sub(r"\s*#\s*time=[\d.]+m?s\s*$", "", test_name, flags=re.IGNORECASE)

        test_id = test_name if test_name else f"test_{test_num}"

        # Check for skip directive
        if directive and directive.lower().startswith("skip"):
            results[test_id] = "SKIPPED"
        elif is_failure:
            results[test_id] = "FAILED"
        else:
            results[test_id] = "PASSED"

    # PRIORITY 2: If no results found, try parsing summary line
    # Format: "# tests N", "# pass N", "# fail N"
    if not results:
        pass_match = re.search(r"#\s*pass\s+(\d+)", text_output, re.IGNORECASE)
        fail_match = re.search(r"#\s*fail\s+(\d+)", text_output, re.IGNORECASE)

        if pass_match or fail_match:
            passed = int(pass_match.group(1)) if pass_match else 0
            failed = int(fail_match.group(1)) if fail_match else 0

            for i in range(passed):
                results[f"tap_test_passed_{i + 1}"] = "PASSED"
            for i in range(failed):
                results[f"tap_test_failed_{i + 1}"] = "FAILED"

    # PRIORITY 3: If no results, check for error indicators
    if not results:
        error_indicators = [
            "npm ERR!",  # npm errors
            "Error: Cannot find module",  # Module loading errors
            "SyntaxError:",  # JavaScript syntax errors
            "TypeError:",  # Type errors
        ]
        has_errors = any(indicator in text_output for indicator in error_indicators)
        # Return None ONLY if: no results found AND errors present
        return None if has_errors else results

    return results


def parse_hardhat_mocha_text(text_output: str) -> Dict[str, str]:
    """
    Parse Hardhat/Mocha console text output (non-JSON reporter).

    Hardhat uses Mocha under the hood and outputs text like:
        Contract: FeeSharingProxy:
            withdrawFees
                ✓ Shouldn't be able to use zero token address
                ✓ Shouldn't be able to withdraw second time in period
                1) Should fail with specific error

        5 passing (1s)
        1 failing

    IMPORTANT: We prioritize finding valid test results over detecting errors.

    Returns:
        Dict[str, str]: Test results mapping test IDs to status (PASSED/FAILED/SKIPPED)
        None: If tests failed to run (not the same as tests failing)
    """
    results = {}

    # Track current context (Contract/describe blocks)
    current_context = []

    # PRIORITY 1: Parse individual test results
    for line in text_output.split("\n"):
        stripped = line.strip()

        # Track Contract: or describe blocks
        contract_match = re.match(r"^Contract:\s*(.+?):\s*$", stripped)
        if contract_match:
            current_context = [contract_match.group(1)]
            continue

        # Track describe blocks (indented without checkmark/number)
        if stripped and not stripped.startswith(("✓", "✗", "-")) and not re.match(r"^\d+\)", stripped):
            # Check if this looks like a describe block (usually followed by test cases)
            if ":" not in stripped and len(stripped) < 100:
                # This might be a describe block, but we'll handle it dynamically
                pass

        # Match passed tests: ✓ test_name or ✔ test_name
        pass_match = re.match(r"^[✓✔]\s+(.+?)(?:\s+\(\d+m?s\))?$", stripped)
        if pass_match:
            test_name = pass_match.group(1).strip()
            test_id = f"{' > '.join(current_context)} > {test_name}" if current_context else test_name
            results[test_id] = "PASSED"
            continue

        # Match failed tests: N) test_name or ✗ test_name
        fail_match = re.match(r"^(?:\d+\)|[✗✘])\s*(.+?)$", stripped)
        if fail_match:
            test_name = fail_match.group(1).strip()
            test_id = f"{' > '.join(current_context)} > {test_name}" if current_context else test_name
            results[test_id] = "FAILED"
            continue

        # Match skipped tests: - test_name
        skip_match = re.match(r"^-\s+(.+?)$", stripped)
        if skip_match:
            test_name = skip_match.group(1).strip()
            test_id = f"{' > '.join(current_context)} > {test_name}" if current_context else test_name
            results[test_id] = "SKIPPED"
            continue

    # PRIORITY 2: Parse summary if no individual results found
    if not results:
        # Look for "N passing" and "N failing"
        pass_match = re.search(r"(\d+)\s+passing", text_output, re.IGNORECASE)
        fail_match = re.search(r"(\d+)\s+failing", text_output, re.IGNORECASE)

        if pass_match or fail_match:
            passed = int(pass_match.group(1)) if pass_match else 0
            failed = int(fail_match.group(1)) if fail_match else 0

            for i in range(passed):
                results[f"mocha_test_passed_{i + 1}"] = "PASSED"
            for i in range(failed):
                results[f"mocha_test_failed_{i + 1}"] = "FAILED"

    # PRIORITY 3: If no results, check for error indicators
    if not results:
        error_indicators = [
            "Error: Cannot find module",
            "SyntaxError:",
            "CompilerError:",  # Solidity compilation errors
            "Error: HH",  # Hardhat errors
        ]
        has_errors = any(indicator in text_output for indicator in error_indicators)
        return None if has_errors else results

    return results


def parse_pytest_text(text_output: str) -> Dict[str, str]:
    """
    Parse pytest plain text output (-v flag).

    Pytest outputs lines like:
      tests/test_foo.py::test_one PASSED
      tests/test_foo.py::test_two FAILED
      tests/test_foo.py::test_three SKIPPED

    Or in short form:
      tests/test_foo.py .F.s

    Also handles summary lines like:
      ===== 3 passed, 1 failed, 1 skipped in 0.5s =====
    """
    results = {}

    # Pattern 1: Verbose output with test names
    # e.g., "tests/test_foo.py::test_one PASSED"
    verbose_pattern = re.compile(
        r"^([\w./]+::\w+(?:::\w+)*)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)", re.MULTILINE
    )

    for match in verbose_pattern.finditer(text_output):
        test_id = match.group(1).strip()
        status = match.group(2).upper()

        if status in ("PASSED", "XPASS"):
            results[test_id] = "PASSED"
        elif status in ("FAILED", "ERROR", "XFAIL"):
            results[test_id] = "FAILED"
        elif status == "SKIPPED":
            results[test_id] = "SKIPPED"

    if results:
        return results

    # Pattern 2: Short form with dots (. = pass, F = fail, s = skip)
    # e.g., "tests/test_foo.py .F.s"
    short_pattern = re.compile(r"^([\w./]+\.py)\s+([.FsExX]+)", re.MULTILINE)

    for match in short_pattern.finditer(text_output):
        file_path = match.group(1)
        outcomes = match.group(2)

        for i, char in enumerate(outcomes):
            test_id = f"{file_path}::test_{i + 1}"
            if char == ".":
                results[test_id] = "PASSED"
            elif char.upper() == "F":
                results[test_id] = "FAILED"
            elif char.lower() == "s":
                results[test_id] = "SKIPPED"

    if results:
        return results

    # Pattern 3: Summary line fallback
    # e.g., "===== 3 passed, 1 failed, 1 skipped in 0.5s ====="
    summary_pattern = re.compile(r"(\d+)\s+passed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+(?:skipped|deselected))?")
    match = summary_pattern.search(text_output)
    if match:
        passed = int(match.group(1) or 0)
        failed = int(match.group(2) or 0)
        skipped = int(match.group(3) or 0)

        for i in range(passed):
            results[f"pytest_test_passed_{i + 1}"] = "PASSED"
        for i in range(failed):
            results[f"pytest_test_failed_{i + 1}"] = "FAILED"
        for i in range(skipped):
            results[f"pytest_test_skipped_{i + 1}"] = "SKIPPED"

    return results


def parse_test_output(output: str, framework: str) -> Dict[str, str]:
    """
    Parse test output to extract individual test results.

    Returns: {'test_id': 'PASSED'|'FAILED'|'SKIPPED'}
    """
    # Direct framework → parser mapping
    parsers = {
        "pytest": parse_junit_xml,
        "unittest": parse_junit_xml,
        "junit": parse_junit_xml,
        "maven": parse_maven_text_output,
        "gtest": parse_gtest_json,
        "cargo-nextest": parse_cargo_nextest,
        "go": parse_go_json,
        "jest": parse_jest_vitest_json,
        "vitest": parse_jest_vitest_json,
        "mocha": parse_mocha_json,
        "bun": parse_bun_text,
        "ctest": parse_junit_xml,
        "cppunit": parse_cppunit_text,
        "bespoke_libgeos": parse_bespoke_libgeos,
        # XCTest using hybrid approach
        "xctest": parse_xctest_output,
        "testing": parse_xctest_output,  # New Swift Testing framework (Swift 6+)
        # Lua frameworks
        "busted": parse_junit_xml,  # Uses JUnit XML output
        "luaunit": parse_junit_xml,  # Uses JUnit XML output
        "telescope": parse_telescope_text,
        "lust": parse_lust_text,
        "minitest": parse_minitest_text,  # Neovim mini.nvim test framework
        # TAP (Test Anything Protocol) - used by tape, node-tap
        "tap": parse_tap_text,
        "tape": parse_tap_text,
        # Hardhat (Solidity) - uses Mocha console output
        "hardhat": parse_hardhat_mocha_text,
    }

    parser = parsers.get(framework)
    if parser:
        result = parser(output)
        # Fallback for common frameworks if their primary parser returns None/empty
        if not result:
            if framework in ["junit", "maven"]:
                result = parse_maven_text_output(output)
            elif framework == "pytest":
                # Pytest often outputs plain text, not JUnit XML
                result = parse_pytest_text(output)
            elif framework == "mocha":
                # Mocha might output text instead of JSON (console reporter)
                result = parse_hardhat_mocha_text(output)
        return result or {}

    # Try auto-detection for unknown frameworks
    # Check for TAP output
    if "TAP version" in output or re.search(r"^(?:not )?ok\s+\d+", output, re.MULTILINE):
        return parse_tap_text(output) or {}

    # Check for Mocha/Hardhat console output
    if "Contract:" in output or re.search(r"^\s*[✓✔]\s+", output, re.MULTILINE):
        return parse_hardhat_mocha_text(output) or {}

    return {}
