# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Per-framework parser coverage for ``parsing.parse_test_output``.

The multilingual flat-grading path dispatches raw test output through ~13 per-framework parsers
(jest/mocha/gtest/maven/cargo-nextest/bun/cppunit/minitest/...), and the resulting PASS/FAIL map
feeds reward directly — so a parser bug silently corrupts reward. These cases pin the
node-id/status extraction for the framework families the rest of the suite does not exercise.
"""

from responses_api_agents.swe_env.parsing.parsing import parse_test_output


def test_parse_test_output_jest_json():
    out = (
        '{"testResults":[{"name":"a.test.js","status":"failed",'
        '"assertionResults":['
        '{"fullName":"math adds","status":"passed"},'
        '{"fullName":"math subtracts","status":"failed"},'
        '{"fullName":"math pending","status":"skipped"}'
        "]}]}"
    )
    assert parse_test_output(out, "jest") == {
        "a.test.js::math adds": "PASSED",
        "a.test.js::math subtracts": "FAILED",
        "a.test.js::math pending": "SKIPPED",
    }


def test_parse_test_output_maven_summary():
    out = "[INFO] Building MyModule 1.0 [1/1]\nTests run: 3, Failures: 1, Errors: 0, Skipped: 0\n"
    res = parse_test_output(out, "maven")
    assert res["MyModule::test_1"] == "PASSED"
    assert res["MyModule::test_2"] == "PASSED"
    assert res["MyModule::test_failed_1"] == "FAILED"


def test_parse_test_output_maven_build_failure():
    assert parse_test_output("...\nBUILD FAILURE\n...", "maven") == {"maven::build": "FAILED"}


def test_parse_test_output_cargo_nextest():
    out = "PASS [   1.588s] mycrate tests::test_alpha\nFAIL [   0.500s] mycrate tests::test_beta\n"
    assert parse_test_output(out, "cargo-nextest") == {
        "mycrate tests::test_alpha": "PASSED",
        "mycrate tests::test_beta": "FAILED",
    }


def test_parse_test_output_cargo_nextest_compile_error_is_none_to_empty():
    assert parse_test_output("error[E0425]: cannot find value\n", "cargo-nextest") == {}


def test_parse_test_output_pytest_text_shortform_fallback():
    res = parse_test_output("tests/test_foo.py .F.s\n", "pytest")
    assert res == {
        "tests/test_foo.py::test_1": "PASSED",
        "tests/test_foo.py::test_2": "FAILED",
        "tests/test_foo.py::test_3": "PASSED",
        "tests/test_foo.py::test_4": "SKIPPED",
    }


def test_parse_test_output_unknown_framework_returns_empty():
    assert parse_test_output("some random output", "totally-unknown-framework") == {}
