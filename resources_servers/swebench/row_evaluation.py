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
#
# Portions adapted from SWE-bench (https://github.com/SWE-bench/SWE-bench), tag v5.0.1, commit
# 87ab1f6ced28f75ba73ca899dc759b019310944a, MIT License, Copyright (c) 2023 Carlos E Jimenez, John Yang,
# Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik R Narasimhan:
#   - swebench/harness/utils.py: parse_eval_script, record_test_exit_code, make_test_spec (as make_row_test_spec)
#   - swebench/harness/grading.py: SUITE_RAN, parse_test_exit_code, _resolve_case, test_passed, test_maintained,
#     test_failed, get_logs_eval, get_eval_tests_report, get_resolution_status, get_eval_report
#   - swebench/harness/run_evaluation.py: _inject_asset_restore (as insert_test_asset_restore)
# The grading functions keep SWE-bench's logic. Changes: the FAIL_TO_FAIL / PASS_TO_FAIL "x to fail" metrics and
# the advisory infra-failure classification are left out (neither affects `resolved`), and the parsers come from
# row_log_parsers.py.
"""Evaluation defined by the row: the image, evaluation script, log parser and evaluation type a SWE-bench row
carries.

Since swebench 5.0, a SWE-bench row names its prebuilt image (``image``), its full evaluation script
(``eval_script``), the log parser that reads the test output (``log_parser``) and whether passing tests are read
from the log or assumed (``eval_type``). The pinned swebench 4.1.0 harness derives all four from per-repository
specs instead, and has none for the repositories of the SWE-bench Multimodal test split. A resources server with
``evaluation_spec_source: row`` builds the test spec from the row and grades with SWE-bench 5's rules, below.
"""

import json
import re
from dataclasses import dataclass
from shlex import quote
from typing import Any, Sequence

from resources_servers.swebench.row_log_parsers import ROW_LOG_PARSERS, TestStatus


START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
TEST_EXIT_CODE = ">>>>> Test Exit Code"
TEST_EXIT_CODE_VAR = "SWEBENCH_TEST_EXIT_CODE"
# Markers SWE-bench's harness writes into the log when a run went wrong before the tests.
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"

EVAL_TYPES = ("pass_and_fail", "fail_only")


@dataclass
class RowTestSpec:
    """The test spec of a row that defines its own evaluation, with the attributes ``run_instance`` reads."""

    instance_id: str
    repo: str
    version: str
    base_commit: str
    image: str
    eval_script_list: list[str]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    log_parser: str
    eval_type: str
    # The row's image is prebuilt; nothing is built locally.
    is_remote_image: bool = True

    @property
    def instance_image_key(self) -> str:
        return self.image

    @property
    def eval_script(self) -> str:
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list) + "\n"


def parse_eval_script(eval_script: str) -> list[str]:
    """The script's commands, without the shebang and shell options ``eval_script`` adds back."""
    return [line for line in eval_script.strip().split("\n") if line not in ("#!/bin/bash", "set -uxo pipefail")]


def record_test_exit_code(eval_script_list: list[str]) -> list[str]:
    """Record the test command's own exit status after the end-of-output marker.

    The scripts end by resetting the test files and run without ``set -e``, so the script's exit status is the
    reset's. Grading compares the recorded status with the parsed log.
    """
    for index, line in enumerate(eval_script_list):
        if END_TEST_OUTPUT in line:
            return [
                *eval_script_list[:index],
                f"{TEST_EXIT_CODE_VAR}=$?",
                line,
                f'echo "{TEST_EXIT_CODE}: ${TEST_EXIT_CODE_VAR}"',
                *eval_script_list[index + 1 :],
            ]
    return eval_script_list


def make_row_test_spec(
    *,
    instance_id: str,
    repo: str,
    version: str,
    base_commit: str,
    image: str | None,
    eval_script: str | None,
    log_parser: str | None,
    eval_type: str | None,
    fail_to_pass: str | list[str],
    pass_to_pass: str | list[str],
) -> RowTestSpec:
    """The test spec of a row that defines its own evaluation; raises ``ValueError`` when the row cannot."""
    missing = [
        name
        for name, value in (
            ("image", image),
            ("eval_script", eval_script),
            ("log_parser", log_parser),
            ("eval_type", eval_type),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Row {instance_id} defines no evaluation: it has no {', '.join(missing)}")
    if log_parser not in ROW_LOG_PARSERS:
        raise ValueError(f"Row {instance_id} names log parser {log_parser!r}, which this server does not have")
    if eval_type not in EVAL_TYPES:
        raise ValueError(f"Row {instance_id} has eval_type {eval_type!r}; expected one of {EVAL_TYPES}")
    return RowTestSpec(
        instance_id=instance_id,
        repo=repo,
        version=version,
        base_commit=base_commit,
        image=image,
        eval_script_list=record_test_exit_code(parse_eval_script(eval_script)),
        FAIL_TO_PASS=json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else list(fail_to_pass),
        PASS_TO_PASS=json.loads(pass_to_pass) if isinstance(pass_to_pass, str) else list(pass_to_pass),
        log_parser=log_parser,
        eval_type=eval_type,
    )


def insert_test_asset_restore(test_spec: RowTestSpec, staged_paths: Sequence[tuple[str, str]]) -> None:
    """Copy each staged binary file of the test patch into the repository before the tests start.

    ``staged_paths`` pairs each file's staged upload path with its path in the repository. The copies go right
    before the start-of-output marker, after the script has reset and applied the test patch, as SWE-bench does.
    """
    if not staged_paths:
        return
    restore = [
        f"mkdir -p $(dirname {quote(path)}) && cp {quote(staged)} {quote(path)}" for staged, path in staged_paths
    ]
    for index, line in enumerate(test_spec.eval_script_list):
        if START_TEST_OUTPUT in line:
            test_spec.eval_script_list[index:index] = restore
            return
    test_spec.eval_script_list.extend(restore)


# Evidence that a test runner executed, which tells "ran with no failures" apart from "never ran" when the parsed
# status map is empty. Every count must be non-zero: under fail_only an empty status map scores every FAIL_TO_PASS
# test as passing, so a runner that lost its browser ("Executed 0 of 0") must not count.
SUITE_RAN = re.compile(
    r"Executed [1-9]\d* of \d+"
    r"|TOTAL: [1-9]\d* (?:SUCCESS|FAILED)"
    r"|[1-9]\d* passing"
    r"|Tests:\s+[1-9]\d*"
    r"|Test Suites:\s+(?:\d+ \w+, )*[1-9]\d* total"
    r"|^# tests [1-9]\d*"
    r"|[1-9]\d* specs?, \d+ failures?"
    r"|': ok$",
    re.M,
)
TEST_EXIT_CODE_RE = re.compile(rf"{re.escape(TEST_EXIT_CODE)}:\s*(-?\d+)")


def parse_test_exit_code(content: str) -> int | None:
    match = TEST_EXIT_CODE_RE.search(content)
    return int(match.group(1)) if match else None


def _resolve_case(case: str, status_map: dict[str, str]) -> str | None:
    """The status-map key for ``case``; a truncated parametrized id matches by prefix when the matches agree."""
    if case in status_map:
        return case
    if case.count("[") > case.count("]"):
        matches = [key for key in status_map if key.startswith(case)]
        passing = {TestStatus.PASSED.value, TestStatus.XFAIL.value}
        if matches and len({status_map[key] in passing for key in matches}) == 1:
            return matches[0]
    return None


def test_passed(case: str, status_map: dict[str, str]) -> bool:
    key = _resolve_case(case, status_map)
    return key is not None and status_map[key] in (TestStatus.PASSED.value, TestStatus.XFAIL.value)


def test_maintained(case: str, status_map: dict[str, str]) -> bool:
    """PASS_TO_PASS: a skipped test is not a regression."""
    key = _resolve_case(case, status_map)
    return test_passed(case, status_map) or (key is not None and status_map[key] == TestStatus.SKIPPED.value)


def test_failed(case: str, status_map: dict[str, str]) -> bool:
    """FAIL_TO_PASS: a missing, failed, errored or skipped test is not a resolution."""
    key = _resolve_case(case, status_map)
    return key is None or status_map[key] in (
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
        TestStatus.SKIPPED.value,
    )


def get_logs_eval(test_spec: RowTestSpec, test_output: str) -> tuple[dict[str, str], bool]:
    """The test status map from ``test_output``, and whether the output describes a test run at all."""
    log_parser = ROW_LOG_PARSERS[test_spec.log_parser]
    if any(code in test_output for code in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT)):
        return {}, False
    if not (START_TEST_OUTPUT in test_output and END_TEST_OUTPUT in test_output):
        return {}, False

    sliced = test_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    status_map = log_parser(sliced, test_spec)
    if not status_map:
        # Some runners print their results outside the markers (bash's trace and the runner's output arrive on
        # different streams), so read the whole output before deciding there are no results.
        status_map = log_parser(test_output, test_spec)
    if not status_map and not SUITE_RAN.search(test_output):
        # No results and no sign the suite ran: under fail_only this would otherwise read as resolved.
        return {}, False

    # A patch can print its own "passed" lines; a test command that exited non-zero while the log shows no failure
    # does not describe the run that happened.
    exit_code = parse_test_exit_code(test_output)
    if (
        exit_code not in (None, 0)
        and status_map
        and not any(status in (TestStatus.FAILED.value, TestStatus.ERROR.value) for status in status_map.values())
    ):
        return {}, False
    return status_map, True


def get_eval_tests_report(
    status_map: dict[str, str], fail_to_pass: Sequence[str], pass_to_pass: Sequence[str], *, eval_type: str
) -> dict[str, dict[str, list[str]]]:
    """Which FAIL_TO_PASS and PASS_TO_PASS tests succeeded and failed."""

    def check_pass_and_fail(case: str, success: list[str], failed: list[str]) -> None:
        if test_passed(case, status_map):
            success.append(case)
        elif test_failed(case, status_map):
            failed.append(case)

    def check_maintained(case: str, success: list[str], failed: list[str]) -> None:
        if test_maintained(case, status_map):
            success.append(case)
        elif test_failed(case, status_map):
            failed.append(case)

    def check_fail_only(case: str, success: list[str], failed: list[str]) -> None:
        if case in status_map and status_map[case] == TestStatus.FAILED.value:
            failed.append(case)
        else:
            success.append(case)

    check_f2p = check_pass_and_fail if eval_type == "pass_and_fail" else check_fail_only
    check_p2p = check_maintained if eval_type == "pass_and_fail" else check_fail_only
    f2p_success: list[str] = []
    f2p_failure: list[str] = []
    for case in fail_to_pass:
        check_f2p(case, f2p_success, f2p_failure)
    p2p_success: list[str] = []
    p2p_failure: list[str] = []
    for case in pass_to_pass:
        check_p2p(case, p2p_success, p2p_failure)
    return {
        "FAIL_TO_PASS": {"success": f2p_success, "failure": f2p_failure},
        "PASS_TO_PASS": {"success": p2p_success, "failure": p2p_failure},
    }


def is_fully_resolved(report: dict[str, dict[str, list[str]]]) -> bool:
    """Every FAIL_TO_PASS test now passes and every PASS_TO_PASS test still does (an empty list counts as met)."""
    return not report["FAIL_TO_PASS"]["failure"] and not report["PASS_TO_PASS"]["failure"]


def get_row_eval_report(
    test_spec: RowTestSpec, prediction: dict[str, Any], test_log_path: str, include_tests_status: bool
) -> dict[str, dict[str, Any]]:
    """SWE-bench's report for one prediction, in the shape ``run_instance`` reads: ``{instance_id: {...}}``."""
    instance_id = prediction["instance_id"]
    result: dict[str, Any] = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "resolved": False,
    }
    report_map = {instance_id: result}
    if prediction["model_patch"] is None:
        result["patch_is_None"] = True
        return report_map
    result["patch_exists"] = True

    with open(test_log_path, errors="replace") as f:
        test_output = f.read()
    status_map, found = get_logs_eval(test_spec, test_output)
    if not found:
        return report_map
    result["patch_successfully_applied"] = True

    report = get_eval_tests_report(
        status_map, test_spec.FAIL_TO_PASS, test_spec.PASS_TO_PASS, eval_type=test_spec.eval_type
    )
    result["resolved"] = is_fully_resolved(report)
    if include_tests_status:
        result["tests_status"] = report
    return report_map


# Gym's own check, not SWE-bench's: SUITE_RAN accepts any non-zero Karma count, so a run that stopped part way
# ("Executed 1723 of 3052", a hook failure that ends the suite) still reads as one that ran. Under fail_only every
# test that never ran then counts as passing.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_KARMA_SUMMARY = re.compile(r"Executed (\d+) of (\d+)((?: \([^)\n]*\))*)")
_KARMA_SKIPPED = re.compile(r"\(skipped (\d+)\)")
KARMA_COMPLETE_FRACTION = 0.95


def karma_runs_completed(test_output: str) -> bool:
    """Whether Karma, if the output shows it ran, executed or skipped nearly every test it planned in some run.

    Karma prints a running count ("Executed 3 of 1604") and a final summary such as "Executed 1339 of 1346 (skipped 7)
    SUCCESS". A few planned tests may never run in a completed suite (bpmn-js-1011: 1262 + 7 skipped of 1273 in every
    run), while a suite a hook error ended stops far short ("Executed 1723 of 3052"), so a summary that accounts for
    ``KARMA_COMPLETE_FRACTION`` of the planned tests counts as complete. Output without Karma summaries passes.
    """
    summaries = _KARMA_SUMMARY.findall(_ANSI_ESCAPE.sub("", test_output))
    if not summaries:
        return True
    for done, total, notes in summaries:
        skipped = sum(int(count) for count in _KARMA_SKIPPED.findall(notes))
        if int(total) > 0 and int(done) + skipped >= KARMA_COMPLETE_FRACTION * int(total):
            return True
    return False
