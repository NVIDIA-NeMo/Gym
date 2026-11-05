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
"""Evidence that a failure-only test suite ran to completion.

SWE-bench grades Chart.js, marked and p5.js from failures alone (``FAIL_ONLY_REPOS``): their log parsers record
failing tests only, and a test missing from the log counts as passing. A suite that never started (a browser that
did not launch, a missing test runner) or stopped part way therefore reads as resolved. Each check below looks for
the summary the repository's test runner prints once its suite has run. The summaries were taken from gold-patch
evaluations of every Chart.js, marked and p5.js task of the SWE-bench Multimodal dev split.
"""

import re
from typing import Callable

from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Karma, once per browser: "Chrome 120.0.0.0 (Linux x86_64): Executed 1604 of 1604 (16 FAILED) (43 secs / 33 secs)".
# Karma also prints the running count ("Executed 3 of 1604"), and "Executed 1590 of 1604 (skipped 14)" when tests are
# skipped. The Chart.js parser reads Chrome's failures only.
_KARMA_CHROME_SUMMARY = re.compile(r"^Chrome [^\n]*?: Executed (\d+) of (\d+)", re.MULTILINE)

# Jasmine: "1410 specs, 0 failures", and "Incomplete: ..." when it found no specs or only focused ones.
_JASMINE_SUMMARY = re.compile(r"^(\d+) specs?, \d+ failures?", re.MULTILINE)
_JASMINE_INCOMPLETE = re.compile(r"^Incomplete:", re.MULTILINE)

# Grunt prints 'Running "<task>:<target>" (<task>) task' before each task. p5.js runs its browser suites as
# mochaChrome tasks (older versions: mocha, in PhantomJS), each ending with a Mocha summary such as
# "  921 passing (8s)". mochaTest runs a separate suite in Node.
_GRUNT_TASK = re.compile(r'^Running "([^"]+)"', re.MULTILINE)
_MOCHA_BROWSER_TASK = re.compile(r"^mocha(Chrome)?:")
_MOCHA_PASSING = re.compile(r"^\s*(\d+) passing\b", re.MULTILINE)


def _chrome_suite_completed(log: str) -> bool:
    """Chrome executed every test Karma planned, and there was at least one."""
    return any(int(done) == int(total) > 0 for done, total in _KARMA_CHROME_SUMMARY.findall(log))


def _jasmine_suite_completed(log: str) -> bool:
    """Jasmine ran at least one spec and did not report the run incomplete."""
    return any(int(specs) > 0 for specs in _JASMINE_SUMMARY.findall(log)) and not _JASMINE_INCOMPLETE.search(log)


def _mocha_browser_suites_completed(log: str) -> bool:
    """Every browser suite grunt started printed a Mocha summary with passing tests, and there was at least one."""
    tasks = list(_GRUNT_TASK.finditer(log))
    browser_suites = 0
    for index, task in enumerate(tasks):
        if not _MOCHA_BROWSER_TASK.match(task.group(1)):
            continue
        end = tasks[index + 1].start() if index + 1 < len(tasks) else len(log)
        if not any(int(passing) > 0 for passing in _MOCHA_PASSING.findall(log[task.end() : end])):
            return False
        browser_suites += 1
    return browser_suites > 0


FAILURE_ONLY_SUITE_CHECKS: dict[str, Callable[[str], bool]] = {
    "chartjs/Chart.js": _chrome_suite_completed,
    "markedjs/marked": _jasmine_suite_completed,
    "processing/p5.js": _mocha_browser_suites_completed,
}


def failure_only_suite_completed(repo: str, test_output: str) -> bool:
    """Whether ``test_output`` shows that ``repo``'s test suite ran to completion.

    Only the output between SWE-bench's test-output markers counts. A repository without a check is never accepted.
    """
    check = FAILURE_ONLY_SUITE_CHECKS.get(repo)
    if check is None:
        return False
    log = _ANSI_ESCAPE.sub("", test_output)
    if START_TEST_OUTPUT in log and END_TEST_OUTPUT in log:
        log = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    return check(log)
