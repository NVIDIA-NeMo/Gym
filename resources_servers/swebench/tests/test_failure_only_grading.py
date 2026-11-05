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
"""The suite-completion checks, on log excerpts shaped like gold-patch evaluations of SWE-bench Multimodal."""

import pytest
from swebench.harness.constants import FAIL_ONLY_REPOS

from resources_servers.swebench.failure_only_grading import (
    FAILURE_ONLY_SUITE_CHECKS,
    failure_only_suite_completed,
)


def _test_output(body: str) -> str:
    return f"+ pnpm run build\nbuilt\n+ : '>>>>> Start Test Output'\n{body}\n+ : '>>>>> End Test Output'\n+ git checkout\n"


CHROME_DONE = "Chrome 120.0.6099.109 (Linux x86_64): Executed 1545 of 1545 (12 FAILED) (43.019 secs / 33.247 secs)"
FIREFOX_DONE = "Firefox 131.0 (Linux x86_64): Executed 1545 of 1545 SUCCESS (56.968 secs / 48.098 secs)"
CHROME_GAVE_UP = (
    "04 11 2026 21:00:01.000:ERROR [launcher]: Cannot start Chrome\n"
    "04 11 2026 21:00:09.000:ERROR [launcher]: Chrome failed 2 times (cannot start). Giving up."
)

JASMINE_DONE = "Randomized with seed 49913\nStarted\n....\n\n658 specs, 0 failures\nFinished in 0.851 seconds"

P5_YUI = 'Running "mochaChrome:yui" (mochaChrome) task\n\n  739 passing (22s)\n'
P5_TEST = 'Running "mochaChrome:test" (mochaChrome) task\n\n  921 passing (8s)\n  3 pending\n'
P5_NODE = 'Running "mochaTest:test" (mochaTest) task\n\n  8 passing (13ms)\n\nDone.'


@pytest.mark.parametrize(
    ("repo", "body"),
    [
        ("chartjs/Chart.js", f"{CHROME_DONE}\n{FIREFOX_DONE}"),
        # Older Karma prints the running count before the final summary, with colour codes.
        (
            "chartjs/Chart.js",
            "\x1b[32mChrome 120.0.0.0 (Linux x86_64): Executed 0 of 1150 SUCCESS\x1b[39m\n"
            "Chrome 120.0.0.0 (Linux x86_64): Executed 1150 of 1150 (14 FAILED) (30.588 secs / 23.626 secs)",
        ),
        ("markedjs/marked", JASMINE_DONE),
        ("processing/p5.js", f"{P5_YUI}{P5_TEST}{P5_NODE}"),
        # Older p5.js runs its suite in PhantomJS through grunt-mocha.
        ("processing/p5.js", 'Running "mocha:yui" (mocha) task\n\n  419 passing (31s)\n  109 failing\n'),
    ],
)
def test_a_completed_suite_is_accepted(repo: str, body: str) -> None:
    assert failure_only_suite_completed(repo, _test_output(body))


@pytest.mark.parametrize(
    ("repo", "body"),
    [
        # Chrome never started; the Chart.js parser reads Chrome failures only, so Firefox cannot stand in for it.
        ("chartjs/Chart.js", f"{CHROME_GAVE_UP}\n{FIREFOX_DONE}"),
        ("chartjs/Chart.js", f"{CHROME_GAVE_UP}\nFirefox failed 2 times (cannot start). Giving up."),
        ("chartjs/Chart.js", "Chrome 120.0.0.0 (Linux x86_64): Executed 512 of 1545 (3 FAILED) DISCONNECTED"),
        ("chartjs/Chart.js", "Chrome 120.0.0.0 (Linux x86_64): Executed 3 of 1545 (skipped 1542) SUCCESS"),
        ("chartjs/Chart.js", "Chrome 120.0.0.0 (Linux x86_64): Executed 0 of 0 ERROR"),
        ("markedjs/marked", "/eval.sh: line 22: ./node_modules/.bin/jasmine: No such file or directory"),
        ("markedjs/marked", "Started\n\n\n0 specs, 0 failures\nIncomplete: No specs found"),
        ("markedjs/marked", "Started\n....\n\n4 specs, 0 failures\nIncomplete: fit() or fdescribe() was found"),
        # The browser suite stopped without a summary; the Node suite that follows cannot stand in for it.
        (
            "processing/p5.js",
            f'{P5_YUI}Running "mochaChrome:test" (mochaChrome) task\n'
            "Warning: Failed to launch the browser process! Used --force, continuing.\n"
            f"{P5_NODE}",
        ),
        ("processing/p5.js", P5_NODE),
        ("processing/p5.js", 'Running "mochaChrome:yui" (mochaChrome) task\n\n  0 passing (1ms)\n'),
    ],
)
def test_a_suite_that_did_not_complete_is_rejected(repo: str, body: str) -> None:
    assert not failure_only_suite_completed(repo, _test_output(body))


def test_only_the_test_output_section_counts() -> None:
    log = f"{CHROME_DONE}\n" + _test_output(CHROME_GAVE_UP)
    assert not failure_only_suite_completed("chartjs/Chart.js", log)


def test_every_failure_only_repository_has_a_check() -> None:
    assert set(FAIL_ONLY_REPOS) <= set(FAILURE_ONLY_SUITE_CHECKS)
    assert not failure_only_suite_completed("example/unchecked", _test_output(CHROME_DONE))
