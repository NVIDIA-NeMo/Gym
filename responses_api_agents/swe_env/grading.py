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

"""Pure grading helpers shared by harnesses + the verifier server.

These functions never touch a sandbox; they decide ``resolved`` from parsed
test status and map a report to a (non-nullable) reward.
"""

from __future__ import annotations

from collections.abc import Iterable

from responses_api_agents.swe_env.harness import SweEvalReport


def compute_resolved(
    *,
    fail_to_pass: Iterable[str],
    pass_to_pass: Iterable[str],
    passed: Iterable[str],
) -> bool:
    """Apply the SWE-bench resolution rule.

    A task is resolved when every FAIL_TO_PASS and PASS_TO_PASS test passes.

    Args:
        fail_to_pass (Iterable[str]): Tests that must transition from failing to
            passing.
        pass_to_pass (Iterable[str]): Tests that must remain passing.
        passed (Iterable[str]): The tests that actually passed.

    Returns:
        bool: ``True`` if all required tests passed, ``False`` if there are no
            required tests or any required test did not pass.
    """
    passed_set = set(passed)
    required = list(fail_to_pass) + list(pass_to_pass)
    if not required:
        return False
    return all(test in passed_set for test in required)


def reward_from_report(report: SweEvalReport) -> float:
    """Map a graded report to a reward.

    An infra or eval failure (``error_kind`` set) yields ``0.0`` and is masked
    via the flag downstream; the result is always a ``float`` and never ``None``.

    Args:
        report (SweEvalReport): The graded result to convert.

    Returns:
        float: ``1.0`` if the task resolved with no error, otherwise ``0.0``.
    """
    if report.error_kind is not None:
        return 0.0
    return 1.0 if report.resolved else 0.0
