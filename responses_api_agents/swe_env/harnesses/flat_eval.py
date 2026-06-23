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

"""Flat (host-graded) eval-script mode for SWE dataset families.

Flat mode runs an instance's eval script directly in the sandbox and parses the
produced log host-side, computing ``resolved`` from ``FAIL_TO_PASS`` /
``PASS_TO_PASS`` via :func:`compute_resolved`. Because there is no nested
container, this runs on any exec-capable provider (docker / opensandbox).

The eval script resets the repo, applies the gold/model patch plus the test
patch, runs the repo's test command, and wraps the test output between two
sentinel markers::

    >>>>> Start Test Output
    ... per-test "PASSED <id>" / "FAILED <id>" lines ...
    >>>>> End Test Output

It also emits patch-apply / reset / timeout status codes
(``>>>>> Applied Patch`` etc.). The host-side parser in this module recognises
these markers and per-test status tokens without importing ``swebench``, so
grading can run in environments where that package (and its Docker
dependencies) is absent.

Flat mode is selected when the harness is constructed in flat mode
(``flat_eval=True`` on the constructor) or when a task opts in via
``SweTask.metadata["flat_eval"]``. Only the harness-level flag lifts the
apptainer-only ``supports_provider`` restriction; a per-task flag alone affects
only ``run_eval`` / ``grade`` dispatch on an already-flat-capable harness,
because the provider is chosen at provisioning time from the harness capability
before task metadata is consulted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from responses_api_agents.swe_env.grading import compute_resolved
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# SWE-bench eval-log sentinels, kept here so we never import swebench at grade
# time.
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

# Codes that mean the harness/patch/test setup failed before tests could be
# trusted; their presence forces an empty status map + patch_applied=False.
_BAD_CODES = (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT)

# Per-test status tokens a pytest-style test runner emits at the start of a line
# ("PASSED tests/test_x.py::test_a"). XFAIL counts as a pass.
_PASS_TOKENS = ("PASSED", "XFAIL")
_FAIL_TOKENS = ("FAILED", "ERROR")
_STATUS_TOKENS = _PASS_TOKENS + _FAIL_TOKENS + ("SKIPPED",)

# Where the flat path writes the eval script and its captured log inside the
# sandbox.
EVAL_SCRIPT_PATH = "/root/eval.sh"
EVAL_LOG_PATH = "/root/eval_output.log"


def parse_eval_log(log: str) -> tuple[dict[str, str], bool]:
    """Parse a SWE-bench eval-script log host-side.

    For the common pytest-style runner:

    1. If any "bad code" (patch-apply / reset / tests-error / timeout) is
       present, the run is untrustworthy -> return ``({}, False)``.
    2. If the ``Start``/``End`` test-output markers are missing, the test patch
       never applied -> return ``({}, False)``.
    3. Otherwise extract the slice between the markers and parse per-test
       ``"<STATUS> <node_id>"`` lines into a ``{node_id: STATUS}`` map. As a
       fallback (output sometimes escapes the markers, e.g. to stderr) the whole
       log is scanned when the slice yields nothing.

    Args:
        log: The combined stdout/stderr captured from running the eval script.

    Returns:
        A tuple ``(status_map, patch_applied)``. ``status_map`` maps each test
        node id to its status token. ``patch_applied`` is ``True`` only when the
        markers were found and no bad code fired.
    """
    if any(code in log for code in _BAD_CODES):
        return {}, False
    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        return {}, False

    between = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    status_map = _parse_pytest_status_lines(between)
    if not status_map:
        # Fallback: some runners emit per-test lines outside the markers.
        status_map = _parse_pytest_status_lines(log)
    return status_map, True


def _parse_pytest_status_lines(text: str) -> dict[str, str]:
    """Parse ``"<STATUS> <node_id>"`` pytest-style lines into a status map.

    A status line starts with one of the recognised status tokens, and the node
    id is the second whitespace field. FAILED lines may read
    ``"FAILED <id> - <reason>"``; the trailing reason is stripped by rewriting
    ``" - "`` to ``" "``.

    Args:
        text: Text containing zero or more per-test status lines.

    Returns:
        A mapping from each test node id to its status token. When a node id
        appears more than once, the last occurrence wins.
    """
    status_map: dict[str, str] = {}
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        token = next((t for t in _STATUS_TOKENS if line.startswith(t)), None)
        if token is None:
            continue
        if token == "FAILED":
            line = line.replace(" - ", " ")
        fields = line.split()
        if len(fields) <= 1:
            continue
        node_id = fields[1]
        # Last status wins for a duplicated node id: a later line overwrites an
        # earlier one, so a runner that re-reports a node (e.g. a rerun plugin)
        # ends up with its final status.
        status_map[node_id] = fields[0]
    return status_map


def passed_tests(status_map: dict[str, str]) -> list[str]:
    """Return node ids whose status counts as a pass (PASSED or XFAIL).

    Args:
        status_map: A mapping from test node id to its status token.

    Returns:
        The list of node ids whose status is a passing token.
    """
    return [node for node, status in status_map.items() if status in _PASS_TOKENS]


async def flat_run_eval(env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
    """Run the instance's eval script in the sandbox and capture its log.

    The eval script must be supplied on the task via
    ``task.metadata["eval_script"]``. It is written into the sandbox and run,
    teeing its combined output to :data:`EVAL_LOG_PATH`; the captured
    stdout/stderr already contain the ``>>>>>`` markers, so ``test_output`` is
    graded directly. The log file is read back as a fallback when the streamed
    output is empty.

    Args:
        env: The SWE environment used to write files and execute commands in the
            sandbox.
        task: The task whose ``metadata["eval_script"]`` is run.

    Returns:
        An :class:`EvalArtifacts` holding the captured test output, the script's
        return code, whether a model patch existed, and raw metadata. When no
        eval script is present the artifacts carry an ``eval_error``.
    """
    eval_script = task.metadata.get("eval_script", "")
    if not eval_script:
        # No script to run -> mask as an eval error rather than scoring 0.
        return EvalArtifacts(
            test_output="",
            return_code=1,
            patch_applied=False,
            raw={"error_type": "eval_error", "flat": True},
        )

    await env.write_text(EVAL_SCRIPT_PATH, eval_script if eval_script.endswith("\n") else eval_script + "\n")
    # The script is self-contained (it resets + applies patches + runs tests);
    # `|| true` keeps the captured log even on a non-zero test exit so grade()
    # can parse per-test status. Combined output is also tee'd to a log file.
    result = await env.execute(
        f"bash {EVAL_SCRIPT_PATH} 2>&1 | tee {EVAL_LOG_PATH}; exit ${{PIPESTATUS[0]}}",
        cwd=task.repo_workdir,
        is_eval=True,
        timeout_s=task.metadata.get("tests_timeout"),
    )
    log_text = result["output"]
    if not log_text.strip() and result.get("error_type") not in {"sandbox", "timeout"}:
        # Streamed output was empty; fall back to the tee'd log file.
        cat = await env.execute(f"cat {EVAL_LOG_PATH}", cwd=task.repo_workdir)
        if cat["returncode"] == 0:
            log_text = cat["output"]

    return EvalArtifacts(
        test_output=log_text,
        return_code=result["returncode"],
        patch_applied=bool(task.model_patch),
        raw={"error_type": result.get("error_type"), "flat": True},
    )


def flat_grade(task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
    """Grade a flat eval-script log host-side.

    Infra failures (sandbox/timeout) are masked via ``error_kind``. A log with a
    bad code or missing markers grades as unresolved with ``patch_applied`` set
    from the parse, since a failed setup is a legitimate unresolved rather than
    an infra mask.

    Args:
        task: The task being graded, supplying the instance id, expected
            ``fail_to_pass`` / ``pass_to_pass`` tests, and model patch.
        artifacts: The eval artifacts produced by :func:`flat_run_eval`.

    Returns:
        A :class:`SweEvalReport` describing whether the task was resolved,
        whether the patch applied and existed, any masking ``error_kind``, and
        the per-test status breakdown.
    """
    if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            patch_applied=artifacts.patch_applied,
            error_kind=artifacts.raw["error_type"],
        )
    # A missing eval script is an eval error (masked), not a 0 score.
    if artifacts.raw.get("error_type") == "eval_error":
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            patch_applied=artifacts.patch_applied,
            error_kind="eval_error",
        )

    status_map, log_patch_applied = parse_eval_log(artifacts.test_output)
    passed = passed_tests(status_map)
    resolved = log_patch_applied and compute_resolved(
        fail_to_pass=task.fail_to_pass,
        pass_to_pass=task.pass_to_pass,
        passed=passed,
    )
    return SweEvalReport(
        instance_id=task.instance_id,
        resolved=resolved,
        patch_applied=log_patch_applied,
        patch_exists=bool(task.model_patch),
        tests_status={"passed": passed, "all": status_map},
    )


def flat_eval_enabled(harness_flag: bool, task: SweTask) -> bool:
    """Return whether flat mode should be used for this task.

    Flat mode is selected when the harness was constructed in flat mode
    (``harness_flag``) or the task opts in via ``metadata["flat_eval"]``. The
    harness flag lifts the ``supports_provider`` apptainer-only gate; the
    per-task key only affects ``run_eval`` / ``grade`` dispatch on an
    already-flat-capable harness.

    Args:
        harness_flag: Whether the harness instance was constructed in flat mode.
        task: The task whose ``metadata["flat_eval"]`` is consulted.

    Returns:
        ``True`` when flat mode applies to this task, otherwise ``False``.
    """
    return bool(harness_flag) or bool(task.metadata.get("flat_eval", False))
