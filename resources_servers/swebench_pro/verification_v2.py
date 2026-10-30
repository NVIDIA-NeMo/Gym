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

"""SWE-bench Pro V2 verification: the task's own Harbor verifier, run on a pristine sandbox.

Upstream source (tag v2.0.0):
https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2

* The task's ``tests/`` files (``test.sh``, ``run_script.sh``, ``parser.py``, ``config.json``,
  ``test_patch.patch``) run unchanged from ``/tests``; ``test.sh`` writes ``/logs/verifier/reward.txt``.
* The model patch is applied the way ``v2/tooling/patch_replay.py`` applies it: ``git apply``, then a
  3-way apply, then GNU ``patch``.
* The sandbox daemon's ``OTEL_*`` variables are dropped first, as in V1: tests read ``OTEL_SERVICE_NAME``.

Verdicts follow upstream: the ``reward.txt`` that ``test.sh`` writes is final, also without parser output, and a
``test.sh`` that exceeds its budget scores 0. The budget is enforced in the sandbox with ``timeout``, because a
provider may report a command it killed as a plain exit code. Only what the environment caused has no verdict: no
repository, a rejected test patch, no reward or a failed exec. Test output is not evidence of a fault (unlike V1's
DNS rule): the patched code writes it.
"""

import json
import shlex
from collections.abc import Mapping
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox
from resources_servers.swebench_pro.verification import SCRUB_HARNESS_ENV, VerificationResult, strip_binary_hunks


VERIFIER_FILES = ("test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch")
TESTS_DIR = "/tests"
LOGS_DIR = "/logs/verifier"
WORKSPACE_DIR = "/workspace"
PATCH_PATH = f"{WORKSPACE_DIR}/patch.diff"
SCRIPT_PATH = f"{WORKSPACE_DIR}/verify_v2.sh"
TEST_SH_LOG_PATH = f"{WORKSPACE_DIR}/test_sh.log"
PATCH_STATUS_PATH = f"{WORKSPACE_DIR}/patch_apply_status"
TEST_SH_STATUS_PATH = f"{WORKSPACE_DIR}/test_sh_status"
TEST_PATCH_FAILURE = "ERROR: Failed to apply test patch"
# Time the replay script gets beyond the test.sh budget: patch application, the kill grace period, slack.
EXEC_MARGIN_S = 300
# Exit statuses of a test.sh that timeout stopped: GNU 124 (TERM) or 137 (KILL after -k), BusyBox 143.
TIMEOUT_STATUSES = frozenset({124, 137, 143})


def replay_script(test_timeout_s: int | None) -> str:
    """Apply the patch as patch_replay.py does, then run test.sh within its budget.

    GNU timeout exits 124 or 137 when it stops test.sh, BusyBox's (the Alpine images) 143, and an old BusyBox has no
    -k. So the script records test.sh's status and run time; see ``stopped_at_budget``.
    """
    select_timeout = ""
    if test_timeout_s is not None:
        budget = int(test_timeout_s)
        select_timeout = f"""if command -v timeout >/dev/null 2>&1; then
  if timeout -k 1 5 true >/dev/null 2>&1; then ng_timeout="timeout -k 30 {budget}"; else ng_timeout="timeout {budget}"; fi
fi"""
    return f"""set -u
{SCRUB_HARNESS_ENV}
mkdir -p {LOGS_DIR}
cd /app 2>/dev/null || cd /testbed 2>/dev/null || {{ echo "NG_APPLY: neither /app nor /testbed exists"; exit 2; }}
if [ -s {PATCH_PATH} ]; then
  git apply --verbose {PATCH_PATH} || git apply --3way {PATCH_PATH} || patch --fuzz=3 -p1 -i {PATCH_PATH}
  echo $? > {PATCH_STATUS_PATH}
else
  echo "NG_APPLY: patch is empty, nothing to apply"
  echo 0 > {PATCH_STATUS_PATH}
fi
ng_timeout=""
{select_timeout}
ng_start=$(date +%s)
$ng_timeout bash {TESTS_DIR}/test.sh > {TEST_SH_LOG_PATH} 2>&1
ng_status=$?
echo "$ng_status $(( $(date +%s) - ng_start ))" > {TEST_SH_STATUS_PATH}
exit 0
"""


def stopped_at_budget(status_line: str, test_timeout_s: int | None) -> bool:
    """Whether the recorded ``<status> <seconds>`` line shows a test.sh that timeout stopped at its budget."""
    try:
        status, elapsed = (int(value) for value in status_line.split())
    except ValueError:
        return False
    return test_timeout_s is not None and status in TIMEOUT_STATUSES and elapsed >= test_timeout_s


def verification_files(verifier_files: Mapping[str, str], patch: str, *, test_timeout_s: int | None) -> dict[str, str]:
    """Sandbox files for one V2 verification: the task's verifier, the model patch and the replay script."""
    missing = [name for name in VERIFIER_FILES if name not in verifier_files]
    if missing:
        raise ValueError(f"SWE-bench Pro V2 row is missing verifier files: {missing}")
    files = {f"{TESTS_DIR}/{name}": verifier_files[name] for name in VERIFIER_FILES}
    files[PATCH_PATH] = strip_binary_hunks(patch)
    files[SCRIPT_PATH] = replay_script(test_timeout_s)
    return files


async def run_v2_verification(
    sandbox: AsyncSandbox,
    log_dir: Path,
    timeout_s: int | None,
) -> VerificationResult:
    """Apply the uploaded patch, run the task's ``test.sh`` (budget ``timeout_s``) and read its reward."""
    log_dir.mkdir(parents=True, exist_ok=True)
    execution = None
    execution_error = None
    exec_timeout_s = None if timeout_s is None else timeout_s + EXEC_MARGIN_S
    try:
        execution = await sandbox.exec(f"bash {shlex.quote(SCRIPT_PATH)}", timeout_s=exec_timeout_s)
    except Exception as exc:
        execution_error = exc

    async def capture(remote_path: str, local_name: str) -> str:
        try:
            result = await sandbox.exec(f"cat {shlex.quote(remote_path)}")
            contents = (result.stdout or "") if result.return_code == 0 else ""
        except Exception:
            contents = ""
        (log_dir / local_name).write_text(contents, encoding="utf-8")
        return contents

    test_sh_log = await capture(TEST_SH_LOG_PATH, "test_sh.log")
    test_stdout = await capture(f"{LOGS_DIR}/run-script-stdout.txt", "test_stdout.log")
    test_stderr = await capture(f"{LOGS_DIR}/run-script-stderr.txt", "test_stderr.log")
    output_text = await capture(f"{LOGS_DIR}/output.json", "output.json")
    reward_text = (await capture(f"{LOGS_DIR}/reward.txt", "reward.txt")).strip()
    patch_applied = (await capture(PATCH_STATUS_PATH, "patch_apply_status")).strip() == "0"
    test_sh_status = (await capture(TEST_SH_STATUS_PATH, "test_sh_status")).strip()
    test_output = f"TEST.SH:\n{test_sh_log}\n\nSTDOUT:\n{test_stdout}\n\nSTDERR:\n{test_stderr}"

    def inconclusive(error: str, *, timed_out: bool = False) -> VerificationResult:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            test_output=test_output,
            error=error,
            timed_out=timed_out,
        )

    if execution_error is not None:
        return inconclusive(
            f"Evaluation execution failed: {execution_error}", timed_out=isinstance(execution_error, TimeoutError)
        )
    if execution.error_type:
        return inconclusive(
            f"Evaluation execution {execution.error_type} (limit: {exec_timeout_s}s)",
            timed_out=execution.error_type == "timeout",
        )
    if execution.return_code != 0:
        return inconclusive(f"Verification script exited with {execution.return_code}: {execution.stdout or ''}")
    if stopped_at_budget(test_sh_status, timeout_s):
        # Upstream's verifier timeout: the patched suite did not finish, which scores 0.
        return VerificationResult(
            completed=True,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            test_output=test_output,
            error=f"test.sh exceeded its {timeout_s}s budget; scored 0",
            timed_out=True,
        )
    if TEST_PATCH_FAILURE in test_sh_log:
        return inconclusive("The task's test patch did not apply")
    if reward_text not in ("0", "1"):
        return inconclusive(f"test.sh wrote no reward (reward.txt: {reward_text!r})")
    try:
        test_results = json.loads(output_text)
    except json.JSONDecodeError:
        test_results = None

    return VerificationResult(
        completed=True,
        resolved=reward_text == "1",
        patch_applied=patch_applied,
        test_results=test_results if isinstance(test_results, dict) else None,
        test_output=test_output,
    )


def v2_inconclusive_reason(result: VerificationResult) -> str | None:
    """Say why a V2 verification has no verdict, or ``None`` if ``test.sh`` gave one."""
    if not result.completed:
        return f"evaluation did not complete ({result.error or 'unknown error'})"
    return None
