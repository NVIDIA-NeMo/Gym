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

"""Run the pinned task's test.sh and preserve its final reward and inconclusive outcomes."""

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox
from resources_servers.swebench_pro.verification import (
    SCRUB_HARNESS_ENV,
    VerificationInputs,
    VerificationResult,
    build_environment_repairs,
    strip_binary_hunks,
)


TESTS_DIR = "/tests"
TEST_FILES = ("test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch")
ENTRY_SCRIPT_PATH = "/workspace/entryscript.sh"
TEST_SH_STATUS_PATH = "/workspace/test_sh_status"
TEST_SH_EXIT_PATH = "/workspace/test_sh_exit_status"
EXEC_MARGIN_S = 300
TIMEOUT_STATUSES = frozenset({124, 137, 143})


@dataclass(frozen=True)
class VerificationInputsV2(VerificationInputs):
    environment_repairs: tuple[str, ...] = ()
    test_sh: str = ""
    test_config: str = ""
    test_patch: str = ""
    evaluation_timeout: int | None = None


def create_entryscript(sample: dict[str, Any]) -> str:
    """Apply the candidate patch and run the task's unchanged upstream verifier."""
    timeout_s = sample.get("evaluation_timeout")
    select_timeout = "ng_timeout=''"
    if timeout_s is not None:
        if not isinstance(timeout_s, int) or timeout_s <= 0:
            raise ValueError("SWE-bench Pro V2 evaluation_timeout must be a positive integer")
        select_timeout = f"""command -v timeout >/dev/null 2>&1 || {{ echo 'Verifier image has no timeout command'; exit 2; }}
if timeout -k 1 5 true >/dev/null 2>&1; then
  ng_timeout='timeout -k 30 {timeout_s}'
else
  ng_timeout='timeout {timeout_s}'
fi"""
    return f"""set -u
{SCRUB_HARNESS_ENV}
if git -C /app rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cd /app
elif git -C /testbed rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cd /testbed
else
  echo 'No task git repository at /app or /testbed'; exit 2
fi
if [ -s /workspace/patch.diff ]; then
  git apply --verbose /workspace/patch.diff || git apply --3way /workspace/patch.diff || patch --fuzz=3 -p1 -i /workspace/patch.diff
  printf '%s\\n' "$?" > /workspace/patch_apply_status
else
  echo 0 > /workspace/patch_apply_status
fi
{build_environment_repairs(sample.get("environment_repairs", ()))}
mkdir -p {TESTS_DIR} /logs/verifier
cp {" ".join(f"/workspace/{name}" for name in TEST_FILES)} {TESTS_DIR}/ || exit 2
rm -f {TEST_SH_EXIT_PATH} /logs/verifier/reward.txt
{select_timeout}
$ng_timeout bash -c 'bash {TESTS_DIR}/test.sh; ng_exit=$?; printf "%s\\n" "$ng_exit" > {TEST_SH_EXIT_PATH}; exit "$ng_exit"' > /workspace/test_sh.log 2>&1
printf '%s\\n' "$?" > {TEST_SH_STATUS_PATH}
exit 0
"""


def assemble_workspace_files(
    uid: str,
    scripts_dir: str | None,
    patch: str,
    sample: dict[str, Any],
) -> tuple[dict[str, str], str]:
    """Build the fresh verifier's workspace from the checksum-verified task files."""
    del uid, scripts_dir
    entryscript = create_entryscript(sample)
    return {
        "patch.diff": strip_binary_hunks(patch),
        "entryscript.sh": entryscript,
        "test.sh": sample["test_sh"],
        "run_script.sh": sample["run_script"],
        "parser.py": sample["parser_script"],
        "config.json": sample["test_config"],
        "test_patch.patch": sample["test_patch"],
    }, entryscript


async def run_verification(
    sandbox: AsyncSandbox,
    inputs: VerificationInputsV2,
    log_dir: Path,
    timeout_s: int | None,
) -> VerificationResult:
    """Preserve upstream reward.txt; keep provider, wrapper and artifact transport faults inconclusive."""
    log_dir.mkdir(parents=True, exist_ok=True)
    files, script = assemble_workspace_files(inputs.instance_id, None, inputs.patch, asdict(inputs))
    (log_dir / "patch.diff").write_text(files["patch.diff"], encoding="utf-8")
    (log_dir / "eval.sh").write_text(script, encoding="utf-8")
    execution = None
    execution_error = None
    exec_timeout_s = None if timeout_s is None else timeout_s + EXEC_MARGIN_S
    try:
        execution = await sandbox.exec(f"bash {ENTRY_SCRIPT_PATH}", timeout_s=exec_timeout_s)
    except Exception as exc:
        execution_error = exc
    (log_dir / "execution_stdout.log").write_text(
        (execution.stdout if execution is not None else "") or "", encoding="utf-8"
    )
    (log_dir / "execution_stderr.log").write_text(
        (execution.stderr if execution is not None else "") or "", encoding="utf-8"
    )

    capture_errors: list[str] = []

    async def capture(remote_path: str, local_name: str) -> str:
        try:
            result = await sandbox.exec(f"cat {shlex.quote(remote_path)}")
            if result.error_type is not None:
                capture_errors.append(f"{remote_path}: {result.error_type}")
            contents = (result.stdout or "") if result.return_code == 0 and result.error_type is None else ""
        except Exception as exc:
            capture_errors.append(f"{remote_path}: {type(exc).__name__}: {exc}")
            contents = ""
        (log_dir / local_name).write_text(contents, encoding="utf-8")
        return contents

    test_sh_log = await capture("/workspace/test_sh.log", "test_sh.log")
    stdout = await capture("/logs/verifier/run-script-stdout.txt", "test_stdout.log")
    stderr = await capture("/logs/verifier/run-script-stderr.txt", "test_stderr.log")
    output_text = await capture("/logs/verifier/output.json", "output.json")
    reward = (await capture("/logs/verifier/reward.txt", "reward.txt")).strip()
    patch_status = (await capture("/workspace/patch_apply_status", "patch_apply_status")).strip()
    test_status = (await capture(TEST_SH_STATUS_PATH, "test_sh_status")).strip()
    test_exit = (await capture(TEST_SH_EXIT_PATH, "test_sh_exit_status")).strip()
    test_output = f"TEST.SH:\n{test_sh_log}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

    def inconclusive(error: str, *, timed_out: bool = False) -> VerificationResult:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_status == "0",
            test_results=None,
            test_output=test_output,
            error=error,
            timed_out=timed_out,
        )

    if execution_error is not None:
        return inconclusive(
            f"Evaluation execution failed: {type(execution_error).__name__}: {execution_error}",
            timed_out=isinstance(execution_error, TimeoutError),
        )
    if execution.error_type is not None:
        return inconclusive(
            f"Evaluation execution {execution.error_type}", timed_out=execution.error_type == "timeout"
        )
    if execution.return_code != 0:
        return inconclusive(f"Verification script exited with {execution.return_code}: {execution.stdout or ''}")
    if capture_errors:
        return inconclusive("Failed to collect verifier artifacts: " + "; ".join(capture_errors))
    if timeout_s is not None and test_status in {str(status) for status in TIMEOUT_STATUSES} and not test_exit:
        return VerificationResult(
            completed=True,
            resolved=False,
            patch_applied=patch_status == "0",
            test_results=None,
            test_output=test_output,
            error=f"test.sh exceeded its {timeout_s}s budget; scored 0",
            timed_out=True,
        )
    if "ERROR: Failed to apply test patch" in test_sh_log:
        return inconclusive("The task's test patch did not apply")
    if reward not in ("0", "1"):
        return inconclusive(f"test.sh wrote no reward (reward.txt: {reward!r})")
    try:
        output = json.loads(output_text)
    except json.JSONDecodeError:
        output = None
    return VerificationResult(
        completed=True,
        resolved=reward == "1",
        patch_applied=patch_status == "0",
        test_results=output if isinstance(output, dict) else None,
        test_output=test_output,
    )
