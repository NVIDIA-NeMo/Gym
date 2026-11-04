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
"""Run one internal-v1 SWE task's tests in a sandbox and grade the result.

Every row carries its own ``run_script.sh`` (how to run the suite, or a selected list of test
files) and ``parsing_script.py`` (turn the suite's stdout/stderr into ``{"tests": [{"name",
"status"}]}``), authored per task by the dataset vendor. This is the same contract the
``swe_agents`` harness' ``NVInternalDatasetProcessor`` grades with, reproduced here step for
step so a verdict from this server matches one from that harness:

1. ``export`` every ``ENV`` line of the task's Dockerfiles (``env_exports``),
2. ``git reset --hard <base_commit>`` in the repo, apply the candidate patch with
   ``git apply --reject`` (partial applies keep what applied),
3. install the hidden tests: the row's checkout command (``git checkout <fix_commit> -- <test
   files>``) first, the ``test_patch`` diff as a fallback,
4. ``bash run_script.sh <comma-separated test files>``, then ``parsing_script.py stdout stderr
   output.json``,
5. resolved iff ``FAIL_TO_PASS ∪ PASS_TO_PASS`` is non-empty and every id is ``PASSED``.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from resources_servers.swemer_v2.verification import drop_patch_sections, drop_test_patch_files


__all__ = [
    "VerificationInputs",
    "VerificationResult",
    "build_eval_script",
    "drop_patch_sections",
    "drop_test_patch_files",
    "grade",
    "parse_verification_output",
    "run_verification",
    "verification_files",
]

PATCH_FILE = "/tmp/nemo_gym_patch.diff"
TEST_PATCH_FILE = "/tmp/nemo_gym_test_patch.diff"
RUN_SCRIPT_FILE = "/tmp/nemo_gym_run_script.sh"
PARSING_SCRIPT_FILE = "/tmp/nemo_gym_parsing_script.py"
STDOUT_FILE = "/tmp/nemo_gym_stdout.log"
STDERR_FILE = "/tmp/nemo_gym_stderr.log"
OUTPUT_FILE = "/tmp/nemo_gym_output.json"

PATCH_APPLIED_MARK = "___NEMO_GYM_SWE_INTERNAL_V1_PATCH_APPLIED___"
TEST_PATCH_FAILED = "___NEMO_GYM_SWE_INTERNAL_V1_TEST_PATCH_FAILED___"
TEST_OUTPUT_BEGIN = "___NEMO_GYM_SWE_INTERNAL_V1_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SWE_INTERNAL_V1_TEST_END___"
RESULT_FILE_BEGIN = "___NEMO_GYM_SWE_INTERNAL_V1_RESULT_FILE_BEGIN___"
RESULT_FILE_END = "___NEMO_GYM_SWE_INTERNAL_V1_RESULT_FILE_END___"

PASSED = "PASSED"
WORKDIR_MISSING_EXIT = 97

# How much of the suite's stdout / stderr comes back for the log (the parsed verdict is separate).
STDOUT_TAIL_BYTES = 300_000
STDERR_TAIL_BYTES = 100_000


@dataclass
class VerificationInputs:
    instance_id: str
    workdir: str
    base_commit: str
    patch: str
    run_script: str
    parsing_script: str
    test_files: Sequence[str] = field(default_factory=tuple)
    test_patch: str = ""
    # ``git checkout <fix_commit> -- <test files>``: how the vendor's harness installs the hidden
    # tests (the fix commit is present in the image's history). Empty -> test_patch only.
    test_patch_checkout_cmd: str = ""
    env_exports: Sequence[str] = field(default_factory=tuple)
    fail_to_pass: Sequence[str] = field(default_factory=tuple)
    pass_to_pass: Sequence[str] = field(default_factory=tuple)


@dataclass
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None = None
    test_patch_failed: bool = False


def _env_block(env_exports: Iterable[str]) -> str:
    lines = [line.strip() for line in env_exports if line and line.strip()]
    return "\n".join(line if line.startswith("export ") else f"export {line}" for line in lines)


def build_eval_script(inputs: VerificationInputs) -> str:
    """The script run inside the sandbox; exits with the test command's own exit code."""
    wd = shlex.quote(inputs.workdir)
    base = shlex.quote(inputs.base_commit) if inputs.base_commit else ""
    reset_block = f"git reset -q --hard {base} 2>/dev/null; git checkout -q {base} 2>/dev/null\n" if base else ""

    if inputs.patch.strip():
        apply_block = (
            f"__before=$(git status --porcelain 2>/dev/null | md5sum)\n"
            f"if git apply --ignore-space-change --ignore-whitespace --reject -v {PATCH_FILE} "
            f">/tmp/nemo_gym_patch_apply.log 2>&1; then patch_applied=1; else patch_applied=0; fi\n"
            f"__after=$(git status --porcelain 2>/dev/null | md5sum)\n"
            '[ "$__before" != "$__after" ] && tree_changed=1 || tree_changed=0\n'
            "tail -c 4000 /tmp/nemo_gym_patch_apply.log\n"
        )
    else:
        apply_block = "patch_applied=1\ntree_changed=1\n"

    checkout = inputs.test_patch_checkout_cmd.strip()
    test_patch_steps: list[str] = []
    if checkout:
        test_patch_steps.append(f"( {checkout} ) >/tmp/nemo_gym_test_checkout.log 2>&1")
    if inputs.test_patch.strip():
        test_patch_steps.append(
            f"git apply --ignore-space-change --ignore-whitespace --reject -v {TEST_PATCH_FILE} "
            ">/tmp/nemo_gym_test_patch.log 2>&1"
        )
    if test_patch_steps:
        install_tests = (
            "if " + " || ".join(test_patch_steps) + "; then :; else\n"
            f'  echo "{TEST_PATCH_FAILED}"; cat /tmp/nemo_gym_test_checkout.log /tmp/nemo_gym_test_patch.log 2>/dev/null | tail -c 4000\n'
            "fi\n"
        )
    else:
        install_tests = ""

    test_files_arg = shlex.quote(",".join(inputs.test_files)) if inputs.test_files else ""

    return (
        "#!/bin/bash\n"
        f"cd {wd} || exit {WORKDIR_MISSING_EXIT}\n"
        "set +e\n"
        f"{_env_block(inputs.env_exports)}\n"
        f"cd {wd}\n"
        f"{reset_block}"
        f"{apply_block}"
        f'echo "{PATCH_APPLIED_MARK} $patch_applied $tree_changed"\n'
        f"{install_tests}"
        f'echo "{TEST_OUTPUT_BEGIN}"\n'
        f"bash {RUN_SCRIPT_FILE} {test_files_arg} > {STDOUT_FILE} 2> {STDERR_FILE}\n"
        "__test_exit=$?\n"
        f"__py=$(command -v python3 || command -v python)\n"
        f'"$__py" {PARSING_SCRIPT_FILE} {STDOUT_FILE} {STDERR_FILE} {OUTPUT_FILE} >/tmp/nemo_gym_parse.log 2>&1 '
        '|| echo "parsing_script failed: $(tail -c 2000 /tmp/nemo_gym_parse.log)"\n'
        f"tail -c {STDOUT_TAIL_BYTES} {STDOUT_FILE}\n"
        'echo; echo "--- stderr ---"\n'
        f"tail -c {STDERR_TAIL_BYTES} {STDERR_FILE}\n"
        f'echo "{TEST_OUTPUT_END}"\n'
        f'echo "{RESULT_FILE_BEGIN}"\n'
        f"cat {OUTPUT_FILE} 2>/dev/null\n"
        f'echo "{RESULT_FILE_END}"\n'
        "exit $__test_exit\n"
    )


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    """Files placed in the verification sandbox before the eval script runs."""
    files = {
        "/tmp/nemo_gym_eval.sh": build_eval_script(inputs),
        RUN_SCRIPT_FILE: inputs.run_script,
        PARSING_SCRIPT_FILE: inputs.parsing_script,
    }
    if inputs.patch.strip():
        files[PATCH_FILE] = inputs.patch
    if inputs.test_patch.strip():
        files[TEST_PATCH_FILE] = inputs.test_patch
    return files


def _slice(log: str, begin: str, end: str) -> str:
    start = log.find(begin)
    if start == -1:
        return ""
    start += len(begin)
    stop = log.find(end, start)
    return log[start:stop] if stop != -1 else log[start:]


def grade(statuses: dict[str, str], fail_to_pass: Iterable[str], pass_to_pass: Iterable[str]) -> dict[str, Any]:
    """Resolved only when every required test is observed AND passing (the vendor harness'
    ``check_tests_passed``: ``required <= passed`` with both sets non-empty). A test absent from
    the parsed output counts as not passing."""

    def split(names: Iterable[str]) -> tuple[list[str], list[str]]:
        passed, failed = [], []
        for name in names:
            (passed if statuses.get(name) == PASSED else failed).append(name)
        return passed, failed

    f2p_passed, f2p_failed = split(fail_to_pass)
    p2p_passed, p2p_failed = split(pass_to_pass)
    required = len(f2p_passed) + len(f2p_failed) + len(p2p_passed) + len(p2p_failed)
    return {
        "FAIL_TO_PASS": {"success": f2p_passed, "failure": f2p_failed},
        "PASS_TO_PASS": {"success": p2p_passed, "failure": p2p_failed},
        "tests_observed": len(statuses),
        "tests_passed": sum(1 for s in statuses.values() if s == PASSED),
        "resolved": required > 0 and not f2p_failed and not p2p_failed,
    }


def parse_output_json(text: str) -> dict[str, str] | None:
    """``parsing_script.py``'s ``{"tests": [{"name", "status"}]}`` -> name -> status, or None when
    the file is missing/unparseable."""
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    tests = payload.get("tests") if isinstance(payload, dict) else None
    if not isinstance(tests, list):
        return None
    statuses: dict[str, str] = {}
    for test in tests:
        if isinstance(test, dict) and test.get("name") is not None:
            statuses[str(test["name"])] = str(test.get("status", "")).upper()
    return statuses


_PATCH_APPLIED_RE = re.compile(re.escape(PATCH_APPLIED_MARK) + r"\s+([01])\s+([01])")


def parse_verification_output(output: str, return_code: int, inputs: VerificationInputs) -> VerificationResult:
    """Turn the eval script's combined output into a verdict.

    ``completed`` is false when there is no verdict to read: the hidden tests could not be
    installed, or ``parsing_script.py`` produced no result file. Those are row/infra faults, not
    evidence about the patch, and a sweep retries or buckets them separately.
    """
    if return_code == WORKDIR_MISSING_EXIT:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error=f"workdir {inputs.workdir} not present in the image",
        )

    applied_match = _PATCH_APPLIED_RE.search(output)
    if applied_match is None:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error="eval script did not reach the patch step",
        )
    patch_applied = applied_match.group(1) == "1"
    tree_changed = applied_match.group(2) == "1"
    patch_touched_tree = patch_applied or tree_changed
    test_patch_failed = TEST_PATCH_FAILED in output

    statuses = parse_output_json(_slice(output, RESULT_FILE_BEGIN, RESULT_FILE_END))
    report = grade(statuses or {}, inputs.fail_to_pass, inputs.pass_to_pass)
    report["test_exit_code"] = return_code
    report["patch_applied"] = patch_applied
    report["patch_partially_applied"] = tree_changed and not patch_applied
    report["test_patch_failed"] = test_patch_failed

    error: str | None = None
    if test_patch_failed:
        error = "held-out tests did not install (checkout and test_patch both failed)"
    elif statuses is None:
        error = "parsing_script produced no result file"
    elif inputs.patch.strip() and not patch_touched_tree:
        error = "candidate patch did not apply"
    elif inputs.patch.strip() and not patch_applied:
        error = "candidate patch applied only partially"

    completed = error is None or error.startswith("candidate patch")
    resolved = completed and bool(report["resolved"]) and (patch_touched_tree or not inputs.patch.strip())
    report["resolved"] = resolved
    return VerificationResult(
        completed=completed,
        resolved=resolved,
        patch_applied=patch_touched_tree,
        test_results=report,
        test_output=output,
        error=error,
        test_patch_failed=test_patch_failed,
    )


async def run_verification(
    sandbox: Any,
    inputs: VerificationInputs,
    timeout_s: float | None = None,
    log_dir: Path | None = None,
) -> VerificationResult:
    import asyncio

    result = await sandbox.exec("bash /tmp/nemo_gym_eval.sh", timeout_s=timeout_s)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

    if log_dir is not None:

        def _persist() -> None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "test_output.log").write_text(output, errors="replace")
            except OSError:
                pass

        await asyncio.to_thread(_persist)

    return parse_verification_output(output, result.return_code, inputs)
