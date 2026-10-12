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
"""Run one OML SWE-bench Extended task's own grader in a sandbox and read its verdict.

Every task package ships a self-contained verifier under ``tests/``: ``test.sh`` restores the
graded test surface from the image's pristine snapshot, installs the hidden ``test.patch``, runs
the suite and calls ``grade.py`` (per-test FAIL_TO_PASS enforcement over ``config.json``), which
writes ``1`` or ``0`` to ``/logs/verifier/reward.txt``. The frameworks span js/maven/gradle/
ctest/cargo-nextest and a long tail, and ``grade.py`` differs per task (1,510 distinct copies in
the 5k delivery), so nothing here re-parses test output: the package's grader is the verdict.

The sandbox receives the four ``tests/`` files at ``/tests`` (the layout ``test.sh`` assumes via
``SCRIPT_DIR``), the candidate patch is applied with the same fallback chain the package's own
``solution/solve.sh`` uses, then ``test.sh`` runs and the reward file is read back.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from resources_servers.swemer_v1.verification import mirror_files
from resources_servers.swemer_v2.verification import drop_patch_sections, drop_test_patch_files


__all__ = [
    "VerificationInputs",
    "VerificationResult",
    "build_eval_script",
    "drop_patch_sections",
    "drop_test_patch_files",
    "parse_verification_output",
    "run_verification",
    "verification_files",
]

TESTS_DIR = "/tests"
PATCH_FILE = "/tmp/nemo_gym_patch.diff"
TEST_SH_LOG = "/tmp/nemo_gym_test_sh.log"
REWARD_FILE = "/logs/verifier/reward.txt"

PATCH_APPLIED_MARK = "___NEMO_GYM_SWEMER_OML_PATCH_APPLIED___"
TEST_OUTPUT_BEGIN = "___NEMO_GYM_SWEMER_OML_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SWEMER_OML_TEST_END___"
REWARD_BEGIN = "___NEMO_GYM_SWEMER_OML_REWARD_BEGIN___"
REWARD_END = "___NEMO_GYM_SWEMER_OML_REWARD_END___"

# Strings test.sh itself prints when it fails closed before/without a real verdict.
TEST_PATCH_FAILED_TEXT = "ERROR: test.patch failed to apply"
NO_REWARD_TEXT = "VERIFIER: grader produced no reward"

# How much of test.sh's output comes back; the tail carries the grader summary.
TEST_OUTPUT_TAIL_BYTES = 400_000

WORKDIR_MISSING_EXIT = 97


@dataclass
class VerificationInputs:
    instance_id: str
    workdir: str
    patch: str
    test_sh: str
    config_json: str
    grade_py: str
    test_patch: str = ""
    test_framework: str = ""
    fail_to_pass: Sequence[str] = field(default_factory=tuple)
    pass_to_pass: Sequence[str] = field(default_factory=tuple)
    # Extra flag for the first `patch -p1` attempt; the package's solve.sh passes "--fuzz=5" for
    # the golden patch (its lockfile hunks often drift) and nothing for anything else.
    apply_primary: str = ""


@dataclass
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None = None
    test_patch_failed: bool = False


# The package's own apply chain (solution/solve.sh), verbatim, so a golden patch is applied the
# way the vendor's QC applied it and a model patch gets the same leniency.
_APPLY_PATCH_FN = """apply_patch() {
  local repo="$1" pf="$2" primary="${3:-}"
  patch -p1 -d "$repo" $primary -i "$pf" 2>/dev/null && return 0
  ( cd "$repo" && git apply --3way --ignore-whitespace "$pf" ) 2>/dev/null && return 0
  ( cd "$repo" && git apply --ignore-whitespace --whitespace=nowarn "$pf" ) 2>/dev/null && return 0
  patch -p1 -d "$repo" --fuzz=5 -i "$pf" 2>/dev/null && return 0
  patch -p0 -d "$repo" --fuzz=5 -i "$pf" 2>/dev/null && return 0
  ( cd "$repo" && git ls-files -z 2>/dev/null | xargs -0 -r sed -i 's/\\r$//' ) 2>/dev/null
  ( cd "$repo" && git apply --3way --ignore-whitespace "$pf" ) 2>/dev/null && return 0
  patch -p1 -d "$repo" --fuzz=5 -i "$pf" 2>/dev/null && return 0
  return 1
}
"""


def build_eval_script(inputs: VerificationInputs) -> str:
    """The script run inside the sandbox: apply the candidate patch, run the package's test.sh,
    echo the reward file between markers. Exits with test.sh's own exit code."""
    wd = shlex.quote(inputs.workdir)
    apply_block = (
        f"if apply_patch {wd} {PATCH_FILE} {shlex.quote(inputs.apply_primary)}; then patch_applied=1; "
        "else patch_applied=0; fi\n"
        # The chain's exit code alone under-reports: `patch` applies what it can before failing
        # (a drifted lockfile hunk after the image's npm install), so also record whether the
        # tree changed at all.
        f"__after=$(git -C {wd} status --porcelain 2>/dev/null | md5sum)\n"
        '[ "$__before" != "$__after" ] && tree_changed=1 || tree_changed=0\n'
        if inputs.patch.strip()
        else "patch_applied=1\n"
    )
    return (
        "#!/bin/bash\n"
        f"cd {wd} || exit {WORKDIR_MISSING_EXIT}\n"
        "set +e\n"
        f"mkdir -p /logs/verifier /workspace/test-results {TESTS_DIR} 2>/dev/null\n"
        f"rm -f {REWARD_FILE}\n"
        f"{_APPLY_PATCH_FN}"
        f"__before=$(git -C {wd} status --porcelain 2>/dev/null | md5sum)\n"
        "tree_changed=1\n"
        f"{apply_block}"
        f'echo "{PATCH_APPLIED_MARK} $patch_applied $tree_changed"\n'
        f'echo "{TEST_OUTPUT_BEGIN}"\n'
        f"bash {TESTS_DIR}/test.sh > {TEST_SH_LOG} 2>&1\n"
        "__test_exit=$?\n"
        f"tail -c {TEST_OUTPUT_TAIL_BYTES} {TEST_SH_LOG}\n"
        f'echo "{TEST_OUTPUT_END}"\n'
        f'echo "{REWARD_BEGIN}"\n'
        f"cat {REWARD_FILE} 2>/dev/null\n"
        f'echo "{REWARD_END}"\n'
        "exit $__test_exit\n"
    )


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    """Files placed in the verification sandbox before the eval script runs."""
    files = {
        "/tmp/nemo_gym_eval.sh": build_eval_script(inputs),
        f"{TESTS_DIR}/test.sh": inputs.test_sh,
        f"{TESTS_DIR}/config.json": inputs.config_json,
        f"{TESTS_DIR}/grade.py": inputs.grade_py,
    }
    # Most packages embed test.patch in test.sh; a few `cp "$SCRIPT_DIR/test.patch"` instead.
    if inputs.test_patch.strip():
        files[f"{TESTS_DIR}/test.patch"] = inputs.test_patch
    if inputs.patch.strip():
        files[PATCH_FILE] = inputs.patch
    # Maven Central 429s under concurrent load (seen in this dataset's first smoke); the
    # Google-hosted mirror fix from swemer_v1 is harmless for non-JVM rows.
    files.update(mirror_files())
    return files


def _slice(log: str, begin: str, end: str) -> str:
    start = log.find(begin)
    if start == -1:
        return ""
    start += len(begin)
    stop = log.find(end, start)
    return log[start:stop] if stop != -1 else log[start:]


_PATCH_APPLIED_RE = re.compile(re.escape(PATCH_APPLIED_MARK) + r"\s+([01])(?:\s+([01]))?")


def parse_verification_output(output: str, return_code: int, inputs: VerificationInputs) -> VerificationResult:
    """Turn the eval script's combined output into a verdict.

    ``completed`` means the package's grader produced a real reward: a missing reward file, a
    hidden test patch that did not install, or test.sh failing closed are infrastructure/row
    faults, not evidence about the patch, and are reported as incomplete so a sweep can retry or
    bucket them separately.
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
    patch_applied = applied_match is not None and applied_match.group(1) == "1"
    # A non-zero apply chain that still changed the tree is a partial apply; the package grader's
    # verdict on the result stands (it is what an agent's own partially-applying patch would get).
    tree_changed = applied_match is not None and applied_match.group(2) != "0"
    patch_touched_tree = patch_applied or tree_changed
    reward_text = _slice(output, REWARD_BEGIN, REWARD_END).strip()
    test_patch_failed = TEST_PATCH_FAILED_TEXT in output
    grader_silent = NO_REWARD_TEXT in output

    error: str | None = None
    if applied_match is None:
        error = "eval script did not reach the patch step"
    elif test_patch_failed:
        error = "held-out test patch did not apply"
    elif grader_silent:
        error = "package grader produced no reward (test.sh failed closed)"
    elif reward_text not in {"0", "1"}:
        error = f"no reward in {REWARD_FILE} (got {reward_text[:40]!r})"
    elif not patch_touched_tree:
        error = "candidate patch did not apply"
    elif not patch_applied:
        error = "candidate patch applied only partially"

    completed = error is None or error.startswith("candidate patch")
    reward = 1 if (completed and reward_text == "1") else 0
    resolved = completed and patch_touched_tree and reward == 1
    test_results = {
        "reward": reward,
        "test_exit_code": return_code,
        "patch_applied": patch_applied,
        "patch_partially_applied": tree_changed and not patch_applied,
        "test_patch_failed": test_patch_failed,
        "framework": inputs.test_framework,
        "FAIL_TO_PASS": list(inputs.fail_to_pass),
        "PASS_TO_PASS": list(inputs.pass_to_pass),
        "resolved": resolved,
    }
    return VerificationResult(
        completed=completed,
        resolved=resolved,
        patch_applied=patch_touched_tree,
        test_results=test_results if applied_match is not None else None,
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
