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
"""Run one Swemer Agentic-v2 task's tests in a sandbox and grade the result.

Every row ships its own hand-authored ``test_command`` (there is no per-language template like
Scale-SWE/SWE-rebench have): the dataset authors already ran the actual test suite and recorded
FAIL_TO_PASS/PASS_TO_PASS test ids in a dotted convention (e.g. ``tests.pkg.test_mod::test_name``
for pytest) rather than each framework's own real node-id format.

Grading is delegated to ``responses_api_agents.swe_agents.swe_bench_ext`` -- the same
output-flag injection and test-result parsing the swe-bench-ext harness itself uses -- instead of
a hand-rolled parser per framework (an earlier version of this file had one; see git history).
``normalize_test_id`` in ``grade()`` is what makes swe_bench_ext's real-node-id-keyed parser
output match this dataset's dotted ids: confirmed for real against the existing golden-patch-
validated dataset (149/149 rows across all 5 frameworks, pytest/go/jest/mocha/vitest, resolved
identically to the previous hand-rolled parsers) before this replaced them.

Only ``SUPPORTED_FRAMEWORKS`` are handled -- kept at the same 5 the hand-rolled parsers covered,
not swe_bench_ext's full ~22, since the other ~15 aren't in ``data/swemer_v2_training.jsonl`` at
all (this dataset's raw source isn't available locally to test whether they could be added; see
swemer_v1's README for the sibling dataset where it was).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from responses_api_agents.swe_agents.swe_bench_ext.frameworks import (
    get_framework_config,
    get_test_command_with_output,
)
from responses_api_agents.swe_agents.swe_bench_ext.parsing import normalize_test_id, parse_test_output


TEST_OUTPUT_BEGIN = "___NEMO_GYM_SWEMER_V2_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SWEMER_V2_TEST_END___"
TEST_PATCH_FAILED = "___NEMO_GYM_SWEMER_V2_TEST_PATCH_FAILED___"
RESULT_FILE_BEGIN = "___NEMO_GYM_SWEMER_V2_RESULT_FILE_BEGIN___"
RESULT_FILE_END = "___NEMO_GYM_SWEMER_V2_RESULT_FILE_END___"

SUPPORTED_FRAMEWORKS = frozenset({"pytest", "go", "jest", "mocha", "vitest"})

PASSED = "PASSED"


@dataclass
class VerificationInputs:
    instance_id: str
    workdir: str
    patch: str
    test_patch: str
    test_framework: str
    test_command: str
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


def patch_section_path(section: str) -> str | None:
    """Return the repository-relative path one ``diff --git`` section targets."""
    a_path: str | None = None
    b_path: str | None = None
    for line in section.splitlines():
        if line.startswith("@@"):
            break
        if line.startswith("--- ") and a_path is None:
            value = line[4:].strip()
            a_path = None if value == "/dev/null" else value.removeprefix("a/")
        elif line.startswith("+++ ") and b_path is None:
            value = line[4:].strip()
            b_path = None if value == "/dev/null" else value.removeprefix("b/")

    if b_path or a_path:
        return b_path or a_path

    header = re.match(r"^diff --git a/(.+?) b/(.+)$", section.splitlines()[0] if section else "")
    return header.group(2) if header else None


def drop_patch_sections(patch: str, paths: Iterable[str]) -> str:
    """Drop the diff sections targeting ``paths``.

    ``git add -N . && git diff`` picks up every untracked file, including ones that were already
    untracked before the agent touched anything. Excluding those paths keeps the extracted patch
    to what the agent actually changed.
    """
    dropped = set(paths)
    if not patch or not dropped:
        return patch

    kept: list[str] = []
    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        if not section.strip():
            continue
        path = patch_section_path(section)
        if path is not None and path in dropped:
            continue
        kept.append(section)

    return "".join(kept)


def drop_test_patch_files(patch: str, test_patch: str) -> str:
    sections = re.split(r"(?=^diff --git )", test_patch, flags=re.MULTILINE)
    return drop_patch_sections(patch, {patch_section_path(section) for section in sections if section.strip()})


def _result_file_read_command(result_file: str) -> str:
    """Shell snippet printing a result file's content, or concatenated matches for a
    ``find:<base>:<path_glob>:<file_glob>`` pattern (not used by this dataset's 5 supported
    frameworks, but kept for parity with swemer_v1's identical helper).
    """
    if result_file.startswith("find:"):
        _, base, path_glob, file_glob = result_file.split(":", 3)
        return (
            f"find {shlex.quote(base)} -path {shlex.quote(path_glob)} -name {shlex.quote(file_glob)} "
            f"-exec cat {{}} \\; 2>/dev/null"
        )
    return f"cat {shlex.quote(result_file)} 2>/dev/null"


def build_eval_script(inputs: VerificationInputs) -> str:
    """The script run inside the sandbox.

    Patch order mirrors Scale-SWE: the candidate patch (implementation) first, then the
    dataset's held-out ``test_patch``. The two touch disjoint files in every sample inspected
    (implementation vs. test tree), so order has not been observed to matter, but this keeps the
    convention consistent across the SWE resources servers.
    """
    apply_patch = (
        "git apply --reject --recount --ignore-space-change --whitespace=nowarn /tmp/nemo_gym_patch.diff || true"
        if inputs.patch.strip()
        else ""
    )
    apply_test_patch = (
        "git apply --reject --recount --ignore-space-change --whitespace=nowarn "
        "/tmp/nemo_gym_test_patch.diff 2>&1 | tee /tmp/nemo_gym_test_patch.log\n"
        f"grep -q '^error: ' /tmp/nemo_gym_test_patch.log && echo {TEST_PATCH_FAILED}"
        if inputs.test_patch.strip()
        else ""
    )

    injected_command = get_test_command_with_output(inputs.test_command, inputs.test_framework)
    config = get_framework_config(inputs.test_framework, inputs.test_command)
    result_file = config.get("result_file")
    result_file_cmd = _result_file_read_command(result_file) if result_file else "true"

    return f"""#!/bin/bash
cd {shlex.quote(inputs.workdir)} || exit 97
set +e
mkdir -p /workspace/test-results 2>/dev/null
{apply_patch}
{apply_test_patch}

echo "{TEST_OUTPUT_BEGIN}"
{injected_command}
__test_exit=$?
echo "{TEST_OUTPUT_END}"
echo "{RESULT_FILE_BEGIN}"
{result_file_cmd}
echo "{RESULT_FILE_END}"
exit $__test_exit
"""


def _slice(log: str, begin: str, end: str) -> str:
    start = log.find(begin)
    if start == -1:
        return ""
    start += len(begin)
    stop = log.find(end, start)
    return log[start:stop] if stop != -1 else log[start:]


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    if inputs.patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = inputs.patch
    if inputs.test_patch.strip():
        files["/tmp/nemo_gym_test_patch.diff"] = inputs.test_patch
    return files


def grade(
    statuses: dict[str, str],
    fail_to_pass: Iterable[str],
    pass_to_pass: Iterable[str],
    test_framework: str = "",
) -> dict[str, Any]:
    """Resolved only when every required test is observed AND passing.

    A test absent from the output counts as not passing: treating "absent" as success is how a
    test command that never ran scores as a resolved instance.

    Exact id match first, ``normalize_test_id`` as a fallback -- this dataset's ids are dotted
    (``tests.pkg.test_mod::test_name``), not swe_bench_ext's own real-node-id convention, so an
    exact match rarely hits; the normalized fallback (delimiter unification) is what actually
    carries most rows. Confirmed for real: 149/149 rows across a stratified sample of the
    already-golden-patch-validated dataset resolved identically to the previous hand-rolled
    parsers with this fallback in place.
    """
    normalized_index: dict[str, str] | None = None

    def resolve_status(name: str) -> str | None:
        nonlocal normalized_index
        status = statuses.get(name)
        if status is not None:
            return status
        if normalized_index is None:
            normalized_index = {}
            for raw_name, raw_status in statuses.items():
                normalized_index[normalize_test_id(raw_name, test_framework)] = raw_status
        return normalized_index.get(normalize_test_id(name, test_framework))

    def split(names: Iterable[str]) -> tuple[list[str], list[str]]:
        passed, failed = [], []
        for name in names:
            (passed if resolve_status(name) == PASSED else failed).append(name)
        return passed, failed

    f2p_passed, f2p_failed = split(fail_to_pass)
    p2p_passed, p2p_failed = split(pass_to_pass)
    return {
        "FAIL_TO_PASS": {"success": f2p_passed, "failure": f2p_failed},
        "PASS_TO_PASS": {"success": p2p_passed, "failure": p2p_failed},
        "tests_observed": len(statuses),
        "resolved": not f2p_failed and not p2p_failed,
    }


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

    if result.return_code == 97:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error=f"workdir {inputs.workdir} not present in the image",
        )

    result_file_content = _slice(output, RESULT_FILE_BEGIN, RESULT_FILE_END)
    stdout_section = _slice(output, TEST_OUTPUT_BEGIN, TEST_OUTPUT_END)
    parse_target = result_file_content.strip() or stdout_section

    try:
        statuses = await asyncio.to_thread(parse_test_output, parse_target, inputs.test_framework)
        if not statuses and stdout_section.strip() and parse_target is not stdout_section:
            # The result file was empty/missing (e.g. the framework never got far enough to
            # write it) -- stdout may still carry a plain-text fallback report.
            statuses = await asyncio.to_thread(parse_test_output, stdout_section, inputs.test_framework)
    except Exception as exc:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=True,
            test_results=None,
            test_output=output,
            error=f"parse failure ({inputs.test_framework}): {exc}",
        )
    report = grade(statuses or {}, inputs.fail_to_pass, inputs.pass_to_pass, inputs.test_framework)
    test_patch_failed = TEST_PATCH_FAILED in output
    return VerificationResult(
        completed=not test_patch_failed,
        resolved=bool(report["resolved"]) and not test_patch_failed,
        patch_applied=True,
        test_results=report,
        test_output=output,
        error="held-out test patch did not apply" if test_patch_failed else None,
        test_patch_failed=test_patch_failed,
    )
