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
"""Grading for SWE-Gym rows: build the in-sandbox eval script, parse pytest output, decide resolution.

SWE-Gym (https://huggingface.co/datasets/SWE-Gym/SWE-Gym) is SWE-bench-format data: every row has a
prebuilt image (``docker.io/xingyaoww/sweb.eval.x86_64.<owner>_s_<repo>-<pr>``) with the repo checked
out at ``base_commit`` under ``/testbed`` and a conda env named ``testbed``. The eval script below is the
official SWE-bench harness recipe (``make_eval_script_list_py``): activate the env, apply the candidate
patch, re-run the repo's ``install`` step, reset the test files, apply the held-out test patch, run the
repo's pytest command on the touched test files, and reset the test files again.

Nothing in here touches the sandbox an agent works in. ``verification_files`` are delivered to a FRESH
sandbox created for grading, so the held-out tests and the golden patch never sit next to the model.
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from resources_servers.swe_gym.swebench_specs import (
    PASSING_STATUSES,
    normalize_test_id,
    parse_log_pytest,
    pytest_command,
    spec_for,
    touched_test_files,
)


REPO_DIRECTORY = "/testbed"
CONDA_ENV = "testbed"

TEST_OUTPUT_BEGIN = "___NEMO_GYM_SWE_GYM_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SWE_GYM_TEST_END___"
PATCH_FAILED = "___NEMO_GYM_SWE_GYM_PATCH_FAILED___"
TEST_PATCH_FAILED = "___NEMO_GYM_SWE_GYM_TEST_PATCH_FAILED___"

# Exit codes the script reserves for "the image is not what the row says it is"; both are infra
# verdicts (evaluation incomplete), never a 0 for the patch.
EXIT_NO_REPO = 97
EXIT_NO_CONDA = 96


def as_list(value: Any) -> list[str]:
    """Hub rows store FAIL_TO_PASS / PASS_TO_PASS either as a list or as its JSON string."""
    if value is None:
        return []
    if isinstance(value, str):
        import json

        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except ValueError:
            return [value]
        return [str(x) for x in parsed] if isinstance(parsed, list) else [str(parsed)]
    return [str(x) for x in value]


@dataclass
class VerificationInputs:
    instance_id: str
    repo: str
    version: str
    base_commit: str
    patch: str
    test_patch: str = ""
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
    for line in section.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                return parts[3][2:]
            return None
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                return None
            return target[2:] if target.startswith("b/") else target
    return None


def drop_patch_sections(patch: str, paths: Iterable[str]) -> str:
    """Remove the per-file sections of ``patch`` whose target is in ``paths``."""
    drop = set(paths)
    if not patch.strip() or not drop:
        return patch
    kept: list[str] = []
    for section in patch.split("diff --git "):
        if not section.strip():
            continue
        section = "diff --git " + section
        if patch_section_path(section) in drop:
            continue
        kept.append(section)
    return "".join(kept)


def drop_test_patch_files(patch: str, test_patch: str) -> str:
    """A model may not edit the held-out tests: its sections for those files are discarded before
    the test patch is applied, exactly as the SWE-bench harness resets them."""
    return drop_patch_sections(patch, [p for p in map(patch_section_path, _sections(test_patch)) if p])


def _sections(patch: str) -> list[str]:
    return ["diff --git " + s for s in patch.split("diff --git ") if s.strip()]


def build_eval_script(inputs: VerificationInputs) -> str:
    """The SWE-bench eval recipe as one bash script, run from a sandbox created for grading.

    The candidate patch is applied strictly (``git apply`` then ``patch --fuzz=5`` as the harness does);
    a patch that does not apply is a real verdict about the patch, reported through ``PATCH_FAILED``
    rather than by exiting, so the caller still gets the log. The install step is advisory (``set +e``)
    because on a prebuilt image it mostly re-confirms what is already there.
    """
    spec = spec_for(inputs.repo, inputs.version)
    test_files = touched_test_files(inputs.repo, inputs.test_patch)
    reset_tests = f"git checkout {shlex.quote(inputs.base_commit)} {' '.join(shlex.quote(f) for f in test_files)}"
    eval_commands = "\n".join(spec.get("eval_commands") or [])
    install = spec.get("install") or ""
    run_tests = pytest_command(inputs.repo, inputs.version, inputs.test_patch)

    apply_patch = ""
    if inputs.patch.strip():
        apply_patch = (
            "if ! git apply -v /tmp/nemo_gym_patch.diff; then\n"
            "    if ! patch --batch --fuzz=5 -p1 -i /tmp/nemo_gym_patch.diff; then\n"
            f"        echo {PATCH_FAILED}\n"
            "        exit 0\n"
            "    fi\n"
            "fi"
        )
    apply_test_patch = ""
    if inputs.test_patch.strip():
        apply_test_patch = (
            "git apply -v /tmp/nemo_gym_test_patch.diff 2>&1 | tee /tmp/nemo_gym_test_patch.log\n"
            f"grep -q '^error: ' /tmp/nemo_gym_test_patch.log && echo {TEST_PATCH_FAILED}"
        )

    return f"""#!/bin/bash
set +u
source /opt/miniconda3/bin/activate >/dev/null 2>&1 || exit {EXIT_NO_CONDA}
conda activate {CONDA_ENV} >/dev/null 2>&1 || exit {EXIT_NO_CONDA}
cd {REPO_DIRECTORY} || exit {EXIT_NO_REPO}
git config --global --add safe.directory {REPO_DIRECTORY} >/dev/null 2>&1
{eval_commands}
git -c core.fileMode=false status --short | head -20

{apply_patch}

set +e
{install}

{reset_tests}
{apply_test_patch}
echo "{TEST_OUTPUT_BEGIN}"
{run_tests}
__test_exit=$?
echo "{TEST_OUTPUT_END}"
{reset_tests}
exit $__test_exit
"""


def slice_test_output(log: str) -> str:
    """Only the region between the markers reaches the parser, so install noise cannot grade."""
    start = log.find(TEST_OUTPUT_BEGIN)
    if start == -1:
        return log
    start += len(TEST_OUTPUT_BEGIN)
    end = log.find(TEST_OUTPUT_END, start)
    return log[start:end] if end != -1 else log[start:]


def grade(statuses: dict[str, str], fail_to_pass: Iterable[str], pass_to_pass: Iterable[str]) -> dict[str, Any]:
    """SWE-bench resolution: every FAIL_TO_PASS and PASS_TO_PASS test observed and passing.

    A test missing from the output counts as failed; PASSED and XFAIL both count as passing. Ids are
    compared after ``normalize_test_id`` so pytest's escaped non-ASCII parameters match the dataset's.
    """

    observed = {normalize_test_id(name): status for name, status in statuses.items()}

    def split(names: Iterable[str]) -> tuple[list[str], list[str]]:
        passed, failed = [], []
        for name in names:
            (passed if observed.get(normalize_test_id(name)) in PASSING_STATUSES else failed).append(name)
        return passed, failed

    f2p_passed, f2p_failed = split(fail_to_pass)
    p2p_passed, p2p_failed = split(pass_to_pass)
    return {
        "FAIL_TO_PASS": {"success": f2p_passed, "failure": f2p_failed},
        "PASS_TO_PASS": {"success": p2p_passed, "failure": p2p_failed},
        "tests_observed": len(statuses),
        "resolved": not f2p_failed and not p2p_failed,
    }


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    """Files the grading sandbox is created with (one provider round trip, large patches included)."""
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    if inputs.patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = inputs.patch
    if inputs.test_patch.strip():
        files["/tmp/nemo_gym_test_patch.diff"] = inputs.test_patch
    return files


async def run_verification(
    sandbox: Any,
    inputs: VerificationInputs,
    timeout_s: float | None = None,
    log_dir: Path | None = None,
) -> VerificationResult:
    """Run the eval script in a sandbox already seeded with ``verification_files``."""
    result = await sandbox.exec("bash /tmp/nemo_gym_eval.sh", timeout_s=timeout_s)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

    if log_dir is not None:

        def _persist_log() -> None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "test_output.log").write_text(output, errors="replace")
            except OSError:
                pass

        await asyncio.to_thread(_persist_log)

    if result.return_code == EXIT_NO_REPO:
        return VerificationResult(False, False, False, None, output, f"{REPO_DIRECTORY} is not present in the image")
    if result.return_code == EXIT_NO_CONDA:
        return VerificationResult(False, False, False, None, output, f"conda env {CONDA_ENV!r} missing in the image")
    if PATCH_FAILED in output:
        # A verdict, not an infrastructure fault: the patch does not apply to base_commit.
        return VerificationResult(True, False, False, None, output, "patch does not apply")

    statuses = await asyncio.to_thread(parse_log_pytest, slice_test_output(output))
    report = grade(statuses, inputs.fail_to_pass, inputs.pass_to_pass)
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
