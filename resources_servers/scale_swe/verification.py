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
"""Run one AweAI-Team/Scale-SWE instance's tests in a sandbox and grade the result.

Scale-SWE is 20,181 rows and entirely Python, so unlike the multi-language sets there is one
test runner (pytest) and one log format. That removes the parser-dispatch problem completely.

Each row carries its own image, an explicit ``workdir``, and a ``pre_commands`` string that
performs the checkout of ``parent_commit`` and scrubs the repo's git history so the fix commit
is not reachable. The failing tests arrive one of two ways -- a literal test file in
``f2p_script`` (92% of rows) or a patch in ``f2p_patch`` (69%) -- and many rows have both.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


TEST_OUTPUT_BEGIN = "___NEMO_GYM_SCALE_SWE_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SCALE_SWE_TEST_END___"

# Where f2p_script is written. The dataset's own FAIL_TO_PASS ids name this file, so the path
# is part of the data contract rather than a choice.
F2P_SCRIPT_NAME = "test_fail_to_pass.py"

PASSED = "PASSED"

_STATUS_WORDS = ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS")


def extract_statuses(log: str, target_ids: Iterable[str]) -> dict[str, str]:
    """Map each of ``target_ids`` to its pytest status, read from a ``-rA`` short summary.

    A naive ``STATUS <id>`` regex (``\\S+`` up to the first whitespace) is wrong here: this
    dataset parametrizes tests over literal address strings, so a real summary line looks like
    ``PASSED tests/test_parser.py::test_parse_address[No address here-None]`` -- the id itself
    contains spaces. Worse, ``FAILED``/``ERROR`` lines append `` - <reason>``, and some of the
    same ids ALSO contain a literal `` - `` inside their brackets (e.g.
    ``test_parse_address[2590 Elm Road NE - Warren, OH 44483-expected1]``), so no fixed
    delimiter reliably separates "the id" from "the reason" in general.

    Measured cost of getting this wrong: one row in a 200-row sample had 954/954 tests pass,
    with every PASSED line truncated at the first space by the naive parser -- 494 of that
    row's PASS_TO_PASS ids then read as "absent" (not passing), scoring a fully green run as a
    near-total regression.

    The fix sidesteps the ambiguity: rather than parsing an id out of free text, check whether
    the text after the status word STARTS WITH one of the ids we already know we need (the
    row's own FAIL_TO_PASS/PASS_TO_PASS). Real ids always end at a syntactic boundary --
    ``]`` for a parametrized id, an identifier character otherwise -- so a full-string
    ``startswith`` match is exact there; it is only ambiguous for two candidate ids where one
    is a literal prefix of the other (e.g. ``test_foo`` vs. ``test_foo_extra``), which trying
    the longest ids first resolves.
    """
    ids_by_length = sorted(set(target_ids), key=len, reverse=True)
    statuses: dict[str, str] = {}
    for line in log.splitlines():
        for status in _STATUS_WORDS:
            prefix = status + " "
            if not line.startswith(prefix):
                continue
            rest = line[len(prefix) :]
            for node_id in ids_by_length:
                if rest.startswith(node_id) and (len(rest) == len(node_id) or rest[len(node_id)] in " \t-"):
                    # Later occurrences win: pytest prints progress first and the summary
                    # last, and the summary is authoritative.
                    statuses[node_id] = status
                    break
            break
    return statuses


def as_id_list(value: Any) -> list[str]:
    """FAIL_TO_PASS / PASS_TO_PASS arrive as JSON-encoded strings in this dataset."""
    if not value:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return [str(item) for item in decoded] if isinstance(decoded, list) else []
    return [str(item) for item in value]


def unique_test_files(node_ids: Iterable[str]) -> list[str]:
    """Unique test files named by a set of node ids, order preserved.

    pytest is pointed at files rather than the ids themselves: a row can carry hundreds of ids
    across up to ~69 files, which risks the command-line length limit, and running the file and
    grading by id afterwards gives the same verdict without that fragility.
    """
    files: list[str] = []
    for node_id in node_ids:
        path = node_id.split("::", 1)[0].strip()
        if path and path not in files:
            files.append(path)
    return files


@dataclass
class VerificationInputs:
    instance_id: str
    workdir: str
    patch: str
    pre_commands: str = ""
    f2p_patch: str = ""
    f2p_script: str = ""
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


def _clean_commands(value: str) -> str:
    """``pre_commands`` ends with a literal backslash-n in the published data."""
    return value.replace("\\n", "\n").strip()


def build_eval_script(inputs: VerificationInputs) -> str:
    """The script run inside the sandbox.

    ``pre_commands`` runs FIRST and unconditionally: it performs the checkout of parent_commit
    and rewrites git history, so applying a patch before it would be undone immediately.
    """
    files = unique_test_files(list(inputs.fail_to_pass) + list(inputs.pass_to_pass))
    pytest_targets = " ".join(shlex.quote(path) for path in files)

    pre = _clean_commands(inputs.pre_commands)
    apply_patch = (
        "git apply --reject --recount --ignore-space-change --whitespace=nowarn /tmp/nemo_gym_patch.diff || true"
        if inputs.patch.strip()
        else ""
    )
    apply_f2p_patch = (
        "git apply --reject --recount --ignore-space-change --whitespace=nowarn /tmp/nemo_gym_f2p.diff || true"
        if inputs.f2p_patch.strip()
        else ""
    )
    # The script is uploaded to /tmp and copied in, rather than written into the repo before
    # pre_commands, because `git clean -fd` there would delete it.
    install_f2p_script = (
        f"cp /tmp/nemo_gym_f2p_script.py {shlex.quote(F2P_SCRIPT_NAME)}" if inputs.f2p_script.strip() else ""
    )

    return f"""#!/bin/bash
cd {shlex.quote(inputs.workdir)} || exit 97

# Checkout and history scrub, from the dataset. Non-fatal: a partial setup that still reaches
# the tests yields a real verdict, which is more informative than an unexplained infra error.
set +e
{pre}

{apply_patch}
{apply_f2p_patch}
{install_f2p_script}

echo "{TEST_OUTPUT_BEGIN}"
python -m pytest -rA --tb=no -p no:cacheprovider {pytest_targets}
__test_exit=$?
echo "{TEST_OUTPUT_END}"
exit $__test_exit
"""


def slice_test_output(log: str) -> str:
    start = log.find(TEST_OUTPUT_BEGIN)
    if start == -1:
        return log
    start += len(TEST_OUTPUT_BEGIN)
    end = log.find(TEST_OUTPUT_END, start)
    return log[start:end] if end != -1 else log[start:]


def grade(statuses: dict[str, str], fail_to_pass: Iterable[str], pass_to_pass: Iterable[str]) -> dict[str, Any]:
    """Resolved only when every required test is observed AND passing.

    A test absent from the output counts as not passing: treating "absent" as success is how a
    test command that never ran scores as a resolved instance.
    """

    def split(names: Iterable[str]) -> tuple[list[str], list[str]]:
        passed, failed = [], []
        for name in names:
            (passed if statuses.get(name) == PASSED else failed).append(name)
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
    """Files the sandbox is created with; see build_eval_script for why they live in /tmp."""
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    if inputs.patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = inputs.patch
    if inputs.f2p_patch.strip():
        files["/tmp/nemo_gym_f2p.diff"] = inputs.f2p_patch
    if inputs.f2p_script.strip():
        files["/tmp/nemo_gym_f2p_script.py"] = inputs.f2p_script
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
        # Off the loop: a shared-filesystem write per task stalls every concurrent verification.
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

    target_ids = list(inputs.fail_to_pass) + list(inputs.pass_to_pass)
    statuses = await asyncio.to_thread(extract_statuses, slice_test_output(output), target_ids)
    report = grade(statuses, inputs.fail_to_pass, inputs.pass_to_pass)
    return VerificationResult(
        completed=True,
        resolved=bool(report["resolved"]),
        patch_applied=True,
        test_results=report,
        test_output=output,
    )
