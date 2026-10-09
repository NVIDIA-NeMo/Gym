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
"""Grading for R2E-Gym rows: build the in-sandbox eval script, parse pytest output, decide resolution.

R2E-Gym-Subset (https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) tasks are synthetic issues
over real commits in ten Python repos. Each row's image (``namanjain12/<repo>_final:<commit>``) holds
the repository at ``/testbed`` in its pre-fix state with its own venv on PATH, the held-out tests under
``/r2e_tests`` and the runner ``/testbed/run_tests.sh`` (untracked). The row's ``expected_output_json``
is the per-test status the fixed code produces.

The eval script mirrors R2E-Gym's ``LocalRuntime`` recipe (``run_local_evaluation.py`` +
``DockerRuntime.setup_env`` / ``_calculate_reward_r2e``): apply the candidate patch with ``git apply
--whitespace=fix`` excluding the image's untracked files, clear bytecode caches, move the runner and
the tests next to each other under ``/root`` (``/testbed/r2e_tests`` becomes a symlink), run the
runner, parse pytest's summary. Resolution is R2E-Gym's rule: the observed statuses must equal the
expected ones exactly -- same set of tests, same status for each.

Nothing in here touches the sandbox an agent works in: ``seed_session`` removes ``/r2e_tests`` and the
runner there, and grading runs in a FRESH sandbox from the same image where they are intact.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_DIRECTORY = "/testbed"
HIDDEN_TESTS_DIR = "/r2e_tests"
RUNNER_PATH = f"{REPO_DIRECTORY}/run_tests.sh"
ALT_DIR = "/root"

TEST_OUTPUT_BEGIN = "___NEMO_GYM_R2E_GYM_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_R2E_GYM_TEST_END___"
PATCH_FAILED = "___NEMO_GYM_R2E_GYM_PATCH_FAILED___"

# Exit codes the script reserves for "the image is not what the row says it is": infra verdicts
# (evaluation incomplete), never a 0 for the patch.
EXIT_NO_REPO = 97
EXIT_NO_TESTS = 96

# Everything R2E-Gym's harness hides from the agent (``SKIP_FILES``); older images ship the JSON
# sidecars inside /testbed, current ones only the runner and the tests. Removed from the AGENT sandbox
# at seed time -- they describe the fix and the hidden tests.
AGENT_HIDDEN_PATHS = (
    HIDDEN_TESTS_DIR,
    RUNNER_PATH,
    f"{REPO_DIRECTORY}/syn_issue.json",
    f"{REPO_DIRECTORY}/expected_test_output.json",
    f"{REPO_DIRECTORY}/execution_result.json",
    f"{REPO_DIRECTORY}/parsed_commit.json",
    f"{REPO_DIRECTORY}/modified_files.json",
    f"{REPO_DIRECTORY}/modified_entities.json",
    "/expected_test_output.json",
    "/run_tests.sh",
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def hide_from_agent_command() -> str:
    """Shell that removes the hidden-test artifacts from an agent sandbox (idempotent)."""
    return "rm -rf " + " ".join(AGENT_HIDDEN_PATHS)


@dataclass
class VerificationInputs:
    instance_id: str
    repo_name: str
    patch: str
    expected: dict[str, str] = field(default_factory=dict)


@dataclass
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None = None


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


def _sections(patch: str) -> list[str]:
    return ["diff --git " + s for s in patch.split("diff --git ") if s.strip()]


def drop_patch_sections(patch: str, paths: Iterable[str]) -> str:
    """Remove the per-file sections of ``patch`` whose target is in ``paths``."""
    drop = set(paths)
    if not patch.strip() or not drop:
        return patch
    return "".join(s for s in _sections(patch) if patch_section_path(s) not in drop)


# Repo-relative paths a candidate patch may never touch: the hidden tests' directory (the eval script
# links the real one in at this path) and the files R2E-Gym's harness hides from the agent.
HIDDEN_TEST_PREFIXES = ("r2e_tests/",)
HIDDEN_TEST_FILES = frozenset(
    {
        "r2e_tests",
        "run_tests.sh",
        "syn_issue.json",
        "expected_test_output.json",
        "execution_result.json",
        "parsed_commit.json",
        "modified_files.json",
        "modified_entities.json",
    }
)


def is_hidden_test_path(path: str | None) -> bool:
    return bool(path) and (path in HIDDEN_TEST_FILES or path.startswith(HIDDEN_TEST_PREFIXES))


def drop_hidden_test_sections(patch: str) -> str:
    """The R2E analogue of SWE-bench's test-file reset: a model may not add or edit anything under
    ``r2e_tests/`` or the runner, so those sections are discarded before the patch is applied."""
    if not patch.strip():
        return patch
    return "".join(s for s in _sections(patch) if not is_hidden_test_path(patch_section_path(s)))


def parse_expected(value: Any) -> dict[str, str]:
    """``expected_output_json`` as the Hub ships it (a JSON string) or already decoded."""
    if value is None:
        return {}
    if isinstance(value, str):
        return {str(k): str(v) for k, v in json.loads(value).items()} if value.strip() else {}
    return {str(k): str(v) for k, v in dict(value).items()}


def build_eval_script(inputs: VerificationInputs) -> str:
    """The R2E-Gym local-evaluation recipe as one bash script, run from a sandbox created for grading.

    The patch is applied strictly, as the harness does; a patch that does not apply is a real verdict
    about the patch and is reported through ``PATCH_FAILED`` so the caller still gets the log. The
    untracked files the image ships next to the repo (the runner, ``install.sh``) are excluded from
    ``git apply`` exactly as upstream excludes them, so a patch that happens to mention them still
    applies.
    """
    apply_patch = ""
    if inputs.patch.strip():
        apply_patch = f"""EXCLUDES=""
for f in $(git ls-files --others --exclude-standard); do EXCLUDES="$EXCLUDES --exclude=$f"; done
if ! git apply --whitespace=fix $EXCLUDES /tmp/nemo_gym_patch.diff; then
    echo {PATCH_FAILED}
    exit 0
fi"""

    return f"""#!/bin/bash
cd {REPO_DIRECTORY} || exit {EXIT_NO_REPO}
[ -d {HIDDEN_TESTS_DIR} ] && [ -f {RUNNER_PATH} ] || exit {EXIT_NO_TESTS}
git config --global --add safe.directory {REPO_DIRECTORY} >/dev/null 2>&1
git -c core.fileMode=false status --short | head -20

{apply_patch}

set +e
# R2E-Gym's setup_env: expose the venv tools, add the one extra package its runner expects, drop stale
# bytecode, and stage the runner and the tests under {ALT_DIR} with a symlink back into the repo.
ln -s {REPO_DIRECTORY}/.venv {ALT_DIR}/.venv 2>/dev/null
mkdir -p {ALT_DIR}/.local/bin
find {REPO_DIRECTORY}/.venv/bin -type f -executable -exec ln -sf {{}} {ALT_DIR}/.local/bin/ \\; 2>/dev/null
uv pip install chardet >/dev/null 2>&1 || true
find . -name '*.pyc' -delete 2>/dev/null
find . -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null
find {HIDDEN_TESTS_DIR} -name '*.pyc' -delete 2>/dev/null
find {HIDDEN_TESTS_DIR} -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null
mv {RUNNER_PATH} {ALT_DIR}/run_tests.sh
mv {HIDDEN_TESTS_DIR} {ALT_DIR}/r2e_tests
rm -rf {REPO_DIRECTORY}/r2e_tests
ln -s {ALT_DIR}/r2e_tests {REPO_DIRECTORY}/r2e_tests

echo "{TEST_OUTPUT_BEGIN}"
bash {ALT_DIR}/run_tests.sh
__test_exit=$?
echo "{TEST_OUTPUT_END}"
exit $__test_exit
"""


def slice_test_output(log: str) -> str:
    """Only the region between the markers reaches the parser."""
    start = log.find(TEST_OUTPUT_BEGIN)
    if start == -1:
        return log
    start += len(TEST_OUTPUT_BEGIN)
    end = log.find(TEST_OUTPUT_END, start)
    return log[start:end] if end != -1 else log[start:]


def decolor(text: str) -> str:
    return _ANSI.sub("", text).replace("\r", "")


def parse_log_pytest(log: str) -> dict[str, str]:
    """R2E-Gym's pytest parser: statuses from the ``short test summary info`` block, keyed by the
    ``::``-joined test path without the file (``Class.test_name``), ``" - <reason>"`` stripped."""
    log = decolor(log or "")
    if "short test summary info" not in log:
        return {}
    statuses: dict[str, str] = {}
    for line in log.split("short test summary info", 1)[1].strip().split("\n"):
        if "PASSED" in line:
            statuses[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return statuses


def _normalize(statuses: dict[str, str]) -> dict[str, str]:
    return {decolor(k).split(" - ")[0]: v for k, v in sorted(statuses.items())}


def grade(statuses: dict[str, str], expected: dict[str, str]) -> dict[str, Any]:
    """R2E-Gym's reward: observed statuses must equal the expected ones exactly.

    Same number of tests, every expected test present, every status identical. A missing test, an
    extra test, or any status change is a 0 -- the expected map was produced by the real fix, so
    anything else means the candidate behaves differently somewhere the tests can see.
    """
    observed = _normalize(statuses)
    wanted = _normalize(expected)
    missing = sorted(k for k in wanted if k not in observed)
    unexpected = sorted(k for k in observed if k and k not in wanted)
    mismatched = sorted(k for k in wanted if k in observed and observed[k] != wanted[k])
    resolved = bool(wanted) and len(observed) == len(wanted) and not missing and not unexpected and not mismatched
    return {
        "tests_expected": len(wanted),
        "tests_observed": len(observed),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": {k: {"expected": wanted[k], "observed": observed[k]} for k in mismatched},
        "resolved": resolved,
    }


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    """Files the grading sandbox is created with (one provider round trip)."""
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    if inputs.patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = inputs.patch
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
    if result.return_code == EXIT_NO_TESTS:
        return VerificationResult(
            False, False, False, None, output, f"{HIDDEN_TESTS_DIR} or {RUNNER_PATH} missing in the image"
        )
    if PATCH_FAILED in output:
        return VerificationResult(True, False, False, None, output, "patch does not apply")

    statuses = await asyncio.to_thread(parse_log_pytest, slice_test_output(output))
    report = grade(statuses, inputs.expected)
    return VerificationResult(
        completed=True,
        resolved=bool(report["resolved"]),
        patch_applied=True,
        test_results=report,
        test_output=output,
        error=None,
    )
