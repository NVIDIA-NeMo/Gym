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
"""Run one SWE-rebench instance's tests in a sandbox and grade the result.

The dataset ships everything needed per row: a prebuilt image, the commands to install and to
test, and the name of the parser that reads the resulting log. So verification is: restore the
repo, apply the test patch and the candidate patch, install, test, parse, grade.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# The marker delimits the graded test output from the install noise that precedes it. Parsers
# are line-oriented and several will happily interpret dependency-resolver chatter as test
# results, so the log is sliced before parsing rather than handed over whole.
TEST_OUTPUT_BEGIN = "___NEMO_GYM_SWE_REBENCH_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SWE_REBENCH_TEST_END___"

PASSED = "PASSED"

# Test names in several ecosystems carry per-run timings, so the recorded expectation and the
# observed name differ only by a duration. Stripping them is what makes the comparison stable.
# Kept identical to the swe_agents harness's normalization so both graders agree.
_TIMING_PATTERNS = (
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
)


def normalize_test_name(name: str) -> str:
    for pattern in _TIMING_PATTERNS:
        name = pattern.sub("", name)
    return name.strip()


def as_command_list(value: Any) -> list[str]:
    """``install`` and ``test_cmd`` are a string in some rows and a list in others."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def repo_directory(repo: str) -> str:
    """Where the image checks the repo out: ``/<repo-name>``, e.g. ``elastic/synthetics`` -> ``/synthetics``."""
    return "/" + (repo.split("/", 1)[1] if "/" in repo else repo)


@dataclass
class VerificationInputs:
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str = ""
    install: Sequence[str] = field(default_factory=tuple)
    test_cmd: Sequence[str] = field(default_factory=tuple)
    log_parser: str = ""
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


def build_eval_script(inputs: VerificationInputs) -> str:
    """The script run inside the sandbox.

    Patches are applied with the same tolerant flags the existing SWE-rebench harness uses.
    ``|| true`` on the patch steps is deliberate: a partially applying patch must still reach
    the test phase, because the tests are the actual verdict — refusing to run them here would
    convert a recoverable patch into an unexplained infrastructure failure.
    """
    work_dir = repo_directory(inputs.repo)
    install_block = "\n".join(as_command_list(inputs.install))
    test_block = "\n".join(as_command_list(inputs.test_cmd))

    apply_patch = ""
    if inputs.patch.strip():
        apply_patch = (
            "git apply --reject --recount --ignore-space-change --whitespace=nowarn /tmp/nemo_gym_patch.diff || true"
        )
    apply_test_patch = ""
    if inputs.test_patch.strip():
        apply_test_patch = (
            "git apply --reject --recount --ignore-space-change --whitespace=nowarn "
            "/tmp/nemo_gym_test_patch.diff || true"
        )

    return f"""#!/bin/bash
cd {shlex.quote(work_dir)} || exit 97
git reset --hard HEAD >/dev/null 2>&1 || true
git checkout {shlex.quote(inputs.base_commit)} >/dev/null 2>&1 || true

{apply_patch}
{apply_test_patch}

# Install steps are advisory: several rows ship commands that fail harmlessly on an image that
# is already provisioned, and aborting there would mask an otherwise gradeable run.
set +e
{install_block}

echo "{TEST_OUTPUT_BEGIN}"
{test_block}
__test_exit=$?
echo "{TEST_OUTPUT_END}"
exit $__test_exit
"""


def slice_test_output(log: str) -> str:
    """Return only the region between the markers, or the whole log if they are absent."""
    start = log.find(TEST_OUTPUT_BEGIN)
    if start == -1:
        return log
    start += len(TEST_OUTPUT_BEGIN)
    end = log.find(TEST_OUTPUT_END, start)
    return log[start:end] if end != -1 else log[start:]


def grade(
    statuses: dict[str, str],
    fail_to_pass: Iterable[str],
    pass_to_pass: Iterable[str],
) -> dict[str, Any]:
    """An instance resolves only when every required test is observed AND passing.

    A missing test counts as not passing. Treating "absent" as success is the classic way a
    broken test command scores as a resolved instance.
    """
    observed = {normalize_test_name(name): status for name, status in statuses.items()}

    def split(names: Iterable[str]) -> tuple[list[str], list[str]]:
        passed, failed = [], []
        for raw in names:
            name = normalize_test_name(raw)
            (passed if observed.get(name) == PASSED else failed).append(raw)
        return passed, failed

    f2p_passed, f2p_failed = split(fail_to_pass)
    p2p_passed, p2p_failed = split(pass_to_pass)
    return {
        "FAIL_TO_PASS": {"success": f2p_passed, "failure": f2p_failed},
        "PASS_TO_PASS": {"success": p2p_passed, "failure": p2p_failed},
        "tests_observed": len(observed),
        "resolved": not f2p_failed and not p2p_failed,
    }


# Where Maven and Gradle look by default when running as root, which these images do.
MAVEN_SETTINGS_PATH = "/root/.m2/settings.xml"
GRADLE_INIT_PATH = "/root/.gradle/init.d/nemo_gym_mirror.gradle"


def _mirror_files() -> dict[str, str]:
    """Redirect Maven Central to its Google-hosted mirror.

    Measured on a 200-row pass: every JVM row that failed did so on
    ``repo.maven.apache.org ... Network is unreachable``, while crates.io, proxy.golang.org,
    registry.npmjs.org and packagist all fetched successfully from the same sandboxes. So egress
    works and Maven Central specifically does not. The mirror is a read-through copy on
    googleapis.com, which the Go module proxy already demonstrates is reachable here.

    Reused verbatim from responses_api_agents/swe_agents/maven_mirror/, where it was added for
    the same class of problem.
    """
    mirror_dir = Path(__file__).resolve().parents[2] / "responses_api_agents" / "swe_agents" / "maven_mirror"
    files: dict[str, str] = {}
    settings = mirror_dir / "settings.xml"
    init_gradle = mirror_dir / "init.gradle"
    if settings.exists():
        files[MAVEN_SETTINGS_PATH] = settings.read_text()
    if init_gradle.exists():
        files[GRADLE_INIT_PATH] = init_gradle.read_text()
    return files


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    """Files the sandbox must be created with.

    Delivered through ``SandboxSpec.files`` rather than written after start: the patches can be
    large, and a create-time file is one provider round trip instead of several.
    """
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    # Harmless for non-JVM rows, which never read them.
    files.update(_mirror_files())
    if inputs.patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = inputs.patch
    if inputs.test_patch.strip():
        files["/tmp/nemo_gym_test_patch.diff"] = inputs.test_patch
    return files


async def run_verification(
    sandbox: Any,
    inputs: VerificationInputs,
    parser: Callable[[str], dict[str, str]],
    timeout_s: float | None = None,
    log_dir: Path | None = None,
) -> VerificationResult:
    """Run the eval script in a sandbox already seeded with ``verification_files``."""
    result = await sandbox.exec("bash /tmp/nemo_gym_eval.sh", timeout_s=timeout_s)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

    if log_dir is not None:
        # Off the loop: this is a shared-filesystem write, and at high concurrency a few ms per
        # task on the event loop becomes a stall for every other in-flight verification.
        def _persist_log() -> None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "test_output.log").write_text(output, errors="replace")
            except OSError:
                pass

        await asyncio.to_thread(_persist_log)

    if result.return_code == 97:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error=f"repo directory {repo_directory(inputs.repo)} not present in the image",
        )

    try:
        # Also off the loop. The parsers are regex passes over the whole test log, so they are
        # CPU-bound; running them inline serialises every concurrent verification behind them.
        statuses = await asyncio.to_thread(parser, slice_test_output(output))
    except Exception as exc:  # a parser crash is a grading failure, never a silent zero
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=True,
            test_results=None,
            test_output=output,
            error=f"log parser {inputs.log_parser!r} raised: {exc}",
        )

    report = grade(statuses, inputs.fail_to_pass, inputs.pass_to_pass)
    return VerificationResult(
        completed=True,
        resolved=bool(report["resolved"]),
        patch_applied=True,
        test_results=report,
        test_output=output,
    )
