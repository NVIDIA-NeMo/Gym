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
"""Run one Swemer Agentic-v1 task's tests in a sandbox and grade the result.

Unlike swemer_v2 (which hand-rolls a parser per framework), grading here is delegated to
``responses_api_agents.swe_agents.swe_bench_ext`` -- the same output-flag injection and
test-result parsing the swe-bench-ext harness itself uses, covering the ~22 frameworks v1's raw
task_metadata.json spans (pytest/go/jest/junit/maven/cargo/mocha/ctest/gtest/vitest/xctest/...),
not just the 5 swemer_v2 supports. FAIL_TO_PASS/PASS_TO_PASS ids here are real node ids (pytest
slash-paths, JUnit ``classname::name``, ...) exactly as swe_bench_ext's own parsers key their
results, not swemer_v2's dotted-path convention -- there is no id-translation step.
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


# Where Maven and Gradle look by default when running as root, which these images do.
MAVEN_SETTINGS_PATH = "/root/.m2/settings.xml"
GRADLE_INIT_PATH = "/root/.gradle/init.d/nemo_gym_mirror.gradle"

TEST_OUTPUT_BEGIN = "___NEMO_GYM_SWEMER_V1_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SWEMER_V1_TEST_END___"
TEST_PATCH_FAILED = "___NEMO_GYM_SWEMER_V1_TEST_PATCH_FAILED___"
RESULT_FILE_BEGIN = "___NEMO_GYM_SWEMER_V1_RESULT_FILE_BEGIN___"
RESULT_FILE_END = "___NEMO_GYM_SWEMER_V1_RESULT_FILE_END___"

PASSED = "PASSED"

# frameworks.FRAMEWORK_CONFIGS minus bazel/jasmine (4 rows total across the whole v1 delivery
# set -- not worth a custom eval-script path for), matching what parse_test_output can actually
# dispatch to a real parser rather than falling through to its best-effort auto-detection.
SUPPORTED_FRAMEWORKS = frozenset(
    {
        "pytest",
        "unittest",
        "junit",
        "maven",
        "gtest",
        "cargo-nextest",
        "cargo",
        "go",
        "jest",
        "vitest",
        "mocha",
        "bun",
        "ctest",
        "cppunit",
        "bespoke_libgeos",
        "xctest",
        "testing",
        "busted",
        "luaunit",
        "telescope",
        "lust",
        "minitest",
        "tap",
        "tape",
        "hardhat",
    }
)


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


def _result_file_read_command(result_file: str) -> str:
    """Shell snippet printing a result file's content, or concatenated matches for a
    ``find:<base>:<path_glob>:<file_glob>`` pattern (junit/maven's surefire-reports layout).

    ``parse_junit_xml`` explicitly handles multiple concatenated ``<?xml ...?>`` documents (one
    per matched file), so a plain concatenation here is exactly what it expects.
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

    Patch order mirrors scale_swe/swemer_v2: candidate patch first, then the dataset's held-out
    ``test_patch``.
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


def mirror_files() -> dict[str, str]:
    """Redirect Maven Central to its Google-hosted mirror.

    Confirmed for real on this same cluster (see swe_rebench.verification.mirror_files): under
    concurrent load, JVM rows fail with a 429 from ``repo.maven.apache.org`` while crates.io,
    proxy.golang.org, registry.npmjs.org, and packagist all fetch fine from the same sandboxes.

    A local copy (``maven_mirror/``, not ``responses_api_agents/swe_agents/maven_mirror/``) since
    ``init.gradle`` here has a swemer_v1-specific fix: the original wraps registration in
    ``gradle.beforeSettings``, added in Gradle 6.8, which throws ``MissingMethodException`` at
    script-evaluation time on older Gradle -- failing the whole build outright regardless of
    whether dependencies would have resolved fine. Confirmed for real: 95 JVM rows failed with
    exactly this. Harmless for non-JVM rows, which never read these files.
    """
    mirror_dir = Path(__file__).resolve().parent / "maven_mirror"
    files: dict[str, str] = {}
    settings = mirror_dir / "settings.xml"
    init_gradle = mirror_dir / "init.gradle"
    if settings.exists():
        files[MAVEN_SETTINGS_PATH] = settings.read_text()
    if init_gradle.exists():
        files[GRADLE_INIT_PATH] = init_gradle.read_text()
    return files


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    files.update(mirror_files())
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

    Exact id match first, ``normalize_test_id`` as a fallback only -- not applied unconditionally,
    to avoid changing behavior for frameworks whose raw keys already match the dataset's ids
    cleanly. The fallback exists for cases like cargo-nextest's ``(N/M)`` progress-counter prefix:
    real dataset ids observed to bake this in verbatim (e.g. ``( 4/10) mod::test``), but the
    counter reflects PARALLEL completion order, not a stable per-test identity, so it can differ
    between the run that recorded FAIL_TO_PASS and any later run of the exact same test. Confirmed
    for real: several cargo rows showed "N tests run: N passed" in raw output while still grading
    as unresolved, because the counter prefix didn't match the one baked into the stored id.
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
        completed=True,
        resolved=bool(report["resolved"]) and not test_patch_failed,
        patch_applied=True,
        test_results=report,
        test_output=output,
        test_patch_failed=test_patch_failed,
    )
