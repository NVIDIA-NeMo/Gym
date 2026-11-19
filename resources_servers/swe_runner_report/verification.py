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
"""Grading from the task's own test-runner report.

A task names the exact argv that ran its tests when the task was validated. ``{results}`` in an argument is replaced
by the report path and a bare ``{specs}`` argument by the task's spec files. The runner writes a JSON report there:

* ``vitest``: the Jest-compatible JSON reporter (``testResults[].assertionResults[]``; Jest's ``--json`` output has
  the same shape);
* ``mocha``: mocha's ``json`` reporter (``passes`` / ``failures`` / ``pending``).

A test id is ``<spec path relative to the workdir>::<full test name>``, the form recorded at validation, so ids match
without translation. A task resolves only when every FAIL_TO_PASS and PASS_TO_PASS test passed and the test command
exited 0 within its timeout: a report alone is not enough, because candidate code runs inside the test process and
can leave a passing report behind before failing or hanging.
"""

import asyncio
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence


ResultsFormat = Literal["vitest", "mocha"]

RESULTS_PATH = "/tmp/nemo_gym_runner_report/report.json"
MAX_REPORT_BYTES = 50_000_000
MAX_LISTED_TESTS = 50
NO_WORKDIR = 97
TIMEOUT_RETURN_CODES = (124, 137)

TEST_OUTPUT_BEGIN = "___NEMO_GYM_RUNNER_REPORT_TEST_OUTPUT_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_RUNNER_REPORT_TEST_OUTPUT_END___"
TEST_EXIT = "___NEMO_GYM_RUNNER_REPORT_TEST_EXIT___"
REPORT_BEGIN = "___NEMO_GYM_RUNNER_REPORT_BEGIN___"
REPORT_END = "___NEMO_GYM_RUNNER_REPORT_END___"
TEST_PATCH_FAILED = "___NEMO_GYM_RUNNER_REPORT_TEST_PATCH_FAILED___"
SETUP_FAILED = "___NEMO_GYM_RUNNER_REPORT_SETUP_FAILED___"


@dataclass
class VerificationInputs:
    instance_id: str
    workdir: str
    patch: str
    test_patch: str
    test_argv: Sequence[str]
    results_format: ResultsFormat
    spec_files: Sequence[str] = field(default_factory=tuple)
    setup_commands: Sequence[Sequence[str]] = field(default_factory=tuple)
    setup_timeout_s: int = 900
    test_timeout_s: int = 1200
    fail_to_pass: Sequence[str] = field(default_factory=tuple)
    pass_to_pass: Sequence[str] = field(default_factory=tuple)
    source_prefixes: Sequence[str] = field(default_factory=tuple)
    source_suffixes: Sequence[str] = field(default_factory=tuple)

    def replayed_patch(self) -> tuple[str, list[str]]:
        """The part of ``patch`` the verifier applies, and the paths it leaves out (see keep_editable_sections)."""
        return keep_editable_sections(
            self.patch,
            prefixes=self.source_prefixes,
            suffixes=self.source_suffixes,
            test_paths=patch_paths(self.test_patch),
        )


@dataclass
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None = None
    test_patch_failed: bool = False


# ------------------------------------------------------------------------------------------------ patches
def patch_section_path(section: str) -> str | None:
    """Repository-relative path one ``diff --git`` section writes (its new path for a rename)."""
    lines = section.splitlines()
    new_path = old_path = None
    for line in lines[1:]:
        if line.startswith("@@"):
            break
        if line.startswith("rename to "):
            new_path = line[len("rename to ") :].strip()
        elif line.startswith("+++ ") and line[4:].strip() != "/dev/null":
            new_path = line[4:].strip().removeprefix("b/")
        elif line.startswith("--- ") and line[4:].strip() != "/dev/null":
            old_path = line[4:].strip().removeprefix("a/")
    if new_path or old_path:
        return new_path or old_path
    header = re.match(r"^diff --git a/(.+?) b/(.+)$", lines[0] if lines else "")
    return header.group(2) if header else None


def _sections(patch: str) -> list[str]:
    return [s for s in re.split(r"(?=^diff --git )", patch or "", flags=re.MULTILINE) if s.strip()]


def patch_paths(patch: str) -> set[str]:
    return {path for section in _sections(patch) if (path := patch_section_path(section))}


def editable(path: str, prefixes: Sequence[str], suffixes: Sequence[str]) -> bool:
    return any(path.startswith(p) for p in prefixes) and any(path.endswith(s) for s in suffixes)


def keep_editable_sections(
    patch: str, *, prefixes: Sequence[str], suffixes: Sequence[str], test_paths: set[str]
) -> tuple[str, list[str]]:
    """Keep the sections that only touch editable source files; return (kept patch, dropped paths).

    Editable means under one of the task's source prefixes, with one of its source suffixes, and not a path the hidden
    test patch writes. Both sides of a rename must qualify. The verifier replays nothing else, so an edit to tests,
    dependency manifests or tooling cannot change what is graded.
    """
    kept, dropped = [], []
    for section in _sections(patch):
        path = patch_section_path(section)
        sources = {path}
        if renamed := re.search(r"^rename from (.+)$", section, flags=re.MULTILINE):
            sources.add(renamed.group(1).strip())
        if path and all(p and editable(p, prefixes, suffixes) and p not in test_paths for p in sources):
            kept.append(section)
        else:
            dropped.append(path or "<unknown>")
    return "".join(kept), dropped


# ------------------------------------------------------------------------------------------------ script
def expand_test_argv(argv: Sequence[str], spec_files: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for part in argv:
        if part == "{specs}":
            expanded.extend(spec_files)
        else:
            expanded.append(part.replace("{results}", RESULTS_PATH))
    return expanded


def build_eval_script(inputs: VerificationInputs) -> str:
    """Candidate patch, hidden test patch (must apply), setup, tests; then the exit code and report between markers."""
    q = shlex.quote
    report_dir = str(Path(RESULTS_PATH).parent)
    lines = [
        "#!/bin/bash",
        f"cd {q(inputs.workdir)} || exit {NO_WORKDIR}",
        "set +e",
        f"rm -rf {q(report_dir)} && mkdir -p {q(report_dir)}",
    ]
    if inputs.replayed_patch()[0].strip():
        lines.append(
            "git apply --whitespace=nowarn /tmp/nemo_gym_patch.diff >/tmp/nemo_gym_patch.log 2>&1 || "
            "git apply --reject --recount --whitespace=nowarn /tmp/nemo_gym_patch.diff >>/tmp/nemo_gym_patch.log 2>&1"
        )
    if inputs.test_patch.strip():
        lines.append(
            f"git apply --whitespace=nowarn /tmp/nemo_gym_test_patch.diff || {{ echo {TEST_PATCH_FAILED}; exit 0; }}"
        )
    for command in inputs.setup_commands:
        lines.append(
            f"timeout -k 10 {int(inputs.setup_timeout_s)} {shlex.join(command)} || {{ echo {SETUP_FAILED}; exit 0; }}"
        )
    lines += [
        f"echo {TEST_OUTPUT_BEGIN}",
        f"timeout -k 10 {int(inputs.test_timeout_s)} "
        f"{shlex.join(expand_test_argv(inputs.test_argv, inputs.spec_files))} 2>&1 | tail -c 100000",
        "__test_exit=${PIPESTATUS[0]}",
        f"echo {TEST_OUTPUT_END}",
        f"echo {TEST_EXIT}$__test_exit",
        f"echo {REPORT_BEGIN}",
        f"[ -f {q(RESULTS_PATH)} ] && [ ! -L {q(RESULTS_PATH)} ] && head -c {MAX_REPORT_BYTES} {q(RESULTS_PATH)}",
        "echo",
        f"echo {REPORT_END}",
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    patch = inputs.replayed_patch()[0]
    if patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = patch
    if inputs.test_patch.strip():
        files["/tmp/nemo_gym_test_patch.diff"] = inputs.test_patch
    return files


# ------------------------------------------------------------------------------------------------ grading
def _relative(name: str, workdir: str) -> str:
    prefix = workdir.rstrip("/") + "/"
    return name[len(prefix) :] if name.startswith(prefix) else name


def runner_statuses(report: object, results_format: ResultsFormat, workdir: str) -> dict[str, str] | None:
    """Status per test id from a runner report; None when the report does not have the runner's shape."""
    if not isinstance(report, dict):
        return None
    statuses: dict[str, str] = {}
    if results_format == "mocha":
        for key, status in (("pending", "pending"), ("failures", "failed"), ("passes", "passed")):
            tests = report.get(key, [])
            if not isinstance(tests, list):
                return None
            for test in tests:
                if not isinstance(test, dict):
                    return None
                path, title = test.get("file") or "", test.get("fullTitle") or test.get("title")
                if not isinstance(path, str) or not isinstance(title, str):
                    return None
                statuses[f"{_relative(path, workdir)}::{title}"] = status
        return statuses
    suites = report.get("testResults")
    if not isinstance(suites, list):
        return None
    for suite in suites:
        if not isinstance(suite, dict) or not isinstance(suite.get("assertionResults", []), list):
            return None
        path = suite.get("name") or ""
        if not isinstance(path, str):
            return None
        for test in suite.get("assertionResults", []):
            if not isinstance(test, dict):
                return None
            title, status = test.get("fullName") or test.get("title"), test.get("status")
            if not isinstance(title, str) or not isinstance(status, str):
                return None
            statuses[f"{_relative(path, workdir)}::{title}"] = status
    return statuses


def grade(
    statuses: dict[str, str] | None,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    *,
    returncode: int | None,
) -> dict[str, Any]:
    """Required tests must all be reported as passed, and the test command must have exited 0 in time."""
    timed_out = returncode in TIMEOUT_RETURN_CODES
    if statuses is None:
        return {
            "resolved": False,
            "classification": "tests_timed_out" if timed_out else "no_report",
            "test_returncode": returncode,
        }

    def split(names: Sequence[str]) -> dict[str, Any]:
        groups: dict[str, list[str]] = {"success": [], "failure": [], "missing": []}
        for name in names:
            status = statuses.get(name)
            groups["success" if status == "passed" else "missing" if status is None else "failure"].append(name)
        return {key: value[:MAX_LISTED_TESTS] for key, value in groups.items()} | {
            "counts": {key: len(value) for key, value in groups.items()}
        }

    f2p, p2p = split(fail_to_pass), split(pass_to_pass)
    all_passed = all(group["counts"]["success"] == sum(group["counts"].values()) for group in (f2p, p2p))
    resolved = all_passed and returncode == 0
    return {
        "resolved": resolved,
        "classification": "resolved"
        if resolved
        else "tests_timed_out"
        if timed_out
        else "unresolved"
        if not all_passed
        else "nonzero_exit",
        "test_returncode": returncode,
        "tests_observed": len(statuses),
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
    }


def _slice(log: str, begin: str, end: str) -> str | None:
    start = log.find(begin)
    if start == -1:
        return None
    start += len(begin)
    stop = log.find(end, start)
    return log[start : stop if stop != -1 else None]


def parse_eval_output(output: str, inputs: VerificationInputs) -> VerificationResult:
    """Turn the verifier script output into a result. Infrastructure problems are incomplete; candidate failures
    (setup broken by the patch, no report, failing tests, nonzero exit) are completed zeros."""
    test_output = _slice(output, TEST_OUTPUT_BEGIN, TEST_OUTPUT_END) or output
    if TEST_PATCH_FAILED in output:
        # The hidden test patch applies to the task's base tree and edits to its paths are never replayed.
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=bool(inputs.replayed_patch()[0].strip()),
            test_results=None,
            test_output=output,
            error="held-out test patch did not apply",
            test_patch_failed=True,
        )
    if SETUP_FAILED in output:
        return VerificationResult(
            completed=True,
            resolved=False,
            patch_applied=bool(inputs.replayed_patch()[0].strip()),
            test_results={"resolved": False, "classification": "setup_failed"},
            test_output=output,
        )
    exit_match = re.search(re.escape(TEST_EXIT) + r"(-?\d+)", output)
    if exit_match is None:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=bool(inputs.replayed_patch()[0].strip()),
            test_results=None,
            test_output=output,
            error="verifier script did not reach the test run",
        )
    report_text = (_slice(output, REPORT_BEGIN, REPORT_END) or "").strip()
    statuses = None
    if report_text:
        try:
            statuses = runner_statuses(json.loads(report_text), inputs.results_format, inputs.workdir)
        except ValueError:
            statuses = None
    report = grade(statuses, inputs.fail_to_pass, inputs.pass_to_pass, returncode=int(exit_match.group(1)))
    return VerificationResult(
        completed=True,
        resolved=bool(report["resolved"]),
        patch_applied=bool(inputs.replayed_patch()[0].strip()),
        test_results=report,
        test_output=test_output,
    )


async def run_verification(
    sandbox: Any,
    inputs: VerificationInputs,
    timeout_s: float | None = None,
    log_dir: Path | None = None,
) -> VerificationResult:
    result = await sandbox.exec("bash /tmp/nemo_gym_eval.sh", timeout_s=timeout_s)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

    if log_dir is not None:

        def persist() -> None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "test_output.log").write_text(output, errors="replace")
            except OSError:
                pass

        await asyncio.to_thread(persist)

    if result.error_type is not None:
        return VerificationResult(
            # Candidate code can hang the test command; that consumes its grading budget.
            completed=result.error_type == "timeout",
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error=f"Test execution did not complete: {result.error_type}",
        )
    if result.return_code == NO_WORKDIR:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error=f"workdir {inputs.workdir} not present in the image",
        )
    return await asyncio.to_thread(parse_eval_output, output, inputs)
