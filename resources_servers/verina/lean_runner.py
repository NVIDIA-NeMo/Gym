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
"""Run Lean on a check file and read the verdict from its messages."""

import asyncio
import json
import os
import re
import resource
import signal
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from resources_servers.verina.lean_file import PASS_MARKER, CheckFile


# The axioms every Lean proof that uses classical reasoning may depend on. Anything else, such as `sorryAx`,
# `Lean.ofReduceBool` (from `native_decide`), or an axiom the solution declares, fails the proof.
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

_AXIOMS = re.compile(r"depends on axioms: \[(.*)\]", re.DOTALL)
_NO_AXIOMS = "does not depend on any axioms"


@dataclass
class LeanMessage:
    severity: str
    line: int
    text: str


@dataclass
class CheckOutcome:
    """What Lean reported about one check file."""

    timed_out: bool = False
    # Errors located before the test commands, i.e. in the task text or the solution.
    errors: list[LeanMessage] = field(default_factory=list)
    # Errors reported on test commands, such as a test that does not type-check or evaluates `sorry`.
    test_errors: list[LeanMessage] = field(default_factory=list)
    tests_passed: int = 0
    tests_total: int = 0
    # None when Lean printed no axiom list for the theorem (it does not exist or the file stopped early).
    axioms: list[str] | None = None

    @property
    def compiled(self) -> bool:
        return not self.timed_out and not self.errors

    @property
    def all_tests_passed(self) -> bool:
        return not self.timed_out and self.tests_total > 0 and self.tests_passed == self.tests_total

    @property
    def proof_verified(self) -> bool:
        """The file has no errors and the theorem uses only the standard axioms."""
        return self.compiled and self.axioms is not None and set(self.axioms) <= STANDARD_AXIOMS


def parse_messages(stdout: str) -> list[LeanMessage]:
    """Parse the JSON message lines that `lean --json` writes, skipping anything else."""
    messages = []
    for raw in stdout.splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or "severity" not in item:
            continue
        line = (item.get("pos") or {}).get("line", 0)
        messages.append(LeanMessage(severity=item["severity"], line=int(line), text=str(item.get("data", ""))))
    return messages


def read_outcome(check: CheckFile, messages: list[LeanMessage]) -> CheckOutcome:
    """Turn Lean's messages into an outcome, trusting only output located on the checker's own lines."""
    first_test_line = min(check.test_lines, default=check.axioms_line)
    outcome = CheckOutcome(tests_total=len(check.test_lines))
    passed = set()
    for message in messages:
        if message.severity == "error":
            if message.line < first_test_line:
                outcome.errors.append(message)
            elif message.line in check.test_lines:
                outcome.test_errors.append(message)
        if message.severity != "information":
            continue
        if message.line in check.test_lines:
            index = check.test_lines.index(message.line)
            if message.text.strip().strip('"') == f"{PASS_MARKER} {index}":
                passed.add(index)
        elif message.line == check.axioms_line:
            if _NO_AXIOMS in message.text:
                outcome.axioms = []
            else:
                found = _AXIOMS.search(message.text)
                if found:
                    outcome.axioms = [name.strip() for name in found.group(1).split(",") if name.strip()]
    # A test that printed its fail marker, or nothing at all, does not count.
    outcome.tests_passed = len(passed)
    return outcome


class LeanRunner:
    """Checks files with a local Lean toolchain, at most `max_concurrency` at a time."""

    def __init__(
        self,
        *,
        project_dir: Path,
        lean_path: str,
        env: dict[str, str],
        max_concurrency: int,
        memory_limit_bytes: int | None = None,
    ) -> None:
        self._project_dir = project_dir
        self._env = {**env, "LEAN_PATH": lean_path}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._memory_limit_bytes = memory_limit_bytes

    def _limit_resources(self) -> None:
        # RLIMIT_DATA counts the heap but not Mathlib's read-only memory-mapped .olean files, so it stops a solution
        # that allocates without bound (Lean aborts with bad_alloc) while leaving the import unaffected.
        if self._memory_limit_bytes is not None:
            resource.setrlimit(resource.RLIMIT_DATA, (self._memory_limit_bytes, self._memory_limit_bytes))
        # An aborted Lean process would otherwise leave a core file of several GB.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    async def check(self, check: CheckFile, *, timeout_s: float) -> CheckOutcome:
        async with self._semaphore:
            with tempfile.TemporaryDirectory(prefix="verina_") as tmp:
                path = Path(tmp) / "Check.lean"
                path.write_text(check.source)
                stdout, timed_out = await self._run(path, timeout_s)
        if timed_out:
            return CheckOutcome(timed_out=True, tests_total=len(check.test_lines))
        return read_outcome(check, parse_messages(stdout))

    async def _run(self, path: Path, timeout_s: float) -> tuple[str, bool]:
        # A new session lets a timeout kill Lean together with any process it started.
        process = await asyncio.create_subprocess_exec(
            "lean",
            "--json",
            str(path),
            cwd=self._project_dir,
            env=self._env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=self._limit_resources,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            return "", True
        return stdout.decode("utf-8", errors="replace"), False
