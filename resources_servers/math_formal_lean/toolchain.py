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

"""Confirm the sandbox runs the Mathlib a benchmark's statements were written against.

Each Lean benchmark pins its own Mathlib, and a NeMo-Skills sandbox serves exactly one:
``/execute`` takes no project parameter, so the version is fixed when the container starts.
A mismatch does not error at startup; tasks fail with ordinary compile errors and the run
reports a plausible but meaningless score.

A server constructs a :class:`ToolchainCheck` with the version its rows expect and awaits
:meth:`ToolchainCheck.run` before its first compile. The probe runs once per process.
"""

import asyncio
import logging
import re
from typing import Any, Dict, Optional


LOG = logging.getLogger(__name__)

# `import Mathlib` is included so a sandbox without Mathlib fails the probe outright.
TOOLCHAIN_PROBE = "import Mathlib\n#eval Lean.versionString\n"

# A cold `import Mathlib` can take minutes; independent of the per-task compile timeout.
PROBE_TIMEOUT = 600.0

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+)\b")


def parse_lean_version(compiler_output: Dict[str, Any]) -> Optional[str]:
    """The ``x.y.z`` Lean version from the probe's output, or None if the probe did not run."""
    combined = f"{compiler_output.get('stdout', '')}\n{compiler_output.get('stderr', '')}"
    if "error:" in combined.lower():
        return None
    match = _VERSION_RE.search(combined)
    return match.group(1) if match else None


def normalize_version(value: Optional[str]) -> str:
    """``leanprover/lean4:v4.19.0``, ``v4.19.0`` and ``4.19.0`` all mean the same thing."""
    return (value or "").removeprefix("leanprover/lean4:").lstrip("v")


class ToolchainCheck:
    """One-shot sandbox version probe, safe to await from every concurrent verify.

    Logs an error rather than raising so a run in flight is not killed.
    """

    def __init__(self, expected: Optional[str]):
        self.expected = normalize_version(expected)
        self._checked = False
        self._lock: Optional[asyncio.Lock] = None

    async def run(self, sandbox_client: Any, expected_override: Optional[str] = None) -> None:
        """Probe once. ``expected_override`` is a row's own pin and beats the server default."""
        if self._checked:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            # Re-check under the lock so concurrent verifies share one probe.
            if self._checked:
                return
            self._checked = True

            result = await sandbox_client.execute_lean4(code=TOOLCHAIN_PROBE, timeout=PROBE_TIMEOUT)
            found = parse_lean_version(result)
            want = normalize_version(expected_override) or self.expected

            if found is None:
                LOG.error(
                    "Could not determine the sandbox's Lean version: `import Mathlib` did not compile. "
                    "Every task will fail for reasons unrelated to the model."
                )
            elif want and found != want:
                LOG.error(
                    "SANDBOX MATHLIB MISMATCH: sandbox is Lean/Mathlib %s, but these tasks are written "
                    "against %s. Statements may fail to compile regardless of the model, so scores from "
                    "this run are not comparable to published ones.",
                    found,
                    want,
                )
            else:
                LOG.info("Lean sandbox toolchain verified: %s", found)
