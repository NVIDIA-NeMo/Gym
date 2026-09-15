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

"""Confirm the sandbox is running the Mathlib a benchmark's statements were written against.

This exists because the failure is otherwise invisible. Each Lean benchmark pins a different
Mathlib -- miniF2F v4.12.0, LeanCat v4.19.0, Formal Conjectures v4.33.1 -- and the sandbox
carries exactly one. Point a benchmark at the wrong one and it does not error at startup: it
fails individual tasks with ordinary-looking "unknown identifier" and "invalid field" errors,
indistinguishable at a glance from a model that could not do the problem. Measured on LeanCat
against a v4.12.0 sandbox: 36 of the 100 reference statements fail to compile *with their
`sorry` still intact*, so those score 0 no matter what the model writes and the run reports a
plausible number that is not a model result.

``/execute`` takes no project parameter and NeMo-Skills hardcodes the project path, so the
Mathlib version is fixed when the container starts. This check is what turns that from a silent
assumption into a loud one.
"""

import asyncio
import logging
import re
from typing import Any, Dict, Optional


LOG = logging.getLogger(__name__)

# `import Mathlib` is part of the probe deliberately: a bare version query would happily report a
# version from a sandbox that has no Mathlib at all and cannot state a single problem.
TOOLCHAIN_PROBE = "import Mathlib\n#eval Lean.versionString\n"

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+)\b")


def parse_lean_version(compiler_output: Dict[str, Any]) -> Optional[str]:
    """Pull the ``x.y.z`` Lean version out of the probe's output, or None if it did not run."""
    combined = f"{compiler_output.get('stdout', '')}\n{compiler_output.get('stderr', '')}"
    if "error:" in combined.lower():
        return None
    match = _VERSION_RE.search(combined)
    return match.group(1) if match else None


def normalise_version(value: Optional[str]) -> str:
    """``leanprover/lean4:v4.33.1`` and ``v4.33.1`` and ``4.33.1`` all mean the same thing."""
    return (value or "").removeprefix("leanprover/lean4:").lstrip("v")


class ToolchainCheck:
    """One-shot sandbox version probe, safe to await from every concurrent verify.

    Warns rather than raises: a run already in flight should not die on this, and the operator
    needs the message, not a traceback. The pre-flight equivalent is each benchmark's
    ``check_sandbox.py``.
    """

    def __init__(self, expected: str):
        self.expected = normalise_version(expected)
        self._checked = False
        self._lock: Optional[asyncio.Lock] = None

    async def run(self, sandbox_client: Any, expected_override: Optional[str] = None) -> None:
        if self._checked:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            # Guarded: with hundreds of concurrent verifies an unguarded probe would mean
            # hundreds of full Mathlib compiles just to read a version string.
            if self._checked:
                return
            self._checked = True

            result = await sandbox_client.execute_lean4(code=TOOLCHAIN_PROBE, timeout=600.0)
            found = parse_lean_version(result)
            want = normalise_version(expected_override) or self.expected

            if found is None:
                LOG.error(
                    "Could not determine the sandbox's Lean version -- `import Mathlib` did not compile. "
                    "Every task will fail for reasons unrelated to the model. Run check_sandbox.py "
                    "before trusting any number from this run."
                )
            elif want and found != want:
                LOG.error(
                    "SANDBOX MATHLIB MISMATCH: sandbox is Lean/Mathlib %s, these tasks target %s. "
                    "An older Mathlib does NOT fail loudly -- it fails individual statements with "
                    "ordinary-looking errors, so the run reports a plausible but meaningless number. "
                    "Scores from this run are not comparable to published ones.",
                    found,
                    want,
                )
            else:
                LOG.info("Lean sandbox toolchain verified: %s", found)
