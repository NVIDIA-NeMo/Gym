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

"""Shared `proof_status` vocabulary and the sandbox-result mapping.

One vocabulary across the Lean benchmarks so a rollout dump means the same thing everywhere,
and so the distinction that matters -- a wrong proof versus broken infrastructure -- is made
once. A benchmark with a rule of its own adds a status next to these rather than redefining
them.
"""

import re
from typing import Any, Dict, Optional, Tuple


STATUS_COMPLETED = "completed"
STATUS_EMPTY_GENERATION = "empty_generation"
STATUS_BANNED_TOKENS = "banned_tokens"
STATUS_STATEMENT_MODIFIED = "statement_modified"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_TIMEOUT = "timeout"
STATUS_SANDBOX_ERROR = "sandbox_error"


def determine_proof_status(compiler_output: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Map a sandbox result onto a proof status and, when it failed, a one-line reason.

    A zero exit is not sufficient: a build that declares a ``sorry`` exits zero with only a
    warning, so the output is scanned for ``error:`` and ``sorry`` as well.

    ``error_type`` separates the sandbox failing to run the command from Lean rejecting the
    proof. A non-zero exit with no ``error_type`` is an ordinary compile error, which is what
    most wrong proofs look like and must not be reported as infrastructure trouble.
    """
    error_type = compiler_output.get("error_type")
    if error_type:
        if "timeout" in str(error_type).lower():
            return STATUS_TIMEOUT, "Lean compilation timed out."
        return STATUS_SANDBOX_ERROR, f"Sandbox reported {error_type!r}."

    stdout = compiler_output.get("stdout") or ""
    stderr = compiler_output.get("stderr") or ""
    combined = f"{stdout}\n{stderr}".lower()

    return_code = compiler_output.get("return_code", 0)
    # `timeout` exits 124 when it had to stop Lean, and 137 when Lean ignored TERM and was
    # killed -- which is also what an out-of-memory kill looks like, hence the hedged reason.
    # Without this a killed compile reads as an ordinary non-zero exit, i.e. a rejected proof.
    if return_code == 124:
        return STATUS_TIMEOUT, "Lean compilation timed out."
    if return_code == 137:
        return STATUS_TIMEOUT, "Lean was killed (timeout grace expired, or out of memory)."
    if return_code != 0:
        return STATUS_COMPILE_ERROR, "Lean rejected the proof."
    if "error:" in combined:
        return STATUS_COMPILE_ERROR, "Lean reported compilation errors."
    if re.search(r"\bsorry\b", combined) is not None:
        return STATUS_COMPILE_ERROR, "Lean reported a declaration that uses 'sorry'."

    return STATUS_COMPLETED, None
