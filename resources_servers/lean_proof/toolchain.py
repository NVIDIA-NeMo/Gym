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

Each Lean benchmark pins its own Mathlib, and a sandbox image carries exactly one:
A sandbox image carries one Mathlib build, so the version is fixed when the container starts.
A mismatch does not error at startup; tasks fail with ordinary compile errors and the run
reports a plausible but meaningless score.

``LeanSandbox.check_toolchain`` runs the probe once per process and compares what it finds
against the version a benchmark's rows expect.
"""

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
