# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Bridge to the agentic-if instruction_pool verifier system.

The constraint verifiers, grading semantics, and reward formula live in the
agentic-if repo (single source of truth — see agentic-if
instruction_pool/rubrics/constraint_sets.py). This module locates the checkout,
puts it on sys.path, and re-exports the framework-free grading core:

  parse_trajectory   Responses-API output items -> typed trajectory steps
  grade_constraints  steps + [{type, params}] -> GradingResult (scope filtering,
                     injection awareness, N/A handling, partial credit)
  compute_reward     shaped multiplicative reward: task * (1 + alpha * constraint)

Checkout resolution order:
  1. AGENTIC_IF_REPO environment variable
  2. explicit path passed to ensure_agentic_if()
  3. sibling checkout next to the Gym repo root (../agentic-if)
"""

import os
import sys
from pathlib import Path
from typing import Optional


_GYM_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CHECKOUT = _GYM_ROOT.parent / "agentic-if"

_resolved_repo: Optional[Path] = None


def find_agentic_if_repo(repo_path: Optional[str] = None) -> Optional[Path]:
    """Return the agentic-if checkout root, or None if not found."""
    candidates = []
    if env_path := os.environ.get("AGENTIC_IF_REPO"):
        candidates.append(Path(env_path))
    if repo_path:
        candidates.append(Path(repo_path) if Path(repo_path).is_absolute() else _GYM_ROOT / repo_path)
    candidates.append(_DEFAULT_CHECKOUT)
    for candidate in candidates:
        if (candidate / "instruction_pool" / "rubrics" / "verifiers" / "trajectory.py").exists():
            return candidate.resolve()
    return None


def ensure_agentic_if(repo_path: Optional[str] = None) -> Path:
    """Put the agentic-if checkout on sys.path (idempotent) and return its root.

    Raises FileNotFoundError with remediation steps when no checkout is found.
    """
    global _resolved_repo
    if _resolved_repo is not None:
        return _resolved_repo
    repo = find_agentic_if_repo(repo_path)
    if repo is None:
        raise FileNotFoundError(
            "agentic-if checkout not found. The swerl_constrained server verifies constraints "
            "with the verifier registry in the agentic-if repo (instruction_pool/rubrics/). "
            "Clone it next to the Gym repo, or set AGENTIC_IF_REPO=/path/to/agentic-if, "
            "or set agentic_if_repo in the server config."
        )
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    _resolved_repo = repo
    return repo


def load_grading_core(repo_path: Optional[str] = None):
    """Import and return (parse_trajectory, grade_constraints, compute_reward, InjectionMode)."""
    ensure_agentic_if(repo_path)
    from instruction_pool.rubrics.if_format.constraints import InjectionMode
    from instruction_pool.rubrics.reward import compute_reward
    from instruction_pool.rubrics.verifiers.trajectory import grade_constraints, parse_trajectory

    return parse_trajectory, grade_constraints, compute_reward, InjectionMode


def coerce_constraint_declarations(raw: list) -> list[dict]:
    """Normalize metadata constraint declarations to the [{type, params}] schema.

    Accepts the current agentic-if schema ({"type": ..., "params": {...}}) and
    coerces legacy bare-string entries ("unified_diff") for older datasets.
    """
    declarations = []
    for entry in raw or []:
        if isinstance(entry, str):
            declarations.append({"type": entry, "params": {}})
        elif isinstance(entry, dict) and "type" in entry:
            declarations.append({"type": entry["type"], "params": entry.get("params") or {}})
        else:
            raise ValueError(f"Malformed constraint declaration: {entry!r}")
    return declarations
