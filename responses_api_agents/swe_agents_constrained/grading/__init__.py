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
"""Constraint grading core — single source of truth.

Constraint semantics (registries + injectable instruction text), deterministic
verifiers, trajectory grading, and the shaped reward formula::

    reward = task_reward * (1 + alpha * constraint_reward)

This package is self-contained (stdlib + pydantic only) so it can be imported
without the rest of nemo_gym. The agentic-if repo — where constraint
*development* happens (curation ledger, keep-list audits, generation and
calibration pipelines, dataset builders) — re-exports these modules via shims
rather than duplicating them; this package must never import from agentic-if.

Public grading surface:

  parse_trajectory   Responses-API output items -> typed trajectory steps
  grade_constraints  steps + [{type, params}] -> GradingResult (scope filtering,
                     injection awareness, N/A handling, partial credit)
  compute_reward     shaped multiplicative reward (see formula above)
  InjectionMode      where the constraint text was placed in the prompt
"""

from responses_api_agents.swe_agents_constrained.grading.if_format.constraints import InjectionMode
from responses_api_agents.swe_agents_constrained.grading.reward import compute_reward
from responses_api_agents.swe_agents_constrained.grading.verifiers.trajectory import (
    grade_constraints,
    parse_trajectory,
)

__all__ = [
    "InjectionMode",
    "compute_reward",
    "grade_constraints",
    "parse_trajectory",
]
