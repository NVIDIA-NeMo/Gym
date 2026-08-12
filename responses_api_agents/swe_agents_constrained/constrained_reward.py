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
"""Constraint grading + reward shaping for the constrained SWE-bench wrapper.

Free of nemo_gym / OpenHands imports so the shaping semantics are unit-testable
in any venv that can see the agentic-if checkout.
"""

import json
import logging
from typing import Any

from responses_api_agents.swe_agents_constrained.agentic_if_bridge import coerce_constraint_declarations


log = logging.getLogger(__name__)

DEFAULT_CONSTRAINT_ALPHA = 1.0  # FORMAT mode default (agentic-if reward.py)


def grade_and_shape(
    output_items: Any,
    metadata: dict[str, str],
    task_reward: float,
    default_alpha: float,
    grading_core: tuple,
) -> dict[str, Any]:
    """Grade a completed trajectory and shape the task reward.

    Returns the constraint response fields (including the final ``reward``) to
    overlay on the base verify response. Grading failures never crash the
    rollout: the task reward passes through unshaped and the error is recorded
    in ``violations``.
    """
    parse_trajectory, grade_constraints, compute_reward, injection_mode_cls = grading_core
    alpha = float(metadata.get("constraint_alpha", default_alpha))
    fields: dict[str, Any] = {
        "reward": task_reward,
        "reward_components": {"task": task_reward},
        "task_reward": task_reward,
        "constraint_reward": None,
        "constraint_graded": False,
        "constraint_alpha": alpha,
    }

    try:
        # Canonical rows carry a JSON string (Responses metadata is
        # Dict[str, str]); tolerate an already-parsed list from older files.
        raw = metadata.get("constraints", "[]") or "[]"
        if isinstance(raw, str):
            raw = json.loads(raw)
        constraints = coerce_constraint_declarations(raw)
        if not constraints:
            return fields
        steps = parse_trajectory(output_items)
        grading = grade_constraints(
            steps,
            constraints,
            injection_mode=injection_mode_cls(metadata.get("injection_mode", injection_mode_cls.SYSTEM_PROMPT)),
            injection_step=int(metadata.get("injection_step", 0)),
            grading_mode=metadata.get("grading_mode", "fraction"),
            step_aggregation=metadata.get("step_aggregation", "mean"),
        )
    except Exception as e:
        log.exception("Constraint grading failed; passing task reward through unshaped")
        fields["violations"] = [f"constraint grading error: {e}"]
        return fields

    fields.update(
        constraint_graded=grading.any_graded,
        constraint_results=grading.constraint_results,
        constraint_scores=grading.constraint_scores,
        constraint_applicable=grading.constraint_applicable,
        violations=grading.violations,
    )
    if grading.any_graded:
        fields["reward"] = compute_reward(task_reward, grading.reward, alpha=alpha).total
        fields["constraint_reward"] = grading.reward
        fields["reward_components"]["constraint"] = grading.reward
        for name, score in grading.constraint_scores.items():
            if grading.constraint_applicable.get(name):
                fields["reward_components"][f"constraint_{name}"] = score
    return fields
