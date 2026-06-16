# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEvolve contrib helpers for running candidate programs through NeMo Gym."""

from openevolve_gym.openhands_evaluator import (
    OpenHandsEvaluationConfig,
    evaluate_candidate,
    parse_rollout_scores,
    render_openhands_candidate,
    run_gym_rollout,
)


__all__ = [
    "OpenHandsEvaluationConfig",
    "evaluate_candidate",
    "parse_rollout_scores",
    "render_openhands_candidate",
    "run_gym_rollout",
]
