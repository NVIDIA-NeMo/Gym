# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI operations for agent-harness optimization."""

import asyncio
import json

from nemo_gym.harness_optimization import HarnessOptimizationRecipe, run_harness_optimization


def optimize(recipe: HarnessOptimizationRecipe) -> None:
    """Run a HarnessOptimizer against Gym's fixed evaluator."""

    result = asyncio.run(run_harness_optimization(recipe))
    if result is not None:
        print(json.dumps(result, indent=2))
