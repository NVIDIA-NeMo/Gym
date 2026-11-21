# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The committed verifier cases hold against the original graders through the bridge."""

import pytest

from nemo_gym.verification.verifier_fixture import exercise_verifier_fixture
from resources_servers.aga_visual_tasks.verifier_fixture import VERIFIER_FIXTURE


@pytest.mark.asyncio
async def test_committed_cases_cover_each_original_grader() -> None:
    results = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0.0, 1.0), determinism="stochastic")
    scored = {(result.name.split("_")[0], result.kind) for result in results}
    assert {
        (grader, kind) for grader in ("math", "mcqa", "string") for kind in ("full_reward", "zero_reward")
    } <= scored
    assert sum(result.kind == "malformed" for result in results) == 2
