# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.gym_v.app import VERIFIER_FIXTURE


def test_exported_verifier_fixture():
    results = asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=[0.0, 1.0],
            determinism="unknown",
        )
    )
    assert [(r.name, r.observed_rewards) for r in results] == [
        ("solved", (1.0,)),
        ("positive_partial_failure", (0.0,)),
        ("missing_state", (0.0,)),
        ("missing_response", ()),
    ]
