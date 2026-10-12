# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.vision2code.app import VERIFIER_FIXTURE


async def test_verifier_fixture():
    results = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0.0, 1.0), determinism="stochastic")
    assert len(results) == 12
