# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.opencode_multimodal_benchmarks.app import VERIFIER_FIXTURE


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", [VERIFIER_FIXTURE])
async def test_catalog_verifier_contract(fixture):
    await exercise_verifier_fixture(fixture, reward_range=(0, 1), determinism="unknown")
