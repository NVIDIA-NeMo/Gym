# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The exported verifier fixture: full, zero and malformed cases, through Gym's own fixture runner."""

import pytest

from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.agentic_if.app import VERIFIER_FIXTURE


@pytest.mark.asyncio
async def test_verifier_fixture_pins_both_reward_endpoints_and_rejects_a_malformed_request():
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
    )
    assert {result.kind: result.observed_rewards for result in results} == {
        "full_reward": (1.0,),
        "zero_reward": (0.0,),
        "malformed": (),
    }


@pytest.mark.asyncio
async def test_fixture_cases_regrade_instruction_following_and_keep_the_persisted_task_verdict():
    server = VERIFIER_FIXTURE.server_factory()
    cases = [line for line in VERIFIER_FIXTURE.cases_path.read_text().splitlines() if line]
    import json

    full = VERIFIER_FIXTURE.request_model.model_validate(json.loads(cases[0])["request"])
    result = await VERIFIER_FIXTURE.invoke(server, full)
    assert result.reward == 1.0 and result.mask_sample is False and result.evaluation_completed is True
    assert result.if_result.if_all == 1
    assert result.test_output == "fixture"  # the task backend's own fields are kept
    assert result.deliverables_dir == str(VERIFIER_FIXTURE.cases_path.parent / "full")
