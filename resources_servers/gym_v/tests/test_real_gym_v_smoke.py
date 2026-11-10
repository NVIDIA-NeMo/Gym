# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native simulator smoke; not a model rollout or full-suite qualification."""

import math

import numpy as np
import pytest


gym_v = pytest.importorskip("gym_v")


def test_real_frozenlake_reset_and_step() -> None:
    env = gym_v.make("Games/FrozenLake-v0", size=4, num_holes=3, tile_size=32)
    try:
        observations, _ = env.reset(seed=900000)
        assert observations["agent_0"].image is not None
        observations, rewards, terminated, truncated, _ = env.step({"agent_0": "[up]"})
        assert observations["agent_0"].image is not None
        assert math.isfinite(rewards["agent_0"])
        assert isinstance(terminated["__all__"], (bool, np.bool_))
        assert isinstance(truncated["__all__"], (bool, np.bool_))
    finally:
        env.close()


def test_real_frozenlake_invalid_action_is_normal_feedback() -> None:
    env = gym_v.make("Games/FrozenLake-v0", size=4, num_holes=3, tile_size=32)
    try:
        env.reset(seed=900000)
        observations, rewards, _, _, info = env.step({"agent_0": "not-an-action"})
        assert observations["agent_0"].image is not None
        assert math.isfinite(rewards["agent_0"])
        assert info["agent_0"]["invalid_action"]
    finally:
        env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("env_id", ["Spatial/Empty-v0", "Spatial/DoorKey-v0"])
async def test_spatial_invalid_action_counts_toward_horizon_without_mask(monkeypatch, env_id) -> None:
    from resources_servers.gym_v import app as gymv_app
    from resources_servers.gym_v.schemas import GymVCloseRequest, GymVSeedSessionRequest, GymVStepRequest
    from resources_servers.gym_v.tests.test_server import _request, _row, _server, _verify_request

    provider_make = gym_v.make
    server = _server(monkeypatch, None)
    monkeypatch.setattr(gymv_app.gym_v, "make", provider_make)
    row = _row(env_id=env_id, horizon_cap=2)
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=row))
    env = server.env_id_to_env[seeded.env_id]
    initial_native_steps = env.unwrapped._minigrid_env.unwrapped.step_count
    try:
        first = await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="not-an-action"))
        assert first.done is False
        assert "Invalid action" in first.obs[0].content[0]["text"]
        second = await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="not-an-action"))
        assert second.done is True
        assert second.horizon_terminated is True
        assert server.env_id_to_turn_count[seeded.env_id] == 2
        assert env.unwrapped._minigrid_env.unwrapped.step_count == initial_native_steps
    finally:
        await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))
    assert verified.reward == 0.0
    assert verified.mask_sample is False
