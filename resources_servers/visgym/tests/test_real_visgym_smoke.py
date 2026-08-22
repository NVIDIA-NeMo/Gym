# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from conftest import requires_visgym


@requires_visgym
@pytest.mark.parametrize("size", [5, 7, 9, 11])
def test_real_maze_2d_easy_reset_and_step(size: int) -> None:
    import gymnasium as gym

    env = gym.make("maze_2d/easy", maze_width=size, maze_height=size)
    obs, info = env.reset(seed=1234)

    assert obs is not None
    assert "distance" in info

    obs, reward, terminated, truncated, info = env.step("('move', 0)")

    assert obs is not None
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated or truncated), bool)
    assert "env_feedback" in info

    env.close()
