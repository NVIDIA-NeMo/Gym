# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from openair_congestion.render import to_user_text
from openair_congestion.replay_env import ReplayEnv
from openair_congestion.schemas import SUPPORTED_REGIMES


def test_user_text_omits_generator_metadata():
    env = ReplayEnv(pool_size=1, max_steps_default=2)
    observation, meta = env.reset(
        seed=555,
        difficulty=0.95,
        regime_mix={"interference": 1.0},
        scenario_id="interference",
        tier="replay",
        max_steps=2,
    )
    env.close(meta.episode_id)

    text = to_user_text(observation)
    normalized = text.lower().replace("-", "_").replace(" ", "_")

    assert "difficulty" not in text.lower()
    assert "regime mix" not in text.lower()
    assert not any(regime in normalized for regime in SUPPORTED_REGIMES)
