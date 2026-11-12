# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from responses_api_agents.osworld_agent.reward_modes import reward_from_score


@pytest.mark.parametrize(
    ("mode", "score", "terminal", "expected"),
    [
        ("binary", 1.0, "DONE", 1.0),
        ("binary", 0.75, "DONE", 0.0),
        ("raw", 0.75, "DONE", 0.75),
        ("signed_v1", 1.0, "DONE", 1.0),
        ("signed_v1", 0.75, "DONE", -1.0),  # partial credit is still a false DONE
        ("signed_v1", 0.0, "done", -1.0),
        ("signed_v1", 0.0, "FAIL", 0.0),
        ("signed_v1", 0.0, None, 0.0),  # horizon or policy stop
        ("signed_v1", 1.0, "FAIL", 1.0),  # OSWorld's infeasible evaluator: FAIL scores 1
    ],
)
def test_reward_from_score(mode, score, terminal, expected):
    assert reward_from_score(mode, score, terminal) == expected


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        reward_from_score("signed_v2", 1.0, "DONE")
