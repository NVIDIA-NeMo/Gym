# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.openair_congestion.client import choose_action, drive_episode
from resources_servers.openair_congestion.openair_congestion.render import (
    to_policy_text,
)
from resources_servers.openair_congestion.openair_congestion.replay_env import (
    ReplayEnv,
)


def _observation(
    *,
    p99: int,
    fairness: float,
    sinr: float,
    bler: int,
) -> str:
    return (
        f"- Cell 0: DL PRB util p50=70%, p99={p99}%; "
        f"Jain fairness {fairness:.2f}; 0 SLA violation(s) in last 5s.\n"
        f"    UE 0 (5QI 9): delivered 5.0 Mbps, SINR {sinr:.1f} dB, "
        f"BLER {bler}%, buffer 100 kB."
    )


def test_choose_action_conditions_relief_on_visible_kpis():
    assert choose_action(_observation(p99=90, fairness=0.99, sinr=12.0, bler=5), 0)["name"] == ("set_ul_power_control")
    assert choose_action(_observation(p99=50, fairness=0.90, sinr=12.0, bler=5), 0) == {
        "name": "set_scheduler_policy",
        "arguments": {"cell_id": 0, "policy": "RR"},
    }
    assert choose_action(_observation(p99=50, fairness=0.99, sinr=-2.0, bler=30), 0) == {
        "name": "set_handover_trigger",
        "arguments": {"cell_id": 0, "a3_offset_db": -24.0, "ttt_ms": 0},
    }
    assert choose_action(_observation(p99=50, fairness=0.99, sinr=12.0, bler=5), 0) == {
        "name": "noop",
        "arguments": {},
    }
    assert choose_action(_observation(p99=90, fairness=0.99, sinr=12.0, bler=5), 1) == {
        "name": "noop",
        "arguments": {},
    }


def test_choose_action_uses_compact_t2_decision_contract():
    env = ReplayEnv(pool_size=1, max_steps_default=16)
    observation, meta = env.reset(
        seed=911117,
        difficulty=0.9,
        regime_mix={"interference": 1.0},
        scenario_id="interference",
        tier="T2",
        max_steps=16,
    )
    env.close(meta.episode_id)
    observation = observation.model_copy(
        update={
            "cells": [
                cell.model_copy(
                    update={
                        "ues": [
                            ue.model_copy(
                                update={
                                    "ue_id": cell.cell_id * 8 + ue.ue_id,
                                }
                            )
                            for ue in cell.ues
                        ]
                    }
                )
                for cell in observation.cells
            ]
        }
    )

    policy_text = to_policy_text(observation)
    action = choose_action(policy_text, 0)

    assert action == {
        "name": "set_prb_cap",
        "arguments": {
            "cell_id": 2,
            "target": "ue",
            "target_id": 19,
            "max_prb": 200,
        },
    }
    text_with_global_ue_history = policy_text.replace(
        "\nA|one_tool_call_or_noop",
        '\nL|set_prb_cap|{"cell_id":2,"max_prb":200,"target":"ue","target_id":19}|none\nA|one_tool_call_or_noop',
    )
    assert choose_action(text_with_global_ue_history, 0) == action


@pytest.mark.asyncio
async def test_drive_episode_closes_session_when_step_fails():
    calls: list[str] = []

    async def post(path: str, payload: dict) -> dict:
        del payload
        calls.append(path)
        if path == "/reset":
            return {
                "observation": "- Cell 0: p99=90%; 1 SLA violation",
                "info": {"episode_id": "episode-1", "seed": 7, "scenario_id": "test"},
            }
        if path == "/close":
            return {"ok": True, "already_closed": False, "summary": {}}
        raise RuntimeError("step transport failed")

    with pytest.raises(RuntimeError, match="step transport failed"):
        await drive_episode(post)

    assert calls == ["/reset", "/step", "/close"]


@pytest.mark.asyncio
async def test_drive_episode_preserves_completed_return_when_close_fails():
    calls: list[str] = []

    async def post(path: str, payload: dict) -> dict:
        calls.append(path)
        if path == "/reset":
            return {
                "observation": "- Cell 0: p99=50%; Jain fairness 1.00",
                "info": {"episode_id": "episode-1", "seed": 7, "scenario_id": "test"},
            }
        if path == "/step":
            return {
                "observation": "- Cell 0: p99=50%; Jain fairness 1.00",
                "reward": -0.25,
                "terminated": True,
                "truncated": False,
                "info": {"guardrail_accepted": True},
            }
        if path == "/close":
            raise RuntimeError("close transport failed")
        raise AssertionError((path, payload))

    assert await drive_episode(post) == pytest.approx(-0.25)
    assert calls == ["/reset", "/step", "/close"]
