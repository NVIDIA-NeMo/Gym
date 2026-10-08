# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.openair_congestion.client import choose_action, drive_episode


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
