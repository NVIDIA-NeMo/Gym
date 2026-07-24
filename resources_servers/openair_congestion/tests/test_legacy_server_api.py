# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
from openair_congestion import server


class _FakeLiveEnv:
    scenario_mode = "off"

    def n_episodes_live(self) -> int:
        return 0

    def close_all(self) -> None:
        return None

    def close(self, episode_id: str) -> dict:
        return {
            "ok": True,
            "n_steps": 3,
            "scenario": {
                "ok": True,
                "n_deliveries": 1,
                "work_dir": "/tmp/private-run",
                "snapshot_path": "/tmp/private-run/snapshot.json",
            },
        }


@pytest.mark.asyncio
async def test_health_does_not_enumerate_gpu_hardware(monkeypatch):
    server._BUILD_REVISION = "test"

    def _unexpected_subprocess(*args, **kwargs):
        raise AssertionError("health must not execute nvidia-smi")

    monkeypatch.setattr(server.subprocess, "check_output", _unexpected_subprocess)
    app = server.create_app(_FakeLiveEnv())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["gpu_status"] == {"required": False}


@pytest.mark.asyncio
async def test_close_response_redacts_internal_filesystem_paths():
    server._BUILD_REVISION = "test"
    app = server.create_app(_FakeLiveEnv())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/close", json={"episode_id": "episode-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["n_steps"] == 3
    assert body["summary"]["scenario"]["n_deliveries"] == 1
    assert "work_dir" not in body["summary"]["scenario"]
    assert "snapshot_path" not in body["summary"]["scenario"]
    assert "/tmp/private-run" not in response.text
