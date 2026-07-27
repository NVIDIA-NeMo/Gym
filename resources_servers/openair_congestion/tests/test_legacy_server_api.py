# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
from openair_congestion import server
from openair_congestion.kpi_client import KpiScrapeError


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


class _FailingKpiEnv(_FakeLiveEnv):
    kpi_url = "http://internal-kpi-exporter.cluster.local:9101/metrics"

    def reset(self, **kwargs):
        del kwargs
        raise KpiScrapeError(f"kpi-exporter {self.kpi_url} unreachable")

    def step(self, episode_id, action):
        del episode_id, action
        raise KpiScrapeError(f"kpi-exporter {self.kpi_url} returned HTTP 503")


class _LeakingFailureEnv(_FakeLiveEnv):
    def reset(self, **kwargs):
        del kwargs
        raise RuntimeError("sampler failed under /private/run/secret.json")

    def step(self, episode_id, action):
        del episode_id, action
        raise RuntimeError("closed state at /private/run/secret.json")


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "payload"),
    [
        ("/reset", {}),
        ("/step", {"episode_id": "episode-1", "action": {"name": "noop", "arguments": {}}}),
    ],
)
async def test_kpi_errors_do_not_expose_internal_endpoint(route, payload):
    server._BUILD_REVISION = "test"
    app = server.create_app(_FailingKpiEnv())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(route, json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "error": "kpi_scrape_error",
        "message": "KPI source unavailable",
    }
    assert "internal-kpi-exporter" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "payload", "status", "expected"),
    [
        (
            "/reset",
            {},
            503,
            {"error": "env_unavailable", "message": "Environment unavailable"},
        ),
        (
            "/step",
            {"episode_id": "episode-1", "action": {"name": "noop", "arguments": {}}},
            409,
            {
                "error": "step_invalid_state",
                "message": "Episode is not in a step-able state",
            },
        ),
    ],
)
async def test_runtime_errors_do_not_expose_internal_details(route, payload, status, expected):
    server._BUILD_REVISION = "test"
    app = server.create_app(_LeakingFailureEnv())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(route, json=payload)

    assert response.status_code == status
    assert response.json()["detail"] == expected
    assert "/private/run" not in response.text
