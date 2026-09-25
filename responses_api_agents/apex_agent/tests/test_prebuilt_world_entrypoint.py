# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.apex_agent import prebuilt_world_entrypoint as entrypoint


def test_startup_failure_includes_world_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment_log = tmp_path / "environment.log"
    environment_log.write_text("start.sh exited\n", encoding="utf-8")
    world_bundle = tmp_path / "world_bundle.txt"
    world_bundle.write_text("wiki-js did not listen on its port\n", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "WORLD_BUNDLE_LOG", world_bundle)

    async def healthy_gateway(_url: str, timeout_seconds: float) -> None:
        assert timeout_seconds == 1800

    monkeypatch.setattr(entrypoint, "wait_for_gateway", healthy_gateway)
    process = SimpleNamespace(returncode=1)

    with pytest.raises(RuntimeError, match="wiki-js did not listen on its port") as failure:
        asyncio.run(entrypoint.wait_for_startup(process, environment_log))
    assert "start.sh exited" in str(failure.value)


def test_gateway_timeout_includes_world_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment_log = tmp_path / "environment.log"
    environment_log.write_text("gateway startup\n", encoding="utf-8")
    world_bundle = tmp_path / "world_bundle.txt"
    world_bundle.write_text("populate failed\n", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "WORLD_BUNDLE_LOG", world_bundle)

    async def failed_gateway(_url: str, timeout_seconds: float) -> None:
        raise TimeoutError("Archipelago gateway did not become healthy")

    monkeypatch.setattr(entrypoint, "wait_for_gateway", failed_gateway)
    process = SimpleNamespace(returncode=None)

    with pytest.raises(TimeoutError, match="populate failed"):
        asyncio.run(entrypoint.wait_for_startup(process, environment_log))


def test_gateway_and_mcp_share_one_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment_log = tmp_path / "environment.log"
    environment_log.write_text("waiting for MCP\n", encoding="utf-8")
    world_bundle = tmp_path / "world_bundle.txt"
    world_bundle.write_text("app still starting\n", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "WORLD_BUNDLE_LOG", world_bundle)

    async def slow_gateway(_url: str, timeout_seconds: float) -> None:
        await asyncio.sleep(0.02)

    monkeypatch.setattr(entrypoint, "wait_for_gateway", slow_gateway)
    process = SimpleNamespace(returncode=None)

    with pytest.raises(TimeoutError, match="did not finish MCP startup") as failure:
        asyncio.run(entrypoint.wait_for_startup(process, environment_log, timeout_seconds=0.1))
    assert "app still starting" in str(failure.value)
