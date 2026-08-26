# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from resources_servers.osworld.config import OSWorldResourcesServerConfig
from resources_servers.osworld.models import (
    OSWorldSeedSessionRequest,
    OSWorldStepRequest,
)
from resources_servers.osworld.session_manager import (
    CapacityUnavailableError,
    OSWorldSessionManager,
)


class FakeEnvironment:
    instances: list["FakeEnvironment"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.task: dict[str, Any] = {}
        self.step_calls = 0
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self, *, task_config: dict[str, Any]) -> dict[str, Any]:
        self.task = task_config
        return {
            "screenshot": b"initial",
            "instruction": task_config["instruction"],
        }

    def _get_obs(self) -> dict[str, Any]:
        return {
            "screenshot": b"observed",
            "instruction": self.task["instruction"],
        }

    def step(self, action: Any, pause: float):
        self.step_calls += 1
        return (
            {"screenshot": b"next", "instruction": self.task["instruction"]},
            0.25,
            action == "DONE",
            {"pause": pause},
        )

    def evaluate(self) -> float:
        return 0.75

    def close(self) -> None:
        self.closed = True


def make_config(tmp_path: Path, **overrides: Any) -> OSWorldResourcesServerConfig:
    payload: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 8080,
        "entrypoint": "app.py",
        "name": "osworld",
        "require_auth": False,
        "cache_dir": str(tmp_path / "cache"),
        "state_dir": str(tmp_path / "state"),
        "cleanup_orphans_on_start": False,
        "discover_workers": False,
        "workers": [
            {
                "name": "worker-a",
                "remote_host": "worker-a.example",
                "data_host": "10.0.0.8",
                "capacity": 1,
            }
        ],
    }
    payload.update(overrides)
    return OSWorldResourcesServerConfig(**payload)


@pytest.mark.asyncio
async def test_session_lifecycle_and_operation_idempotency(tmp_path: Path) -> None:
    FakeEnvironment.instances.clear()
    manager = OSWorldSessionManager(make_config(tmp_path), env_factory=FakeEnvironment)
    await manager.start()
    try:
        seeded = await manager.seed_session(
            "session-a",
            OSWorldSeedSessionRequest(
                task_config={
                    "id": "task-a",
                    "instruction": "Open settings",
                    "proxy": False,
                }
            ),
        )
        assert seeded.session_id == "session-a"
        assert seeded.worker == "worker-a"
        assert seeded.observation.instruction == "Open settings"

        env = FakeEnvironment.instances[0]
        assert env.kwargs["provider_name"] == "remote_docker"
        assert env.kwargs["provider_options"]["session_id"] == "session-a"

        request = OSWorldStepRequest(operation_id="operation-a", action="DONE", pause=0.1)
        first = await manager.step("session-a", request)
        second = await manager.step("session-a", request)
        assert first == second
        assert first.done is True
        assert env.step_calls == 1

        assert (await manager.observe("session-a")).instruction == "Open settings"
        assert (await manager.evaluate("session-a")).score == 0.75
        assert (await manager.session_status("session-a")).status == "ready"

        with pytest.raises(CapacityUnavailableError, match="all remote Docker workers are full"):
            await manager.seed_session(
                "session-b",
                OSWorldSeedSessionRequest(
                    task_config={
                        "id": "task-b",
                        "instruction": "Open a file",
                        "proxy": False,
                    }
                ),
            )

        assert await manager.close_session("session-a") is True
        assert env.closed is True
        assert (await manager.health())["sessions"] == 0
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_proxy_task_is_rejected_before_environment_creation(tmp_path: Path) -> None:
    FakeEnvironment.instances.clear()
    manager = OSWorldSessionManager(make_config(tmp_path), env_factory=FakeEnvironment)
    await manager.start()
    try:
        with pytest.raises(RuntimeError, match="task requires proxy"):
            await manager.seed_session(
                "session-proxy",
                OSWorldSeedSessionRequest(
                    task_config={
                        "id": "task-proxy",
                        "instruction": "Open a proxied page",
                        "proxy": True,
                    }
                ),
            )
        assert FakeEnvironment.instances == []
    finally:
        await manager.stop()
