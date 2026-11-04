# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from resources_servers.osworld.config import (
    OSWorldResourcesServerConfig,
    RemoteDockerWorkerConfig,
)
from resources_servers.osworld.models import OSWorldObservation


def test_direct_worker_requires_control_and_data_addresses() -> None:
    with pytest.raises(ValidationError, match="remote_host must not be empty"):
        RemoteDockerWorkerConfig(name="worker", data_host="127.0.0.1")

    with pytest.raises(ValidationError, match="data_host must not be empty"):
        RemoteDockerWorkerConfig(
            name="worker",
            remote_host="worker.example",
            data_host="",
        )


def test_static_server_requires_unique_workers() -> None:
    worker = {
        "name": "worker",
        "remote_host": "worker.example",
        "data_host": "10.0.0.8",
    }
    with pytest.raises(ValidationError, match="worker names must be unique"):
        OSWorldResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="osworld",
            discover_workers=False,
            workers=[worker, worker],
        )


def test_observation_serializes_binary_screenshot() -> None:
    observation = OSWorldObservation.from_observation(
        {
            "screenshot": b"\x89PNG\r\n",
            "accessibility_tree": {"role": "desktop"},
            "terminal": "ready",
            "instruction": "Open settings",
        }
    )

    assert base64.b64decode(observation.screenshot_b64) == b"\x89PNG\r\n"
    assert observation.accessibility_tree == {"role": "desktop"}
    assert observation.terminal == "ready"
    assert observation.instruction == "Open settings"
