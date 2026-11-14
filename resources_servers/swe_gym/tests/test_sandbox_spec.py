# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The spec of the task sandbox, built from the checked-in recipe's ``sandbox_config``."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.sandbox import SandboxResources, SandboxSpec, cpu_cap_env
from resources_servers.swe_gym import app


RECIPE = Path(__file__).parents[1] / "configs" / "swe_gym.yaml"
NAMED_CONFIGS = {"sandbox": {"opensandbox": {}, "default_metadata": {"sandbox-api": "sdk", "benchmark": "provider"}}}
BODY = app.SWEGymInstanceRequest.model_construct(
    instance_id="o" * 80, image_name="registry/swe-gym:1", language="python"
)
SERVER_ENV = app.TESTBED_ENV


class _RecordingSandbox:
    def __init__(self, provider: dict) -> None:
        self.provider = provider

    async def start(self, spec: SandboxSpec) -> None:
        self.spec = spec


def _config(sandbox_config: dict) -> app.SWEGymResourcesServerConfig:
    return app.SWEGymResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swe_gym",
        sandbox_provider="sandbox",
        sandbox_config=sandbox_config,
    )


def _spec(monkeypatch: pytest.MonkeyPatch, sandbox_config: dict) -> SandboxSpec:
    monkeypatch.setattr(app, "get_global_config_dict", lambda: NAMED_CONFIGS)
    monkeypatch.setattr(app, "AsyncSandbox", _RecordingSandbox)
    server = app.SWEGymResourcesServer(config=_config(sandbox_config), server_client=MagicMock(spec=ServerClient))
    sandbox = asyncio.run(server._create_sandbox(BODY, {"/tmp/task.txt": "task"}))
    assert sandbox.provider == {"opensandbox": {}}
    return sandbox.spec


def test_the_recipe_sandbox_config_builds_the_task_sandbox_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    (server,) = yaml.safe_load(RECIPE.read_text())["swe_gym_resources_server"]["resources_servers"].values()

    spec = _spec(monkeypatch, server["sandbox_config"])

    assert spec == SandboxSpec(
        image="registry/swe-gym:1",
        ttl_s=18000,
        ready_timeout_s=1200,
        workdir=app.REPO_DIRECTORY,
        env=SERVER_ENV | cpu_cap_env(8),
        files={"/tmp/task.txt": "task"},
        metadata={"sandbox-api": "sdk"}
        | {"benchmark": "swe-gym", "nemo_gym_agent": "swe_gym", "instance_id": "o" * 63, "language": "python"},
        resources=SandboxResources(cpu=8, memory_mib=30720, disk_gib=30),
        provider_options={"resource_requests": {"cpu": 1, "memory_mib": 4096, "disk_gib": 30}},
    )
    assert spec.env["OMP_NUM_THREADS"] == "8"


def test_the_config_env_wins_and_derive_cpu_env_turns_the_caps_off(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"OMP_NUM_THREADS": "1", "EXTRA": "x"}

    capped = _spec(monkeypatch, {"resources": {"cpu": 2}, "env": env})
    uncapped = _spec(monkeypatch, {"resources": {"cpu": 2}, "env": env, "derive_cpu_env": False})

    assert capped.env == SERVER_ENV | cpu_cap_env(2) | env
    assert uncapped.env == SERVER_ENV | env


def test_a_sandbox_config_key_the_server_does_not_read_is_refused() -> None:
    with pytest.raises(ValidationError, match="enforce_agent_no_network"):
        _config({"enforce_agent_no_network": True})
