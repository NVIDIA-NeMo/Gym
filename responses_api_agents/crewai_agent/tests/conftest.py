# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A lent task sandbox on the local provider, and the host's CrewAI install standing in for the sandbox runtime."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents import SandboxPythonRuntime
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess, SandboxSpec, register_provider
from responses_api_agents.crewai_agent import app, sandbox
from responses_api_agents.crewai_agent.app import CrewAIAgent
from responses_api_agents.crewai_agent.tests.fakes import LendableLocalProvider


@pytest.fixture
def runtime(tmp_path, monkeypatch) -> Path:
    """The host's CrewAI install stands in for the sandbox runtime; harness and session files go under tmp_path."""
    package = tmp_path / "runtime/harness/responses_api_agents/crewai_agent"
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=True))
    monkeypatch.setattr(sandbox, "SANDBOX_PYTHON", sys.executable)
    monkeypatch.setattr(sandbox, "_PACKAGE_DIR", str(package))
    monkeypatch.setattr(app, "SESSION_ROOT", str(tmp_path / "sessions"))
    return package


@pytest.fixture
async def lent_sandbox(tmp_path, monkeypatch):
    """A running task sandbox owned by the test (the Resources Server's role), with a task file in its workdir."""
    register_provider("lendable_local", LendableLocalProvider, override=True)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"lender": {"lendable_local": {}}})
    # The lendable provider shares the host's network as the local one does.
    monkeypatch.setattr(app, "assert_model_url_reachable_from_sandbox", lambda url, *, provider_name: None)
    monkeypatch.setattr(CrewAIAgent, "resolve_model_base_url", lambda self, *args, **kwargs: "http://model/v1")
    workdir = tmp_path / "task"
    workdir.mkdir()
    (workdir / "input.txt").write_text("from the task image\n")
    owner = AsyncSandbox(LendableLocalProvider())
    await owner.start(SandboxSpec(workdir=str(workdir)))
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="lender", descriptor=await owner.serialize()),
        workdir=str(workdir),
        env={"TASK_VAR": "from-the-owner"},
    )
    yield owner, access
    await owner.stop()
