# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
)
from nemo_gym.environment.authoring import load_environment, materialize_single_task
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.single_agent_episode_types import ResponsesResourcesVerifyRequest, ResponsesVerificationInput
from resources_servers.environment_adapter.app import (
    EnvironmentAdapterResourcesServer,
    EnvironmentAdapterResourcesServerConfig,
    SandboxWorkspace,
)


ROOT = Path(__file__).parents[3]
HELLO_WORLD = ROOT / "environments" / "hello_world"


def make_server(*, trusted: bool = True) -> EnvironmentAdapterResourcesServer:
    config = EnvironmentAdapterResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="hello_world_resources_server",
        environment_root=str(HELLO_WORLD),
        runtime_image="ubuntu:24.04",
        sandbox_provider="sandbox",
        trusted_environment_code=trusted,
    )
    return EnvironmentAdapterResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def response() -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def test_requires_explicit_trust_for_local_verifier() -> None:
    with pytest.raises(ValueError, match="trusted_environment_code=true"):
        make_server(trusted=False)


@pytest.mark.asyncio
async def test_seed_verify_and_close_hello_world(monkeypatch: MonkeyPatch) -> None:
    server = make_server()
    task = materialize_single_task(load_environment(HELLO_WORLD))
    episode_id = EpisodeId(rollout_id="hello-rollout")

    async def download(_remote_path: str, local_path: Path) -> None:
        local_path.write_text("Hello from NeMo Gym!\n")

    sandbox = SimpleNamespace(
        serialize=AsyncMock(return_value={"sandbox_id": "sandbox"}),
        download=AsyncMock(side_effect=download),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    seed = await server.seed_session(
        request,
        ResourcesSeedSessionRequest(
            episode_id=episode_id,
            task_id=task.task_id,
            task_data=task.task_input.task_data,
        ),
    )

    assert seed.resources_session_id == "session"
    assert seed.sandbox_access.workdir == "/workspace"
    assert seed.sandbox_access.connection.descriptor == {"sandbox_id": "sandbox"}

    verification = await server.verify(
        request,
        ResponsesResourcesVerifyRequest(
            episode_id=episode_id,
            task_id=task.task_id,
            verification_input=ResponsesVerificationInput(
                responses_create_params=task.task_input.responses_create_params,
                response=response(),
            ),
        ),
    )

    assert verification.reward == 1.0
    sandbox.download.assert_awaited_once()

    closed = await server.close_session(
        request,
        ResourcesCloseSessionRequest(resources_session_id="session", episode_id=episode_id),
    )

    assert closed.resources_session_id == "session"
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_workspace_rejects_path_escape() -> None:
    workspace = SandboxWorkspace(SimpleNamespace(download=AsyncMock()), "/workspace")

    with pytest.raises(OSError, match="escapes"):
        await workspace.read_text("/workspace/../etc/passwd")
