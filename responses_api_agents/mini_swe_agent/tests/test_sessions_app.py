# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi import HTTPException

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import entrypoint_ray_enabled
from responses_api_agents.mini_swe_agent.sessions_app import MiniSWESessionsAgent
from responses_api_agents.mini_swe_agent.tests.test_app import create_run_request, create_test_config


AGENT_DIR = Path(__file__).resolve().parents[1]


def test_the_session_template_starts_an_entrypoint_without_ray() -> None:
    template = yaml.safe_load((AGENT_DIR / "configs/mini_swe_agent_sandboxed.yaml").read_text())
    entrypoint = template["mini_swe_agent"]["responses_api_agents"]["mini_swe_agent"]["entrypoint"]

    assert entrypoint == "sessions_app.py"
    assert entrypoint_ray_enabled(AGENT_DIR / entrypoint) is False
    # The SWE-Gym /run entrypoint still has Gym start Ray.
    assert entrypoint_ray_enabled(AGENT_DIR / "app.py") is True


async def test_run_is_refused_before_ray() -> None:
    server = MiniSWESessionsAgent(config=create_test_config(env="docker"), server_client=MagicMock(spec=ServerClient))

    with patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote") as runner_ray_remote:
        with pytest.raises(HTTPException, match="needs Ray") as error:
            await server.run(create_run_request())
    assert error.value.status_code == 422
    runner_ray_remote.remote.assert_not_called()


def test_the_entrypoint_serves_agent_sessions() -> None:
    server = MiniSWESessionsAgent(config=create_test_config(env=None), server_client=MagicMock(spec=ServerClient))

    paths = {route.path for route in server.setup_webserver().routes}
    assert {"/v1/agent_sessions", "/v1/agent_sessions/close", "/ng-rollout/{rollout_id}/v1/responses"} <= paths
