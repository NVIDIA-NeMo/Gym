# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OSWorld responses-api agent.

Heavy dependencies (``ray``, ``desktop_env``) are mocked at the module
boundary so the suite runs on a login node without OSWorld installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.osworld_agent.app import (
    OSWorldAgent,
    OSWorldAgentConfig,
    OSWorldRunRequest,
    OSWorldVerifyResponse,
)


DEFAULT_OSWORLD_TASK: Dict[str, Any] = {
    "id": "test-task-001",
    "instruction": "Open Chrome and enable Do Not Track.",
    "snapshot": "chrome",
    "config": [],
    "evaluator": {"func": "exact_match"},
    "related_apps": ["chrome"],
}

DEFAULT_RUN_RESULT: Dict[str, Any] = {
    "reward": 1.0,
    "score": 1.0,
    "finished": True,
    "error": None,
    "steps": [
        {
            "step": 0,
            "model_text": "```python\npyautogui.click(100, 200)\n```",
            "actions": ["pyautogui.click(100, 200)"],
            "reward": 0.0,
            "done": False,
            "info": {},
        },
        {
            "step": 1,
            "model_text": "```DONE```",
            "actions": ["DONE"],
            "reward": 1.0,
            "done": True,
            "info": {},
        },
    ],
}


def make_config(**overrides: Any) -> OSWorldAgentConfig:
    base: Dict[str, Any] = dict(
        name="osworld_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        concurrency=1,
        provider_name="docker",
        headless=True,
        screen_width=1280,
        screen_height=800,
        require_a11y_tree=False,
        client_password="password",  # pragma: allowlist secret
        max_steps=3,
        max_trajectory_length=3,
        sleep_after_execution=0.0,
        cache_dir="cache",
        max_tokens=512,
        temperature=1.0,
        top_p=0.9,
    )
    base.update(overrides)
    return OSWorldAgentConfig(**base)


def make_run_request(
    osworld_task: Optional[Dict[str, Any]] = None,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> OSWorldRunRequest:
    metadata: Dict[str, Any] = {"task_id": "test-task-001", "domain": "chrome"}
    if osworld_task is not None:
        metadata["osworld_task"] = osworld_task
    if extra_metadata:
        metadata.update(extra_metadata)
    return OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=temperature, top_p=top_p
        ),
        verifier_metadata=metadata,
    )


def setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict):
    mock_client = MagicMock()
    mock_client.global_config_dict = {
        "policy_model_name": "test-policy",
        "policy_api_key": "test-key",  # pragma: allowlist secret
    }
    mock_load_from_global_config.return_value = mock_client
    mock_get_first_server_config_dict.return_value = {"host": "127.0.0.1", "port": 8000}


class TestApp:
    def test_sanity(self) -> None:
        OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))

    async def test_responses_not_implemented(self) -> None:
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        with pytest.raises(NotImplementedError):
            await agent.responses(NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=1.0, top_p=0.9))

    def test_endpoints_registration(self) -> None:
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        app = agent.setup_webserver()
        client = TestClient(app, raise_server_exceptions=False)

        # /v1/responses raises NotImplementedError -> 500 (not 404).
        resp = client.post("/v1/responses", json={"input": [], "temperature": 1.0, "top_p": 0.9})
        assert resp.status_code == 500

        # /run is registered (anything other than 404 satisfies registration).
        run_resp = client.post("/run", json={})
        assert run_resp.status_code != 404

    async def test_run_missing_task_returns_empty_response(self) -> None:
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        request = make_run_request(osworld_task=None)  # no osworld_task -> short-circuit
        response = await agent.run(request)
        assert isinstance(response, OSWorldVerifyResponse)
        assert response.reward == 0.0
        assert "osworld_error" in response.verifier_metadata
        assert "No 'osworld_task'" in response.verifier_metadata["osworld_error"]

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_successful_execution(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)

        # Mock the Ray-remote ``.options(...).remote(...)`` call chain.
        future = MagicMock()
        mock_remote.options.return_value.remote.return_value = future
        mock_to_thread.return_value = DEFAULT_RUN_RESULT

        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        request = make_run_request(osworld_task=DEFAULT_OSWORLD_TASK, temperature=0.7, top_p=0.95)

        response = await agent.run(request)

        assert isinstance(response, OSWorldVerifyResponse)
        assert response.reward == 1.0
        assert response.verifier_metadata["osworld_score"] == 1.0
        assert response.verifier_metadata["osworld_finished"] is True
        assert response.verifier_metadata["osworld_error"] is None
        # Two model steps -> two output messages. ``response.response`` is a
        # ``NeMoGymResponse`` Pydantic model (coerced from the dict in app.py),
        # so use attribute access.
        assert len(response.response.output) == 2
        # Per-request overrides win over the agent default.
        assert response.response.temperature == 0.7
        assert response.response.top_p == 0.95
        # Ray remote was dispatched exactly once with our task spec.
        mock_remote.options.assert_called_once()
        mock_remote.options.return_value.remote.assert_called_once()
        positional_args, _ = mock_remote.options.return_value.remote.call_args
        assert positional_args[0] == DEFAULT_OSWORLD_TASK

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_ray_failure_returns_empty_response(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.side_effect = RuntimeError("docker daemon unreachable")

        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        request = make_run_request(osworld_task=DEFAULT_OSWORLD_TASK)

        response = await agent.run(request)

        assert response.reward == 0.0
        assert "RuntimeError" in response.verifier_metadata["osworld_error"]
        assert "docker daemon unreachable" in response.verifier_metadata["osworld_error"]

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_partial_score_thresholds_to_zero(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Score < 1.0 -> reward 0.0 (matches gym's 0/1 reward convention)."""
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = {
            "reward": 0.0,
            "score": 0.4,
            "finished": False,
            "error": None,
            "steps": [],
        }

        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        response = await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        assert response.reward == 0.0
        assert response.verifier_metadata["osworld_score"] == 0.4
        assert response.verifier_metadata["osworld_finished"] is False
