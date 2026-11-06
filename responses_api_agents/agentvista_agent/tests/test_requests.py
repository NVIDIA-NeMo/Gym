# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate dispatch prerequisites without requiring the external benchmark runtime."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.agentvista_agent.app import AgentVistaAgent, AgentVistaConfig, AgentVistaRunRequest


def test_step_budget_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AgentVistaConfig(
            host="127.0.0.1",
            port=1234,
            entrypoint="app.py",
            name="fixture",
            source_root=str(tmp_path),
            dataset=str(tmp_path / "tasks.json"),
            image_folder=str(tmp_path),
            output_root=str(tmp_path / "output"),
            model_name="fixture",
            max_steps=0,
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )


@pytest.mark.asyncio
async def test_missing_rollout_identity_never_starts_worker(tmp_path: Path) -> None:
    marker = tmp_path / "agentvista_harness/worker.py"
    marker.parent.mkdir()
    marker.write_text("")
    config = AgentVistaConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="fixture",
        source_root=str(tmp_path),
        dataset=str(tmp_path / "tasks.json"),
        image_folder=str(tmp_path),
        output_root=str(tmp_path / "output"),
        model_name="fixture",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    agent = AgentVistaAgent(config=config, server_client=MagicMock(spec=ServerClient))
    body = AgentVistaRunRequest.model_validate(
        {
            "responses_create_params": {"input": "Act"},
            "verifier_metadata": {"task_id": "fixture", "source_sha256": "source"},
        }
    )
    with pytest.raises(HTTPException, match="_ng_rollout_id"):
        await agent.run(body)
    assert not (tmp_path / "output").exists()
