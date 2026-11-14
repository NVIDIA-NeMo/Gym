# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate dispatch prerequisites without requiring the external benchmark runtime."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
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
    marker = tmp_path / "infer.py"
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
        await agent._execute(body)
    assert not (tmp_path / "output").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["default", "opencode"])
@pytest.mark.parametrize("sampling", [None, {"temperature": 0.6, "top_p": 0.8, "max_output_tokens": 1234}])
async def test_selected_worker_receives_gym_endpoint_and_isolated_storage(
    tmp_path: Path, monkeypatch, harness: str, sampling
) -> None:
    (tmp_path / "infer.py").write_text("")
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
        harness=harness,
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    agent = AgentVistaAgent(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(AgentVistaAgent, "resolve_model_base_url", lambda *args: "http://gym/rollout/test/v1")
    started = []

    async def spawn(*command, **kwargs):
        started.append((command, kwargs))
        worker = json.loads(Path(command[-1]).read_text())
        assert worker["endpoint"] == "http://gym/rollout/test/v1"
        assert worker["harness"] == harness
        assert worker["temperature"] == (sampling["temperature"] if sampling else 0.0)
        assert worker["top_p"] == (sampling["top_p"] if sampling else 1.0)
        assert worker["output_tokens"] == (sampling["max_output_tokens"] if sampling else 8192)
        assert Path(kwargs["env"]["XDG_CACHE_HOME"]).is_relative_to(tmp_path)
        assert Path(kwargs["env"]["HOME"]).is_relative_to(tmp_path)
        Path(worker["output"], "result.json").write_text(
            json.dumps(
                {
                    "artifacts": worker["output"],
                    "mask_sample": False,
                    "verify": {"status": "FAIL", "reward": 0.0},
                    "agent_output": "wrong answer",
                }
            )
        )
        return SimpleNamespace(wait=AsyncMock(return_value=0), returncode=0)

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    body = AgentVistaRunRequest.model_validate(
        {
            "responses_create_params": {"input": "Act", **(sampling or {})},
            "_ng_rollout_id": "dispatch-test",
            "verifier_metadata": {"task_id": "fixture", "source_sha256": "source", "category_id": "test"},
        }
    )
    response = await agent._execute(body)
    worker_config = json.loads(Path(started[0][0][-1]).read_text())
    expected = "default_worker" if harness == "default" else "worker"
    assert started[0][0][-2] == "responses_api_agents.agentvista_agent.runtime." + expected
    assert response["verify"]["reward"] == 0 and not response["mask_sample"]
    assert worker_config["defer_verification"] is True
    assert not any(key.startswith("VERIFIER_") for key in started[0][1]["env"])


@pytest.mark.parametrize("harness", ["default", "opencode"])
def test_benchmark_include_selects_execution_harness(harness: str) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.config import GlobalConfigDictParser

    _, documents = GlobalConfigDictParser().load_extra_config_paths([f"benchmarks/agentvista_{harness}/config.yaml"])
    composed = OmegaConf.merge(*documents)
    assert composed.agentvista_agent.responses_api_agents.agentvista_agent.harness == harness
