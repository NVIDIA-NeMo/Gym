# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from benchmarks.biomlbench.environment import BioMLBenchSandboxEnvironment
from responses_api_agents.harbor_agent.tests.test_environment import _make_environment, _provider
from responses_api_agents.harbor_agent.tests.test_environment import _reset_fake_provider as _reset_fake_provider
from responses_api_agents.harbor_agent_general import app, skillsbench_app
from responses_api_agents.harbor_agent_general.app import HarborRunRequest
from responses_api_agents.harbor_agent_general.biomlbench_app import BioMLBenchHarborAgent
from responses_api_agents.harbor_agent_general.skillsbench_app import SkillsBenchHarborAgentConfig


ROOT = Path(__file__).resolve().parents[3]


def _task_environment(tmp_path: Path, task_id: str) -> None:
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "task.json").write_text(json.dumps({"task_id": task_id, "checksums": {}}))
    (environment / "biomlbench_task.py").write_text("# runtime\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("task_id", ["polarishub/tdcommons-herg", "manual/open-problems-label-projection"])
async def test_start_prepares_data_as_root_before_the_agent(tmp_path, task_id):
    _task_environment(tmp_path, task_id)
    env = _make_environment(tmp_path, environment_class=BioMLBenchSandboxEnvironment)
    await env.start(force_build=False)
    provider = _provider()
    assert set(provider.uploads) >= {"/opt/biomlbench/biomlbench_task.py", "/opt/biomlbench/task.json"}
    setup = [call for call in provider.exec_calls if call["user"] == "root"]
    commands = [call["command"] for call in setup]
    assert any("pip install --quiet awscli" in c for c in commands) == task_id.startswith("manual/open-problems-")
    prepare = next(command for command in commands if " prepare " in command)
    assert f"--task-id {task_id} --data-dir /private/data --public-dir /home/data" in prepare
    assert "--task-json /opt/biomlbench/task.json" in prepare
    # The agent can reach root through the sandbox's exec daemon, so no private split may stay.
    assert prepare.endswith("&& rm -rf /private/data")
    permissions = next(command for command in commands if "chown nonroot:nonroot" in command)
    assert "chmod -R a+rX,go-w /home/data" in permissions and "/logs/agent" in permissions
    assert "serve-validation --public-dir /home/data --port 5000" in commands[-1]
    assert provider.closed_handles == []
    await env.stop(delete=True)


@pytest.mark.asyncio
async def test_failed_preparation_stops_the_sandbox(tmp_path):
    _task_environment(tmp_path, "polarishub/tdcommons-herg")
    env = _make_environment(tmp_path, environment_class=BioMLBenchSandboxEnvironment)
    env.exec = AsyncMock(return_value=SimpleNamespace(return_code=1, stderr="checksums differ", stdout=""))
    with pytest.raises(RuntimeError, match="BioML-bench .* failed.*checksums differ"):
        await env.start(force_build=False)
    assert _provider().closed_handles == ["sbx-123"]


def _config() -> SkillsBenchHarborAgentConfig:
    merged = OmegaConf.merge(OmegaConf.load(ROOT / "benchmarks/biomlbench/harbor_opensandbox.yaml"))
    raw = OmegaConf.to_container(
        merged.biomlbench_harbor_agent.responses_api_agents.harbor_agent_general, resolve=True
    )
    return SkillsBenchHarborAgentConfig.model_validate(
        raw | {"name": "biomlbench_harbor_agent", "host": "127.0.0.1", "port": 8080}
    )


def test_profile_runs_terminus_2_in_the_biomlbench_environment():
    job = _config().build_job_config("polarishub__tdcommons-herg", "t0-r0")
    assert job.agents[0].name == "terminus-2"
    assert job.agents[0].model_name == "hosted_vllm/policy_model"
    assert job.environment.import_path == "benchmarks.biomlbench.environment:BioMLBenchSandboxEnvironment"
    assert job.datasets[0].task_names == ["polarishub__tdcommons-herg"]


def test_rollout_sampling_reaches_terminus_2_arguments(monkeypatch):
    global_config = {
        "policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.2", "port": 9010}}},
        "sandbox": {"opensandbox": {"connection": {"domain": "sandbox.example", "api_key": "synthetic-secret"}}},
    }
    monkeypatch.setattr(skillsbench_app, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(BioMLBenchHarborAgent, "base_url_for_run", lambda self, root, identity: root)
    agent = BioMLBenchHarborAgent.model_construct(config=_config())
    sandbox_config = json.loads(agent.config.harbor_worker_env["BIOMLBENCH_SANDBOX_CONFIG"])
    assert sandbox_config["opensandbox"]["connection"]["api_key"] == "synthetic-secret"
    assert "SKILLSBENCH_SANDBOX_CONFIG" not in agent.config.harbor_worker_env
    body = HarborRunRequest.model_validate(
        {
            "task_name": "polarishub__tdcommons-herg",
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "responses_create_params": {"input": [], "top_p": 0.9},
        }
    )
    body.responses_create_params.temperature = 0.6
    body.responses_create_params.max_output_tokens = 4096
    job = agent.config.build_job_config(body.task_name, "t0-r0")
    agent._configure_job_for_run(job, body)
    kwargs = job.agents[0].kwargs
    assert "responses_create_params" not in kwargs
    assert kwargs["temperature"] == 0.6
    assert kwargs["llm_call_kwargs"] == {"top_p": 0.9, "max_tokens": 4096}
    assert kwargs["api_base"] == "http://127.0.0.2:9010/v1"
